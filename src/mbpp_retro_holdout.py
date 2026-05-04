"""
R5: MBPP retro-holdout redesign.

Original MBPP perturbation = paraphrase of prompt, which over-clarifies and
produces a negative gap. This redesign uses:
  - orig_prompt unchanged (no clarification bias)
  - pert_name (different function name)
  - pert_tests (rewritten with the new name)

So we test: does the model complete correctly even when the function name
it memorized is no longer present? This is a structural, identifier-level
perturbation that preserves difficulty.

Runs on the 7 frontier models already used elsewhere.
"""
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent.parent / "data"
PERT_DIR = DATA_DIR / "perturbations"
EVAL_DIR = DATA_DIR / "evaluations"
OUT_FILE = EVAL_DIR / "mbpp_retro_results.json"

MODELS_GPT = [
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-08-06",
    "gpt-4.1-2025-04-14",
]
MODELS_CLAUDE = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
]

N_ITEMS = 100  # pilot


def build_prompt(prompt_text: str, fn_name: str, tests: list[str]) -> str:
    """Build a completion prompt that forces the target function name."""
    example_test = tests[0] if tests else ""
    return (
        f"Solve the following Python programming task. Your solution must define "
        f"a function named `{fn_name}` that passes the example assertion.\n\n"
        f"Task: {prompt_text}\n\n"
        f"Example test: {example_test}\n\n"
        f"Output only the Python function definition, no markdown fences, "
        f"no explanation:"
    )


def openai_complete(model: str, prompt: str) -> str | None:
    from openai import OpenAI
    client = OpenAI(timeout=60.0, max_retries=2)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        return _strip_fences(text)
    except Exception as e:
        print(f"    OpenAI {model} error: {str(e)[:100]}", flush=True)
        return None


def claude_complete(model: str, prompt: str) -> str | None:
    from anthropic import Anthropic
    client = Anthropic(timeout=60.0, max_retries=2)
    try:
        resp = client.messages.create(
            model=model, max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        return _strip_fences(text)
    except Exception as e:
        print(f"    Claude {model} error: {str(e)[:100]}", flush=True)
        return None


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:python)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text


def execute_solution(code: str, tests: list[str], imports: list[str], timeout_s: int = 5) -> bool:
    """Run code + tests in a subprocess. Returns True if all tests pass."""
    full = ""
    for imp in imports or []:
        full += imp + "\n"
    full += code + "\n\n"
    for t in tests:
        full += t + "\n"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(full)
            path = f.name
        r = subprocess.run(
            ["python3", path],
            capture_output=True, timeout=timeout_s, text=True,
        )
        os.unlink(path)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        try: os.unlink(path)
        except Exception: pass
        return False
    except Exception:
        return False


def eval_one(model, complete_fn, item, which):
    """Evaluate one item. which ∈ {'orig', 'retro'}.
    - orig: use orig_prompt, orig_name, orig_tests
    - retro: use orig_prompt (unchanged!), pert_name, pert_tests
    """
    if which == "orig":
        prompt = item["orig_prompt"]
        name = item["orig_name"]
        tests = item["orig_tests"]
    else:
        prompt = item["orig_prompt"]   # SAME prompt (preserves difficulty)
        name = item["pert_name"]
        tests = item["pert_tests"]
    imports = item.get("orig_test_imports", [])
    full_prompt = build_prompt(prompt, name, tests)
    code = complete_fn(model, full_prompt)
    if code is None:
        return None
    # Check function name appears in the completion
    if f"def {name}" not in code:
        return 0
    return 1 if execute_solution(code, tests, imports) else 0


def run_model(model, complete_fn, items, label):
    print(f"\n=== {label}: {model} ===", flush=True)
    results = {"orig": [], "retro": []}
    for which in ("orig", "retro"):
        scores = []
        for i, it in enumerate(items):
            s = eval_one(model, complete_fn, it, which)
            scores.append(s)
            if (i + 1) % 20 == 0:
                valid = [x for x in scores if x is not None]
                acc = sum(valid) / max(1, len(valid))
                print(f"  [{which}] {i+1}/{len(items)}  acc={acc:.3f}", flush=True)
        results[which] = scores
    valid_o = [x for x in results["orig"] if x is not None]
    valid_r = [x for x in results["retro"] if x is not None]
    o_acc = sum(valid_o) / max(1, len(valid_o))
    r_acc = sum(valid_r) / max(1, len(valid_r))
    gap = o_acc - r_acc
    print(f"  {model}: orig={o_acc:.3f} retro={r_acc:.3f} gap={gap:+.3f}", flush=True)
    return results, gap


def main():
    with open(PERT_DIR / "mbpp_perturbed.json") as f:
        items = json.load(f)[:N_ITEMS]
    existing = {}
    if OUT_FILE.exists():
        existing = json.load(open(OUT_FILE))

    # Run OpenAI
    for m in MODELS_GPT:
        if m in existing and len([x for x in existing[m].get("orig", []) if x is not None]) > 50:
            print(f"Skip {m} (done)", flush=True)
            continue
        res, _ = run_model(m, openai_complete, items, "MBPP-retro GPT")
        existing[m] = res
        with open(OUT_FILE, "w") as f:
            json.dump(existing, f, indent=2)

    # Run Claude
    for m in MODELS_CLAUDE:
        if m in existing and len([x for x in existing[m].get("orig", []) if x is not None]) > 50:
            print(f"Skip {m} (done)", flush=True)
            continue
        res, _ = run_model(m, claude_complete, items, "MBPP-retro Claude")
        existing[m] = res
        with open(OUT_FILE, "w") as f:
            json.dump(existing, f, indent=2)

    # Summary
    print("\n=== Summary ===")
    print(f"{'Model':<30} {'orig':>8} {'retro':>8} {'gap':>8}")
    for m, r in existing.items():
        o = [x for x in r["orig"] if x is not None]
        p = [x for x in r["retro"] if x is not None]
        if not o or not p: continue
        o_acc = sum(o) / len(o)
        p_acc = sum(p) / len(p)
        print(f"{m:<30} {o_acc:>8.3f} {p_acc:>8.3f} {o_acc - p_acc:>+8.3f}")


if __name__ == "__main__":
    main()
