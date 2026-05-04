"""
Run TruthfulQA evaluation on all 4 GPT + 3 Claude models.
Using the new Claude-generated perturbations.
"""
import json
import os
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from openai import OpenAI
from claude_eval import client as anthropic_client, LETTERS, _clean_choice

sys.stdout.reconfigure(line_buffering=True)
openai_client = OpenAI(timeout=30.0, max_retries=2)

DATA_DIR = Path(__file__).parent.parent / "data"
PERT_PATH = DATA_DIR / "perturbations" / "truthfulqa_perturbed.json"
EVAL_PATH = DATA_DIR / "evaluations" / "truthfulqa_results.json"

OPENAI_MODELS = [
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-08-06",
    "gpt-4.1-2025-04-14",
]
CLAUDE_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
]


def eval_openai(model, question, options, correct_idx):
    n = min(len(options), 26)
    letters = list(LETTERS[:n])
    options_str = "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(options[:n]))
    prompt = f"{question}\n\n{options_str}\n\nOutput ONLY the letter of the correct answer."
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        pred = resp.choices[0].message.content.strip().upper()
        m = re.search(r"[A-Z]", pred)
        if m:
            return 1 if letters.index(m.group()) == correct_idx else 0
    except Exception:
        return None
    return 0


def eval_claude(model, question, options, correct_idx):
    n = min(len(options), 26)
    letters = list(LETTERS[:n])
    options_str = "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(options[:n]))
    prompt = f"{question}\n\n{options_str}\n\nOutput ONLY the letter of the correct answer."
    try:
        resp = anthropic_client.messages.create(
            model=model, max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        pred = resp.content[0].text.strip().upper()
        m = re.search(r"[A-Z]", pred)
        if m:
            try:
                return 1 if letters.index(m.group()) == correct_idx else 0
            except ValueError:
                return 0
    except Exception:
        return None
    return 0


def _eval_parallel(eval_fn, tasks, workers=10):
    results = [None] * len(tasks)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(eval_fn, *args): idx for idx, args in tasks}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except:
                results[idx] = None
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(tasks)}", flush=True)
    return results


def main():
    with open(PERT_PATH) as f:
        items = json.load(f)
    print(f"TruthfulQA: {len(items)} items × 7 models × 2 versions", flush=True)

    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = {}  # always rebuild since prev results used different perturbations

    all_models = OPENAI_MODELS + CLAUDE_MODELS
    for model in all_models:
        is_claude = model.startswith("claude-")
        eval_fn = eval_claude if is_claude else eval_openai

        print(f"\n=== {model} ===", flush=True)
        results[model] = {"orig": [], "pert": []}
        tasks_orig = [(i, (model, items[i]["orig_question"], items[i]["orig_options"], items[i]["correct_idx"])) for i in range(len(items))]
        tasks_pert = [(i, (model, items[i]["pert_question"], items[i]["pert_options"], items[i]["correct_idx"])) for i in range(len(items))]
        print("  [orig]", flush=True)
        results[model]["orig"] = _eval_parallel(eval_fn, tasks_orig)
        print("  [pert]", flush=True)
        results[model]["pert"] = _eval_parallel(eval_fn, tasks_pert)

        with open(EVAL_PATH, 'w') as f:
            json.dump(results, f, indent=2)

        o = [x for x in results[model]["orig"] if x is not None]
        p = [x for x in results[model]["pert"] if x is not None]
        oa = sum(o)/max(1, len(o)); pa = sum(p)/max(1, len(p))
        print(f"  {model}: orig={oa:.3f}({len(o)}/{len(items)}) pert={pa:.3f}({len(p)}/{len(items)}) gap={oa-pa:+.3f}", flush=True)

    print("\nAll done!", flush=True)


if __name__ == "__main__":
    main()
