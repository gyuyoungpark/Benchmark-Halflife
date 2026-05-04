"""
E2: HumanEval perturbation evaluation.
Strategy: rename function + variables + rephrase docstring; same operations.
Test code is regenerated to match the new function name.
"""
import json
import os
import re
import sys
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from claude_eval import client as anthropic_client

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent.parent / "data"
ITEMS_PATH = DATA_DIR / "items" / "humaneval_items.json"
PERT_PATH = DATA_DIR / "perturbations" / "humaneval_perturbed.json"
EVAL_PATH = DATA_DIR / "evaluations" / "humaneval_results.json"

PARAPHRASE_PROMPT = """You are paraphrasing a Python function specification. Generate a renamed version with:
- New function name (synonym or descriptive variant): {orig_name} → <new name>
- New parameter names (semantically equivalent)
- Rewritten docstring with same examples and same meaning
- The function body should NOT be included; we only need the signature + docstring
- Internal logic must be solvable identically

Original function spec:
```python
{prompt}
```

Original test (do not paraphrase, just for reference):
```python
{test_snippet}
```

Output ONLY a JSON object with these keys:
- "new_name": new function name (string)
- "new_prompt": full new function spec including imports, signature, and docstring (string)

The new_prompt should be a complete Python prompt that ends with the function signature and docstring (no body)."""


def paraphrase_humaneval(item):
    test_snippet = item["test"][:300]
    prompt = PARAPHRASE_PROMPT.format(
        orig_name=item["entry_point"],
        prompt=item["prompt"],
        test_snippet=test_snippet,
    )
    try:
        resp = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        m = re.search(r'\{[\s\S]*\}', text)
        if not m:
            return None
        parsed = json.loads(m.group())
        if "new_name" in parsed and "new_prompt" in parsed:
            return parsed
    except Exception:
        return None
    return None


def adapt_test(orig_test, orig_name, new_name):
    """Replace function name in test code."""
    return orig_test.replace(orig_name, new_name)


def generate_perturbations():
    with open(ITEMS_PATH) as f:
        items = json.load(f)
    print(f"Paraphrasing {len(items)} HumanEval items via Claude...", flush=True)

    PERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    perturbed = []
    errors = 0
    done = 0

    def _one(item):
        pert = paraphrase_humaneval(item)
        if pert is None:
            return None
        adapted_test = adapt_test(item["test"], item["entry_point"], pert["new_name"])
        return {
            "task_id": item["task_id"],
            "orig_prompt": item["prompt"],
            "orig_entry_point": item["entry_point"],
            "orig_canonical_solution": item["canonical_solution"],
            "orig_test": item["test"],
            "pert_prompt": pert["new_prompt"],
            "pert_entry_point": pert["new_name"],
            "pert_test": adapted_test,
        }

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_one, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r is None:
                errors += 1
            else:
                perturbed.append(r)
            if done % 20 == 0:
                print(f"  [{done}/{len(items)}] errors={errors}", flush=True)
                with open(PERT_PATH, 'w') as f:
                    json.dump(perturbed, f)

    with open(PERT_PATH, 'w') as f:
        json.dump(perturbed, f, indent=2)
    print(f"Done: {len(perturbed)}/{len(items)} ({errors} errors)", flush=True)


def _strip_md(text):
    text = re.sub(r"^```python\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()


def execute_solution(completion, test, entry_point, timeout=10):
    """Execute completion against test. completion must be a FULL Python file with def.
    Returns 1 if pass, 0 if fail, None if error."""
    # Ensure necessary imports
    code = "from typing import *\nimport math, re, collections, itertools, functools\n"
    code += completion + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(code)
            tmp = tf.name
        result = subprocess.run(
            ["python3", tmp],
            capture_output=True, text=True, timeout=timeout
        )
        os.unlink(tmp); tmp = None
        return 1 if result.returncode == 0 else 0
    except subprocess.TimeoutExpired:
        if tmp:
            try: os.unlink(tmp)
            except: pass
        return 0
    except Exception:
        return None


COMPLETION_INSTRUCTION = """Implement the following Python function. Output ONLY a single Python code block containing the COMPLETE function definition (with imports if needed, the def line, the docstring, and the body). Do not include test code, do not include explanations.

```python
{prompt}
```"""


def get_completion_openai(model, prompt, max_tokens=1000):
    from openai import OpenAI
    client = OpenAI(timeout=30.0, max_retries=1)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert Python programmer. Always output complete, runnable Python code in a single ```python block."},
                {"role": "user", "content": COMPLETION_INSTRUCTION.format(prompt=prompt)}
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return _strip_md(resp.choices[0].message.content)
    except Exception:
        return None


def get_completion_claude(model, prompt, max_tokens=1000):
    try:
        resp = anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": COMPLETION_INSTRUCTION.format(prompt=prompt)}],
        )
        return _strip_md(resp.content[0].text)
    except Exception:
        return None


def eval_one(model, item, version):
    prompt_key = f"{version}_prompt"
    entry_key = f"{version}_entry_point"
    test_key = f"{version}_test"

    if model.startswith("gpt-"):
        completion = get_completion_openai(model, item[prompt_key])
    else:
        completion = get_completion_claude(model, item[prompt_key])

    if completion is None:
        return None
    return execute_solution(completion, item[test_key], item[entry_key])


def run_evaluation(models):
    with open(PERT_PATH) as f:
        items = json.load(f)
    print(f"Evaluating {len(items)} HumanEval items × {len(models)} models × 2 versions", flush=True)

    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if EVAL_PATH.exists():
        with open(EVAL_PATH) as f:
            results = json.load(f)
    else:
        results = {}

    for model in models:
        if model in results and len([x for x in results[model].get("orig", []) if x is not None]) > 100:
            print(f"Skipping {model}", flush=True)
            continue
        print(f"\n=== {model} ===", flush=True)
        results[model] = {"orig": [None]*len(items), "pert": [None]*len(items)}

        for version in ["orig", "pert"]:
            print(f"  [{version}]", flush=True)
            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = {ex.submit(eval_one, model, items[i], version): i for i in range(len(items))}
                done = 0
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        results[model][version][idx] = fut.result()
                    except:
                        results[model][version][idx] = None
                    done += 1
                    if done % 20 == 0:
                        print(f"    {done}/{len(items)}", flush=True)
                        with open(EVAL_PATH, 'w') as f:
                            json.dump(results, f, indent=2)

        with open(EVAL_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        o = [x for x in results[model]["orig"] if x is not None]
        p = [x for x in results[model]["pert"] if x is not None]
        print(f"  {model}: orig={sum(o)/max(1,len(o)):.3f}({len(o)}/{len(items)}) pert={sum(p)/max(1,len(p)):.3f}({len(p)}/{len(items)}) gap={(sum(o)/max(1,len(o)))-(sum(p)/max(1,len(p))):+.3f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("perturb", "all"):
        generate_perturbations()
    if cmd in ("eval", "all"):
        # Run on subset first, then expand
        models = [
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "gpt-3.5-turbo-0125",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-2024-08-06",
            "gpt-4.1-2025-04-14",
        ]
        run_evaluation(models)
    print("Done!", flush=True)
