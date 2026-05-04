"""
MBPP perturbation evaluation.
MBPP items have:
- prompt: natural language description
- code: reference solution
- test_list: assert statements that call the function

Strategy: paraphrase the natural-language prompt + rename function in test_list.
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
ITEMS_PATH = DATA_DIR / "items" / "mbpp_items.json"
PERT_PATH = DATA_DIR / "perturbations" / "mbpp_perturbed.json"
EVAL_PATH = DATA_DIR / "evaluations" / "mbpp_results.json"

PARAPHRASE_PROMPT = """You will paraphrase a Python coding problem and rename its target function.

Given:
- Original prompt: a natural-language problem description
- Reference solution code with function name {orig_name}
- Test cases that call {orig_name}

Generate:
1. A paraphrased prompt (different wording, same meaning, same difficulty)
2. A new function name (semantically equivalent, single word or snake_case)
3. The same test cases with the function name replaced

Output ONLY a JSON object with keys: "new_prompt" (string), "new_name" (string), "new_tests" (array of strings).

Original prompt:
{prompt}

Original solution (for context only):
{code}

Original tests:
{tests}
"""


def extract_function_name(code):
    """Extract the function name from the first `def` in the code."""
    m = re.search(r"def\s+(\w+)\s*\(", code)
    return m.group(1) if m else None


def paraphrase_mbpp(item):
    orig_name = extract_function_name(item["code"])
    if orig_name is None:
        return None
    tests_str = "\n".join(item["test_list"][:5])
    prompt = PARAPHRASE_PROMPT.format(
        orig_name=orig_name,
        prompt=item["prompt"],
        code=item["code"][:600],
        tests=tests_str,
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
        if all(k in parsed for k in ("new_prompt", "new_name", "new_tests")):
            return {
                "orig_name": orig_name,
                "new_prompt": parsed["new_prompt"],
                "new_name": parsed["new_name"],
                "new_tests": parsed["new_tests"],
            }
    except Exception:
        return None


def generate_perturbations():
    with open(ITEMS_PATH) as f:
        items = json.load(f)
    print(f"Paraphrasing {len(items)} MBPP items via Claude...", flush=True)

    PERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    perturbed = []
    errors = 0
    done = 0

    def _one(item):
        pert = paraphrase_mbpp(item)
        if pert is None:
            return None
        return {
            "task_id": item["task_id"],
            "orig_prompt": item["prompt"],
            "orig_code": item["code"],
            "orig_name": pert["orig_name"],
            "orig_tests": item["test_list"],
            "orig_test_imports": item.get("test_imports", []),
            "pert_prompt": pert["new_prompt"],
            "pert_name": pert["new_name"],
            "pert_tests": pert["new_tests"],
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


# ===== Evaluation =====

def execute_mbpp(completion, tests, test_imports):
    """Execute completion against MBPP test list."""
    code = "from typing import *\nimport math, re, collections, itertools, functools\n"
    for imp in (test_imports or []):
        code += imp + "\n"
    code += completion + "\n\n"
    for t in tests:
        code += t + "\n"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(code)
            tmp = tf.name
        result = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=10)
        os.unlink(tmp); tmp = None
        return 1 if result.returncode == 0 else 0
    except subprocess.TimeoutExpired:
        if tmp:
            try: os.unlink(tmp)
            except: pass
        return 0
    except Exception:
        return None


def _strip_md(text):
    text = re.sub(r"^```python\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()


COMPLETION_INSTRUCTION = """Implement a Python function for this problem. Output ONLY a single Python code block with the COMPLETE function definition (signature + body, plus any imports). The function must be named exactly `{name}`. Do not include test code or explanations.

Problem:
{prompt}"""


def get_completion_openai(model, prompt, name, max_tokens=800):
    from openai import OpenAI
    client = OpenAI(timeout=30.0, max_retries=1)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert Python programmer. Always output complete runnable Python in a single ```python block."},
                {"role": "user", "content": COMPLETION_INSTRUCTION.format(prompt=prompt, name=name)}
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return _strip_md(resp.choices[0].message.content)
    except Exception:
        return None


def get_completion_claude(model, prompt, name, max_tokens=800):
    try:
        resp = anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": COMPLETION_INSTRUCTION.format(prompt=prompt, name=name)}],
        )
        return _strip_md(resp.content[0].text)
    except Exception:
        return None


def eval_one(model, item, version):
    prompt_key = f"{version}_prompt"
    name_key = f"{version}_name"
    test_key = f"{version}_tests"

    if model.startswith("gpt-"):
        completion = get_completion_openai(model, item[prompt_key], item[name_key])
    else:
        completion = get_completion_claude(model, item[prompt_key], item[name_key])

    if completion is None:
        return None
    return execute_mbpp(completion, item[test_key], item.get("orig_test_imports", []))


def run_evaluation(models, max_workers=4):
    with open(PERT_PATH) as f:
        items = json.load(f)
    print(f"MBPP eval: {len(items)} items × {len(models)} models × 2 versions", flush=True)

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
        if model not in results:
            results[model] = {"orig": [None]*len(items), "pert": [None]*len(items)}

        for version in ["orig", "pert"]:
            print(f"  [{version}]", flush=True)
            current = results[model].get(version, [None]*len(items))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(eval_one, model, items[i], version): i for i in range(len(items))}
                done = 0
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        current[idx] = fut.result()
                    except:
                        current[idx] = None
                    done += 1
                    if done % 30 == 0:
                        print(f"    {done}/{len(items)}", flush=True)
                        results[model][version] = current
                        with open(EVAL_PATH, 'w') as f:
                            json.dump(results, f, indent=2)
            results[model][version] = current
            with open(EVAL_PATH, 'w') as f:
                json.dump(results, f, indent=2)

        o = [x for x in results[model]["orig"] if x is not None]
        p = [x for x in results[model]["pert"] if x is not None]
        oa = sum(o)/max(1, len(o)); pa = sum(p)/max(1, len(p))
        print(f"  {model}: orig={oa:.3f}({len(o)}/{len(items)}) pert={pa:.3f}({len(p)}/{len(items)}) gap={oa-pa:+.3f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("perturb", "all"):
        generate_perturbations()
    if cmd in ("eval", "all"):
        models = [
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "gpt-3.5-turbo-0125",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-2024-08-06",
            "gpt-4.1-2025-04-14",
        ]
        run_evaluation(models, max_workers=6)
    print("Done!", flush=True)
