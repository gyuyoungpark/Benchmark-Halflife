"""
Evaluate OpenAI models on original + perturbed benchmark items.

Models span multiple generations:
- gpt-3.5-turbo (2022 era)
- gpt-4-0613 (2023-06)
- gpt-4-turbo (2024-04)
- gpt-4o (2024-05)
- gpt-4.1 (2025-04)

For each (model, benchmark, version) combination, compute accuracy and item-level
binary responses.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent.parent / "data"
PERT_DIR = DATA_DIR / "perturbations"
EVAL_DIR = DATA_DIR / "evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(timeout=30.0, max_retries=1)

# Models spanning 2022-2025, with approximate release dates for temporal analysis
MODELS = {
    "gpt-3.5-turbo-0125": "2024-01",      # 2023-era baseline
    "gpt-4-turbo-2024-04-09": "2024-04",   # GPT-4 Turbo
    "gpt-4o-2024-08-06": "2024-08",        # GPT-4o
    "gpt-4.1-2025-04-14": "2025-04",       # GPT-4.1
}


# ============================================================
# Prompts
# ============================================================

MC_SYSTEM = """You are an expert at answering multiple-choice questions. For each question, output ONLY the letter of the correct answer (A, B, C, or D). Do not include explanations."""

GSM_SYSTEM = """You are an expert at solving math word problems. Solve the problem step by step, then give the final numerical answer on a new line prefixed with '####'."""


def _clean_choice(c):
    """Strip 'A) ', 'A. ', etc. prefixes if present."""
    if not isinstance(c, str):
        return str(c)
    return re.sub(r"^\s*[A-Da-d][\)\.\:]\s*", "", c).strip()


def eval_mc(model, question, choices, answer):
    """Evaluate a single MC question."""
    letters = ["A", "B", "C", "D"]
    cleaned = [_clean_choice(c) for c in choices[:4]]
    prompt = f"{question}\n\n"
    for i, c in enumerate(cleaned):
        prompt += f"{letters[i]}) {c}\n"
    prompt += "\nAnswer:"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": MC_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5,
        )
        pred = resp.choices[0].message.content.strip().upper()
        # Extract letter
        m = re.search(r"[ABCD]", pred)
        if m:
            pred_letter = m.group()
            if isinstance(answer, int):
                correct_letter = letters[answer]
            else:
                correct_letter = str(answer).upper()
            return 1 if pred_letter == correct_letter else 0
    except Exception as e:
        return None
    return 0


def eval_gsm(model, question, correct_final):
    """Evaluate a single GSM8K problem."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GSM_SYSTEM},
                {"role": "user", "content": question}
            ],
            temperature=0,
            max_tokens=500,
        )
        answer_text = resp.choices[0].message.content

        # Extract final answer
        m = re.search(r"####\s*([-\d,.]+)", answer_text)
        if not m:
            # Try to find last number
            nums = re.findall(r"[-\d,]+\.?\d*", answer_text)
            if nums:
                pred = nums[-1].replace(",", "")
            else:
                return 0
        else:
            pred = m.group(1).strip().replace(",", "")

        # Normalize
        try:
            pred_val = float(pred)
            correct_val = float(str(correct_final).replace(",", "").strip())
            return 1 if abs(pred_val - correct_val) < 1e-4 else 0
        except:
            return 0
    except Exception as e:
        return None


# ============================================================
# Run evaluation
# ============================================================

def _eval_batch_parallel(eval_fn, tasks, workers=20):
    """
    Run eval_fn in parallel. tasks is a list of (idx, args_tuple).
    Returns results in order of idx.
    """
    results = [None] * len(tasks)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(eval_fn, *args): idx for idx, args in tasks}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = None
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{len(tasks)}", flush=True)
    return results


def _acc(lst):
    v = [x for x in lst if x is not None]
    return sum(v) / len(v) if v else float('nan')


def run_mmlu(models=None):
    models = models or list(MODELS.keys())
    with open(PERT_DIR / "mmlu_perturbed.json") as f:
        items = json.load(f)

    print(f"MMLU: {len(items)} items x {len(models)} models x 2 versions", flush=True)
    results = {m: {"orig": [], "pert": []} for m in models}

    for model in models:
        print(f"\n=== {model} ===", flush=True)
        # Build tasks: original
        tasks_orig = [(i, (model, item["orig_question"], item["orig_choices"], item["answer"])) for i, item in enumerate(items)]
        tasks_pert = [(i, (model, item["pert_question"], item["pert_choices"], item["answer"])) for i, item in enumerate(items)]

        print(f"  [orig]", flush=True)
        results[model]["orig"] = _eval_batch_parallel(eval_mc, tasks_orig)
        print(f"  [pert]", flush=True)
        results[model]["pert"] = _eval_batch_parallel(eval_mc, tasks_pert)

        orig_acc = _acc(results[model]["orig"])
        pert_acc = _acc(results[model]["pert"])
        print(f"  {model}: orig={orig_acc:.3f}, pert={pert_acc:.3f}, delta={orig_acc-pert_acc:+.3f}", flush=True)

        with open(EVAL_DIR / "mmlu_results.json", 'w') as f:
            json.dump(results, f, indent=2)

    return results


def run_arc(models=None):
    models = models or list(MODELS.keys())
    with open(PERT_DIR / "arc_perturbed.json") as f:
        items = json.load(f)

    print(f"\nARC-C: {len(items)} items x {len(models)} models", flush=True)
    results = {m: {"orig": [], "pert": []} for m in models}

    for model in models:
        print(f"\n=== {model} ===", flush=True)
        tasks_orig = [(i, (model, item["orig_question"], item["orig_choices"], item["answer_idx"])) for i, item in enumerate(items)]
        tasks_pert = [(i, (model, item["pert_question"], item["pert_choices"], item["answer_idx"])) for i, item in enumerate(items)]

        print(f"  [orig]", flush=True)
        results[model]["orig"] = _eval_batch_parallel(eval_mc, tasks_orig)
        print(f"  [pert]", flush=True)
        results[model]["pert"] = _eval_batch_parallel(eval_mc, tasks_pert)

        orig_acc = _acc(results[model]["orig"])
        pert_acc = _acc(results[model]["pert"])
        print(f"  {model}: orig={orig_acc:.3f}, pert={pert_acc:.3f}, delta={orig_acc-pert_acc:+.3f}", flush=True)

        with open(EVAL_DIR / "arc_results.json", 'w') as f:
            json.dump(results, f, indent=2)

    return results


def run_gsm(models=None):
    models = models or list(MODELS.keys())
    with open(PERT_DIR / "gsm8k_perturbed.json") as f:
        items = json.load(f)

    print(f"\nGSM8K: {len(items)} items x {len(models)} models", flush=True)
    results = {m: {"orig": [], "pert": []} for m in models}

    for model in models:
        print(f"\n=== {model} ===", flush=True)
        tasks_orig = [(i, (model, item["orig_question"], item["orig_final"])) for i, item in enumerate(items)]
        tasks_pert = [(i, (model, item["pert_question"], item["pert_final"])) for i, item in enumerate(items)]

        print(f"  [orig]", flush=True)
        results[model]["orig"] = _eval_batch_parallel(eval_gsm, tasks_orig)
        print(f"  [pert]", flush=True)
        results[model]["pert"] = _eval_batch_parallel(eval_gsm, tasks_pert)

        orig_acc = _acc(results[model]["orig"])
        pert_acc = _acc(results[model]["pert"])
        print(f"  {model}: orig={orig_acc:.3f}, pert={pert_acc:.3f}, delta={orig_acc-pert_acc:+.3f}", flush=True)

        with open(EVAL_DIR / "gsm8k_results.json", 'w') as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("mmlu", "all"):
        run_mmlu()
    if which in ("arc", "all"):
        run_arc()
    if which in ("gsm", "all"):
        run_gsm()
    print("\nAll evaluations done!")
