"""
Add 3 Claude models to evaluations of MMLU, ARC, GSM8K.
Merges with existing OpenAI results.
"""
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from claude_eval import claude_eval_mc, claude_eval_gsm, CLAUDE_MODELS

DATA_DIR = Path(__file__).parent.parent / "data"
PERT_DIR = DATA_DIR / "perturbations"
EVAL_DIR = DATA_DIR / "evaluations"


def _eval_parallel(eval_fn, tasks, workers=10):
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
            if done % 50 == 0:
                print(f"    {done}/{len(tasks)}", flush=True)
    return results


def run_mmlu_claude():
    with open(PERT_DIR / "mmlu_perturbed.json") as f:
        items = json.load(f)
    with open(EVAL_DIR / "mmlu_results.json") as f:
        results = json.load(f)

    for model in CLAUDE_MODELS:
        if model in results and len([x for x in results[model].get("orig", []) if x is not None]) > 100:
            print(f"Skipping {model} (already done)", flush=True)
            continue
        print(f"\n=== MMLU on {model} ===", flush=True)
        results[model] = {"orig": [], "pert": []}
        tasks_orig = [(i, (model, items[i]["orig_question"], items[i]["orig_choices"], items[i]["answer"])) for i in range(len(items))]
        tasks_pert = [(i, (model, items[i]["pert_question"], items[i]["pert_choices"], items[i]["answer"])) for i in range(len(items))]
        print("  [orig]", flush=True)
        results[model]["orig"] = _eval_parallel(claude_eval_mc, tasks_orig)
        print("  [pert]", flush=True)
        results[model]["pert"] = _eval_parallel(claude_eval_mc, tasks_pert)
        with open(EVAL_DIR / "mmlu_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        o = [x for x in results[model]["orig"] if x is not None]
        p = [x for x in results[model]["pert"] if x is not None]
        print(f"  {model}: orig={sum(o)/max(1,len(o)):.3f} pert={sum(p)/max(1,len(p)):.3f}", flush=True)


def run_arc_claude():
    with open(PERT_DIR / "arc_perturbed.json") as f:
        items = json.load(f)
    with open(EVAL_DIR / "arc_results.json") as f:
        results = json.load(f)

    for model in CLAUDE_MODELS:
        if model in results and len([x for x in results[model].get("orig", []) if x is not None]) > 100:
            print(f"Skipping {model}", flush=True)
            continue
        print(f"\n=== ARC on {model} ===", flush=True)
        results[model] = {"orig": [], "pert": []}
        tasks_orig = [(i, (model, items[i]["orig_question"], items[i]["orig_choices"], items[i]["answer_idx"])) for i in range(len(items))]
        tasks_pert = [(i, (model, items[i]["pert_question"], items[i]["pert_choices"], items[i]["answer_idx"])) for i in range(len(items))]
        print("  [orig]", flush=True)
        results[model]["orig"] = _eval_parallel(claude_eval_mc, tasks_orig)
        print("  [pert]", flush=True)
        results[model]["pert"] = _eval_parallel(claude_eval_mc, tasks_pert)
        with open(EVAL_DIR / "arc_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        o = [x for x in results[model]["orig"] if x is not None]
        p = [x for x in results[model]["pert"] if x is not None]
        print(f"  {model}: orig={sum(o)/max(1,len(o)):.3f} pert={sum(p)/max(1,len(p)):.3f}", flush=True)


def run_gsm_claude():
    with open(PERT_DIR / "gsm8k_perturbed.json") as f:
        items = json.load(f)
    with open(EVAL_DIR / "gsm8k_results.json") as f:
        results = json.load(f)

    for model in CLAUDE_MODELS:
        if model in results and len([x for x in results[model].get("orig", []) if x is not None]) > 100:
            print(f"Skipping {model}", flush=True)
            continue
        print(f"\n=== GSM8K on {model} ===", flush=True)
        results[model] = {"orig": [], "pert": []}
        tasks_orig = [(i, (model, items[i]["orig_question"], items[i]["orig_final"])) for i in range(len(items))]
        tasks_pert = [(i, (model, items[i]["pert_question"], items[i]["pert_final"])) for i in range(len(items))]
        print("  [orig]", flush=True)
        results[model]["orig"] = _eval_parallel(claude_eval_gsm, tasks_orig)
        print("  [pert]", flush=True)
        results[model]["pert"] = _eval_parallel(claude_eval_gsm, tasks_pert)
        with open(EVAL_DIR / "gsm8k_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        o = [x for x in results[model]["orig"] if x is not None]
        p = [x for x in results[model]["pert"] if x is not None]
        print(f"  {model}: orig={sum(o)/max(1,len(o)):.3f} pert={sum(p)/max(1,len(p)):.3f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("mmlu", "all"):
        run_mmlu_claude()
    if cmd in ("arc", "all"):
        run_arc_claude()
    if cmd in ("gsm", "all"):
        run_gsm_claude()
    print("Done!", flush=True)
