"""
Run 2 Gemini models on perturbation eval (GSM8K first; others if quota allows).
Rate-limited due to free-tier quota.
"""
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from gemini_eval import gemini_eval_mc, gemini_eval_gsm

DATA_DIR = Path(__file__).parent.parent / "data"
PERT_DIR = DATA_DIR / "perturbations"
EVAL_DIR = DATA_DIR / "evaluations"

# Free-tier-available models (as of run time)
MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

# Rate limit: ~10-15 RPM free tier. Be conservative.
WORKERS = 2
SLEEP_BETWEEN = 0.0  # backoff handled in gemini_eval


def _eval_parallel(eval_fn, tasks, workers=WORKERS):
    results = [None] * len(tasks)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(eval_fn, *args): idx for idx, args in tasks}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = None
            done += 1
            if done % 25 == 0:
                print(f"    {done}/{len(tasks)}", flush=True)
            time.sleep(SLEEP_BETWEEN)
    return results


def run_benchmark(bench, eval_fn, item_keys):
    """bench: 'gsm8k'|'mmlu'|'arc'|'truthfulqa'; item_keys: tuple of (q_orig, c_orig, a, q_pert, c_pert, a_pert)
       eval_fn signature: fn(model, question, answer_or_choices, answer)"""
    with open(PERT_DIR / f"{bench}_perturbed.json") as f:
        items = json.load(f)
    with open(EVAL_DIR / f"{bench}_results.json") as f:
        results = json.load(f)

    for model in MODELS:
        existing = results.get(model, {}).get("orig", [])
        existing = [x for x in existing if x is not None]
        if len(existing) > 100:
            print(f"Skipping {bench}/{model} (already done, n={len(existing)})", flush=True)
            continue
        print(f"\n=== {bench.upper()} on {model} ===", flush=True)
        results[model] = {"orig": [], "pert": []}

        tasks_orig = [(i, (model,) + item_keys(items[i], "orig")) for i in range(len(items))]
        tasks_pert = [(i, (model,) + item_keys(items[i], "pert")) for i in range(len(items))]

        print("  [orig]", flush=True)
        results[model]["orig"] = _eval_parallel(eval_fn, tasks_orig)
        print("  [pert]", flush=True)
        results[model]["pert"] = _eval_parallel(eval_fn, tasks_pert)

        with open(EVAL_DIR / f"{bench}_results.json", "w") as f:
            json.dump(results, f, indent=2)

        o = [x for x in results[model]["orig"] if x is not None]
        p = [x for x in results[model]["pert"] if x is not None]
        o_mean = sum(o) / max(1, len(o))
        p_mean = sum(p) / max(1, len(p))
        print(f"  {model}: orig={o_mean:.3f} pert={p_mean:.3f} gap={o_mean - p_mean:+.3f}", flush=True)


def gsm_keys(item, which):
    return (item[f"{which}_question"], item[f"{which}_final"])


def mmlu_keys(item, which):
    return (item[f"{which}_question"], item[f"{which}_choices"], item["answer"])


def arc_keys(item, which):
    return (item[f"{which}_question"], item[f"{which}_choices"], item["answer_idx"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gsm"
    if cmd in ("gsm", "all"):
        run_benchmark("gsm8k", gemini_eval_gsm, gsm_keys)
    if cmd in ("mmlu", "all"):
        run_benchmark("mmlu", gemini_eval_mc, mmlu_keys)
    if cmd in ("arc", "all"):
        run_benchmark("arc", gemini_eval_mc, arc_keys)
    print("\nDone.", flush=True)
