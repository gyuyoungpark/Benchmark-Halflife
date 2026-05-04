"""
Gemini GSM8K evaluation: 2 models × 200 items × orig/pert.
Sequential with retry-aware spacing for free-tier.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gemini_eval import gemini_eval_gsm

DATA_DIR = Path(__file__).parent.parent / "data"
PERT_DIR = DATA_DIR / "perturbations"
EVAL_DIR = DATA_DIR / "evaluations"

MODELS = ["gemini-2.5-flash-lite"]  # 2.5-flash daily quota burned by retry loops
N_ITEMS = 200
SLEEP = 7.0  # ~8.5 RPM, well under 10 RPM free-tier cap


def run():
    with open(PERT_DIR / "gsm8k_perturbed.json") as f:
        items = json.load(f)[:N_ITEMS]
    with open(EVAL_DIR / "gsm8k_results.json") as f:
        results = json.load(f)

    for model in MODELS:
        existing = results.get(model, {}).get("orig", [])
        if len([x for x in existing if x is not None]) > 100:
            print(f"Skip {model} (already done)", flush=True)
            continue

        print(f"\n=== GSM8K on {model} ===", flush=True)
        results[model] = {"orig": [], "pert": []}

        for label, key in [("orig", "orig"), ("pert", "pert")]:
            print(f"  [{label}] n={len(items)}", flush=True)
            scores = []
            for i, it in enumerate(items):
                s = gemini_eval_gsm(model, it[f"{key}_question"], it[f"{key}_final"])
                scores.append(s)
                if (i + 1) % 20 == 0:
                    valid = [x for x in scores if x is not None]
                    acc = sum(valid) / max(1, len(valid))
                    print(f"    {i+1}/{len(items)}  acc={acc:.3f} (n_valid={len(valid)})", flush=True)
                time.sleep(SLEEP)
            results[model][label] = scores
            with open(EVAL_DIR / "gsm8k_results.json", "w") as f:
                json.dump(results, f, indent=2)

        o = [x for x in results[model]["orig"] if x is not None]
        p = [x for x in results[model]["pert"] if x is not None]
        o_mean = sum(o) / max(1, len(o))
        p_mean = sum(p) / max(1, len(p))
        print(f"  {model}: orig={o_mean:.3f} pert={p_mean:.3f} gap={o_mean - p_mean:+.3f}", flush=True)

    print("\n=== Summary ===", flush=True)
    for m in MODELS:
        if m not in results: continue
        o = [x for x in results[m]["orig"] if x is not None]
        p = [x for x in results[m]["pert"] if x is not None]
        if not o or not p: continue
        print(f"  {m}: orig={sum(o)/len(o):.3f} pert={sum(p)/len(p):.3f} gap={sum(o)/len(o)-sum(p)/len(p):+.3f}")


if __name__ == "__main__":
    run()
