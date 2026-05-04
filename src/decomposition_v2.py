"""
Decomposition analysis v2: compute G/K/F from orig + perturbation evaluation.

Uses:
- Cross-model variance on original vs perturbation items
- Per-model contamination gap (orig - pert)
- Ceiling saturation estimate

Handles partial GSM8K results gracefully.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

EVAL_DIR = Path(__file__).parent.parent / "data" / "evaluations"
DECOMP_DIR = Path(__file__).parent.parent / "data" / "decomposition"
DECOMP_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DATES = {
    "gpt-3.5-turbo-0125": "2023-01",
    "gpt-4-turbo-2024-04-09": "2024-04",
    "gpt-4o-2024-08-06": "2024-08",
    "gpt-4.1-2025-04-14": "2025-04",
}


def analyze(results, benchmark):
    rows = []
    for model, version_results in results.items():
        orig_items = [x for x in version_results.get("orig", []) if x is not None]
        pert_items = [x for x in version_results.get("pert", []) if x is not None]
        if len(orig_items) < 50 or len(pert_items) < 50:
            continue
        orig_acc = np.mean(orig_items)
        pert_acc = np.mean(pert_items)
        rows.append({
            "model": model,
            "date": MODEL_DATES.get(model, "?"),
            "orig_acc": orig_acc,
            "pert_acc": pert_acc,
            "gap": orig_acc - pert_acc,
            "n_orig": len(orig_items),
            "n_pert": len(pert_items),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("date")

    orig_scores = df["orig_acc"].values
    pert_scores = df["pert_acc"].values
    var_orig = np.var(orig_scores, ddof=1) if len(orig_scores) > 1 else 0.0
    var_pert = np.var(pert_scores, ddof=1) if len(pert_scores) > 1 else 0.0

    # Contamination-driven variance collapse
    K = max(0.0, var_pert - var_orig)

    # Ceiling fraction
    ceiling_frac = np.mean(orig_scores >= 0.95)

    # Mean and max contamination gap
    mean_gap = df["gap"].mean()
    max_gap = df["gap"].abs().max()

    return {
        "benchmark": benchmark,
        "scores": df.to_dict(orient="records"),
        "var_orig": float(var_orig),
        "var_pert": float(var_pert),
        "K_contamination_variance": float(K),
        "mean_gap": float(mean_gap),
        "max_gap": float(max_gap),
        "ceiling_frac": float(ceiling_frac),
    }


def main():
    all_decomp = {}
    for bench in ["mmlu", "arc", "gsm8k"]:
        path = EVAL_DIR / f"{bench}_results.json"
        if not path.exists():
            continue
        with open(path) as f:
            results = json.load(f)

        d = analyze(results, bench)
        if d is None:
            print(f"\n{bench.upper()}: insufficient data")
            continue

        all_decomp[bench] = d
        print(f"\n=== {bench.upper()} ===")
        for r in d["scores"]:
            print(f"  {r['model']:<30} orig={r['orig_acc']:.3f} pert={r['pert_acc']:.3f} gap={r['gap']:+.3f}")
        print(f"  var_orig={d['var_orig']:.4f}  var_pert={d['var_pert']:.4f}")
        print(f"  K (contam variance) = {d['K_contamination_variance']:.4f}")
        print(f"  Mean gap = {d['mean_gap']:+.3f}  Max gap = {d['max_gap']:.3f}")
        print(f"  Ceiling fraction (orig acc >= 0.95) = {d['ceiling_frac']:.2f}")

    with open(DECOMP_DIR / "decomposition_results.json", 'w') as f:
        json.dump(all_decomp, f, indent=2)
    print(f"\nSaved: {DECOMP_DIR / 'decomposition_results.json'}")


if __name__ == "__main__":
    main()
