"""
Compute G/K/F decomposition from original + perturbed evaluation results.

For each benchmark at each model-epoch:
  - D_orig_t = inter-model variance on original items
  - D_pert_t = inter-model variance on perturbation holdouts
  - K_t (contamination) = D_pert_t - D_orig_t  (original has less variance than holdout => contamination collapsed it)
  - F_t (ceiling)      = ceiling-induced variance loss (fraction of items at >95% correct)
  - G_t (genuine)      = (D_0 - D_orig_t) - K_t - F_t
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

EVAL_DIR = Path(__file__).parent.parent / "data" / "evaluations"
DECOMP_DIR = Path(__file__).parent.parent / "data" / "decomposition"
DECOMP_DIR.mkdir(parents=True, exist_ok=True)

# Model release dates (for temporal analysis)
MODEL_DATES = {
    "gpt-3.5-turbo-0125": "2023-01",
    "gpt-4-0613": "2023-06",
    "gpt-4-turbo-2024-04-09": "2024-04",
    "gpt-4o-2024-08-06": "2024-08",
    "gpt-4.1-2025-04-14": "2025-04",
}


def compute_scores(results):
    """Compute per-model accuracy for orig and pert versions."""
    scores = {}
    for model, versions in results.items():
        orig = [x for x in versions["orig"] if x is not None]
        pert = [x for x in versions["pert"] if x is not None]
        scores[model] = {
            "orig_acc": np.mean(orig) if orig else np.nan,
            "pert_acc": np.mean(pert) if pert else np.nan,
            "orig_n": len(orig),
            "pert_n": len(pert),
        }
    return scores


def decompose(results, benchmark_name):
    """
    Decompose discriminative power loss into G/K/F components.

    For 5 models (one per epoch):
    - Treat each as a single-model cohort
    - Compute inter-item variance instead (since we have 5 models)
    - Compare original vs perturbation
    """
    scores = compute_scores(results)

    # Build a dataframe
    model_order = list(MODEL_DATES.keys())
    rows = []
    for m in model_order:
        if m in scores:
            rows.append({
                "model": m,
                "date": MODEL_DATES[m],
                "orig_acc": scores[m]["orig_acc"],
                "pert_acc": scores[m]["pert_acc"],
                "gap": scores[m]["orig_acc"] - scores[m]["pert_acc"],
            })

    df = pd.DataFrame(rows)

    # Cross-model variance at each version
    orig_scores = df["orig_acc"].values
    pert_scores = df["pert_acc"].values

    var_orig = np.var(orig_scores, ddof=1)
    var_pert = np.var(pert_scores, ddof=1)

    # Contamination-driven variance loss
    # If perturbation has more variance than original, contamination is compressing the original
    K = max(0, var_pert - var_orig)

    # Design ceiling: fraction of top-model scoring above 0.95
    top_orig = df["orig_acc"].max()
    ceiling_fraction = 1.0 if top_orig >= 0.95 else 0.0
    F = ceiling_fraction * var_orig * 0.3  # heuristic

    # Genuine variance (the variance that appears in the perturbation holdout)
    G = var_pert - K - F

    # Mean contamination gap
    mean_gap = df["gap"].mean()

    result = {
        "benchmark": benchmark_name,
        "scores": df.to_dict(orient="records"),
        "var_orig": var_orig,
        "var_pert": var_pert,
        "contamination_gap_mean": mean_gap,
        "contamination_gap_per_model": df.set_index("model")["gap"].to_dict(),
        "K_contamination": K,
        "F_ceiling": F,
        "G_genuine": max(0, G),
    }

    return result


def main():
    all_results = {}
    for bench in ["mmlu", "arc", "gsm8k"]:
        path = EVAL_DIR / f"{bench}_results.json"
        if not path.exists():
            print(f"Missing: {path}")
            continue

        with open(path) as f:
            results = json.load(f)

        decomp = decompose(results, bench)
        all_results[bench] = decomp

        print(f"\n=== {bench.upper()} ===")
        print(f"Per-model accuracy:")
        for row in decomp["scores"]:
            print(f"  {row['model']:<30} orig={row['orig_acc']:.3f} pert={row['pert_acc']:.3f} gap={row['gap']:+.3f}")
        print(f"Cross-model variance: orig={decomp['var_orig']:.4f} pert={decomp['var_pert']:.4f}")
        print(f"Contamination gap (mean): {decomp['contamination_gap_mean']:+.3f}")
        print(f"Decomposition: G={decomp['G_genuine']:.4f}  K={decomp['K_contamination']:.4f}  F={decomp['F_ceiling']:.4f}")

    with open(DECOMP_DIR / "all_decomp.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved: {DECOMP_DIR / 'all_decomp.json'}")


if __name__ == "__main__":
    main()
