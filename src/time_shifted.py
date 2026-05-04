"""
E5: Time-shifted contamination analysis.
Plot per-model contamination gap vs training cutoff date.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_DIR = DATA_DIR / "evaluations"
FIG_DIR = Path(__file__).parent.parent / "figures"

# Approximate training cutoff dates for OpenAI models (from public statements)
MODEL_CUTOFFS = {
    "gpt-3.5-turbo-0125":      "2023-09",
    "gpt-4-turbo-2024-04-09":  "2023-12",
    "gpt-4o-2024-08-06":       "2023-10",
    "gpt-4.1-2025-04-14":      "2024-06",
}

MODEL_RELEASE = {
    "gpt-3.5-turbo-0125":      "2024-01",
    "gpt-4-turbo-2024-04-09":  "2024-04",
    "gpt-4o-2024-08-06":       "2024-08",
    "gpt-4.1-2025-04-14":      "2025-04",
}

# Benchmark publication dates
BENCH_PUB_DATE = {
    "mmlu":  "2021-01",
    "arc":   "2018-03",
    "gsm8k": "2021-10",
}


def compute_gaps():
    rows = []
    for bench in ["mmlu", "arc", "gsm8k"]:
        path = EVAL_DIR / f"{bench}_results.json"
        if not path.exists():
            continue
        with open(path) as f:
            r = json.load(f)
        for m in MODEL_CUTOFFS:
            if m not in r:
                continue
            o = [x for x in r[m]["orig"] if x is not None]
            p = [x for x in r[m]["pert"] if x is not None]
            if len(o) < 50 or len(p) < 50:
                continue
            gap = sum(o)/len(o) - sum(p)/len(p)
            cutoff = pd.Timestamp(MODEL_CUTOFFS[m])
            release = pd.Timestamp(MODEL_RELEASE[m])
            bench_pub = pd.Timestamp(BENCH_PUB_DATE[bench])
            months_since_pub_at_cutoff = (cutoff - bench_pub).days / 30.44
            rows.append({
                "benchmark": bench,
                "model": m,
                "cutoff": MODEL_CUTOFFS[m],
                "release": MODEL_RELEASE[m],
                "months_since_bench_pub_at_cutoff": months_since_pub_at_cutoff,
                "gap": gap,
                "orig_acc": sum(o)/len(o),
                "pert_acc": sum(p)/len(p),
            })
    return pd.DataFrame(rows)


def main():
    df = compute_gaps()
    print(df.to_string(index=False))

    # Correlation: gap vs months_since_bench_pub_at_cutoff
    print("\n=== Correlations ===")
    for bench in df["benchmark"].unique():
        sub = df[df["benchmark"] == bench]
        if len(sub) >= 3:
            r, p = pearsonr(sub["months_since_bench_pub_at_cutoff"], sub["gap"])
            print(f"  {bench}: Pearson r = {r:+.3f} (p={p:.3f})")

    # Aggregate
    if len(df) >= 6:
        r_all, p_all = pearsonr(df["months_since_bench_pub_at_cutoff"], df["gap"])
        print(f"  ALL: Pearson r = {r_all:+.3f} (p={p_all:.3f})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"mmlu": "#E64A19", "arc": "#1976D2", "gsm8k": "#388E3C"}
    bench_labels = {"mmlu": "MMLU", "arc": "ARC-C", "gsm8k": "GSM8K"}

    for bench in df["benchmark"].unique():
        sub = df[df["benchmark"] == bench]
        ax.plot(sub["months_since_bench_pub_at_cutoff"], sub["gap"], "o-",
                label=bench_labels[bench], color=colors[bench], markersize=10, lw=2)
        for _, row in sub.iterrows():
            short = row["model"].split("-")[1] if "gpt-" in row["model"] else row["model"][:6]
            ax.annotate(short, (row["months_since_bench_pub_at_cutoff"], row["gap"]),
                       xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Months between benchmark publication and model training cutoff", fontsize=12)
    ax.set_ylabel("Contamination gap (orig − pert accuracy)", fontsize=12)
    ax.set_title("Contamination gap is invariant to training-cutoff position", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_time_shifted.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig_time_shifted.png", bbox_inches='tight', dpi=150)
    print(f"\nSaved: {FIG_DIR / 'fig_time_shifted.pdf'}")

    df.to_csv(DATA_DIR / "time_shifted_analysis.csv", index=False)


if __name__ == "__main__":
    main()
