"""
P3: Benchmark triage simulation.

Task: rank a set of frontier models using a budget of K benchmarks chosen from a pool.
Compare policies for choosing which benchmarks to use:
  1. Random
  2. Akhtar saturation index (low S = use it)
  3. Half-life (high tau = use it)
  4. Naive: use all benchmarks (oracle)

Metric: rank correlation (Spearman) between the policy's ranking and the oracle ranking
on a held-out set of frontier models.

Setup:
- Pool of benchmarks: 11 LLM benchmarks
- Pool of models: 4 GPT + 3 Claude = 7 frontier models
- Their "true" ability ordering is from the average of all benchmark accuracies on perturbation holdouts
- For each policy, pick K=3 benchmarks; compute rank correlation between
  the K-benchmark average score ranking and the true ranking.
- Also report: average top-stratum variance of selected benchmarks (the higher the better - more discriminative)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_DIR = DATA_DIR / "evaluations"
FIG_DIR = Path(__file__).parent.parent / "figures"
OUT_DIR = DATA_DIR / "triage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Half-lives (best AIC) from prior analysis
HALFLIFE = {
    "winogrande": 1.7,
    "hellaswag":  2.5,
    "arc":        3.4,
    "gsm8k":      4.9,
    "truthfulqa": 5.7,
    "bbh":       19.6,
    # the following had no measurable decay in main runs; treat as "long"
    "mmlu":       50.0,  # MMLU original (we use this as proxy for noisy/young)
    "mmlu_pro":   60.0,
    "gpqa":       60.0,
    "math":       60.0,
    "ifeval":     60.0,
    "musr":       60.0,
}

# Akhtar saturation index (computed previously)
AKHTAR_S = {
    "arc": 0.252, "hellaswag": 0.153, "truthfulqa": 0.003, "winogrande": 0.106, "gsm8k": 0.071,
    "bbh": 0.534, "mmlu_pro": 0.966, "gpqa": 0.784, "ifeval": 0.994, "math": 0.630, "musr": 0.416,
    "mmlu": 0.40,  # proxy
}

MODELS = [
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-08-06",
    "gpt-4.1-2025-04-14",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
]


def load_per_model_acc():
    """Load per-model accuracies on PERTURBATION items (uncontaminated proxy for true ability)."""
    benches = ["mmlu", "arc", "gsm8k", "truthfulqa", "humaneval"]
    table = {}
    for b in benches:
        path = EVAL_DIR / f"{b}_results.json"
        if not path.exists():
            continue
        with open(path) as f:
            r = json.load(f)
        col = {}
        for m in MODELS:
            if m not in r:
                continue
            pert = [x for x in r[m]["pert"] if x is not None]
            if len(pert) >= 50:
                col[m] = sum(pert) / len(pert)
        table[b] = col
    return table


def true_ranking(acc_table, models):
    """Compute true ability ordering as mean accuracy across ALL available benchmarks."""
    means = {}
    for m in models:
        scores = []
        for b, col in acc_table.items():
            if m in col:
                scores.append(col[m])
        if scores:
            means[m] = np.mean(scores)
    return means


def policy_score(policy_name, acc_table, models, k):
    """For a given policy, pick k benchmarks and return per-model scores."""
    benches = list(acc_table.keys())
    np.random.seed(42)
    chosen = []

    if policy_name == "random":
        np.random.seed(0)
        chosen = list(np.random.choice(benches, size=min(k, len(benches)), replace=False))
    elif policy_name == "akhtar_low":
        # Pick k benchmarks with lowest saturation index (most informative per Akhtar)
        benches_sorted = sorted(benches, key=lambda b: AKHTAR_S.get(b, 1.0))
        chosen = benches_sorted[:k]
    elif policy_name == "akhtar_high":
        # Pick k with highest S (the wrong way — most "saturated")
        benches_sorted = sorted(benches, key=lambda b: -AKHTAR_S.get(b, 0.0))
        chosen = benches_sorted[:k]
    elif policy_name == "halflife_long":
        # Pick k with longest half-life (most measurement-stable)
        benches_sorted = sorted(benches, key=lambda b: -HALFLIFE.get(b, 60.0))
        chosen = benches_sorted[:k]
    elif policy_name == "halflife_short":
        # Pick k with shortest half-life (most decayed)
        benches_sorted = sorted(benches, key=lambda b: HALFLIFE.get(b, 60.0))
        chosen = benches_sorted[:k]

    # Compute per-model average over chosen benchmarks
    scores = {}
    for m in models:
        vals = [acc_table[b][m] for b in chosen if b in acc_table and m in acc_table[b]]
        if vals:
            scores[m] = np.mean(vals)
    return chosen, scores


def main():
    acc_table = load_per_model_acc()
    print(f"Loaded {len(acc_table)} benchmarks")

    # True ranking (oracle): all benchmarks
    true_means = true_ranking(acc_table, MODELS)
    sorted_models = sorted(true_means.keys(), key=lambda m: -true_means[m])
    print("\nOracle ranking (all benchmarks):")
    for m in sorted_models:
        print(f"  {m:<35} {true_means[m]:.3f}")

    true_rank = {m: i for i, m in enumerate(sorted_models)}

    # For each policy and each k, compute rank correlation
    policies = ["random", "akhtar_low", "akhtar_high", "halflife_long", "halflife_short"]
    policy_labels = {
        "random": "Random selection",
        "akhtar_low": "Akhtar S (low = informative)",
        "akhtar_high": "Akhtar S (high = saturated, intentionally bad)",
        "halflife_long": "Half-life (long = stable, ours)",
        "halflife_short": "Half-life (short = decayed, intentionally bad)",
    }

    results = {}
    for k in [1, 2, 3]:
        results[k] = {}
        for p in policies:
            chosen, scores = policy_score(p, acc_table, MODELS, k)
            if not scores:
                continue
            ranked = sorted(scores.keys(), key=lambda m: -scores[m])
            # Spearman vs oracle
            common = [m for m in MODELS if m in scores and m in true_rank]
            ranks_policy = [{m: i for i, m in enumerate(ranked)}[m] for m in common]
            ranks_oracle = [true_rank[m] for m in common]
            rho, _ = spearmanr(ranks_policy, ranks_oracle)
            results[k][p] = {
                "chosen": chosen,
                "rho": float(rho),
                "n_models": len(common),
            }
            print(f"\nk={k}, {p:<25} chosen={chosen} rho={rho:+.3f}")

    with open(OUT_DIR / "triage_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Plot: bar chart of rho by policy, by k
    fig, ax = plt.subplots(figsize=(11, 6))
    n_pol = len(policies)
    width = 0.18
    x = np.arange(3)  # k = 1, 2, 3
    colors = {
        "random": "#9E9E9E",
        "akhtar_low": "#1976D2",
        "akhtar_high": "#0D47A1",
        "halflife_long": "#388E3C",
        "halflife_short": "#1B5E20",
    }
    for i, p in enumerate(policies):
        rhos = [results[k][p]["rho"] for k in [1, 2, 3]]
        ax.bar(x + i*width - 2*width, rhos, width, label=policy_labels[p], color=colors[p], alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["k=1", "k=2", "k=3"], fontsize=11)
    ax.set_ylabel("Spearman ρ (policy ranking vs oracle ranking)", fontsize=11)
    ax.set_title("Benchmark Triage: Rank Recovery vs Selection Policy", fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5, label='Oracle (ρ=1)')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.5, 1.05)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_triage.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig_triage.png", bbox_inches='tight', dpi=150)
    print(f"\nSaved: {FIG_DIR / 'fig_triage.pdf'}")


if __name__ == "__main__":
    main()
