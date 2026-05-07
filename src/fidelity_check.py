"""
R3: Perturbation fidelity verification.

Tests:
1. Item-level correlation between orig and pert correctness across the 4 GPT models
2. IRT difficulty correlation between orig and pert
3. Per-item agreement rate (4 models simultaneously correct on both vs different)
4. Sample paraphrase examples
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
PERT_DIR = DATA_DIR / "perturbations"
OUT_DIR = DATA_DIR / "fidelity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path(__file__).parent.parent / "figures"

MODELS = [
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-08-06",
    "gpt-4.1-2025-04-14",
]


def analyze(name):
    path = EVAL_DIR / f"{name}_results.json"
    with open(path) as f:
        results = json.load(f)
    n_items = len(results[MODELS[0]]['orig'])

    # Item-level acc across models (excluding any None)
    orig_acc = np.array([
        np.mean([results[m]['orig'][i] for m in MODELS if results[m]['orig'][i] is not None])
        for i in range(n_items)
    ])
    pert_acc = np.array([
        np.mean([results[m]['pert'][i] for m in MODELS if results[m]['pert'][i] is not None])
        for i in range(n_items)
    ])

    # Empirical "difficulty" (logit of mean accuracy)
    eps = 1e-3
    p_o = np.clip(orig_acc, eps, 1 - eps)
    p_p = np.clip(pert_acc, eps, 1 - eps)
    diff_o = -np.log(p_o / (1 - p_o))
    diff_p = -np.log(p_p / (1 - p_p))

    # Correlations
    pear_acc, _ = pearsonr(orig_acc, pert_acc)
    spear_acc, _ = spearmanr(orig_acc, pert_acc)
    pear_diff, _ = pearsonr(diff_o, diff_p)

    # Item-level agreement: same correct count across 4 models on orig and pert
    agree_count = sum(1 for i in range(n_items) if abs(orig_acc[i] - pert_acc[i]) < 0.001)
    agree_within_25 = sum(1 for i in range(n_items) if abs(orig_acc[i] - pert_acc[i]) < 0.25)

    # Mean shift (positive = orig is easier on average)
    mean_shift = orig_acc.mean() - pert_acc.mean()

    return {
        'name': name,
        'n_items': n_items,
        'pearson_acc': pear_acc,
        'spearman_acc': spear_acc,
        'pearson_difficulty': pear_diff,
        'mean_shift': mean_shift,
        'frac_exact_match': agree_count / n_items,
        'frac_within_25pct': agree_within_25 / n_items,
        'orig_acc': orig_acc.tolist(),
        'pert_acc': pert_acc.tolist(),
    }


def main():
    results = {}
    for bench in ['mmlu', 'arc', 'gsm8k']:
        r = analyze(bench)
        results[bench] = r
        print(f"\n=== {bench.upper()} ({r['n_items']} items) ===")
        print(f"  Pearson r (orig acc vs pert acc):     {r['pearson_acc']:.3f}")
        print(f"  Spearman ρ:                            {r['spearman_acc']:.3f}")
        print(f"  Pearson r (orig logit vs pert logit): {r['pearson_difficulty']:.3f}")
        print(f"  Mean shift (orig - pert):             {r['mean_shift']:+.3f}")
        print(f"  Items with exact match:                {r['frac_exact_match']*100:.1f}%")
        print(f"  Items with within ±25%:                {r['frac_within_25pct']*100:.1f}%")

    with open(OUT_DIR / "fidelity_check.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Plot scatter of orig vs pert difficulty (logit space) for each benchmark
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, bench in zip(axes, ['mmlu', 'arc', 'gsm8k']):
        r = results[bench]
        orig_acc = np.array(r['orig_acc'])
        pert_acc = np.array(r['pert_acc'])
        ax.scatter(orig_acc, pert_acc, alpha=0.4, s=30, color='#1976D2')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('Original item accuracy (mean across 4 models)', fontsize=10)
        ax.set_ylabel('Perturbation item accuracy', fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{bench.upper()}  $r = {r['pearson_acc']:.3f}$,  shift $ = {r['mean_shift']:+.3f}$",
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_fidelity_check.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig_fidelity_check.png", bbox_inches='tight', dpi=150)
    print(f"\nSaved: {FIG_DIR / 'fig_fidelity_check.pdf'}")

    # Sample paraphrases for human inspection
    print("\n=== Sample perturbation pairs ===")
    for bench in ['mmlu', 'arc', 'gsm8k']:
        with open(PERT_DIR / f"{bench}_perturbed.json") as f:
            items = json.load(f)
        print(f"\n--- {bench.upper()} ---")
        for i in [0, 50, 100]:
            if i < len(items):
                it = items[i]
                if 'orig_question' in it:
                    print(f"\n  [item {i}] ORIG: {it['orig_question'][:150]}")
                    print(f"           PERT: {it['pert_question'][:150]}")


if __name__ == "__main__":
    main()
