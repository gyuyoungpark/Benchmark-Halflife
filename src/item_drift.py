"""
T1: Item Parameter Drift analysis on real evaluation data.

Approach: We have 4 GPT models × 628 items × 2 versions (orig/pert).
With only 4 models, traditional IRT is underdetermined.
But we can directly track item-level drift between model generations.

For each item i:
- Compute correctness vector across 4 models in temporal order
- Identify drift category:
  * Stable: no significant pattern
  * Difficulty-drifting: monotonically becoming "easier" (b decreasing)
  * Discrimination-collapsing: 0->1 transition that doesn't track ability
  * Ceiling: all 4 models correct
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_DIR = DATA_DIR / "evaluations"
OUT_DIR = DATA_DIR / "item_drift"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path(__file__).parent.parent / "figures"

MODEL_ORDER = [
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-08-06",
    "gpt-4.1-2025-04-14",
]

# Approximate ability ordering (from MMLU/GSM8K aggregate scores)
MODEL_ABILITY = {
    "gpt-3.5-turbo-0125": 0.0,
    "gpt-4-turbo-2024-04-09": 1.0,
    "gpt-4o-2024-08-06": 1.5,
    "gpt-4.1-2025-04-14": 1.5,
}


def classify_item(orig_responses, pert_responses):
    """
    Classify a single item's drift pattern based on responses across model generations.

    orig_responses: list of {0,1,None} length 4 (one per model in order)
    pert_responses: list of {0,1,None} length 4

    Categories:
    - 'all_correct': all 4 models correct on both (ceiling item)
    - 'all_wrong': all 4 models wrong (genuinely hard)
    - 'monotone_increasing': 0...0,1...1 pattern (genuine difficulty drift)
    - 'noisy': non-monotone or contains None
    - 'contamination_signature': models correct on orig but not pert
    - 'discrimination_loss': originally split, now all correct/wrong
    """
    valid_orig = [x for x in orig_responses if x is not None]
    valid_pert = [x for x in pert_responses if x is not None]
    if len(valid_orig) < 4 or len(valid_pert) < 4:
        return 'incomplete'

    n_correct_orig = sum(orig_responses)
    n_correct_pert = sum(pert_responses)

    # Ceiling items
    if n_correct_orig == 4 and n_correct_pert == 4:
        return 'all_correct'
    if n_correct_orig == 0 and n_correct_pert == 0:
        return 'all_wrong'

    # Contamination signature: orig consistently right, pert wrong
    contam_count = sum(1 for o, p in zip(orig_responses, pert_responses) if o == 1 and p == 0)
    anti_contam = sum(1 for o, p in zip(orig_responses, pert_responses) if o == 0 and p == 1)
    if contam_count >= 2 and anti_contam == 0:
        return 'contamination_signature'

    # Monotone increasing in orig (genuine progress)
    is_monotone_orig = True
    for i in range(1, len(orig_responses)):
        if orig_responses[i] < orig_responses[i-1]:
            is_monotone_orig = False
            break
    if is_monotone_orig and n_correct_orig > 0 and n_correct_orig < 4:
        return 'difficulty_drift'

    # Discrimination loss: originally discriminating but no longer
    if n_correct_orig in {1, 2, 3}:
        return 'discriminating'

    return 'noisy'


def analyze_benchmark(name):
    path = EVAL_DIR / f"{name}_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        results = json.load(f)

    # Restrict to MODEL_ORDER models
    available = [m for m in MODEL_ORDER if m in results]
    if len(available) < 4:
        print(f"{name}: only {len(available)} models")
        return None

    # Get response matrices: orig and pert
    n_items = len(results[available[0]]['orig'])
    orig_mat = np.array([[results[m]['orig'][i] if results[m]['orig'][i] is not None else None
                          for i in range(n_items)] for m in available], dtype=object)
    pert_mat = np.array([[results[m]['pert'][i] if results[m]['pert'][i] is not None else None
                          for i in range(n_items)] for m in available], dtype=object)

    # Per-item classification
    categories = []
    for i in range(n_items):
        orig_resp = list(orig_mat[:, i])
        pert_resp = list(pert_mat[:, i])
        cat = classify_item(orig_resp, pert_resp)
        categories.append(cat)

    cat_series = pd.Series(categories)
    counts = cat_series.value_counts()
    fractions = counts / len(categories)

    return {
        'n_items': n_items,
        'category_counts': counts.to_dict(),
        'category_fractions': fractions.to_dict(),
    }


def main():
    all_results = {}
    for bench in ['mmlu', 'arc', 'gsm8k']:
        r = analyze_benchmark(bench)
        if r is None:
            continue
        all_results[bench] = r
        print(f"\n=== {bench.upper()} ({r['n_items']} items) ===")
        for cat, frac in sorted(r['category_fractions'].items(), key=lambda x: -x[1]):
            cnt = r['category_counts'][cat]
            print(f"  {cat:<25} {cnt:>4} ({frac*100:.1f}%)")

    with open(OUT_DIR / "item_drift_analysis.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Plot: stacked bars showing category distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    categories_to_plot = ['all_correct', 'difficulty_drift', 'contamination_signature',
                           'discriminating', 'all_wrong', 'noisy']
    colors = {
        'all_correct': '#FFA000',           # ceiling
        'difficulty_drift': '#1976D2',      # genuine progress
        'contamination_signature': '#D32F2F',# contamination
        'discriminating': '#4CAF50',        # still useful
        'all_wrong': '#616161',             # too hard
        'noisy': '#9E9E9E',                 # noise
    }
    pretty = {
        'all_correct': 'Ceiling (all correct)',
        'difficulty_drift': 'Difficulty drift (genuine progress)',
        'contamination_signature': 'Contamination signature',
        'discriminating': 'Still discriminating',
        'all_wrong': 'All wrong (too hard)',
        'noisy': 'Noisy',
    }

    benchmarks = list(all_results.keys())
    bench_labels = ['MMLU', 'ARC-C', 'GSM8K']
    x = np.arange(len(benchmarks))

    bottom = np.zeros(len(benchmarks))
    for cat in categories_to_plot:
        vals = [all_results[b]['category_fractions'].get(cat, 0) for b in benchmarks]
        ax.bar(x, vals, bottom=bottom, label=pretty[cat], color=colors[cat],
               edgecolor='black', linewidth=0.5)
        for i, v in enumerate(vals):
            if v > 0.05:
                ax.text(i, bottom[i] + v/2, f'{v*100:.0f}%', ha='center', va='center',
                       color='white', fontweight='bold', fontsize=10)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(bench_labels, fontsize=12)
    ax.set_ylabel('Fraction of items', fontsize=12)
    ax.set_title('Item-Level Drift Categories Across 4 GPT Generations',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_item_drift.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig_item_drift.png", bbox_inches='tight', dpi=150)
    print(f"\nSaved: {FIG_DIR / 'fig_item_drift.pdf'}")


if __name__ == "__main__":
    main()
