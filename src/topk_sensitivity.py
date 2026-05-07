"""
T5: Top-k stratification sensitivity analysis.

Question: how does halflife depend on the choice of top-k stratum?
We sweep k in {5%, 10%, 15%, 20%, 25%, 30%, 40%, 50%} for all 11 benchmarks.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bootstrap_halflife import bootstrap_v1_benchmark, bootstrap_v2_benchmark

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"
OUT_DIR = DATA_DIR / "topk_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    k_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    all_results = {}

    print("V2 benchmarks")
    for bench in ['bbh', 'mmlu_pro', 'gpqa', 'ifeval', 'math_lvl5', 'musr']:
        all_results[f"{bench}_v2"] = {}
        print(f"\n  {bench}")
        for k in k_values:
            r = bootstrap_v2_benchmark(bench, top_k=k, n_boot=300)
            if r and r.get('point') is not None:
                all_results[f"{bench}_v2"][k] = {
                    'point': r['point'],
                    'ci_lo': r.get('ci_lo'),
                    'ci_hi': r.get('ci_hi'),
                }
                ci = f"[{r['ci_lo']:.1f},{r['ci_hi']:.1f}]" if r.get('ci_lo') else 'N/A'
                print(f"    k={int(k*100)}%: τ½={r['point']:.1f}mo {ci}")
            else:
                all_results[f"{bench}_v2"][k] = None
                print(f"    k={int(k*100)}%: no decay")

    print("\nV1 benchmarks")
    for bench in ['arc_challenge', 'hellaswag', 'truthfulqa', 'winogrande', 'gsm8k']:
        all_results[f"{bench}_v1"] = {}
        print(f"\n  {bench}")
        for k in k_values:
            r = bootstrap_v1_benchmark(bench, top_k=k, n_boot=300)
            if r and r.get('point') is not None:
                all_results[f"{bench}_v1"][k] = {
                    'point': r['point'],
                    'ci_lo': r.get('ci_lo'),
                    'ci_hi': r.get('ci_hi'),
                }
                ci = f"[{r['ci_lo']:.1f},{r['ci_hi']:.1f}]" if r.get('ci_lo') else 'N/A'
                print(f"    k={int(k*100)}%: τ½={r['point']:.1f}mo {ci}")
            else:
                all_results[f"{bench}_v1"][k] = None
                print(f"    k={int(k*100)}%: no decay")

    with open(OUT_DIR / "topk_sweep.json", 'w') as f:
        # serialize numpy/None
        s = json.dumps(all_results, indent=2, default=lambda x: None if x is None else float(x))
        f.write(s)

    # Plot: halflife vs k for all 11 benchmarks
    fig, ax = plt.subplots(figsize=(11, 6))

    # Stable range shading drawn first (background)
    ax.axvspan(10, 30, color='#888', alpha=0.10, zorder=0)
    ax.text(20, 0.55, 'k=20% (recommended)', fontsize=8.5, color='#555',
            ha='center', va='bottom', style='italic', zorder=1)

    # v1 benchmarks (warm reds): 5 lines, all have multiple points
    v1_specs = [
        ('arc_challenge_v1', 'ARC-Challenge (v1)', '#B71C1C', '-'),
        ('hellaswag_v1',     'HellaSwag (v1)',    '#C62828', '-'),
        ('truthfulqa_v1',    'TruthfulQA (v1)',   '#D84315', '-'),
        ('winogrande_v1',    'WinoGrande (v1)',   '#AD1457', '-'),
        ('gsm8k_v1',         'GSM8K (v1)',        '#E53935', '-'),
    ]
    # v2 benchmarks (warm oranges/teals): split by data availability
    v2_lines = [
        ('bbh_v2',    'BBH (v2)',    '#E65100', '--'),
        ('ifeval_v2', 'IFEval (v2)', '#FF8F00', '--'),
    ]
    v2_markers = [
        ('mmlu_pro_v2',  'MMLU-PRO (v2)',  '#00897B', 's'),
        ('math_lvl5_v2', 'MATH Lvl 5 (v2)', '#1565C0', 'D'),
        ('gpqa_v2',      'GPQA (v2)',      '#6A1B9A', '^'),
    ]

    # Lines (multi-point)
    for key, label, color, linestyle in v1_specs + v2_lines:
        if key not in all_results:
            continue
        ks, hls, lows, highs = [], [], [], []
        for k in k_values:
            v = all_results[key].get(k)
            if v is not None and v.get('point') is not None:
                ks.append(int(k * 100))
                hls.append(v['point'])
                lows.append(v.get('ci_lo') or v['point'])
                highs.append(v.get('ci_hi') or v['point'])
        if len(ks) >= 2:
            ax.plot(ks, hls, marker='o', linestyle=linestyle, label=label,
                    color=color, lw=1.8, markersize=6, zorder=3)
            ax.fill_between(ks, lows, highs, color=color, alpha=0.12, zorder=2)

    # Single-point markers (v2 with only k=10% measurable)
    for key, label, color, marker in v2_markers:
        if key not in all_results:
            continue
        for k in k_values:
            v = all_results[key].get(k)
            if v is not None and v.get('point') is not None:
                ax.scatter([int(k * 100)], [v['point']], marker=marker,
                           s=70, color=color, edgecolor='white', linewidth=1.0,
                           label=label + ' (single $k$)', zorder=4)
                break  # only need one marker

    ax.set_xlabel('Top-$k$ stratum (\\%)', fontsize=12)
    ax.set_ylabel('Discriminative half-life (months)', fontsize=12)
    ax.set_title('Sensitivity of half-life to top-$k$ stratum (11 benchmarks)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks([5, 10, 15, 20, 25, 30, 40, 50])
    ax.set_xlim(3, 53)
    ax.set_yscale('log')
    ax.set_ylim(0.5, 250)
    ax.grid(True, alpha=0.25, which='both')
    ax.legend(fontsize=8.5, loc='upper right', ncol=2, framealpha=0.95,
              edgecolor='#888', fancybox=True)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_topk_sensitivity.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig_topk_sensitivity.png", bbox_inches='tight', dpi=150)
    print(f"\nSaved: {FIG_DIR / 'fig_topk_sensitivity.pdf'}")


if __name__ == "__main__":
    main()
