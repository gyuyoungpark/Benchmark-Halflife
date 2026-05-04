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

    # Plot: halflife vs k for benchmarks that have measurable decay at multiple k
    fig, ax = plt.subplots(figsize=(11, 6))

    # V1 benchmarks
    v1_names = {'arc_challenge_v1': 'ARC-C', 'hellaswag_v1': 'HellaSwag',
                'truthfulqa_v1': 'TruthfulQA', 'winogrande_v1': 'WinoGrande', 'gsm8k_v1': 'GSM8K'}
    v2_names = {'bbh_v2': 'BBH'}

    colors = {'arc_challenge_v1': '#D32F2F', 'hellaswag_v1': '#C62828',
              'truthfulqa_v1': '#E64A19', 'winogrande_v1': '#B71C1C',
              'gsm8k_v1': '#E53935', 'bbh_v2': '#F57C00'}

    for key, label in {**v1_names, **v2_names}.items():
        if key not in all_results:
            continue
        ks, hls, lows, highs = [], [], [], []
        for k in k_values:
            v = all_results[key].get(k)
            if v is not None and v.get('point') is not None:
                ks.append(int(k*100))
                hls.append(v['point'])
                lows.append(v.get('ci_lo') or v['point'])
                highs.append(v.get('ci_hi') or v['point'])
        if ks:
            ax.plot(ks, hls, 'o-', label=label, color=colors[key], lw=2, markersize=7)
            ax.fill_between(ks, lows, highs, color=colors[key], alpha=0.15)

    ax.set_xlabel('Top-k stratum (%)', fontsize=12)
    ax.set_ylabel('Discriminative half-life (months)', fontsize=12)
    ax.set_title('Sensitivity of Half-Life Estimates to Stratum Size', fontsize=13, fontweight='bold')
    ax.set_xticks([5, 10, 15, 20, 25, 30, 40, 50])
    ax.legend(fontsize=10, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axvspan(10, 30, color='gray', alpha=0.15, label='Stable range')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_topk_sensitivity.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig_topk_sensitivity.png", bbox_inches='tight', dpi=150)
    print(f"\nSaved: {FIG_DIR / 'fig_topk_sensitivity.pdf'}")


if __name__ == "__main__":
    main()
