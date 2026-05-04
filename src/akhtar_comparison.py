"""
T6: Compare our half-life vs Akhtar et al. (2026) saturation index.

Akhtar saturation index: S = exp(-R_norm^2)
where R_norm = (top score - k-th score) / standard error

When R_norm is large (top model far above field), S is small (not saturated).
When R_norm is small (all models close), S is near 1 (saturated).

We compute S at the LATEST quarter for each benchmark and compare to our half-life.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "akhtar_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def saturation_index(scores, k_th=5):
    """
    Akhtar et al. (2026) saturation index.
    S = exp(-R_norm^2) where R_norm = score range normalized by score std.
    Near 1 = saturated (all models close); near 0 = wide spread.
    """
    if len(scores) < k_th + 1:
        return None
    sorted_scores = np.sort(scores)[::-1]
    top = sorted_scores[0]
    kth = sorted_scores[k_th - 1]
    score_range = top - kth
    sd = np.std(scores, ddof=1)
    if sd < 1e-9:
        return 1.0
    R_norm = score_range / sd
    return float(np.exp(-(R_norm ** 2)))


def compute_v1():
    df = pd.read_csv(DATA_DIR / "leaderboard" / "v1_scores_full.csv", parse_dates=['eval_date'])
    df = df.dropna(subset=['eval_date'])
    df['quarter'] = df['eval_date'].dt.to_period('Q')

    results = {}
    for bench in ['arc_challenge', 'hellaswag', 'truthfulqa', 'winogrande', 'gsm8k']:
        sub = df.dropna(subset=[bench])
        latest_q = sub['quarter'].max()
        latest_scores = sub[sub['quarter'] == latest_q][bench].values
        if len(latest_scores) >= 6:
            S = saturation_index(latest_scores)
            results[bench] = {
                'saturation_index': S,
                'n_models': len(latest_scores),
                'mean_score': float(np.mean(latest_scores)),
                'top_score': float(np.max(latest_scores)),
            }
    return results


def compute_v2():
    results = {}
    for bench in ['bbh', 'mmlu_pro', 'gpqa', 'ifeval', 'math_lvl5', 'musr']:
        df = pd.read_csv(DATA_DIR / "leaderboard" / f"{bench}_v2.csv", parse_dates=['release_date'])
        df = df.dropna(subset=['release_date', 'score'])
        df['quarter'] = df['release_date'].dt.to_period('Q')
        latest_q = df['quarter'].max()
        latest_scores = df[df['quarter'] == latest_q]['score'].values
        if len(latest_scores) >= 6:
            S = saturation_index(latest_scores)
            results[bench] = {
                'saturation_index': S,
                'n_models': len(latest_scores),
                'mean_score': float(np.mean(latest_scores)),
                'top_score': float(np.max(latest_scores)),
            }
    return results


def main():
    v1 = compute_v1()
    v2 = compute_v2()

    # Our half-life results
    our_results = {
        'winogrande': {'halflife': 1.0, 'best_model': 'stretched_exp', 'tau_best': 1.7},
        'hellaswag': {'halflife': 2.5, 'best_model': 'exponential', 'tau_best': 2.5},
        'arc_challenge': {'halflife': 2.7, 'best_model': 'logistic', 'tau_best': 3.4},
        'gsm8k': {'halflife': 4.6, 'best_model': 'logistic', 'tau_best': 4.9},
        'truthfulqa': {'halflife': 9.0, 'best_model': 'logistic', 'tau_best': 5.7},
        'bbh': {'halflife': 39.3, 'best_model': 'logistic', 'tau_best': 19.6},
        'mmlu_pro': {'halflife': None},
        'gpqa': {'halflife': None},
        'ifeval': {'halflife': None},
        'math_lvl5': {'halflife': None},
        'musr': {'halflife': None},
    }

    rows = []
    bench_labels = {
        'arc_challenge': 'ARC-Challenge', 'hellaswag': 'HellaSwag', 'truthfulqa': 'TruthfulQA',
        'winogrande': 'WinoGrande', 'gsm8k': 'GSM8K',
        'bbh': 'BBH', 'mmlu_pro': 'MMLU-PRO', 'gpqa': 'GPQA',
        'ifeval': 'IFEval', 'math_lvl5': 'MATH Lvl5', 'musr': 'MUSR'
    }

    print(f"{'Benchmark':<15} {'S_Akhtar':<12} {'S_status':<15} {'τ½ (mo)':<12} {'τ½ status':<15}")
    print("-" * 75)

    for bench, label in bench_labels.items():
        sat = v1.get(bench, {}).get('saturation_index') or v2.get(bench, {}).get('saturation_index')
        ours = our_results.get(bench, {}).get('halflife')
        tau_best = our_results.get(bench, {}).get('tau_best')

        s_status = ''
        if sat is not None:
            if sat > 0.9:
                s_status = 'SATURATED'
            elif sat > 0.5:
                s_status = 'partial'
            else:
                s_status = 'unsaturated'

        h_status = ''
        if ours is not None:
            if ours < 6:
                h_status = 'EXHAUSTED'
            elif ours < 12:
                h_status = 'fast decay'
            elif ours < 36:
                h_status = 'slow decay'
            else:
                h_status = 'very slow'
        else:
            h_status = 'not measurable'

        sat_str = f"{sat:.3f}" if sat is not None else "N/A"
        ours_str = f"{ours:.1f}" if ours is not None else "N/A"
        print(f"{label:<15} {sat_str:<12} {s_status:<15} {ours_str:<12} {h_status:<15}")

        rows.append({
            'benchmark': label,
            'saturation_index': sat,
            'sat_status': s_status,
            'halflife_exp': ours,
            'halflife_best': tau_best,
            'hl_status': h_status,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "comparison.csv", index=False)
    with open(OUT_DIR / "comparison.json", 'w') as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved: {OUT_DIR}")

    # Find disagreements
    print("\n=== Disagreements (different conclusions) ===")
    for r in rows:
        if r['saturation_index'] is None or r['halflife_exp'] is None:
            continue
        sat_says_done = r['saturation_index'] > 0.7
        hl_says_done = r['halflife_exp'] < 12
        if sat_says_done != hl_says_done:
            print(f"  {r['benchmark']}: S={r['saturation_index']:.3f} ({'SATURATED' if sat_says_done else 'OK'}) vs τ½={r['halflife_exp']:.1f}mo ({'DEAD' if hl_says_done else 'OK'})")


if __name__ == "__main__":
    main()
