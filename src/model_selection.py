"""
T4: AIC/BIC table for decay model selection.

Compares 3 decay models (exponential, stretched exponential, logistic decay)
on each (benchmark, top-k=20%) pair. Reports AIC/BIC and ΔAIC.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "model_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def exp_decay(t, D0, lam, D_inf):
    return D0 * np.exp(-lam * t) + D_inf


def stretched_exp(t, D0, lam, beta, D_inf):
    return D0 * np.exp(-(lam * t) ** beta) + D_inf


def logistic_decay(t, D0, lam, t0, D_inf):
    return (D0 - D_inf) / (1.0 + np.exp(lam * (t - t0))) + D_inf


def aic_bic(rss, n, k):
    """Compute AIC and BIC from residual sum of squares."""
    if rss <= 0:
        rss = 1e-12
    nll = (n / 2) * (np.log(2 * np.pi * rss / n) + 1)  # gaussian NLL
    aic = 2 * k + 2 * nll
    bic = k * np.log(n) + 2 * nll
    return aic, bic


def fit_all_models(t, D):
    """Fit all 3 decay models, return AIC/BIC for each."""
    n = len(t)
    if n < 3:
        return None
    if np.polyfit(t, D, 1)[0] >= 0:
        return None  # not decaying

    results = {}

    # Exponential (3 params)
    try:
        popt, _ = curve_fit(exp_decay, t, D,
            p0=[D[0], 0.05, D[-1]*0.3 if D[-1]>0 else 0],
            bounds=([0, 1e-6, 0], [D.max()*3, 1.0, D.max()]),
            maxfev=10000)
        rss = np.sum((D - exp_decay(t, *popt))**2)
        aic, bic = aic_bic(rss, n, 3)
        hl = np.log(2) / popt[1]
        results['exponential'] = {'params': popt.tolist(), 'rss': float(rss),
                                   'aic': float(aic), 'bic': float(bic),
                                   'halflife': float(hl), 'k': 3}
    except Exception:
        pass

    # Stretched exponential (4 params)
    try:
        popt, _ = curve_fit(stretched_exp, t, D,
            p0=[D[0], 0.05, 1.0, D[-1]*0.3 if D[-1]>0 else 0],
            bounds=([0, 1e-6, 0.1, 0], [D.max()*3, 1.0, 3.0, D.max()]),
            maxfev=10000)
        rss = np.sum((D - stretched_exp(t, *popt))**2)
        aic, bic = aic_bic(rss, n, 4)
        # halflife: solve D0*exp(-(lam*t)^beta) = D0/2
        hl = (np.log(2) ** (1/popt[2])) / popt[1]
        results['stretched_exp'] = {'params': popt.tolist(), 'rss': float(rss),
                                     'aic': float(aic), 'bic': float(bic),
                                     'halflife': float(hl), 'k': 4}
    except Exception:
        pass

    # Logistic decay (4 params)
    try:
        popt, _ = curve_fit(logistic_decay, t, D,
            p0=[D[0], 0.1, np.median(t), D[-1]*0.3 if D[-1]>0 else 0],
            bounds=([0, 1e-6, 0, 0], [D.max()*3, 2.0, t.max()*2, D.max()]),
            maxfev=10000)
        rss = np.sum((D - logistic_decay(t, *popt))**2)
        aic, bic = aic_bic(rss, n, 4)
        hl = popt[2]  # inflection point
        results['logistic_decay'] = {'params': popt.tolist(), 'rss': float(rss),
                                      'aic': float(aic), 'bic': float(bic),
                                      'halflife': float(hl), 'k': 4}
    except Exception:
        pass

    if not results:
        return None
    # ΔAIC
    min_aic = min(r['aic'] for r in results.values())
    for r in results.values():
        r['delta_aic'] = r['aic'] - min_aic
    return results


def get_v1_data(bench, top_k=0.2):
    df = pd.read_csv(DATA_DIR / "leaderboard" / "v1_scores_full.csv", parse_dates=['eval_date'])
    df = df.dropna(subset=[bench, 'eval_date'])
    df['quarter'] = df['eval_date'].dt.to_period('Q')
    quarters = sorted(df['quarter'].unique())
    rows = []
    for q in quarters:
        scores = df[df['quarter'] == q][bench].values
        if len(scores) >= 15:
            thresh = np.percentile(scores, 100 * (1 - top_k))
            top = scores[scores >= thresh]
            if len(top) >= 3:
                months = (q.start_time - quarters[0].start_time).days / 30.44
                rows.append((months, np.var(top, ddof=1)))
    if not rows:
        return None, None
    t, D = zip(*rows)
    return np.array(t), np.array(D)


def get_v2_data(bench, top_k=0.2):
    df = pd.read_csv(DATA_DIR / "leaderboard" / f"{bench}_v2.csv", parse_dates=['release_date'])
    df = df.dropna(subset=['release_date', 'score'])
    df['quarter'] = df['release_date'].dt.to_period('Q')
    quarters = sorted(df['quarter'].unique())
    rows = []
    for q in quarters:
        q_df = df[df['quarter'] == q]
        if len(q_df) >= 10:
            thresh = q_df['score'].quantile(1 - top_k)
            top = q_df[q_df['score'] >= thresh]
            if len(top) >= 3:
                months = (q.start_time - quarters[0].start_time).days / 30.44
                rows.append((months, np.var(top['score'].values, ddof=1)))
    if not rows:
        return None, None
    t, D = zip(*rows)
    return np.array(t), np.array(D)


def main():
    benchmarks = [
        ('arc_challenge_v1', 'ARC-Challenge', get_v1_data, 'arc_challenge'),
        ('hellaswag_v1', 'HellaSwag', get_v1_data, 'hellaswag'),
        ('truthfulqa_v1', 'TruthfulQA', get_v1_data, 'truthfulqa'),
        ('winogrande_v1', 'WinoGrande', get_v1_data, 'winogrande'),
        ('gsm8k_v1', 'GSM8K', get_v1_data, 'gsm8k'),
        ('bbh_v2', 'BBH', get_v2_data, 'bbh'),
    ]

    all_results = {}
    print(f"{'Benchmark':<15} {'Model':<18} {'AIC':<10} {'BIC':<10} {'ΔAIC':<8} {'τ½ (mo)':<10}")
    print("-" * 75)

    for key, label, getter, bench in benchmarks:
        t, D = getter(bench)
        if t is None or len(t) < 3:
            print(f"{label:<15} insufficient data")
            continue
        r = fit_all_models(t, D)
        if r is None:
            print(f"{label:<15} no decay")
            continue
        all_results[key] = r
        for model, vals in sorted(r.items(), key=lambda x: x[1]['aic']):
            best_marker = ' *' if vals['delta_aic'] < 0.01 else ''
            print(f"{label:<15} {model:<18} {vals['aic']:<10.2f} {vals['bic']:<10.2f} {vals['delta_aic']:<8.2f} {vals['halflife']:<10.1f}{best_marker}")
        print()

    with open(OUT_DIR / "aic_bic_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {OUT_DIR / 'aic_bic_results.json'}")


if __name__ == "__main__":
    main()
