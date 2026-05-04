"""
P2: Prospective forecasting validation.

Fit decay model on early subset of data, predict later subset.
For each v2 benchmark with measurable decay (BBH, MMLU-PRO, etc.):
  Train: 2022Q1 - 2024Q2 (10 quarters)
  Test:  2024Q3 - 2025Q1 (3 quarters)
  Metric: predicted vs actual top-stratum variance + ceiling-crossing time

For v1 benchmarks (only 4 quarters total), use leave-last-quarter-out.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path(__file__).parent.parent / "figures"


def exp_decay(t, D0, lam, D_inf):
    return D0 * np.exp(-lam * t) + D_inf


def logistic_decay(t, D0, lam, t0, D_inf):
    return (D0 - D_inf) / (1.0 + np.exp(lam * (t - t0))) + D_inf


def collect_v2_quarterly(bench, top_k=0.20):
    df = pd.read_csv(DATA_DIR / "leaderboard" / f"{bench}_v2.csv", parse_dates=['release_date'])
    df = df.dropna(subset=['release_date', 'score'])
    df['quarter'] = df['release_date'].dt.to_period('Q')
    quarters = sorted(df['quarter'].unique())
    rows = []
    for q in quarters:
        q_df = df[df['quarter'] == q]
        if len(q_df) < 10:
            continue
        thresh = q_df['score'].quantile(1 - top_k)
        top = q_df[q_df['score'] >= thresh]
        if len(top) >= 3:
            months = (q.start_time - quarters[0].start_time).days / 30.44
            rows.append({
                'quarter': str(q),
                'months': months,
                'variance': float(np.var(top['score'].values, ddof=1)),
                'mean_score': float(top['score'].mean()),
                'max_score': float(top['score'].max()),
                'n': len(top),
            })
    return pd.DataFrame(rows)


def fit_and_predict(train_df, test_df, allow_small=False):
    """Fit on train, predict on test. Returns predicted vs actual variances."""
    min_n = 2 if allow_small else 4
    if len(train_df) < min_n:
        return None
    t_train = train_df['months'].values
    D_train = train_df['variance'].values

    # Try exponential
    try:
        if len(t_train) >= 3:
            popt, _ = curve_fit(exp_decay, t_train, D_train,
                p0=[D_train[0] if D_train[0] > 0 else 0.001, 0.05, D_train[-1]*0.3 if D_train[-1] > 0 else 0],
                bounds=([0, 1e-6, 0], [D_train.max()*5+1e-6, 1.0, D_train.max()+1e-6]),
                maxfev=10000)
        else:
            # 2-parameter exponential without floor
            def exp_2p(t, D0, lam):
                return D0 * np.exp(-lam * t)
            popt2, _ = curve_fit(exp_2p, t_train, D_train,
                p0=[D_train[0] if D_train[0] > 0 else 0.001, 0.05],
                bounds=([0, 1e-6], [D_train.max()*5+1e-6, 1.0]),
                maxfev=10000)
            popt = list(popt2) + [0.0]
        D_pred = exp_decay(test_df['months'].values, *popt)
        return {
            'method': 'exponential',
            'params': list(popt),
            'predicted': D_pred.tolist(),
            'actual': test_df['variance'].tolist(),
            'test_months': test_df['months'].tolist(),
            'train_months': t_train.tolist(),
            'train_variance': D_train.tolist(),
        }
    except Exception as e:
        return None


def collect_v1_quarterly(bench, top_k=0.20):
    df = pd.read_csv(DATA_DIR / "leaderboard" / "v1_scores_full.csv", parse_dates=['eval_date'])
    df = df.dropna(subset=[bench, 'eval_date'])
    df['quarter'] = df['eval_date'].dt.to_period('Q')
    quarters = sorted(df['quarter'].unique())
    rows = []
    for q in quarters:
        scores = df[df['quarter'] == q][bench].values
        if len(scores) < 15:
            continue
        thresh = np.percentile(scores, 100 * (1 - top_k))
        top = scores[scores >= thresh]
        if len(top) >= 3:
            months = (q.start_time - quarters[0].start_time).days / 30.44
            rows.append({
                'quarter': str(q),
                'months': months,
                'variance': float(np.var(top, ddof=1)),
                'mean_score': float(np.mean(top)),
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("V1 LEAVE-LAST-QUARTER-OUT FORECAST")
    print("=" * 60)
    v1_bench = ['arc_challenge', 'hellaswag', 'truthfulqa', 'winogrande', 'gsm8k']
    v1_results = {}
    for bench in v1_bench:
        df = collect_v1_quarterly(bench)
        if len(df) < 3:
            continue
        # Train on first n-1 quarters, predict last quarter
        train = df.iloc[:-1]
        test = df.iloc[-1:]
        result = fit_and_predict(train, test, allow_small=True)
        if result is None:
            continue
        predicted = result['predicted'][0]
        actual = result['actual'][0]
        rel_err = abs(predicted - actual) / max(actual, 1e-6)
        v1_results[bench] = {
            'predicted': predicted,
            'actual': actual,
            'rel_error': rel_err,
            'train_n': len(train),
        }
        print(f"  {bench:<20} predicted={predicted:.5f}  actual={actual:.5f}  rel_err={rel_err*100:.1f}%")

    if v1_results:
        mean_rel = np.mean([v['rel_error'] for v in v1_results.values()])
        print(f"\n  Mean relative error: {mean_rel*100:.1f}%")

    print("\n" + "=" * 60)
    print("V2 70/30 SPLIT FORECAST")
    print("=" * 60)
    benchmarks = ['bbh', 'mmlu_pro', 'gpqa', 'ifeval', 'math_lvl5', 'musr']
    all_results = {}
    all_results['v1'] = v1_results

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, bench in enumerate(benchmarks):
        df = collect_v2_quarterly(bench)
        if len(df) < 6:
            print(f"{bench}: insufficient data")
            continue

        # Train: first 70% of quarters, Test: last 30%
        n = len(df)
        split = int(n * 0.70)
        train = df.iloc[:split].copy()
        test = df.iloc[split:].copy()

        if len(test) < 1:
            continue

        result = fit_and_predict(train, test)
        if result is None:
            print(f"{bench}: fit failed")
            continue

        predicted = np.array(result['predicted'])
        actual = np.array(result['actual'])
        # Skip benchmarks where train data shows no decay
        train_var = np.array(result['train_variance'])
        train_slope = np.polyfit(result['train_months'], train_var, 1)[0]

        ape = np.abs((predicted - actual) / np.maximum(actual, 1e-6))
        mape = np.mean(ape)
        all_results[bench] = {
            'split_quarter': str(df.iloc[split]['quarter']) if split < len(df) else None,
            'n_train': len(train),
            'n_test': len(test),
            'train_slope': float(train_slope),
            'train_decaying': bool(train_slope < 0),
            'predicted': predicted.tolist(),
            'actual': actual.tolist(),
            'mape': float(mape),
            'mean_abs_error': float(np.mean(np.abs(predicted - actual))),
        }
        print(f"{bench}: train_slope={train_slope:+.6f}, MAPE={mape*100:.1f}%, MAE={np.mean(np.abs(predicted - actual)):.4f}")

        ax = axes[idx]
        ax.plot(train['months'], train['variance'], 'bo-', label='Train (observed)', markersize=8)
        ax.plot(test['months'], test['variance'], 'go', markersize=10, label='Test (held out)', markeredgecolor='black', markeredgewidth=1)
        # Fitted curve
        t_full = np.linspace(0, df['months'].max() * 1.1, 100)
        if result['method'] == 'exponential':
            D_full = exp_decay(t_full, *result['params'])
        ax.plot(t_full, D_full, 'r--', alpha=0.7, label='Forecast (exp fit on train)')
        ax.plot(test['months'], predicted, 'rx', markersize=12, mew=2, label='Predicted')
        ax.axvline(x=train['months'].iloc[-1] + 1.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Months')
        ax.set_ylabel('Top-20% variance')
        ax.set_title(f"{bench.upper()}\nMAPE={mape*100:.1f}%", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Prospective Forecasting: Train on First 70%, Predict Last 30%', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_prospective_forecast.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig_prospective_forecast.png", bbox_inches='tight', dpi=150)
    print(f"\nSaved: {FIG_DIR / 'fig_prospective_forecast.pdf'}")

    with open(OUT_DIR / "forecast_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n=== Forecast accuracy ===")
    decaying = {k: v for k, v in all_results.items() if v.get('train_decaying')}
    if decaying:
        mean_mape = np.mean([v['mape'] for v in decaying.values()])
        print(f"Decaying benchmarks (n={len(decaying)}): mean MAPE = {mean_mape*100:.1f}%")
        for k, v in decaying.items():
            print(f"  {k}: MAPE = {v['mape']*100:.1f}%")


if __name__ == "__main__":
    main()
