"""
Benchmark archaeology: compute retrospective half-lives for historical benchmarks.
Uses gap-to-ceiling method (same as GLUE).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data" / "leaderboard"
FIG_DIR = Path(__file__).parent.parent / "figures"
OUT_DIR = Path(__file__).parent.parent / "data" / "archaeology"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def exp_decay(t, D0, lam, D_inf):
    return D0 * np.exp(-lam * t) + D_inf


def compute_halflife_gap_to_ceiling(df, ceiling=100):
    """Fit exponential decay to (ceiling - max_score) trajectory."""
    df = df.copy()
    df = df.sort_values('release_date').reset_index(drop=True)
    t0 = df['release_date'].iloc[0]
    df['months'] = (df['release_date'] - t0).dt.days / 30.44

    # Running max score over time (SOTA trajectory)
    df['running_max'] = df['score'].cummax()
    df['gap_to_ceiling'] = ceiling - df['running_max']

    # Deduplicate by month (keep smallest gap)
    df_fit = df.groupby('months').agg({
        'gap_to_ceiling': 'min',
        'running_max': 'max',
    }).reset_index()

    t = df_fit['months'].values
    gap = df_fit['gap_to_ceiling'].values

    if len(t) < 4:
        return None

    # Fit exponential
    try:
        popt, _ = curve_fit(exp_decay, t, gap,
            p0=[gap[0] if gap[0] > 0 else 10, 0.05, gap[-1] * 0.5],
            bounds=([0, 1e-6, 0], [gap.max()*5+1e-6, 1.0, gap.max()+1e-6]),
            maxfev=10000)
        halflife = np.log(2) / popt[1]
        return {
            'halflife_mo': float(halflife),
            'D0': float(popt[0]),
            'lam': float(popt[1]),
            'D_inf': float(popt[2]),
            'fit_points': df_fit.to_dict(orient='records'),
        }
    except Exception as e:
        return None


BENCHMARKS = {
    'glue':       {'file': 'glue_historical.csv', 'ceiling': 100, 'diversity': 3, 'n_items': 65000},
    'superglue':  {'file': 'superglue_historical.csv', 'ceiling': 100, 'diversity': 4, 'n_items': 5000},
    'squad11':    {'file': 'squad11_historical.csv', 'ceiling': 100, 'diversity': 2, 'n_items': 10570},
    'squad20':    {'file': 'squad20_historical.csv', 'ceiling': 100, 'diversity': 2, 'n_items': 11873},
    'coqa':       {'file': 'coqa_historical.csv', 'ceiling': 100, 'diversity': 2, 'n_items': 7983},
}


def main():
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, (name, info) in enumerate(BENCHMARKS.items()):
        path = DATA_DIR / info['file']
        if not path.exists():
            print(f"Missing: {path}")
            continue
        df = pd.read_csv(path, parse_dates=['release_date'])
        result = compute_halflife_gap_to_ceiling(df, info['ceiling'])
        if result is None:
            print(f"{name}: fit failed")
            continue
        results[name] = {
            'halflife_mo': result['halflife_mo'],
            'D0': result['D0'],
            'diversity': info['diversity'],
            'n_items': info['n_items'],
            'ceiling': info['ceiling'],
            'n_data_points': len(result['fit_points']),
            'first_date': df['release_date'].min().strftime('%Y-%m'),
            'last_date': df['release_date'].max().strftime('%Y-%m'),
        }
        print(f"{name.upper():<12} τ½ = {result['halflife_mo']:.1f} mo ({len(result['fit_points'])} points, {df['release_date'].min():%Y-%m} → {df['release_date'].max():%Y-%m})")

        # Plot
        ax = axes[idx]
        df_sorted = df.sort_values('release_date')
        ax.scatter(df_sorted['release_date'], df_sorted['score'], color='#1976D2', s=60, edgecolor='black', zorder=3)
        for _, row in df_sorted.iterrows():
            ax.annotate(row['model'][:15], (row['release_date'], row['score']),
                       fontsize=6, xytext=(3, 3), textcoords='offset points', alpha=0.6)
        ax.axhline(y=info['ceiling'], color='gray', linestyle=':', alpha=0.5, label=f"Ceiling={info['ceiling']}")
        ax.axhline(y=90, color='red', linestyle='--', alpha=0.4)
        ax.set_xlabel('Release date')
        ax.set_ylabel(f'{name.upper()} score')
        ax.set_title(f"{name.upper()}  τ½ = {result['halflife_mo']:.1f} mo",
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused
    for ax in axes[len(BENCHMARKS):]:
        ax.set_visible(False)

    plt.suptitle('Benchmark Archaeology: Retrospective Half-Lives', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_archaeology.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig_archaeology.png", bbox_inches='tight', dpi=150)
    print(f"\nSaved: {FIG_DIR / 'fig_archaeology.pdf'}")

    with open(OUT_DIR / "archaeology_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
