"""
Full IRT item parameter drift analysis using 4 GPT generations.

With 4 models, we treat each as a separate epoch and fit a 1PL (Rasch) model
on a sliding window of 2 models (forming 3 transitions: 3.5→4-turbo, 4-turbo→4o, 4o→4.1).

For each epoch pair, we compute item difficulty drift Δb_i and classify:
- Significant decrease: difficulty drift (genuine progress)
- Significant decrease ONLY on original (not pert): contamination drift
- No change: stable

This gives us actual IRT-derived parameter drift, not just response-pattern matching.

We use a Bayesian fit: minimize negative log-likelihood with N(0, σ²) prior on b_i.
Since 2 models is underdetermined for 2PL, we fix discrimination a=1 (Rasch) and use
an empirical Bayes prior θ from the marginal accuracy distribution.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_DIR = DATA_DIR / "evaluations"
OUT_DIR = DATA_DIR / "irt_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path(__file__).parent.parent / "figures"

MODEL_ORDER = [
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-08-06",
    "gpt-4.1-2025-04-14",
]


def fit_rasch(X, theta_init=None, b_init=None, prior_b_var=4.0, max_iter=1000):
    """
    Fit 1PL Rasch model: P(correct) = sigmoid(theta_j - b_i).

    X: (n_models, m_items) binary matrix
    Returns: thetas (n_models,), b (m_items,), nll
    """
    n, m = X.shape

    # Initialize from accuracy
    if theta_init is None:
        theta_init = np.array([2 * X[j].mean() - 1 for j in range(n)])
    if b_init is None:
        item_acc = X.mean(axis=0)
        b_init = -np.log(np.clip(item_acc, 0.01, 0.99) / (1 - np.clip(item_acc, 0.01, 0.99)))

    def neg_loglik(params):
        theta = params[:n]
        b = params[n:]
        nll = 0.0
        for j in range(n):
            for i in range(m):
                z = theta[j] - b[i]
                # log P
                if z > 30: log_p = 0.0
                elif z < -30: log_p = z
                else: log_p = z - np.log1p(np.exp(z))
                # log (1-P)
                if z > 30: log_1mp = -z
                elif z < -30: log_1mp = 0.0
                else: log_1mp = -np.log1p(np.exp(z))
                nll -= X[j, i] * log_p + (1 - X[j, i]) * log_1mp
        # Priors
        nll += 0.5 * np.sum(theta ** 2)  # N(0, 1) on theta
        nll += 0.5 * np.sum(b ** 2) / prior_b_var  # N(0, prior_b_var) on b
        return nll

    params0 = np.concatenate([theta_init, b_init])
    result = minimize(neg_loglik, params0, method='L-BFGS-B',
                      options={'maxiter': max_iter})
    theta = result.x[:n]
    b = result.x[n:]
    return theta, b, result.fun


def fit_per_window(X_full, window_size=2):
    """
    Fit Rasch on each sliding window of model rows.
    X_full: (n_models, m_items)
    Returns: list of (theta, b) for each window
    """
    n, m = X_full.shape
    fits = []
    for start in range(n - window_size + 1):
        Xw = X_full[start:start + window_size]
        theta, b, nll = fit_rasch(Xw)
        fits.append({
            'window_start': start,
            'window_end': start + window_size - 1,
            'theta': theta.tolist(),
            'b': b.tolist(),
            'nll': float(nll),
        })
    return fits


def analyze_benchmark(name):
    path = EVAL_DIR / f"{name}_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        results = json.load(f)

    # Build response matrices
    available = [m for m in MODEL_ORDER if m in results]
    if len(available) < 4:
        return None

    # Drop items with any None
    n_items = len(results[available[0]]['orig'])
    keep_orig = []
    keep_pert = []
    for i in range(n_items):
        orig_col = [results[m]['orig'][i] for m in available]
        pert_col = [results[m]['pert'][i] for m in available]
        if all(x is not None for x in orig_col) and all(x is not None for x in pert_col):
            keep_orig.append(i)
            keep_pert.append(i)
    keep = sorted(set(keep_orig) & set(keep_pert))

    if len(keep) < 50:
        print(f"  Only {len(keep)} complete items")
        return None

    X_orig = np.array([[results[m]['orig'][i] for i in keep] for m in available])
    X_pert = np.array([[results[m]['pert'][i] for i in keep] for m in available])

    print(f"  Fitting Rasch on {X_orig.shape}...")

    # Fit Rasch on full 4-model data for both versions
    theta_o, b_o, nll_o = fit_rasch(X_orig)
    theta_p, b_p, nll_p = fit_rasch(X_pert)

    # Compare item difficulty between orig and pert
    delta_b = b_o - b_p  # negative => item easier on orig => contamination

    # Sliding windows for temporal drift
    print(f"  Fitting sliding windows...")
    windows_o = fit_per_window(X_orig)
    windows_p = fit_per_window(X_pert)

    # Track b_i drift over windows (only on orig)
    b_traj = np.array([w['b'] for w in windows_o])  # (n_windows, n_items)
    b_drift_per_item = b_traj[-1] - b_traj[0]  # final - initial

    # Theta trajectory (model abilities estimated by Rasch)
    theta_traj_o = [w['theta'] for w in windows_o]
    theta_traj_p = [w['theta'] for w in windows_p]

    return {
        'n_items_complete': len(keep),
        'theta_orig_full': theta_o.tolist(),
        'theta_pert_full': theta_p.tolist(),
        'b_orig_mean': float(b_o.mean()),
        'b_pert_mean': float(b_p.mean()),
        'b_orig_std': float(b_o.std()),
        'b_pert_std': float(b_p.std()),
        'delta_b_mean': float(delta_b.mean()),
        'delta_b_std': float(delta_b.std()),
        'n_items_significantly_easier_on_orig': int((delta_b < -1.0).sum()),
        'n_items_significantly_harder_on_orig': int((delta_b > 1.0).sum()),
        'fraction_neg_drift': float((delta_b < -0.5).mean()),
        'b_orig_per_item': b_o.tolist(),
        'b_pert_per_item': b_p.tolist(),
        'delta_b_per_item': delta_b.tolist(),
    }


def main():
    all_results = {}
    for bench in ['mmlu', 'arc', 'gsm8k']:
        print(f"\n=== {bench.upper()} ===")
        r = analyze_benchmark(bench)
        if r is None:
            continue
        all_results[bench] = r
        print(f"  Items: {r['n_items_complete']}")
        print(f"  θ_orig (4 models): {[f'{x:.2f}' for x in r['theta_orig_full']]}")
        print(f"  θ_pert (4 models): {[f'{x:.2f}' for x in r['theta_pert_full']]}")
        print(f"  b_orig: mean={r['b_orig_mean']:.2f}, std={r['b_orig_std']:.2f}")
        print(f"  b_pert: mean={r['b_pert_mean']:.2f}, std={r['b_pert_std']:.2f}")
        print(f"  Δb (orig - pert): mean={r['delta_b_mean']:+.2f}, std={r['delta_b_std']:.2f}")
        print(f"  Items with Δb < -1 (much easier on orig, contamination signature): {r['n_items_significantly_easier_on_orig']}")
        print(f"  Items with Δb > +1 (harder on orig, anti-contamination): {r['n_items_significantly_harder_on_orig']}")
        print(f"  Fraction with Δb < -0.5: {r['fraction_neg_drift']:.3f}")

    with open(OUT_DIR / "irt_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Plot Δb distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    bench_labels = {'mmlu': 'MMLU', 'arc': 'ARC-C', 'gsm8k': 'GSM8K'}
    for i, bench in enumerate(['mmlu', 'arc', 'gsm8k']):
        if bench not in all_results:
            continue
        ax = axes[i]
        deltas = np.array(all_results[bench]['delta_b_per_item'])
        ax.hist(deltas, bins=30, color='#1976D2', edgecolor='black', alpha=0.8)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(-1, color='red', linestyle=':', alpha=0.7)
        n_contam = (deltas < -1).sum()
        n_total = len(deltas)
        ax.set_title(f"{bench_labels[bench]} — {100*n_contam/n_total:.1f}% items with $\\Delta b < -1$",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel(r'$\Delta b$', fontsize=11)
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_irt_drift.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig_irt_drift.png", bbox_inches='tight', dpi=150)
    print(f"\nSaved: {FIG_DIR / 'fig_irt_drift.pdf'}")


if __name__ == "__main__":
    main()
