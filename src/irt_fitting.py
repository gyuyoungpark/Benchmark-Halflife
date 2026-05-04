"""
Longitudinal IRT fitting: fit 2PL model per epoch, detect item parameter drift.

Input: item-level response matrices per epoch (data/itemlevel/{benchmark}_epoch_{t}.csv)
Output: IRT parameters per epoch (data/irt_params/{benchmark}_epoch_{t}.json)
        Drift analysis (data/drift/{benchmark}_drift.csv)
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import linregress, chi2

DATA_DIR = Path(__file__).parent.parent / "data"


# ============================================================
# 2PL IRT Model
# ============================================================

def prob_2pl(theta, a, b):
    """P(correct | theta, a, b) under 2PL model."""
    z = a * (theta - b)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def neg_log_likelihood_2pl(params, X):
    """
    Negative log-likelihood for 2PL model.
    params: [a_1, ..., a_m, b_1, ..., b_m, theta_1, ..., theta_n]
    X: (n_models, m_items) binary response matrix
    """
    n, m = X.shape
    a = params[:m]
    b = params[m:2*m]
    theta = params[2*m:]

    # Ensure valid ranges
    a = np.clip(a, 0.1, 5.0)

    nll = 0.0
    for j in range(n):
        p = prob_2pl(theta[j], a, b)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        nll -= np.sum(X[j] * np.log(p) + (1 - X[j]) * np.log(1 - p))

    # Priors: theta ~ N(0, 1), a ~ logN(0, 0.5)
    nll += 0.5 * np.sum(theta ** 2)
    nll += 0.5 * np.sum((np.log(np.maximum(a, 0.1))) ** 2) / 0.25

    return nll


def fit_2pl(X, max_iter=500):
    """
    Fit 2PL IRT model via joint MLE with priors.

    Args:
        X: (n_models, m_items) binary response matrix
    Returns:
        a: (m,) discrimination parameters
        b: (m,) difficulty parameters
        theta: (n,) ability parameters
    """
    n, m = X.shape

    # Initialize
    item_means = X.mean(axis=0)
    b_init = -np.log(np.clip(item_means, 0.01, 0.99) / (1 - np.clip(item_means, 0.01, 0.99)))
    a_init = np.ones(m) * 1.0
    theta_init = np.zeros(n)

    params0 = np.concatenate([a_init, b_init, theta_init])

    result = minimize(
        neg_log_likelihood_2pl, params0, args=(X,),
        method='L-BFGS-B',
        bounds=(
            [(0.1, 5.0)] * m +      # a bounds
            [(-4.0, 4.0)] * m +     # b bounds
            [(-4.0, 4.0)] * n       # theta bounds
        ),
        options={'maxiter': max_iter, 'ftol': 1e-6}
    )

    a = np.clip(result.x[:m], 0.1, 5.0)
    b = result.x[m:2*m]
    theta = result.x[2*m:]

    return a, b, theta, result.fun


# ============================================================
# Scale Equating (Stocking-Lord)
# ============================================================

def stocking_lord_equate(a_ref, b_ref, a_new, b_new, theta_points=None):
    """
    Stocking-Lord scale equating: find A, B such that
    a_new_equated = a_new / A
    b_new_equated = A * b_new + B
    minimizing difference in test characteristic curves.

    Returns: A, B transformation constants
    """
    if theta_points is None:
        theta_points = np.linspace(-3, 3, 61)

    m = len(a_ref)

    def objective(params):
        A, B = params
        a_eq = a_new / A
        b_eq = A * b_new + B

        loss = 0.0
        for theta in theta_points:
            p_ref = prob_2pl(theta, a_ref, b_ref)
            p_eq = prob_2pl(theta, a_eq, b_eq)
            loss += np.sum((p_ref - p_eq) ** 2)
        return loss

    result = minimize(objective, [1.0, 0.0], method='Nelder-Mead')
    return result.x


# ============================================================
# Item Parameter Drift Detection
# ============================================================

def detect_drift(param_series, alpha=0.05, n_items=None):
    """
    Detect significant linear drift in a parameter time series.

    Args:
        param_series: (T,) array of parameter values over epochs
        alpha: significance level (Bonferroni-corrected if n_items given)
        n_items: total number of items (for Bonferroni correction)
    Returns:
        slope, p_value, significant (bool)
    """
    T = len(param_series)
    if T < 3:
        return 0.0, 1.0, False

    t = np.arange(T)
    slope, intercept, r_value, p_value, std_err = linregress(t, param_series)

    corrected_alpha = alpha / n_items if n_items else alpha
    significant = p_value < corrected_alpha

    return slope, p_value, significant


def lord_chi2_test(a1, b1, a2, b2, se_a=None, se_b=None):
    """
    Lord's chi-squared test for item parameter drift between two epochs.
    Simplified version assuming independence of a and b estimates.
    """
    if se_a is None:
        se_a = 0.2  # default SE estimate
    if se_b is None:
        se_b = 0.3

    chi2_stat = ((a2 - a1) / se_a) ** 2 + ((b2 - b1) / se_b) ** 2
    p_value = 1 - chi2.cdf(chi2_stat, df=2)
    return chi2_stat, p_value


def classify_drift(a_slope, b_slope, a_sig, b_sig, ceiling_frac):
    """
    Classify item drift pattern.
    Returns: 'stable', 'difficulty_drift', 'discrimination_collapse', 'ceiling'
    """
    if ceiling_frac > 0.95:
        return 'ceiling'
    if a_sig and a_slope < 0:
        return 'discrimination_collapse'
    if b_sig and b_slope < 0 and not a_sig:
        return 'difficulty_drift'
    return 'stable'


# ============================================================
# Main Pipeline
# ============================================================

def run_longitudinal_irt(benchmark: str, data_dir: Path):
    """
    Full longitudinal IRT pipeline for one benchmark.
    """
    itemlevel_dir = data_dir / "itemlevel"
    irt_dir = data_dir / "irt_params"
    drift_dir = data_dir / "drift"

    irt_dir.mkdir(parents=True, exist_ok=True)
    drift_dir.mkdir(parents=True, exist_ok=True)

    # Discover epochs
    epoch_files = sorted(itemlevel_dir.glob(f"{benchmark}_epoch_*.csv"))
    if not epoch_files:
        print(f"No item-level data found for {benchmark}")
        return

    print(f"Found {len(epoch_files)} epochs for {benchmark}")

    all_params = {}  # epoch -> {a, b, theta}

    # Step 1: Fit IRT per epoch
    for ef in epoch_files:
        epoch_name = ef.stem.split("_epoch_")[1]
        print(f"  Fitting epoch {epoch_name}...")

        X = pd.read_csv(ef).values  # (models x items)
        if X.shape[0] < 3:
            print(f"    Skipping: only {X.shape[0]} models")
            continue

        a, b, theta, nll = fit_2pl(X)
        all_params[epoch_name] = {'a': a.tolist(), 'b': b.tolist(), 'theta': theta.tolist(), 'nll': float(nll)}

        with open(irt_dir / f"{benchmark}_epoch_{epoch_name}.json", 'w') as f:
            json.dump(all_params[epoch_name], f)

        print(f"    Done: {X.shape[0]} models, {X.shape[1]} items, NLL={nll:.1f}")

    if len(all_params) < 3:
        print(f"Need at least 3 epochs for drift analysis, got {len(all_params)}")
        return

    # Step 2: Scale equating
    epochs_sorted = sorted(all_params.keys())
    ref_epoch = epochs_sorted[0]
    a_ref = np.array(all_params[ref_epoch]['a'])
    b_ref = np.array(all_params[ref_epoch]['b'])

    for ep in epochs_sorted[1:]:
        a_new = np.array(all_params[ep]['a'])
        b_new = np.array(all_params[ep]['b'])

        A, B = stocking_lord_equate(a_ref, b_ref, a_new, b_new)
        all_params[ep]['a'] = (a_new / A).tolist()
        all_params[ep]['b'] = (A * b_new + B).tolist()

        print(f"  Equating {ep}: A={A:.3f}, B={B:.3f}")

    # Step 3: Drift detection
    m = len(all_params[ref_epoch]['a'])
    drift_results = []

    for i in range(m):
        a_series = np.array([all_params[ep]['a'][i] for ep in epochs_sorted])
        b_series = np.array([all_params[ep]['b'][i] for ep in epochs_sorted])

        a_slope, a_pval, a_sig = detect_drift(a_series, n_items=m)
        b_slope, b_pval, b_sig = detect_drift(b_series, n_items=m)

        # Ceiling fraction: in latest epoch, what fraction of models got it right
        # (approximated from b: very negative b means almost everyone gets it right)
        ceiling_frac = prob_2pl(0, np.mean(a_series), b_series[-1])

        category = classify_drift(a_slope, b_slope, a_sig, b_sig, ceiling_frac)

        drift_results.append({
            'item_id': i,
            'a_slope': a_slope,
            'a_pvalue': a_pval,
            'a_significant': a_sig,
            'b_slope': b_slope,
            'b_pvalue': b_pval,
            'b_significant': b_sig,
            'a_first': a_series[0],
            'a_last': a_series[-1],
            'b_first': b_series[0],
            'b_last': b_series[-1],
            'drift_category': category,
        })

    drift_df = pd.DataFrame(drift_results)
    drift_df.to_csv(drift_dir / f"{benchmark}_drift.csv", index=False)

    # Summary
    counts = drift_df['drift_category'].value_counts()
    print(f"\nDrift summary for {benchmark}:")
    for cat, cnt in counts.items():
        print(f"  {cat}: {cnt} items ({100*cnt/m:.1f}%)")

    return drift_df


def main():
    parser = argparse.ArgumentParser(description="Longitudinal IRT fitting and drift detection")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    run_longitudinal_irt(args.benchmark, Path(args.data_dir))


if __name__ == "__main__":
    main()
