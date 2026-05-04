"""
Discriminative power computation and decay curve fitting.

Input: leaderboard scores per epoch (data/leaderboard/{benchmark}_merged.csv)
       IRT parameters per epoch (data/irt_params/{benchmark}_epoch_{t}.json)
Output: decay curves and half-life estimates (data/decay/{benchmark}_halflife.json)
        Figures (figures/{benchmark}_decay.pdf)
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import kendalltau

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"


# ============================================================
# Discriminative Power Metrics
# ============================================================

def inter_model_variance(scores):
    """Inter-model score variance."""
    if len(scores) < 2:
        return np.nan
    return np.var(scores, ddof=1)


def ranking_stability(scores_t, scores_t1, models_t, models_t1):
    """Kendall's tau between rankings in consecutive epochs for overlapping models."""
    common = set(models_t) & set(models_t1)
    if len(common) < 3:
        return np.nan

    common = sorted(common)
    s_t = [scores_t[models_t.index(m)] for m in common]
    s_t1 = [scores_t1[models_t1.index(m)] for m in common]

    tau, _ = kendalltau(s_t, s_t1)
    return tau


def ceiling_compression(scores, top_k_frac=0.25):
    """Ceiling compression: 1 - IQR(top-k) / IQR(all)."""
    if len(scores) < 4:
        return np.nan

    iqr_all = np.percentile(scores, 75) - np.percentile(scores, 25)
    if iqr_all < 1e-10:
        return 1.0

    k = max(2, int(len(scores) * top_k_frac))
    top_scores = np.sort(scores)[-k:]
    iqr_top = np.percentile(top_scores, 75) - np.percentile(top_scores, 25)

    return 1.0 - iqr_top / iqr_all


def aggregate_irt_discrimination(irt_params_path):
    """Mean item discrimination from IRT parameters."""
    if not irt_params_path.exists():
        return np.nan
    with open(irt_params_path) as f:
        params = json.load(f)
    return np.mean(params['a'])


# ============================================================
# Decay Models
# ============================================================

def exponential_decay(t, D0, lam, D_inf):
    """D(t) = D0 * exp(-lambda * t) + D_inf"""
    return D0 * np.exp(-lam * t) + D_inf


def stretched_exponential(t, D0, lam, beta, D_inf):
    """D(t) = D0 * exp(-(lambda * t)^beta) + D_inf"""
    return D0 * np.exp(-(lam * t) ** beta) + D_inf


def logistic_decay(t, D0, lam, t0, D_inf):
    """D(t) = (D0 - D_inf) / (1 + exp(lambda * (t - t0))) + D_inf"""
    return (D0 - D_inf) / (1.0 + np.exp(lam * (t - t0))) + D_inf


def compute_aic(n, k, nll):
    """AIC = 2k - 2ln(L), where nll = -ln(L)"""
    return 2 * k + 2 * nll


def fit_decay_models(t, D):
    """
    Fit three decay models, return best by AIC.

    Returns: dict with best model name, parameters, half-life, AIC values
    """
    t = np.array(t, dtype=float)
    D = np.array(D, dtype=float)

    valid = ~np.isnan(D)
    t, D = t[valid], D[valid]

    if len(t) < 4:
        return None

    n = len(t)
    D0_init = D[0]
    D_inf_init = D[-1] * 0.5

    results = {}

    # Model 1: Exponential
    try:
        popt, pcov = curve_fit(
            exponential_decay, t, D,
            p0=[D0_init, 0.05, D_inf_init],
            bounds=([0, 0, 0], [np.inf, 1.0, np.inf]),
            maxfev=10000
        )
        D_pred = exponential_decay(t, *popt)
        ss_res = np.sum((D - D_pred) ** 2)
        nll = n / 2 * np.log(ss_res / n)
        aic = compute_aic(n, 3, nll)
        halflife = np.log(2) / popt[1] if popt[1] > 0 else np.inf
        results['exponential'] = {
            'params': {'D0': popt[0], 'lambda': popt[1], 'D_inf': popt[2]},
            'halflife_months': halflife,
            'aic': aic,
            'residual_ss': ss_res,
        }
    except Exception as e:
        results['exponential'] = {'error': str(e)}

    # Model 2: Stretched exponential
    try:
        popt, pcov = curve_fit(
            stretched_exponential, t, D,
            p0=[D0_init, 0.05, 1.0, D_inf_init],
            bounds=([0, 0, 0.1, 0], [np.inf, 1.0, 3.0, np.inf]),
            maxfev=10000
        )
        D_pred = stretched_exponential(t, *popt)
        ss_res = np.sum((D - D_pred) ** 2)
        nll = n / 2 * np.log(ss_res / n)
        aic = compute_aic(n, 4, nll)
        # Half-life for stretched exponential: solve D0*exp(-(lam*t)^beta) = D0/2
        # t_half = (ln2)^(1/beta) / lam
        halflife = (np.log(2) ** (1 / popt[2])) / popt[1]
        results['stretched_exponential'] = {
            'params': {'D0': popt[0], 'lambda': popt[1], 'beta': popt[2], 'D_inf': popt[3]},
            'halflife_months': halflife,
            'aic': aic,
            'residual_ss': ss_res,
        }
    except Exception as e:
        results['stretched_exponential'] = {'error': str(e)}

    # Model 3: Logistic decay
    try:
        popt, pcov = curve_fit(
            logistic_decay, t, D,
            p0=[D0_init, 0.1, np.median(t), D_inf_init],
            bounds=([0, 0, 0, 0], [np.inf, 1.0, t[-1] * 2, np.inf]),
            maxfev=10000
        )
        D_pred = logistic_decay(t, *popt)
        ss_res = np.sum((D - D_pred) ** 2)
        nll = n / 2 * np.log(ss_res / n)
        aic = compute_aic(n, 4, nll)
        # Half-life for logistic: time to reach midpoint (= t0)
        halflife = popt[2]
        results['logistic_decay'] = {
            'params': {'D0': popt[0], 'lambda': popt[1], 't0': popt[2], 'D_inf': popt[3]},
            'halflife_months': halflife,
            'aic': aic,
            'residual_ss': ss_res,
        }
    except Exception as e:
        results['logistic_decay'] = {'error': str(e)}

    # Select best model by AIC
    valid_models = {k: v for k, v in results.items() if 'aic' in v}
    if not valid_models:
        return results

    best_name = min(valid_models, key=lambda k: valid_models[k]['aic'])
    results['best_model'] = best_name
    results['best_halflife_months'] = valid_models[best_name]['halflife_months']

    return results


# ============================================================
# Bootstrap Confidence Intervals
# ============================================================

def bootstrap_halflife(t, D, n_bootstrap=1000, seed=42):
    """
    Bootstrap CI for half-life estimate.
    Resample data points (t, D) pairs and refit.
    """
    rng = np.random.RandomState(seed)
    t = np.array(t, dtype=float)
    D = np.array(D, dtype=float)

    valid = ~np.isnan(D)
    t, D = t[valid], D[valid]
    n = len(t)

    halflives = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        idx = np.sort(np.unique(idx))  # deduplicate and sort
        if len(idx) < 4:
            continue
        result = fit_decay_models(t[idx], D[idx])
        if result and 'best_halflife_months' in result:
            hl = result['best_halflife_months']
            if np.isfinite(hl) and hl > 0:
                halflives.append(hl)

    if len(halflives) < 10:
        return np.nan, np.nan

    return np.percentile(halflives, 2.5), np.percentile(halflives, 97.5)


# ============================================================
# Main Pipeline
# ============================================================

def compute_metrics_over_time(benchmark, data_dir):
    """Compute all 4 discriminative power metrics over time for a benchmark."""
    scores_path = data_dir / "leaderboard" / f"{benchmark}_merged.csv"
    if not scores_path.exists():
        print(f"No merged scores for {benchmark}")
        return None

    df = pd.read_csv(scores_path)
    df['release_date'] = pd.to_datetime(df['release_date'])

    # Create quarterly epochs
    df['epoch'] = df['release_date'].dt.to_period('Q')
    epochs = sorted(df['epoch'].unique())

    metrics = []
    prev_epoch_data = None

    for ep in epochs:
        ep_df = df[df['epoch'] == ep]
        scores = ep_df['score'].values
        models = ep_df['model'].tolist()

        if len(scores) < 3:
            continue

        # Months since benchmark start
        months = (ep.start_time - epochs[0].start_time).days / 30.44

        row = {
            'epoch': str(ep),
            'months': months,
            'n_models': len(scores),
            'mean_score': np.mean(scores),
            'variance': inter_model_variance(scores),
            'ceiling_compression': ceiling_compression(scores),
        }

        # Ranking stability (requires consecutive epoch)
        if prev_epoch_data is not None:
            row['kendall_tau'] = ranking_stability(
                prev_epoch_data['scores'], scores,
                prev_epoch_data['models'], models
            )
        else:
            row['kendall_tau'] = np.nan

        # IRT discrimination
        irt_path = data_dir / "irt_params" / f"{benchmark}_epoch_{str(ep).replace('Q', 'q')}.json"
        row['mean_discrimination'] = aggregate_irt_discrimination(irt_path)

        metrics.append(row)
        prev_epoch_data = {'scores': scores, 'models': models}

    return pd.DataFrame(metrics)


def run_full_analysis(benchmark, data_dir, output_dir, fig_dir):
    """Full analysis pipeline for one benchmark."""
    print(f"\n{'='*60}")
    print(f"Analyzing {benchmark}")
    print(f"{'='*60}")

    # Step 1: Compute metrics
    metrics_df = compute_metrics_over_time(benchmark, data_dir)
    if metrics_df is None or len(metrics_df) < 4:
        print(f"Insufficient data for {benchmark}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / f"{benchmark}_metrics.csv", index=False)

    t = metrics_df['months'].values

    # Step 2: Fit decay for each metric
    results = {}
    metric_columns = ['variance', 'mean_discrimination']
    # For ceiling compression, we fit 1 - C_t (which decays)
    # For kendall_tau, we fit tau_t directly

    for col in metric_columns:
        D = metrics_df[col].values
        print(f"\n  Fitting decay for {col}...")

        fit_result = fit_decay_models(t, D)
        if fit_result and 'best_model' in fit_result:
            ci_lo, ci_hi = bootstrap_halflife(t, D)
            fit_result['ci_95'] = [ci_lo, ci_hi]
            print(f"    Best model: {fit_result['best_model']}")
            print(f"    Half-life: {fit_result['best_halflife_months']:.1f} months "
                  f"(95% CI: [{ci_lo:.1f}, {ci_hi:.1f}])")
        results[col] = fit_result

    # Save results
    with open(output_dir / f"{benchmark}_halflife.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Step 3: Plot
    try:
        plot_decay(benchmark, metrics_df, results, fig_dir)
    except ImportError:
        print("  matplotlib not available, skipping plots")


def plot_decay(benchmark, metrics_df, results, fig_dir):
    """Generate decay curve plots."""
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Discriminative Decay: {benchmark}", fontsize=14)

    t = metrics_df['months'].values
    t_fine = np.linspace(0, t[-1] * 1.2, 200)

    metrics_config = [
        ('variance', 'Inter-model Variance', axes[0, 0]),
        ('mean_discrimination', 'Mean IRT Discrimination', axes[0, 1]),
        ('kendall_tau', "Kendall's τ", axes[1, 0]),
        ('ceiling_compression', 'Ceiling Compression', axes[1, 1]),
    ]

    for col, title, ax in metrics_config:
        D = metrics_df[col].values
        ax.scatter(t, D, color='black', s=20, zorder=3)
        ax.set_xlabel('Months since publication')
        ax.set_ylabel(title)
        ax.set_title(title)

        # Plot fitted curve if available
        if col in results and results[col] and 'best_model' in results[col]:
            best = results[col]['best_model']
            params = results[col][best]['params']
            hl = results[col]['best_halflife_months']

            if best == 'exponential':
                D_fit = exponential_decay(t_fine, params['D0'], params['lambda'], params['D_inf'])
            elif best == 'stretched_exponential':
                D_fit = stretched_exponential(t_fine, params['D0'], params['lambda'], params['beta'], params['D_inf'])
            elif best == 'logistic_decay':
                D_fit = logistic_decay(t_fine, params['D0'], params['lambda'], params['t0'], params['D_inf'])

            ax.plot(t_fine, D_fit, 'r-', linewidth=2, label=f'{best}\n$\\tau_{{1/2}}$={hl:.1f}mo')
            ax.axvline(hl, color='red', linestyle='--', alpha=0.5)
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / f"{benchmark}_decay.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved figure: {fig_dir / f'{benchmark}_decay.pdf'}")


def main():
    parser = argparse.ArgumentParser(description="Decay curve fitting and half-life estimation")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR / "decay"))
    parser.add_argument("--fig-dir", type=str, default=str(FIG_DIR))
    args = parser.parse_args()

    run_full_analysis(args.benchmark, Path(args.data_dir), Path(args.output_dir), Path(args.fig_dir))


if __name__ == "__main__":
    main()
