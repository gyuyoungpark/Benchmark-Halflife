"""
T7: Extended synthetic validation across configurations.

Sweep over:
  - contamination rates: [0.0, 0.1, 0.3, 0.5]
  - capability growth rates: [0.05, 0.15, 0.30]
  - 3 replications per cell for robustness

For each setting:
  - Generate synthetic benchmark with known G, K, F
  - Apply our decomposition method
  - Compare estimated to ground truth
  - Compute MAE per component

Output: scatter plots and recovery accuracy table.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "synthetic_sweep"
FIG_DIR = Path(__file__).parent.parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def prob_2pl(theta, a, b):
    z = a * (theta - b)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def generate_synthetic(n_items=300, n_epochs=15, models_per_epoch=20,
                        growth_rate=0.15, contam_item_frac=0.3,
                        contam_model_frac=0.5, ceiling_frac=0.05, seed=42):
    """Generate synthetic benchmark with known ground truth components."""
    rng = np.random.RandomState(seed)
    a_true = rng.uniform(0.5, 2.5, n_items)
    b_true = rng.normal(0, 1, n_items)

    # Mark ceiling items
    ceiling_items = rng.choice(n_items, int(n_items * ceiling_frac), replace=False)
    b_true[ceiling_items] = rng.uniform(-3, -2, len(ceiling_items))

    # Mark contamination-susceptible items (excluding ceiling)
    non_ceil = [i for i in range(n_items) if i not in ceiling_items]
    n_contam = int(n_items * contam_item_frac)
    contam_items = rng.choice(non_ceil, min(n_contam, len(non_ceil)), replace=False)

    epoch_data = []
    for ep in range(n_epochs):
        # Model abilities
        mean_theta = growth_rate * ep
        theta = rng.normal(mean_theta, 1.0, models_per_epoch)

        # Generate clean responses
        X_clean = np.zeros((models_per_epoch, n_items), dtype=int)
        for j in range(models_per_epoch):
            for i in range(n_items):
                p = prob_2pl(theta[j], a_true[i], b_true[i])
                X_clean[j, i] = rng.binomial(1, p)

        # Generate contaminated responses (only used for contam items by contam models)
        X = X_clean.copy()
        if ep >= n_epochs // 3:  # contamination starts in second third
            contam_strength = (ep - n_epochs // 3) / (n_epochs - n_epochs // 3)
            n_contam_models = int(models_per_epoch * contam_model_frac * contam_strength)
            contam_models = rng.choice(models_per_epoch, n_contam_models, replace=False)
            for j in contam_models:
                for i in contam_items:
                    X[j, i] = 1

        epoch_data.append({
            'X_orig': X,
            'X_pert': X_clean,  # perturbation = no contamination
            'theta': theta,
        })

    return epoch_data, {'a': a_true, 'b': b_true, 'ceiling_items': ceiling_items, 'contam_items': contam_items}


def compute_decomposition(epoch_data, true_params):
    """Apply our decomposition method to synthetic data."""
    n_epochs = len(epoch_data)

    # Per-epoch metrics
    rows = []
    for ep, data in enumerate(epoch_data):
        scores_orig = data['X_orig'].mean(axis=1)
        scores_pert = data['X_pert'].mean(axis=1)

        var_orig = float(np.var(scores_orig, ddof=1))
        var_pert = float(np.var(scores_pert, ddof=1))

        # Mean contamination gap
        mean_orig = scores_orig.mean()
        mean_pert = scores_pert.mean()
        K_mu = float(mean_orig - mean_pert)

        # K_sigma (signed)
        K_sigma = float(var_pert - var_orig)

        # Ceiling fraction
        n_at_ceiling = (scores_orig >= 0.95).sum()
        F_frac = n_at_ceiling / len(scores_orig)

        rows.append({
            'epoch': ep,
            'var_orig': var_orig,
            'var_pert': var_pert,
            'K_mu': K_mu,
            'K_sigma': K_sigma,
            'F_frac': F_frac,
            'mean_orig': float(mean_orig),
            'mean_pert': float(mean_pert),
        })

    df = pd.DataFrame(rows)

    # Total decay observed
    D0_orig = df['var_orig'].iloc[0]
    Dt_orig = df['var_orig'].iloc[-1]
    delta_D = D0_orig - Dt_orig

    # Estimates at the end
    K_mu_final = df['K_mu'].iloc[-1]
    K_sigma_final = df['K_sigma'].iloc[-1]
    F_final = df['F_frac'].iloc[-1] * D0_orig * 0.3  # heuristic
    G_final = max(0.0, df['var_pert'].iloc[0] - df['var_pert'].iloc[-1] - F_final)

    return {
        'K_mu_estimated': float(K_mu_final),
        'K_sigma_estimated': float(K_sigma_final),
        'F_estimated': float(F_final),
        'G_estimated': float(G_final),
        'delta_D': float(delta_D),
    }


def compute_ground_truth(epoch_data, true_params, contam_item_frac):
    """Compute true G/K/F components."""
    contam_items = set(true_params['contam_items'])
    ceiling_items = set(true_params['ceiling_items'])
    n_items = len(true_params['a'])

    # True K_mu = mean accuracy gap due to contamination at final epoch
    final = epoch_data[-1]
    scores_orig = final['X_orig'].mean(axis=1)
    scores_pert = final['X_pert'].mean(axis=1)
    K_mu_true = float(scores_orig.mean() - scores_pert.mean())

    # True G = variance change in CLEAN setting
    var_clean_init = float(np.var(epoch_data[0]['X_pert'].mean(axis=1), ddof=1))
    var_clean_final = float(np.var(epoch_data[-1]['X_pert'].mean(axis=1), ddof=1))
    G_true = max(0.0, var_clean_init - var_clean_final)

    # True F = ceiling effect (no contamination, due to ceiling items)
    # We approximate as zero for now (synthetic doesn't model ceiling decay precisely)
    F_true = 0.0

    return {
        'K_mu_true': K_mu_true,
        'G_true': G_true,
        'F_true': F_true,
    }


def main():
    contam_rates = [0.0, 0.1, 0.3, 0.5]
    growth_rates = [0.05, 0.15, 0.30]
    n_reps = 3

    results = []
    for cr in contam_rates:
        for gr in growth_rates:
            for rep in range(n_reps):
                seed = 42 + rep * 100 + int(cr * 100) + int(gr * 1000)
                epoch_data, true_params = generate_synthetic(
                    contam_item_frac=cr, growth_rate=gr, seed=seed
                )
                est = compute_decomposition(epoch_data, true_params)
                gt = compute_ground_truth(epoch_data, true_params, cr)

                results.append({
                    'contam_rate': cr,
                    'growth_rate': gr,
                    'rep': rep,
                    **gt,
                    **est,
                    'K_mu_err': abs(gt['K_mu_true'] - est['K_mu_estimated']),
                    'G_err': abs(gt['G_true'] - est['G_estimated']),
                })
                print(f"cr={cr} gr={gr} rep={rep}: K_mu true={gt['K_mu_true']:.3f} est={est['K_mu_estimated']:.3f}  G true={gt['G_true']:.4f} est={est['G_estimated']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "sweep_results.csv", index=False)

    # Summary stats
    print("\n=== Recovery Accuracy ===")
    print(f"K_mu  MAE: {df['K_mu_err'].mean():.4f}")
    print(f"G     MAE: {df['G_err'].mean():.4f}")
    print(f"K_mu  correlation true-vs-est: {df[['K_mu_true', 'K_mu_estimated']].corr().iloc[0,1]:.3f}")
    print(f"G     correlation true-vs-est: {df[['G_true', 'G_estimated']].corr().iloc[0,1]:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # K_mu scatter
    ax = axes[0]
    sc1 = ax.scatter(df['K_mu_true'], df['K_mu_estimated'],
                     c=df['contam_rate'], cmap='Reds', s=60, alpha=0.8, edgecolor='black')
    lim = max(df['K_mu_true'].max(), df['K_mu_estimated'].max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
    ax.set_xlabel('True $K_\\mu$ (mean contamination gap)', fontsize=11)
    ax.set_ylabel('Estimated $K_\\mu$', fontsize=11)
    ax.set_title('Recovery: Mean Contamination Gap', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cb1 = fig.colorbar(sc1, ax=ax, fraction=0.046, pad=0.04)
    cb1.set_label('Injected contamination rate', fontsize=9)

    # G scatter
    ax = axes[1]
    sc2 = ax.scatter(df['G_true'], df['G_estimated'],
                     c=df['growth_rate'], cmap='Blues', s=60, alpha=0.8, edgecolor='black')
    lim_g = max(df['G_true'].max(), df['G_estimated'].max()) * 1.1
    ax.plot([0, lim_g], [0, lim_g], 'k--', alpha=0.5)
    ax.set_xlabel('True $G$ (genuine convergence)', fontsize=11)
    ax.set_ylabel('Estimated $G$', fontsize=11)
    ax.set_title('Recovery: Genuine Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cb2 = fig.colorbar(sc2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label('Capability growth rate', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_synthetic_recovery.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig_synthetic_recovery.png", bbox_inches='tight', dpi=150)
    print(f"\nSaved: {FIG_DIR / 'fig_synthetic_recovery.pdf'}")


if __name__ == "__main__":
    main()
