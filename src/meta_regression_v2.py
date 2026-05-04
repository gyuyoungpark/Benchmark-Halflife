"""
Reformulated meta-regression acknowledging the cross-era methodological difference:
- LLM era (v1/v2): variance decay halflives
- BERT era (archaeology): gap-to-ceiling decay halflives

These measure related but distinct quantities. We run TWO regressions:
1. Within-era LLM only (n=6): reproduces our original finding
2. Cross-era with era fixed effects (n=11): robustness check
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"
OUT_DIR = DATA_DIR / "meta_expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA = [
    ("WinoGrande",   1.7,  1267,  1, "llm"),
    ("HellaSwag",    2.5,  10042, 1, "llm"),
    ("ARC-C",        3.4,  1172,  1, "llm"),
    ("GSM8K",        4.9,  1319,  1, "llm"),
    ("TruthfulQA",   5.7,  817,   1, "llm"),
    ("BBH",         19.6,  6511, 23, "llm"),
    ("GLUE",         4.9,  65000, 8, "bert"),
    ("SuperGLUE",    1.5,  5000, 10, "bert"),
    ("SQuAD 1.1",   10.8,  10570, 1, "bert"),
    ("SQuAD 2.0",    3.9,  11873, 1, "bert"),
    ("CoQA",         1.0,  7983,  1, "bert"),
]

df = pd.DataFrame(DATA, columns=["name", "halflife", "n_items", "diversity", "era"])
df['log_hl'] = np.log(df['halflife'])
df['log_div'] = np.log(df['diversity'])
df['log_n'] = np.log(df['n_items'])
df['is_bert'] = (df['era'] == 'bert').astype(int)


def fit_ols(X, y):
    """OLS fit, return coefs, R², residuals."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return beta, r2, y_pred


def bootstrap_ols(X, y, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    n, p = X.shape
    boot_betas = np.zeros((n_boot, p))
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            boot_betas[b], *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        except:
            boot_betas[b] = np.nan
    ci_lo = np.nanpercentile(boot_betas, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot_betas, 97.5, axis=0)
    return ci_lo, ci_hi


# =====================================================
# Model 1: LLM-only (n=6, variance-decay halflife)
# =====================================================
llm_df = df[df['era'] == 'llm'].reset_index(drop=True)
X1 = np.column_stack([np.ones(len(llm_df)), llm_df['log_div'], llm_df['log_n']])
y1 = llm_df['log_hl'].values
beta1, r2_1, y1_pred = fit_ols(X1, y1)
ci_lo_1, ci_hi_1 = bootstrap_ols(X1, y1)

print("=" * 60)
print("Model 1: LLM-only (n=6, variance-decay halflives)")
print("=" * 60)
names1 = ['intercept', 'log(diversity)', 'log(n_items)']
for n, b, lo, hi in zip(names1, beta1, ci_lo_1, ci_hi_1):
    sig = "*" if not (lo < 0 < hi) else " "
    print(f"  {n:<18} = {b:+.3f}  [{lo:+.3f}, {hi:+.3f}] {sig}")
print(f"  R² = {r2_1:.3f}")

# =====================================================
# Model 2: Cross-era with era fixed effect (n=11)
# =====================================================
X2 = np.column_stack([np.ones(len(df)), df['log_div'], df['log_n'], df['is_bert']])
y2 = df['log_hl'].values
beta2, r2_2, y2_pred = fit_ols(X2, y2)
ci_lo_2, ci_hi_2 = bootstrap_ols(X2, y2)

print("\n" + "=" * 60)
print("Model 2: Cross-era with era fixed effect (n=11)")
print("=" * 60)
names2 = ['intercept', 'log(diversity)', 'log(n_items)', 'is_bert (era dummy)']
for n, b, lo, hi in zip(names2, beta2, ci_lo_2, ci_hi_2):
    sig = "*" if not (lo < 0 < hi) else " "
    print(f"  {n:<22} = {b:+.3f}  [{lo:+.3f}, {hi:+.3f}] {sig}")
print(f"  R² = {r2_2:.3f}")

# =====================================================
# Model 3: BERT-era only (n=5, gap-to-ceiling)
# =====================================================
bert_df = df[df['era'] == 'bert'].reset_index(drop=True)
X3 = np.column_stack([np.ones(len(bert_df)), bert_df['log_div'], bert_df['log_n']])
y3 = bert_df['log_hl'].values
beta3, r2_3, y3_pred = fit_ols(X3, y3)

print("\n" + "=" * 60)
print("Model 3: BERT-era only (n=5, gap-to-ceiling halflives)")
print("=" * 60)
for n, b in zip(names1, beta3):
    print(f"  {n:<18} = {b:+.3f}")
print(f"  R² = {r2_3:.3f}")

# Qualitative cross-era check: do both eras show negative age effect?
print("\n" + "=" * 60)
print("Simple correlations within each era")
print("=" * 60)
for era_name in ['llm', 'bert']:
    sub = df[df['era'] == era_name]
    r_div, p_div = pearsonr(sub['log_div'], sub['log_hl'])
    r_n, p_n = pearsonr(sub['log_n'], sub['log_hl'])
    print(f"{era_name.upper():<6} (n={len(sub)}):")
    print(f"  r(log_div, log_hl) = {r_div:+.3f}, p = {p_div:.3f}")
    print(f"  r(log_n,   log_hl) = {r_n:+.3f}, p = {p_n:.3f}")

# Save all
out = {
    'n_total': len(df),
    'model1_llm_only_n6': {
        'coefs': dict(zip(names1, [float(x) for x in beta1])),
        'ci_lo': dict(zip(names1, [float(x) for x in ci_lo_1])),
        'ci_hi': dict(zip(names1, [float(x) for x in ci_hi_1])),
        'R2': float(r2_1),
    },
    'model2_cross_era_n11': {
        'coefs': dict(zip(names2, [float(x) for x in beta2])),
        'ci_lo': dict(zip(names2, [float(x) for x in ci_lo_2])),
        'ci_hi': dict(zip(names2, [float(x) for x in ci_hi_2])),
        'R2': float(r2_2),
    },
    'model3_bert_only_n5': {
        'coefs': dict(zip(names1, [float(x) for x in beta3])),
        'R2': float(r2_3),
    },
}
with open(OUT_DIR / "results_v2.json", 'w') as f:
    json.dump(out, f, indent=2)


# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: All 11 observed vs predicted (cross-era model)
ax = axes[0]
colors = {'llm': '#D32F2F', 'bert': '#1976D2'}
for era in ['llm', 'bert']:
    mask = df['era'].values == era
    ax.scatter(df[mask]['halflife'], np.exp(y2_pred[mask]),
               s=150, color=colors[era], edgecolor='black',
               label=f"{era.upper()} era", alpha=0.85, zorder=3)

for _, row in df.iterrows():
    idx = df.index.get_loc(row.name)
    ax.annotate(row['name'], (row['halflife'], np.exp(y2_pred[idx])),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.plot([0.5, 30], [0.5, 30], 'k--', alpha=0.4, label='y=x')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Observed half-life (mo, log)', fontsize=11)
ax.set_ylabel('Predicted half-life (mo, log)', fontsize=11)
ax.set_title(f'Cross-era model (n=11, R²={r2_2:.2f})\nWith era fixed effect', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: diversity vs log_hl colored by era (hypothesis check)
ax = axes[1]
for era in ['llm', 'bert']:
    mask = df['era'] == era
    ax.scatter(df[mask]['diversity'] + (0.05 if era == 'bert' else -0.05),
               df[mask]['halflife'],
               s=150, color=colors[era], edgecolor='black',
               label=f"{era.upper()} era", alpha=0.85, zorder=3)
for _, row in df.iterrows():
    ax.annotate(row['name'], (row['diversity'], row['halflife']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_yscale('log')
ax.set_xlabel('Number of subtasks $n_{\\mathrm{sub}}$', fontsize=11)
ax.set_ylabel('Half-life (mo, log scale)', fontsize=11)
ax.set_title('Subtask count vs Half-life, by era\nLLM era: monotone; BERT era: non-monotone',
             fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_meta_v2.pdf", bbox_inches='tight')
plt.savefig(FIG_DIR / "fig_meta_v2.png", bbox_inches='tight', dpi=150)
print(f"\nSaved: {FIG_DIR / 'fig_meta_v2.pdf'}")
