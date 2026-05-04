"""
W6: Expanded hierarchical/meta-regression with n=11 benchmarks + bootstrap CIs.
Incorporates archaeology results.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"
OUT_DIR = DATA_DIR / "meta_expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Full benchmark list: LLM era (v1/v2) + archaeology era
# Columns: name, halflife, age_at_snapshot_mo, n_items, diversity(1-5), era
DATA = [
    # LLM v1 era (our measurements, best-AIC halflife)
    ("WinoGrande",   1.7,   56, 1267,  1, "llm_v1"),
    ("HellaSwag",    2.5,   70, 10042, 1, "llm_v1"),
    ("ARC-C",        3.4,   73, 1172,  2, "llm_v1"),
    ("GSM8K",        4.9,   38, 1319,  1, "llm_v1"),
    ("TruthfulQA",   5.7,   34, 817,   1, "llm_v1"),
    # LLM v2 era
    ("BBH",         19.6,   28, 6511,  5, "llm_v2"),
    # Archaeology era (BERT era, gap-to-ceiling method)
    ("GLUE",         4.9,   42, 65000, 3, "bert"),
    ("SuperGLUE",    1.5,   30, 5000,  4, "bert"),
    ("SQuAD 1.1",   10.8,   48, 10570, 2, "bert"),
    ("SQuAD 2.0",    3.9,   36, 11873, 2, "bert"),
    ("CoQA",         1.0,   36, 7983,  2, "bert"),
]

df = pd.DataFrame(DATA, columns=["name", "halflife", "age_mo", "n_items", "diversity", "era"])
df['log_hl'] = np.log(df['halflife'])
df['log_div'] = np.log(df['diversity'])
df['log_n'] = np.log(df['n_items'])
df['age_years'] = df['age_mo'] / 12.0


print(f"Benchmarks: {len(df)}")
print(df.to_string(index=False))

# Univariate correlations
print("\n=== Univariate (Pearson r, p) ===")
for pred in ['log_div', 'log_n', 'age_mo']:
    slope, intercept, r, p, se = linregress(df[pred], df['log_hl'])
    print(f"  log(halflife) ~ {pred}: r={r:+.3f}, p={p:.4f}, slope={slope:.3f}")

# Multivariate OLS: log(hl) ~ log(div) + log(n) + era_v2 + era_bert
# Dummy encode era (llm_v1 as baseline)
df['era_llm_v2'] = (df['era'] == 'llm_v2').astype(int)
df['era_bert'] = (df['era'] == 'bert').astype(int)

X_cols = ['log_div', 'log_n', 'era_llm_v2', 'era_bert']
X = np.column_stack([np.ones(len(df))] + [df[c].values for c in X_cols])
y = df['log_hl'].values
beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
y_pred = X @ beta
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y.mean())**2)
r2 = 1 - ss_res/ss_tot

print(f"\n=== OLS log(halflife) ~ log(div) + log(n) + era ===")
coef_names = ['intercept'] + X_cols
for name, val in zip(coef_names, beta):
    print(f"  {name:<15} = {val:+.3f}")
print(f"  R² = {r2:.3f}")

# Bootstrap CIs for coefficients
n_boot = 2000
boot_betas = np.zeros((n_boot, len(beta)))
rng = np.random.RandomState(42)
for b in range(n_boot):
    idx = rng.randint(0, len(df), size=len(df))
    Xb = X[idx]
    yb = y[idx]
    try:
        beta_b, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        boot_betas[b] = beta_b
    except:
        boot_betas[b] = np.nan

# 95% bootstrap CIs
ci_lo = np.nanpercentile(boot_betas, 2.5, axis=0)
ci_hi = np.nanpercentile(boot_betas, 97.5, axis=0)

print(f"\n=== Bootstrap 95% CIs ({n_boot} resamples) ===")
for name, val, lo, hi in zip(coef_names, beta, ci_lo, ci_hi):
    sig = "*" if not (lo < 0 < hi) else " "
    print(f"  {name:<15} = {val:+.3f} [{lo:+.3f}, {hi:+.3f}] {sig}")

# Simpler 2-predictor model (div + n_items only, no era) for comparison
X_simple = np.column_stack([
    np.ones(len(df)),
    df['log_div'].values,
    df['log_n'].values,
])
beta_simple, *_ = np.linalg.lstsq(X_simple, y, rcond=None)
y_pred_simple = X_simple @ beta_simple
r2_simple = 1 - np.sum((y - y_pred_simple)**2) / ss_tot

# Bootstrap simple
boot_betas_s = np.zeros((n_boot, 3))
for b in range(n_boot):
    idx = rng.randint(0, len(df), size=len(df))
    try:
        beta_b, *_ = np.linalg.lstsq(X_simple[idx], y[idx], rcond=None)
        boot_betas_s[b] = beta_b
    except:
        boot_betas_s[b] = np.nan

ci_lo_s = np.nanpercentile(boot_betas_s, 2.5, axis=0)
ci_hi_s = np.nanpercentile(boot_betas_s, 97.5, axis=0)

print(f"\n=== Simple model: log(hl) ~ log(div) + log(n) ===")
print(f"  intercept       = {beta_simple[0]:+.3f} [{ci_lo_s[0]:+.3f}, {ci_hi_s[0]:+.3f}]")
print(f"  log(diversity)  = {beta_simple[1]:+.3f} [{ci_lo_s[1]:+.3f}, {ci_hi_s[1]:+.3f}]")
print(f"  log(n_items)    = {beta_simple[2]:+.3f} [{ci_lo_s[2]:+.3f}, {ci_hi_s[2]:+.3f}]")
print(f"  R² = {r2_simple:.3f}")

# Plot: observed vs predicted
fig, ax = plt.subplots(figsize=(9, 7))
colors_era = {'llm_v1': '#D32F2F', 'llm_v2': '#F57C00', 'bert': '#1976D2'}
for era_name, color in colors_era.items():
    mask = df['era'] == era_name
    ax.scatter(df[mask]['halflife'], np.exp(y_pred[mask.values]),
               s=120, color=color, edgecolor='black', label=era_name, alpha=0.85)
for _, row in df.iterrows():
    ax.annotate(row['name'], (row['halflife'], np.exp(y_pred[df.index.get_loc(row.name)])),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.plot([0.5, 30], [0.5, 30], 'k--', alpha=0.4, label='y=x')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Observed half-life (months, log scale)', fontsize=11)
ax.set_ylabel('Predicted half-life (months, log scale)', fontsize=11)
ax.set_title(f'Meta-regression: 11 benchmarks (6 LLM + 5 BERT-era archaeology)\nFull model R²={r2:.2f}', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_meta_expanded.pdf", bbox_inches='tight')
plt.savefig(FIG_DIR / "fig_meta_expanded.png", bbox_inches='tight', dpi=150)
print(f"\nSaved: {FIG_DIR / 'fig_meta_expanded.pdf'}")

out = {
    'n_benchmarks': len(df),
    'full_model': {
        'coefs': dict(zip(coef_names, [float(b) for b in beta])),
        'ci_lo': dict(zip(coef_names, [float(x) for x in ci_lo])),
        'ci_hi': dict(zip(coef_names, [float(x) for x in ci_hi])),
        'R2': float(r2),
    },
    'simple_model': {
        'coefs': {'intercept': float(beta_simple[0]),
                  'log_diversity': float(beta_simple[1]),
                  'log_n_items': float(beta_simple[2])},
        'ci_lo': {'intercept': float(ci_lo_s[0]),
                  'log_diversity': float(ci_lo_s[1]),
                  'log_n_items': float(ci_lo_s[2])},
        'ci_hi': {'intercept': float(ci_hi_s[0]),
                  'log_diversity': float(ci_hi_s[1]),
                  'log_n_items': float(ci_hi_s[2])},
        'R2': float(r2_simple),
    },
}
with open(OUT_DIR / "results.json", 'w') as f:
    json.dump(out, f, indent=2)
print(f"Saved: {OUT_DIR / 'results.json'}")
