"""
Expanded meta-regression: n=6 → n=20

Adds:
- 5 v2 LLM-era benchmarks with observed prospective half-lives
- 4 additional BERT-era benchmarks (SNLI, MNLI, STS-B, CoNLL-03)

Subtask counts (n_sub) from official benchmark documentation:
- MMLU-PRO: 14 categories
- GPQA: 3 domains (physics, chemistry, biology)
- MATH Lvl 5: 7 topics (algebra, counting_prob, geometry, intermed_algebra, number_theory, prealgebra, precalculus)
- IFEval: 25 instruction types
- MUSR: 3 tasks (murder mystery, object placement, team allocation)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"
OUT_DIR = DATA_DIR / "meta_expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Original 6 LLM v1 (best-AIC half-lives)
DATA = [
    # name, halflife, n_items, n_sub, era
    ("WinoGrande",   1.7,  1267,   1, "llm_v1"),
    ("HellaSwag",    2.5,  10042,  1, "llm_v1"),
    ("ARC-C",        3.4,  1172,   1, "llm_v1"),
    ("GSM8K",        4.9,  1319,   1, "llm_v1"),
    ("TruthfulQA",   5.7,  817,    1, "llm_v1"),
    ("BBH",         19.6,  6511,  23, "llm_v1"),
    # 4 v2 LLM (observed prospective half-lives from 4-quarter window)
    # BBH excluded: already in v1 (use updated 6.4mo below in sensitivity)
    ("MMLU-PRO",     8.7,  12032, 14, "llm_v2"),
    ("IFEval",       1.6,   541,  25, "llm_v2"),
    ("MUSR",         5.4,   756,   3, "llm_v2"),
    ("MATH Lvl 5",  18.7,  5000,   7, "llm_v2"),
    # 9 BERT-era (gap-to-ceiling fits)
    ("GLUE",         4.9,  65000,  8, "bert"),
    ("SuperGLUE",    1.5,  5000,  10, "bert"),
    ("SQuAD 1.1",   10.8,  10570,  1, "bert"),
    ("SQuAD 2.0",    3.9,  11873,  1, "bert"),
    ("CoQA",         1.0,  7983,   1, "bert"),
    ("SNLI",         3.6,  10000,  1, "bert"),
    ("MNLI",         8.9,  10000,  1, "bert"),
    ("STS-B",       10.8,  1500,   1, "bert"),
    ("CoNLL-03",    56.3,  23499,  1, "bert"),
]

df = pd.DataFrame(DATA, columns=["name", "halflife", "n_items", "n_sub", "era"])
df['log_hl'] = np.log(df['halflife'])
df['log_sub'] = np.log(df['n_sub'])
df['log_n'] = np.log(df['n_items'])
df['is_llm'] = df['era'].str.startswith('llm').astype(int)


def fit_ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = max(0, 1 - ss_res / ss_tot)
    return beta, r2, y_pred


def bootstrap_ols(X, y, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    boot_betas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            b, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
            boot_betas.append(b)
        except:
            pass
    boot_betas = np.array(boot_betas)
    return np.percentile(boot_betas, 2.5, axis=0), np.percentile(boot_betas, 97.5, axis=0)


def loo_mae(X, y):
    """Leave-one-out log-scale MAE."""
    n = len(y)
    errors = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        b, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        pred = X[i] @ b
        errors.append(abs(pred - y[i]))
    return np.mean(errors)


print("=" * 70)
print("EXPANDED META-REGRESSION (n=20)")
print("=" * 70)

# =====================================================
# Model 1: Original LLM v1 only (n=6)
# =====================================================
v1 = df[df['era'] == 'llm_v1'].reset_index(drop=True)
X1 = np.column_stack([np.ones(len(v1)), v1['log_sub'].values, v1['log_n'].values])
y1 = v1['log_hl'].values
beta1, r2_1, pred1 = fit_ols(X1, y1)
ci1_lo, ci1_hi = bootstrap_ols(X1, y1)
loo1 = loo_mae(X1, y1)

print(f"\nModel 1: LLM v1 only (n={len(v1)}, original)")
print(f"  intercept:     {beta1[0]:+.3f} [{ci1_lo[0]:+.3f}, {ci1_hi[0]:+.3f}]")
print(f"  log(n_sub):    {beta1[1]:+.3f} [{ci1_lo[1]:+.3f}, {ci1_hi[1]:+.3f}]")
print(f"  log(n_items):  {beta1[2]:+.3f} [{ci1_lo[2]:+.3f}, {ci1_hi[2]:+.3f}]")
print(f"  R² = {r2_1:.3f}, LOO MAE = {loo1:.3f} (factor {np.exp(loo1):.2f}x)")

r_sub_v1, p_sub_v1 = pearsonr(v1['log_sub'], v1['log_hl'])
print(f"  Univariate r(log_sub, log_hl) = {r_sub_v1:+.3f}, p = {p_sub_v1:.4f}")

# =====================================================
# Model 2: All LLM era (v1+v2, n=11)
# =====================================================
llm = df[df['era'].str.startswith('llm')].reset_index(drop=True)
X2 = np.column_stack([np.ones(len(llm)), llm['log_sub'].values, llm['log_n'].values])
y2 = llm['log_hl'].values
beta2, r2_2, pred2 = fit_ols(X2, y2)
ci2_lo, ci2_hi = bootstrap_ols(X2, y2)
loo2 = loo_mae(X2, y2)

print(f"\nModel 2: All LLM (v1+v2, n={len(llm)})")
print(f"  intercept:     {beta2[0]:+.3f} [{ci2_lo[0]:+.3f}, {ci2_hi[0]:+.3f}]")
print(f"  log(n_sub):    {beta2[1]:+.3f} [{ci2_lo[1]:+.3f}, {ci2_hi[1]:+.3f}]")
print(f"  log(n_items):  {beta2[2]:+.3f} [{ci2_lo[2]:+.3f}, {ci2_hi[2]:+.3f}]")
print(f"  R² = {r2_2:.3f}, LOO MAE = {loo2:.3f} (factor {np.exp(loo2):.2f}x)")

r_sub_llm, p_sub_llm = pearsonr(llm['log_sub'], llm['log_hl'])
rho_sub_llm, p_rho_llm = spearmanr(llm['log_sub'], llm['log_hl'])
print(f"  Pearson  r(log_sub, log_hl) = {r_sub_llm:+.3f}, p = {p_sub_llm:.4f}")
print(f"  Spearman rho               = {rho_sub_llm:+.3f}, p = {p_rho_llm:.4f}")

# =====================================================
# Model 3: All 20 with era fixed effect
# =====================================================
X3 = np.column_stack([np.ones(len(df)), df['log_sub'].values, df['log_n'].values, df['is_llm'].values])
y3 = df['log_hl'].values
beta3, r2_3, pred3 = fit_ols(X3, y3)
ci3_lo, ci3_hi = bootstrap_ols(X3, y3)
loo3 = loo_mae(X3, y3)

print(f"\nModel 3: All benchmarks with era effect (n={len(df)})")
print(f"  intercept:     {beta3[0]:+.3f} [{ci3_lo[0]:+.3f}, {ci3_hi[0]:+.3f}]")
print(f"  log(n_sub):    {beta3[1]:+.3f} [{ci3_lo[1]:+.3f}, {ci3_hi[1]:+.3f}]")
print(f"  log(n_items):  {beta3[2]:+.3f} [{ci3_lo[2]:+.3f}, {ci3_hi[2]:+.3f}]")
print(f"  is_llm:        {beta3[3]:+.3f} [{ci3_lo[3]:+.3f}, {ci3_hi[3]:+.3f}]")
print(f"  R² = {r2_3:.3f}, LOO MAE = {loo3:.3f} (factor {np.exp(loo3):.2f}x)")

r_sub_all, p_sub_all = pearsonr(df['log_sub'], df['log_hl'])
print(f"  Univariate r(log_sub, log_hl) = {r_sub_all:+.3f}, p = {p_sub_all:.4f}")

# =====================================================
# Model 4: BERT only (n=9, expanded)
# =====================================================
bert = df[df['era'] == 'bert'].reset_index(drop=True)
X4 = np.column_stack([np.ones(len(bert)), bert['log_sub'].values, bert['log_n'].values])
y4 = bert['log_hl'].values
beta4, r2_4, pred4 = fit_ols(X4, y4)

print(f"\nModel 4: BERT only (n={len(bert)})")
print(f"  intercept:     {beta4[0]:+.3f}")
print(f"  log(n_sub):    {beta4[1]:+.3f}")
print(f"  log(n_items):  {beta4[2]:+.3f}")
print(f"  R² = {r2_4:.3f}")

r_sub_bert, p_sub_bert = pearsonr(bert['log_sub'], bert['log_hl'])
print(f"  r(log_sub, log_hl) = {r_sub_bert:+.3f}, p = {p_sub_bert:.4f}")

# =====================================================
# Key comparison: does n_sub predict within LLM v2?
# =====================================================
v2 = df[df['era'] == 'llm_v2'].reset_index(drop=True)
if len(v2) > 2:
    r_v2, p_v2 = pearsonr(v2['log_sub'], v2['log_hl'])
    print(f"\nWithin LLM v2 only (n={len(v2)}): r = {r_v2:+.3f}, p = {p_v2:.4f}")

# =====================================================
# v1 → v2 out-of-sample prediction
# =====================================================
print("\n" + "=" * 70)
print("OUT-OF-SAMPLE: v1 model predicting v2 half-lives")
print("=" * 70)
v2_pred = X2[len(v1):] @ beta1  # Use v1 model to predict v2
v2_obs = y2[len(v1):]
v2_names = llm['name'].values[len(v1):]
for name, obs, pred in zip(v2_names, v2_obs, v2_pred):
    obs_hl = np.exp(obs)
    pred_hl = np.exp(pred)
    ratio = obs_hl / pred_hl
    print(f"  {name:<15} observed={obs_hl:.1f}mo, predicted={pred_hl:.1f}mo, ratio={ratio:.2f}x")

oos_mae = np.mean(np.abs(v2_obs - v2_pred))
print(f"\n  Out-of-sample log-MAE = {oos_mae:.3f} (factor {np.exp(oos_mae):.2f}x)")

# =====================================================
# Figure
# =====================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1: LLM v1 original (n=6)
ax = axes[0]
ax.scatter(v1['n_sub'], v1['halflife'], s=120, c='#D32F2F', edgecolor='black', zorder=3)
for _, row in v1.iterrows():
    ax.annotate(row['name'], (row['n_sub'], row['halflife']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Subtask count $n_{\\mathrm{sub}}$')
ax.set_ylabel('Half-life (months)')
ax.set_title(f'LLM v1 (n=6)\nr = {r_sub_v1:+.2f}, p = {p_sub_v1:.3f}, R² = {r2_1:.2f}', fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 2: All LLM (n=11)
ax = axes[1]
colors_v = {'llm_v1': '#D32F2F', 'llm_v2': '#FF9800'}
for era in ['llm_v1', 'llm_v2']:
    mask = llm['era'] == era
    label = 'v1 (variance decay)' if era == 'llm_v1' else 'v2 (prospective)'
    ax.scatter(llm[mask]['n_sub'], llm[mask]['halflife'], s=120,
               c=colors_v[era], edgecolor='black', label=label, zorder=3)
for _, row in llm.iterrows():
    ax.annotate(row['name'], (row['n_sub'], row['halflife']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Subtask count $n_{\\mathrm{sub}}$')
ax.set_ylabel('Half-life (months)')
ax.set_title(f'All LLM (n=11)\nr = {r_sub_llm:+.2f}, p = {p_sub_llm:.3f}, R² = {r2_2:.2f}', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: All 20 benchmarks
ax = axes[2]
era_colors = {'llm_v1': '#D32F2F', 'llm_v2': '#FF9800', 'bert': '#1976D2'}
era_labels = {'llm_v1': 'LLM v1', 'llm_v2': 'LLM v2', 'bert': 'BERT era'}
for era in ['llm_v1', 'llm_v2', 'bert']:
    mask = df['era'] == era
    ax.scatter(df[mask]['n_sub'], df[mask]['halflife'], s=120,
               c=era_colors[era], edgecolor='black', label=era_labels[era], zorder=3)
for _, row in df.iterrows():
    ax.annotate(row['name'], (row['n_sub'], row['halflife']),
                xytext=(5, 3), textcoords='offset points', fontsize=7)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Subtask count $n_{\\mathrm{sub}}$')
ax.set_ylabel('Half-life (months)')
ax.set_title(f'All benchmarks (n=20)\nr = {r_sub_all:+.2f}, R² = {r2_3:.2f} (with era)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_meta_v3.pdf", bbox_inches='tight')
plt.savefig(FIG_DIR / "fig_meta_v3.png", bbox_inches='tight', dpi=150)
print(f"\nSaved: {FIG_DIR / 'fig_meta_v3.pdf'}")

# Save results
results = {
    'model1_llm_v1_n6': {
        'n': len(v1), 'R2': float(r2_1),
        'r_sub': float(r_sub_v1), 'p_sub': float(p_sub_v1),
        'loo_mae': float(loo1),
        'coefs': {'intercept': float(beta1[0]), 'log_sub': float(beta1[1]), 'log_n': float(beta1[2])},
    },
    'model2_llm_all_n11': {
        'n': len(llm), 'R2': float(r2_2),
        'r_sub': float(r_sub_llm), 'p_sub': float(p_sub_llm),
        'rho_sub': float(rho_sub_llm), 'p_rho': float(p_rho_llm),
        'loo_mae': float(loo2),
        'coefs': {'intercept': float(beta2[0]), 'log_sub': float(beta2[1]), 'log_n': float(beta2[2])},
    },
    'model3_all_n20': {
        'n': len(df), 'R2': float(r2_3),
        'r_sub': float(r_sub_all), 'p_sub': float(p_sub_all),
        'loo_mae': float(loo3),
    },
    'model4_bert_n9': {
        'n': len(bert), 'R2': float(r2_4),
        'r_sub': float(r_sub_bert), 'p_sub': float(p_sub_bert),
    },
    'oos_v1_to_v2': {
        'log_mae': float(oos_mae),
        'factor': float(np.exp(oos_mae)),
    },
}
with open(OUT_DIR / "results_v3.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {OUT_DIR / 'results_v3.json'}")
