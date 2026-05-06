"""
Tier 0: Verify the three numbers in §3 'Practical significance vs. baseline noise':
  1. Label-permutation null 95% CI
  2. Cross-paraphrase / replication-set residual gap
  3. Median accuracy step between adjacent frontier models
"""
import json
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

# 6 frontier models per main text (gpt-3.5-turbo excluded as non-frontier on MMLU-PRO)
FRONTIER = [
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-08-06",
    "gpt-4.1-2025-04-14",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
]

main = json.load(open(DATA / "evaluations" / "mmlu_pro_results.json"))
extra = json.load(open(DATA / "evaluations" / "mmlu_pro_extra_results.json"))


def acc(arr):
    return float(np.mean(arr))


# ---------- 1. Observed gap on main set (sanity check vs paper +2.8pp) ----------
print("=" * 60)
print("Sanity: observed orig-pert gap on main MMLU-PRO 196-item set")
print("=" * 60)
gaps_main = {}
for m in FRONTIER:
    if m not in main:
        continue
    o = np.array(main[m]["orig"])
    p = np.array(main[m]["pert"])
    g = o.mean() - p.mean()
    gaps_main[m] = g
    print(f"  {m:30s} orig={o.mean():.3f} pert={p.mean():.3f} gap={g:+.4f}")
mean_gap = np.mean(list(gaps_main.values()))
print(f"\n6-frontier-model mean gap: {mean_gap:+.4f} ({mean_gap*100:+.2f}pp)")
print(f"(Paper claim: +2.8pp on ~400 items combining main+extra)")


# ---------- 1. Label-permutation null on combined 196+198 = 394 items ----------
print("\n" + "=" * 60)
print("(1) Label-permutation null CI for orig-pert gap")
print("=" * 60)

# Combine main 196 + extra 198 for the 3 Claude models (the ~400-item set)
claude_models = [m for m in FRONTIER if m.startswith("claude")]

# Per-model: build (orig, pert) arrays from combined set
combined = {}
for m in claude_models:
    if m in main and m in extra:
        o = np.array(main[m]["orig"] + extra[m]["orig"])
        p = np.array(main[m]["pert"] + extra[m]["pert"])
    elif m in main:
        o = np.array(main[m]["orig"])
        p = np.array(main[m]["pert"])
    else:
        continue
    combined[m] = (o, p)
    print(f"  {m:30s} combined n={len(o)} orig acc={o.mean():.3f} pert acc={p.mean():.3f}")

# Permutation null: for each item, randomly swap orig <-> pert label, recompute mean gap
rng = np.random.RandomState(42)
n_perm = 10000
null_gaps = []
for _ in range(n_perm):
    # one permutation across all models simultaneously (item-level)
    item_gaps = []
    for m, (o, p) in combined.items():
        n = len(o)
        flip = rng.randint(0, 2, size=n).astype(bool)
        # if flip[i], swap labels
        new_o = np.where(flip, p, o)
        new_p = np.where(flip, o, p)
        item_gaps.append(new_o.mean() - new_p.mean())
    null_gaps.append(np.mean(item_gaps))
null_gaps = np.array(null_gaps)
ci_lo, ci_hi = np.percentile(null_gaps, [2.5, 97.5])
print(f"\n  Null distribution (model-pooled mean gap): mean={null_gaps.mean():+.5f}")
print(f"  95% null CI: [{ci_lo*100:+.2f}pp, {ci_hi*100:+.2f}pp]")
print(f"  Paper claim: '95% null CI of [-0.014, +0.014]' (i.e. ±1.4pp)")


# ---------- 2. Cross-paraphrase / replication residual ----------
print("\n" + "=" * 60)
print("(2) Cross-paraphrase / replication-set residual gap")
print("=" * 60)

# The extra set is independent items, NOT same-items-different-paraphrase.
# So the closest interpretable quantity is: gap on the extra (replication) set
# minus gap on the main set, per model. If memorization is real and consistent,
# residual should be near zero.
print("Note: 'extra' set is 198 INDEPENDENT items (not same-items-paraphrase-2).")
print("Residual = |gap_main - gap_extra| measures within-model gap consistency.\n")

residuals = []
for m in claude_models:
    if m in main and m in extra:
        gm = np.mean(main[m]["orig"]) - np.mean(main[m]["pert"])
        ge = np.mean(extra[m]["orig"]) - np.mean(extra[m]["pert"])
        diff = ge - gm
        residuals.append(diff)
        print(f"  {m:30s} gap_main={gm:+.4f} gap_extra={ge:+.4f} diff={diff:+.4f}")
res = np.array(residuals)
print(f"\n  Mean cross-set residual: {res.mean():+.4f} ({res.mean()*100:+.2f}pp)")
print(f"  Std: {res.std(ddof=1):.4f} ({res.std(ddof=1)*100:.2f}pp)")
print(f"  Paper claim: 'residual gap +0.002 ± 0.008pp'")


# ---------- 3. Frontier-model rank step ----------
print("\n" + "=" * 60)
print("(3) Median accuracy step between adjacent frontier models")
print("=" * 60)

# Use orig accuracy from MMLU-PRO main set
accs = []
for m in FRONTIER:
    if m in main:
        accs.append((m, np.mean(main[m]["orig"])))
accs.sort(key=lambda x: x[1])
print("Sorted by orig accuracy:")
for m, a in accs:
    print(f"  {m:30s} {a:.3f}")
steps = np.diff([a for _, a in accs])
print(f"\nAdjacent steps: {[f'{s*100:.2f}pp' for s in steps]}")
print(f"Median step: {np.median(steps)*100:.2f}pp")
print(f"Mean step: {np.mean(steps)*100:.2f}pp")
print(f"Paper claim: '≈ 3.5pp'")


print("\n" + "=" * 60)
print("Summary table:")
print("=" * 60)
print(f"  Statistic                          | Paper claim     | Computed")
print(f"  Observed mean gap (6 frontier)     | +2.8pp          | {mean_gap*100:+.2f}pp")
print(f"  Null CI half-width (label-perm)    | ±1.4pp          | ±{(ci_hi-ci_lo)/2*100:.2f}pp")
print(f"  Cross-set residual (mean)          | +0.2pp          | {res.mean()*100:+.2f}pp")
print(f"  Cross-set residual (std)           | 0.8pp           | {res.std(ddof=1)*100:.2f}pp")
print(f"  Median frontier rank step          | ~3.5pp          | {np.median(steps)*100:.2f}pp")
