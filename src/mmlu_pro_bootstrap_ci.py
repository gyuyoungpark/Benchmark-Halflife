"""Verify paper's MMLU-PRO claim: '+2.8pp, 95% bootstrap CI [+0.011, +0.046]' on 6 frontier models."""
import json
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

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


# Build combined per-model orig/pert arrays
combined = {}
for m in FRONTIER:
    if m not in main:
        continue
    if m in extra:
        o = np.array(main[m]["orig"] + extra[m]["orig"])
        p = np.array(main[m]["pert"] + extra[m]["pert"])
    else:
        o = np.array(main[m]["orig"])
        p = np.array(main[m]["pert"])
    combined[m] = (o, p)

print("Per-model gaps on combined set:")
gaps = []
for m, (o, p) in combined.items():
    g = o.mean() - p.mean()
    gaps.append(g)
    print(f"  {m:30s} n={len(o):3d}  orig={o.mean():.3f}  pert={p.mean():.3f}  gap={g*100:+5.2f}pp")
mean_gap = np.mean(gaps)
print(f"\n6-model mean gap: {mean_gap*100:+.2f}pp")


# Item-level bootstrap on per-model gap, then mean across models
# Each model has different n; bootstrap separately, average gaps, then resample
rng = np.random.RandomState(42)
n_boot = 5000
boot_means = []
for _ in range(n_boot):
    boot_gaps = []
    for m, (o, p) in combined.items():
        n = len(o)
        ix = rng.randint(0, n, size=n)
        boot_gaps.append(o[ix].mean() - p[ix].mean())
    boot_means.append(np.mean(boot_gaps))
boot_means = np.array(boot_means)
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
print(f"\nItem-bootstrap 95% CI for 6-model mean gap: [{ci_lo*100:+.2f}pp, {ci_hi*100:+.2f}pp]")
print(f"Paper claim: '95% bootstrap CI [+0.011, +0.046]' i.e. [+1.1pp, +4.6pp]")

# Per-model bootstrap CI (model-level resample)
boot_means_v2 = []
gap_arr = np.array(gaps)
n_models = len(gap_arr)
for _ in range(n_boot):
    idx = rng.randint(0, n_models, size=n_models)
    boot_means_v2.append(gap_arr[idx].mean())
ci_lo2, ci_hi2 = np.percentile(boot_means_v2, [2.5, 97.5])
print(f"\nModel-bootstrap CI (n=6 models): [{ci_lo2*100:+.2f}pp, {ci_hi2*100:+.2f}pp]")
