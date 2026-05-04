"""
E4: Bootstrap 95% CI for contamination gaps in Table 5.
Resample items with replacement and recompute per-model gap.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_DIR = DATA_DIR / "evaluations"
OUT_DIR = DATA_DIR / "bootstrap_gaps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-08-06",
    "gpt-4.1-2025-04-14",
]


def bootstrap_gap(orig, pert, n_boot=1000, seed=42):
    """Bootstrap CI for paired accuracy gap."""
    rng = np.random.RandomState(seed)
    orig = np.array(orig, dtype=float)
    pert = np.array(pert, dtype=float)
    valid = ~np.isnan(orig) & ~np.isnan(pert)
    orig, pert = orig[valid], pert[valid]
    n = len(orig)
    if n < 30:
        return None

    point_gap = orig.mean() - pert.mean()
    boot_gaps = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        gap = orig[idx].mean() - pert[idx].mean()
        boot_gaps.append(gap)
    return {
        "point": float(point_gap),
        "ci_lo": float(np.percentile(boot_gaps, 2.5)),
        "ci_hi": float(np.percentile(boot_gaps, 97.5)),
        "n": int(n),
    }


def main():
    results = {}
    for bench in ["mmlu", "arc", "gsm8k"]:
        path = EVAL_DIR / f"{bench}_results.json"
        if not path.exists():
            continue
        with open(path) as f:
            r = json.load(f)
        results[bench] = {}
        print(f"\n=== {bench.upper()} ===")
        for m in MODELS:
            if m not in r:
                continue
            orig = [x if x is not None else np.nan for x in r[m]["orig"]]
            pert = [x if x is not None else np.nan for x in r[m]["pert"]]
            ci = bootstrap_gap(orig, pert)
            if ci is None:
                continue
            results[bench][m] = ci
            print(f"  {m:<30} gap = {ci['point']:+.3f} [{ci['ci_lo']:+.3f}, {ci['ci_hi']:+.3f}] (n={ci['n']})")

    with open(OUT_DIR / "gap_cis.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
