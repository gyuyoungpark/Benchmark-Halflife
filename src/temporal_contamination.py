"""
Wow2: Temporal contamination fingerprinting on GSM8K items.

Per-item gap = orig - pert accuracy, computed separately for each GPT generation.
We look for items where the gap is GROWING over model generations — this is a
progressive-contamination signature (newer models memorized the item, older ones
didn't) which is much stronger evidence than aggregate gap.

Outputs:
  - data/triage/temporal_contamination.json
  - figures/fig_temporal_contamination.pdf
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_DIR = DATA_DIR / "evaluations"
FIG_DIR = Path(__file__).parent.parent / "figures"
OUT_DIR = DATA_DIR / "triage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GPT generations ordered by release date
GPT_ORDER = [
    ("gpt-3.5-turbo-0125",    2024.0),   # Jan 2024 retrained
    ("gpt-4-turbo-2024-04-09", 2024.3),
    ("gpt-4o-2024-08-06",      2024.6),
    ("gpt-4.1-2025-04-14",     2025.3),
]


def load_gap_matrix(bench):
    """Return dict {model: np.array per-item orig - pert (0,±1)}."""
    r = json.load(open(EVAL_DIR / f"{bench}_results.json"))
    out = {}
    for m, _ in GPT_ORDER:
        o = np.array([x if x is not None else np.nan for x in r[m]["orig"]], dtype=float)
        p = np.array([x if x is not None else np.nan for x in r[m]["pert"]], dtype=float)
        out[m] = o - p
    return out


def analyze(bench):
    print(f"\n=== {bench.upper()} ===")
    gaps = load_gap_matrix(bench)
    n = len(next(iter(gaps.values())))
    M = np.stack([gaps[m] for m, _ in GPT_ORDER], axis=0)  # (4, n_items)

    # Mean gap per generation
    print("Mean gap per generation (orig - pert):")
    for (m, t), row in zip(GPT_ORDER, M):
        valid = row[~np.isnan(row)]
        print(f"  {m:<30} t={t:.1f}  gap={valid.mean():+.3f}  n_valid={len(valid)}")

    # Per-item trajectory: fit slope(gap vs model_date) across 4 gens
    times = np.array([t for _, t in GPT_ORDER])
    slopes = np.full(n, np.nan)
    for i in range(n):
        y = M[:, i]
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            continue
        t = times[mask]
        yv = y[mask]
        if np.std(t) < 1e-6:
            continue
        slope = np.polyfit(t, yv, 1)[0]
        slopes[i] = slope

    valid = ~np.isnan(slopes)
    print(f"\nSlope distribution (Δgap / year) on {valid.sum()} items:")
    print(f"  mean={slopes[valid].mean():+.3f}")
    print(f"  median={np.median(slopes[valid]):+.3f}")
    print(f"  % positive={100*(slopes[valid]>0).mean():.1f}")
    print(f"  % with slope > +0.5/yr (progressive contamination signature)={100*(slopes[valid]>0.5).mean():.1f}")

    # Top "progressively contaminated" items
    top_idx = np.argsort(slopes)[-10:][::-1]
    print("Top 10 progressive-contamination items (by slope):")
    for i in top_idx:
        if np.isnan(slopes[i]):
            continue
        traj = [f"{g:+.0f}" if not np.isnan(g) else "NA" for g in M[:, i]]
        print(f"  item {i}: slope={slopes[i]:+.2f}/yr traj={traj}")

    return {
        "bench": bench,
        "n_items": int(n),
        "n_valid": int(valid.sum()),
        "mean_slope": float(slopes[valid].mean()),
        "median_slope": float(np.median(slopes[valid])),
        "pct_positive_slope": float(100 * (slopes[valid] > 0).mean()),
        "pct_steep_positive": float(100 * (slopes[valid] > 0.5).mean()),
        "slopes": slopes.tolist(),
        "gap_matrix": M.tolist(),
        "generations": [{"model": m, "time": t} for m, t in GPT_ORDER],
    }


def plot_distribution(results, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, res in zip(axes, results):
        slopes = np.array(res["slopes"])
        slopes = slopes[~np.isnan(slopes)]
        ax.hist(slopes, bins=25, color='#1976D2', edgecolor='black', alpha=0.8)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(slopes.mean(), color='red', linewidth=2, linestyle='--',
                   label=f"mean={slopes.mean():+.2f}")
        ax.set_title(f"{res['bench'].upper()}\n% slope>0: {res['pct_positive_slope']:.0f}%,  "
                     f"% steep(>0.5/yr): {res['pct_steep_positive']:.0f}%",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("Per-item Δgap / year")
        ax.set_ylabel("Item count")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(str(path).replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f"\nSaved: {path}")


def main():
    results = []
    for bench in ["gsm8k", "mmlu", "arc"]:
        res = analyze(bench)
        results.append(res)

    plot_distribution(results, FIG_DIR / "fig_temporal_contamination.pdf")
    with open(OUT_DIR / "temporal_contamination.json", "w") as f:
        # Drop large fields from JSON summary
        summary = [{k: v for k, v in r.items() if k not in ("slopes", "gap_matrix")} for r in results]
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {OUT_DIR / 'temporal_contamination.json'}")


if __name__ == "__main__":
    main()
