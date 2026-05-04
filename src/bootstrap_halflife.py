"""
T3: Bootstrap 95% CI for all half-life estimates.

For each (benchmark, stratum), bootstrap the model population at each quarter,
recompute variance, refit decay curve, extract halflife.
Repeat 1000 times to get 2.5/97.5 percentiles.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "bootstrap"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def exp_decay(t, D0, lam, D_inf):
    return D0 * np.exp(-lam * t) + D_inf


def fit_halflife(t, D):
    """Fit exponential decay, return halflife in months. Returns None on failure or if not decaying."""
    if len(t) < 3 or np.any(np.isnan(D)):
        return None
    # Only fit if linear trend is decreasing
    if len(t) >= 2:
        slope = np.polyfit(t, D, 1)[0]
        if slope >= 0:
            return None
    try:
        popt, _ = curve_fit(
            exp_decay, t, D,
            p0=[D[0], 0.05, D[-1]*0.3 if D[-1] > 0 else 0.0],
            bounds=([0, 1e-6, 0], [D.max()*3+1e-9, 1.0, D.max()+1e-9]),
            maxfev=10000
        )
        if popt[1] <= 1e-5:
            return None
        hl = float(np.log(2) / popt[1])
        # Sanity check: halflife should be in plausible range
        if hl > 200 or hl <= 0:
            return None
        return hl
    except Exception:
        return None


def bootstrap_v2_benchmark(benchmark_name, top_k=0.20, n_boot=1000, seed=42):
    """Bootstrap halflife CI for v2 benchmark (top-k stratum variance decay)."""
    df = pd.read_csv(DATA_DIR / "leaderboard" / f"{benchmark_name}_v2.csv",
                     parse_dates=['release_date'])
    df = df.dropna(subset=['release_date', 'score'])
    df['quarter'] = df['release_date'].dt.to_period('Q')
    quarters = sorted(df['quarter'].unique())

    rng = np.random.RandomState(seed)
    halflives = []
    point_estimate = None

    # Compute quarter -> models map
    quarter_models = {}
    for q in quarters:
        q_df = df[df['quarter'] == q]
        if len(q_df) >= 10:
            quarter_models[q] = q_df['score'].values

    if len(quarter_models) < 4:
        return None

    # Point estimate
    t_pts, D_pts = [], []
    for q in sorted(quarter_models.keys()):
        scores = quarter_models[q]
        thresh = np.percentile(scores, 100 * (1 - top_k))
        top = scores[scores >= thresh]
        if len(top) >= 3:
            months = (q.start_time - sorted(quarter_models.keys())[0].start_time).days / 30.44
            t_pts.append(months)
            D_pts.append(np.var(top, ddof=1))
    if len(t_pts) < 4:
        return None
    point_estimate = fit_halflife(np.array(t_pts), np.array(D_pts))

    # Bootstrap
    for b in range(n_boot):
        t_boot, D_boot = [], []
        for q in sorted(quarter_models.keys()):
            scores = quarter_models[q]
            n = len(scores)
            idx = rng.randint(0, n, size=n)
            resampled = scores[idx]
            thresh = np.percentile(resampled, 100 * (1 - top_k))
            top = resampled[resampled >= thresh]
            if len(top) >= 3:
                months = (q.start_time - sorted(quarter_models.keys())[0].start_time).days / 30.44
                t_boot.append(months)
                D_boot.append(np.var(top, ddof=1))
        if len(t_boot) >= 4:
            hl = fit_halflife(np.array(t_boot), np.array(D_boot))
            if hl is not None and 0 < hl < 1000:
                halflives.append(hl)

    if len(halflives) < 50:
        return {"point": point_estimate, "n_boot": len(halflives), "ci_lo": None, "ci_hi": None}

    return {
        "point": point_estimate,
        "n_boot": len(halflives),
        "ci_lo": float(np.percentile(halflives, 2.5)),
        "ci_hi": float(np.percentile(halflives, 97.5)),
        "median": float(np.median(halflives)),
    }


def bootstrap_v1_benchmark(benchmark_name, top_k=0.20, n_boot=1000, seed=42):
    """Bootstrap halflife CI for v1 benchmark."""
    df = pd.read_csv(DATA_DIR / "leaderboard" / "v1_scores_full.csv",
                     parse_dates=['eval_date'])
    df = df.dropna(subset=[benchmark_name, 'eval_date'])
    df['quarter'] = df['eval_date'].dt.to_period('Q')
    quarters = sorted(df['quarter'].unique())

    rng = np.random.RandomState(seed)
    halflives = []

    # Quarter -> scores
    quarter_scores = {}
    for q in quarters:
        scores = df[df['quarter'] == q][benchmark_name].values
        if len(scores) >= 5:
            quarter_scores[q] = scores

    if len(quarter_scores) < 3:
        return None

    # Point estimate
    sorted_qs = sorted(quarter_scores.keys())
    t_pts, D_pts = [], []
    for q in sorted_qs:
        scores = quarter_scores[q]
        thresh = np.percentile(scores, 100 * (1 - top_k))
        top = scores[scores >= thresh]
        if len(top) >= 3:
            months = (q.start_time - sorted_qs[0].start_time).days / 30.44
            t_pts.append(months)
            D_pts.append(np.var(top, ddof=1))
    if len(t_pts) < 3:
        return None
    point_estimate = fit_halflife(np.array(t_pts), np.array(D_pts))

    # Bootstrap
    for b in range(n_boot):
        t_boot, D_boot = [], []
        for q in sorted_qs:
            scores = quarter_scores[q]
            n = len(scores)
            idx = rng.randint(0, n, size=n)
            resampled = scores[idx]
            thresh = np.percentile(resampled, 100 * (1 - top_k))
            top = resampled[resampled >= thresh]
            if len(top) >= 3:
                months = (q.start_time - sorted_qs[0].start_time).days / 30.44
                t_boot.append(months)
                D_boot.append(np.var(top, ddof=1))
        if len(t_boot) >= 3:
            hl = fit_halflife(np.array(t_boot), np.array(D_boot))
            if hl is not None and 0 < hl < 1000:
                halflives.append(hl)

    if len(halflives) < 50:
        return {"point": point_estimate, "n_boot": len(halflives), "ci_lo": None, "ci_hi": None}

    return {
        "point": point_estimate,
        "n_boot": len(halflives),
        "ci_lo": float(np.percentile(halflives, 2.5)),
        "ci_hi": float(np.percentile(halflives, 97.5)),
        "median": float(np.median(halflives)),
    }


def main():
    results = {}

    print("=" * 60)
    print("V2 benchmarks")
    print("=" * 60)
    for bench in ['bbh', 'mmlu_pro', 'gpqa', 'ifeval', 'math_lvl5', 'musr']:
        print(f"\n{bench}...")
        r = bootstrap_v2_benchmark(bench)
        results[f"{bench}_v2"] = r
        if r and r.get('point') is not None:
            ci_str = f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]" if r.get('ci_lo') else "[N/A]"
            print(f"  point={r['point']:.1f}mo  CI={ci_str}  n_boot={r['n_boot']}")
        else:
            print(f"  no decay (still growing)")

    print("\n" + "=" * 60)
    print("V1 benchmarks")
    print("=" * 60)
    for bench in ['arc_challenge', 'hellaswag', 'truthfulqa', 'winogrande', 'gsm8k']:
        print(f"\n{bench}...")
        r = bootstrap_v1_benchmark(bench)
        results[f"{bench}_v1"] = r
        if r and r.get('point') is not None:
            ci_str = f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]" if r.get('ci_lo') else "[N/A]"
            print(f"  point={r['point']:.1f}mo  CI={ci_str}  n_boot={r['n_boot']}")
        else:
            print(f"  no fit")

    with open(OUT_DIR / "halflife_bootstrap.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'halflife_bootstrap.json'}")


if __name__ == "__main__":
    main()
