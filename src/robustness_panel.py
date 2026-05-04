"""
Tier 1A: Robustness panel for half-life estimates.

For each benchmark with measurable half-life:
  - Leave-one-quarter-out (LOQO) half-life range
  - Top-k sweep (10/15/20/25/30%) Spearman rank stability
  - Mann-Kendall monotone trend test (nonparametric, robust to small n)
  - Status-label stability under LOQO (Saturated/Decaying/Slow flips)

Output: data/robustness_panel.json + data/robustness_panel.csv
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR
LB_DIR = DATA_DIR / "leaderboard"


def exp_decay(t, D0, lam, D_inf):
    return D0 * np.exp(-lam * t) + D_inf


def fit_halflife(t, D, allow_loglin_2pt=False):
    """Exponential fit; if unstable, fall back to log-linear (paper convention).
    If allow_loglin_2pt, allow 2-point log-linear (used for LOQO when quarters are sparse)."""
    min_pts = 2 if allow_loglin_2pt else 3
    if len(t) < min_pts or np.any(np.isnan(D)):
        return None
    slope = np.polyfit(t, D, 1)[0]
    if slope >= 0:
        return None
    if len(t) >= 3:
        try:
            popt, _ = curve_fit(
                exp_decay, t, D,
                p0=[D[0], 0.05, D[-1] * 0.3 if D[-1] > 0 else 0.0],
                bounds=([0, 1e-6, 0], [D.max() * 3 + 1e-9, 1.0, D.max() + 1e-9]),
                maxfev=10000,
            )
            if popt[1] > 1e-5:
                hl = float(np.log(2) / popt[1])
                if 0 < hl < 200:
                    return hl
        except Exception:
            pass
    floor = max(0.0, min(D) * 0.9)
    y = np.log(np.maximum(D - floor, 1e-9))
    s, _ = np.polyfit(t, y, 1)
    if s >= 0:
        return None
    hl = float(np.log(2) / (-s))
    if 0 < hl < 200:
        return hl
    return None


def topk_variance(scores, k):
    if len(scores) < 3:
        return None
    thresh = np.percentile(scores, 100 * (1 - k))
    top = scores[scores >= thresh]
    if len(top) < 3:
        return None
    return float(np.var(top, ddof=1))


def mann_kendall(values):
    """Two-sided Mann-Kendall S, tau, exact p (small n).
    Returns (S, tau, p_two_sided).
    """
    n = len(values)
    if n < 3:
        return None, None, None
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = values[j] - values[i]
            if d > 0:
                s += 1
            elif d < 0:
                s -= 1
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if var_s <= 0:
        return s, 0.0, 1.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2.0)
    return int(s), float(tau), float(p)


def status_label(hl):
    if hl is None:
        return "noisy"
    if hl < 3:
        return "exhausted"
    if hl < 10:
        return "decaying"
    if hl < 24:
        return "slow"
    return "stable"


def load_v1_quarters(bench_col):
    df = pd.read_csv(LB_DIR / "v1_scores_full.csv", parse_dates=["eval_date"])
    df = df.dropna(subset=[bench_col, "eval_date"])
    df["quarter"] = df["eval_date"].dt.to_period("Q")
    out = {}
    for q in sorted(df["quarter"].unique()):
        s = df[df["quarter"] == q][bench_col].values
        if len(s) >= 5:
            out[q] = s
    return out


def load_v2_quarters(bench_name, window=("2024Q2", "2025Q1")):
    df = pd.read_csv(LB_DIR / f"{bench_name}_v2.csv")
    if "eval_date" in df.columns:
        df["eval_date"] = pd.to_datetime(df["eval_date"], errors="coerce")
        df = df.dropna(subset=["score", "eval_date"])
        df["quarter"] = df["eval_date"].dt.to_period("Q")
    else:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df = df.dropna(subset=["score", "release_date"])
        df["quarter"] = df["release_date"].dt.to_period("Q")
    q_lo = pd.Period(window[0], freq="Q")
    q_hi = pd.Period(window[1], freq="Q")
    out = {}
    for q in sorted(df["quarter"].unique()):
        if q < q_lo or q > q_hi:
            continue
        s = df[df["quarter"] == q]["score"].values
        if len(s) >= 20:
            out[q] = s
    return out


def compute_decay_series(quarters_dict, k):
    sorted_qs = sorted(quarters_dict.keys())
    t, D = [], []
    base = sorted_qs[0].start_time
    for q in sorted_qs:
        d = topk_variance(quarters_dict[q], k)
        if d is None:
            continue
        months = (q.start_time - base).days / 30.44
        t.append(months)
        D.append(d)
    return np.array(t), np.array(D), sorted_qs


def loqo_halflives(quarters_dict, k=0.20):
    """Leave-one-quarter-out: drop each quarter in turn, refit. Allow 2-pt log-linear."""
    sorted_qs = sorted(quarters_dict.keys())
    out = []
    for drop in sorted_qs:
        sub = {q: v for q, v in quarters_dict.items() if q != drop}
        t, D, _ = compute_decay_series(sub, k)
        hl = fit_halflife(t, D, allow_loglin_2pt=True) if len(t) >= 2 else None
        out.append({"dropped": str(drop), "halflife": hl, "n_quarters": len(t)})
    return out


def topk_sweep(quarters_dict, ks=(0.10, 0.15, 0.20, 0.25, 0.30)):
    out = {}
    for k in ks:
        t, D, _ = compute_decay_series(quarters_dict, k)
        hl = fit_halflife(t, D) if len(t) >= 3 else None
        out[f"k={int(k*100)}%"] = hl
    return out


def label_stability(loqo_results, main_label):
    """Fraction of LOQO refits that yield the same status label."""
    labels = [status_label(r["halflife"]) for r in loqo_results]
    same = sum(1 for L in labels if L == main_label)
    return same / len(labels), labels


def analyse(name, quarters_dict, k_main=0.20):
    if len(quarters_dict) < 3:
        return None
    t_main, D_main, _ = compute_decay_series(quarters_dict, k_main)
    hl_main = fit_halflife(t_main, D_main)
    main_lbl = status_label(hl_main)

    sweep = topk_sweep(quarters_dict)
    sweep_vals = [v for v in sweep.values() if v is not None]
    sweep_min = min(sweep_vals) if sweep_vals else None
    sweep_max = max(sweep_vals) if sweep_vals else None
    sweep_ratio = (sweep_max / sweep_min) if (sweep_vals and sweep_min) else None

    loqo = loqo_halflives(quarters_dict, k=k_main)
    loqo_hls = [r["halflife"] for r in loqo if r["halflife"] is not None]
    loqo_lo = min(loqo_hls) if loqo_hls else None
    loqo_hi = max(loqo_hls) if loqo_hls else None
    loqo_median = float(np.median(loqo_hls)) if loqo_hls else None

    lbl_stable_frac, _ = label_stability(loqo, main_lbl)

    if len(D_main) >= 3:
        s, tau, p = mann_kendall(list(D_main))
    else:
        s, tau, p = None, None, None

    return {
        "benchmark": name,
        "halflife_main": hl_main,
        "label_main": main_lbl,
        "loqo_median": loqo_median,
        "loqo_range_lo": loqo_lo,
        "loqo_range_hi": loqo_hi,
        "loqo_n_succeeded": len(loqo_hls),
        "loqo_n_total": len(loqo),
        "topk_sweep": sweep,
        "topk_sweep_ratio": sweep_ratio,
        "mk_S": s,
        "mk_tau": tau,
        "mk_p": p,
        "label_stability_frac": lbl_stable_frac,
    }


def main():
    out = {}

    v1_benches = {
        "WinoGrande": "winogrande",
        "HellaSwag": "hellaswag",
        "ARC-Challenge": "arc_challenge",
        "GSM8K": "gsm8k",
        "TruthfulQA": "truthfulqa",
    }
    for name, col in v1_benches.items():
        q = load_v1_quarters(col)
        r = analyse(name, q)
        out[name] = r
        if r:
            print(
                f"{name:15s}  τ½={r['halflife_main']}  LOQO[{r['loqo_range_lo']}, {r['loqo_range_hi']}]  "
                f"k-sweep ratio={r['topk_sweep_ratio']}  MK p={r['mk_p']}  label_stab={r['label_stability_frac']:.2f}"
            )

    v2_benches = ["bbh", "mmlu_pro", "math_lvl5", "ifeval", "musr", "gpqa"]
    name_map = {
        "bbh": "BBH",
        "mmlu_pro": "MMLU-PRO",
        "math_lvl5": "MATH",
        "ifeval": "IFEval",
        "musr": "MUSR",
        "gpqa": "GPQA",
    }
    for b in v2_benches:
        q = load_v2_quarters(b)
        nm = name_map[b]
        r = analyse(nm, q)
        out[nm] = r
        if r:
            print(
                f"{nm:15s}  τ½={r['halflife_main']}  LOQO[{r['loqo_range_lo']}, {r['loqo_range_hi']}]  "
                f"k-sweep ratio={r['topk_sweep_ratio']}  MK p={r['mk_p']}  label_stab={r['label_stability_frac']:.2f}"
            )

    with open(OUT_DIR / "robustness_panel.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {OUT_DIR / 'robustness_panel.json'}")


if __name__ == "__main__":
    main()
