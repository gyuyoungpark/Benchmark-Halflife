"""
Core half-life computation: given a time-indexed score CSV, compute
discriminative half-life with bootstrap CIs.

Input CSV format:
    model,score,eval_date
    model_a,0.85,2024-01-15
    model_b,0.72,2024-01-20
    ...

Output: dict with halflife_mo, ci_lo, ci_hi, decay_detected, n_quarters, etc.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _top_k_variance(scores: np.ndarray, frac: float = 0.20) -> float:
    n = max(1, int(np.ceil(len(scores) * frac)))
    top = np.sort(scores)[-n:]
    return float(top.var())


def fit_decay(t_months: np.ndarray, variances: np.ndarray):
    """Fit exponential decay: v(t) = A * exp(-lambda * t) + C.
    Returns (halflife_months, params) or (None, None)."""
    if len(t_months) < 3 or np.all(variances == variances[0]):
        return None, None
    try:
        v = np.asarray(variances, dtype=float)
        t = np.asarray(t_months, dtype=float)
        p0 = [max(v[0] - v[-1], v[0] * 0.1), 0.05, min(v)]
        bounds = ([0, 1e-6, 0], [v.max() * 10 + 1e-6, 2.0, v.max() + 1e-6])

        def model(t, a, lam, c):
            return a * np.exp(-lam * t) + c

        popt, _ = curve_fit(model, t, v, p0=p0, bounds=bounds, maxfev=20000)
        lam = popt[1]
        if lam < 1e-5:
            return None, None
        halflife = float(np.log(2) / lam)
        return halflife, {"A": popt[0], "lambda": popt[1], "C": popt[2]}
    except Exception:
        return None, None


def bootstrap_halflife(
    df: pd.DataFrame,
    top_k: float = 0.20,
    n_boot: int = 1000,
    seed: int = 42,
    quarter_col: str = "quarter",
):
    """Bootstrap half-life over model population within each quarter."""
    rng = np.random.RandomState(seed)
    quarters = sorted(df[quarter_col].unique())
    t0 = quarters[0]

    taus = []
    for _ in range(n_boot):
        t_mo, vars_ = [], []
        for q in quarters:
            g = df[df[quarter_col] == q]
            if len(g) < 10:
                continue
            idx = rng.randint(0, len(g), size=len(g))
            boot_scores = g.iloc[idx]["score"].values
            v = _top_k_variance(boot_scores, top_k)
            t_mo.append((q - t0).n * 3.0)
            vars_.append(v)
        tau, _ = fit_decay(np.array(t_mo), np.array(vars_))
        if tau is not None and 0.1 < tau < 500:
            taus.append(tau)

    if not taus:
        return None, None, None
    taus = np.array(taus)
    return float(np.median(taus)), float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))


def compute_halflife(
    csv_path: str,
    top_k: float = 0.20,
    n_boot: int = 1000,
    date_col: str = "eval_date",
    score_col: str = "score",
    model_col: str = "model",
):
    """
    Main entry point. Reads a CSV, computes quarterly top-k variance,
    fits exponential decay, and bootstraps CIs.

    Returns dict with keys:
        halflife_mo, ci_lo, ci_hi, decay_detected, n_quarters,
        n_models, quarterly_variance, fit_params
    """
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[score_col])
    df["quarter"] = df[date_col].dt.to_period("Q")

    quarters = sorted(df["quarter"].unique())
    t_mo, vars_ = [], []
    for q in quarters:
        g = df[df["quarter"] == q]
        if len(g) < 10:
            continue
        v = _top_k_variance(g[score_col].values, top_k)
        t_mo.append((q - quarters[0]).n * 3.0)
        vars_.append(v)

    t_mo = np.array(t_mo)
    vars_ = np.array(vars_)

    tau_point, params = fit_decay(t_mo, vars_)
    tau_med, ci_lo, ci_hi = bootstrap_halflife(df, top_k, n_boot)

    return {
        "halflife_mo": tau_point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "decay_detected": tau_point is not None,
        "n_quarters": len(t_mo),
        "n_models": df[model_col].nunique() if model_col in df.columns else len(df),
        "quarterly_variance": list(zip([str(q) for q in quarters[:len(vars_)]], vars_.tolist())),
        "fit_params": params,
    }
