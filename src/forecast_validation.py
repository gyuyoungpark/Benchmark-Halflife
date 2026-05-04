"""
Wow1: Prospective half-life forecast validation.

Paper forecasts (from meta-regression fit on n=6 v1 benchmarks):
  MMLU-PRO 11.6 mo, GPQA 10.0 mo, MATH Lvl 5 9.0 mo, IFEval 6.5 mo, MUSR 6.4 mo

We now check: across 2024Q2 -> 2025Q1 (9 months of v2 leaderboard data),
does top-20% variance actually decay at approximately the forecast rate?
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
LB_DIR = DATA_DIR / "leaderboard"
FIG_DIR = Path(__file__).parent.parent / "figures"
OUT_DIR = DATA_DIR / "forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORECASTS = {
    "mmlu_pro_v2":   {"name": "MMLU-PRO", "forecast_mo": 11.6},
    "gpqa_v2":       {"name": "GPQA",     "forecast_mo": 10.0},
    "math_lvl5_v2":  {"name": "MATH Lvl5","forecast_mo": 9.0},
    "ifeval_v2":     {"name": "IFEval",   "forecast_mo": 6.5},
    "musr_v2":       {"name": "MUSR",     "forecast_mo": 6.4},
    "bbh_v2":        {"name": "BBH",      "forecast_mo": 19.6},  # baseline recheck
}


def top_k_variance(df, frac=0.2):
    n = max(1, int(np.ceil(len(df) * frac)))
    return df.nlargest(n, "score")["score"].var()


def fit_exp_halflife(t_months, var):
    """Exponential fit: var(t) = A * exp(-lam * t) + C."""
    t = np.asarray(t_months, dtype=float)
    v = np.asarray(var, dtype=float)
    if len(t) < 3 or np.all(v == v[0]):
        return None
    try:
        p0 = [max(v[0] - v[-1], v[0] * 0.1), 0.05, min(v)]
        bounds = ([0, 1e-5, 0], [v.max() * 10 + 1e-6, 1.0, v.max() + 1e-6])
        popt, _ = curve_fit(lambda t, a, l, c: a * np.exp(-l * t) + c, t, v,
                            p0=p0, bounds=bounds, maxfev=20000)
        lam = popt[1]
        if lam < 1e-4:
            return None
        return float(np.log(2) / lam)
    except Exception:
        return None


def log_linear_halflife(t_months, var):
    """Fallback: fit log(var - floor) ~ t, report implied half-life."""
    t = np.asarray(t_months, dtype=float)
    v = np.asarray(var, dtype=float)
    floor = max(0.0, min(v) * 0.9)
    y = np.log(np.maximum(v - floor, 1e-9))
    if np.std(t) < 1e-6:
        return None
    slope, _ = np.polyfit(t, y, 1)
    if slope >= 0:
        return None  # not decaying
    return float(np.log(2) / (-slope))


def main():
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (key, meta) in enumerate(FORECASTS.items()):
        df = pd.read_csv(LB_DIR / f"{key}.csv")
        df["eval_date"] = pd.to_datetime(df["eval_date"])
        df = df.dropna(subset=["score"])
        df["quarter"] = df["eval_date"].dt.to_period("Q")

        quarters = []
        variances = []
        for q, g in df.groupby("quarter"):
            if len(g) < 20:
                continue
            v = top_k_variance(g, frac=0.2)
            quarters.append(q.to_timestamp().to_pydatetime())
            variances.append(float(v))

        if len(quarters) < 3:
            print(f"{key}: <3 quarters, skip")
            continue

        t_mo = [(q - quarters[0]).days / 30.44 for q in quarters]
        # Try exponential fit; fall back to log-linear if unstable.
        tau = fit_exp_halflife(t_mo, variances)
        if tau is None:
            tau = log_linear_halflife(t_mo, variances)
        decaying = (variances[-1] < variances[0])
        net_change = (variances[-1] - variances[0]) / variances[0] * 100

        results[key] = {
            "name": meta["name"],
            "forecast_mo": meta["forecast_mo"],
            "observed_halflife_mo": tau,
            "decaying": bool(decaying),
            "net_var_change_pct": float(net_change),
            "quarters": [q.strftime("%Y-%m") for q in quarters],
            "top20_variance": variances,
        }

        pred = meta["forecast_mo"]
        obs_str = f"{tau:.1f}" if tau is not None else "no-fit"
        ratio_str = f"{tau/pred:.2f}x" if tau is not None else "n/a"
        print(f"{meta['name']:<10} forecast={pred:5.1f}mo  observed={obs_str:>8}mo  ratio={ratio_str:>6}  net_var={net_change:+.0f}%  decaying={decaying}")

        ax = axes[i]
        ax.plot(t_mo, variances, "o-", color="#1976D2", linewidth=2, markersize=8, zorder=3)
        if tau is not None and decaying:
            t_smooth = np.linspace(0, max(t_mo), 50)
            v_smooth = variances[0] * np.exp(-np.log(2) / tau * t_smooth)
            ax.plot(t_smooth, v_smooth, "--", color="red", alpha=0.6,
                    label=f"obs τ½={tau:.1f} mo")
        t_smooth = np.linspace(0, max(t_mo), 50)
        v_fcast = variances[0] * np.exp(-np.log(2) / pred * t_smooth)
        ax.plot(t_smooth, v_fcast, ":", color="black", alpha=0.6,
                label=f"forecast τ½={pred:.1f} mo")
        ax.set_title(f"{meta['name']}", fontweight="bold")
        ax.set_xlabel("Months from Q1 2024")
        ax.set_ylabel("Top-20% variance")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    for j in range(len(FORECASTS), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Prospective validation: forecast vs observed top-20% variance decay (2024Q2–2025Q1)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_forecast_validation.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig_forecast_validation.png", bbox_inches="tight", dpi=150)
    print(f"\nSaved: {FIG_DIR / 'fig_forecast_validation.pdf'}")

    with open(OUT_DIR / "forecast_validation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {OUT_DIR / 'forecast_validation.json'}")


if __name__ == "__main__":
    main()
