"""GLUE retrospective figure: SOTA over time with τ½ fit.

Each historical model gets a distinct marker so it can be identified via the
legend (no legend box). The ceiling and saturation reference lines are
labelled in-plot, just below the line itself, so they don't compete with
model entries in the legend.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from _figure_style import plt

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "leaderboard"
FIG = ROOT / "figures"


def exp_decay(t, D0, lam, D_inf):
    return D0 * np.exp(-lam * t) + D_inf


def main():
    df = pd.read_csv(DATA / "glue_historical.csv", parse_dates=["release_date"])
    df = df.sort_values("release_date").reset_index(drop=True)

    # Recompute τ½ via gap-to-ceiling fit on running max (matches archaeology.py)
    t0 = df["release_date"].iloc[0]
    months = (df["release_date"] - t0).dt.days / 30.44
    running_max = df["score"].cummax()
    gap = 100 - running_max.values
    fit_df = (pd.DataFrame({"m": months.values, "g": gap})
              .groupby("m", as_index=False)["g"].min())
    popt, _ = curve_fit(
        exp_decay, fit_df["m"].values, fit_df["g"].values,
        p0=[max(gap[0], 1.0), 0.05, max(gap[-1] * 0.5, 0.0)],
        bounds=([0, 1e-6, 0], [gap.max() * 5 + 1e-6, 1.0, gap.max() + 1e-6]),
        maxfev=10000,
    )
    halflife = float(np.log(2) / popt[1])

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    # Reference lines (no legend entries — labelled in-plot below each line)
    ax.axhline(100, color="#666", linestyle=":", alpha=0.7, zorder=1)
    ax.axhline(90, color="#c0392b", linestyle="--", alpha=0.55, zorder=1)

    # Per-model markers: 14 distinct markers + chronological tab20 palette so
    # individual models are visually identifiable in the legend.
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "P", "*", "h", "H", "X", "d"]
    cmap = plt.get_cmap("tab20")
    n = len(df)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for i, row in df.iterrows():
        ax.scatter(row["release_date"], row["score"],
                   marker=markers[i % len(markers)], s=85,
                   color=colors[i], edgecolor="black", linewidth=0.6,
                   label=row["model"], zorder=3)

    # In-plot labels for the reference lines, placed below each line with a
    # small pixel-level offset so they don't visually collide with the line.
    ax.annotate("Ceiling", xy=(0.985, 100),
                xycoords=("axes fraction", "data"),
                xytext=(0, -10), textcoords="offset points",
                ha="right", va="top", fontsize=9, color="#666", alpha=0.85)
    ax.annotate("Saturation", xy=(0.985, 90),
                xycoords=("axes fraction", "data"),
                xytext=(0, -10), textcoords="offset points",
                ha="right", va="top", fontsize=9, color="#c0392b", alpha=0.85)

    ax.set_xlabel("Release date", fontsize=11)
    ax.set_ylabel("GLUE score", fontsize=11)
    ax.set_title(f"GLUE retrospective  $\\tau_{{1/2}} = {halflife:.1f}$ mo",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(70, 102)
    ax.grid(True, alpha=0.25)

    # Legend with no box. Two columns to keep the panel compact; placed in
    # the lower-right where there is no data.
    ax.legend(fontsize=8, loc="lower right", ncol=2, frameon=False,
              handletextpad=0.3, columnspacing=0.8, borderpad=0.2)

    plt.tight_layout()
    plt.savefig(FIG / "fig_glue_retrospective.pdf", bbox_inches="tight")
    plt.savefig(FIG / "fig_glue_retrospective.png", bbox_inches="tight", dpi=150)
    print(f"Saved: {FIG / 'fig_glue_retrospective.pdf'}")


if __name__ == "__main__":
    main()
