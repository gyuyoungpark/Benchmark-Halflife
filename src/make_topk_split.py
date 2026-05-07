"""Split top-k figure into two narrower figures.

Figure 2: top-k sensitivity (v1 + v2, single panel, narrow width)
Figure 3 (new): Stratification effect (GSM8K, single panel, narrow width)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from _figure_style import plt  # apply Helvetica + shared sizing

ROOT = Path(__file__).parent.parent
FIG = ROOT / "figures"
DATA = ROOT / "data"

all_results = json.load(open(DATA / "topk_sensitivity" / "topk_sweep.json"))
k_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def fetch(key):
    if key not in all_results:
        return [], [], [], []
    ks, hls, lo, hi = [], [], [], []
    for k in k_values:
        v = all_results[key].get(str(k))
        if v is not None and v.get("point") is not None:
            ks.append(int(k * 100))
            hls.append(v["point"])
            lo.append(v.get("ci_lo") or v["point"])
            hi.append(v.get("ci_hi") or v["point"])
    return ks, hls, lo, hi


# ============================================================
# Figure 2: Top-k sensitivity (combined v1 + v2)
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 5.0))
ax.axvspan(10, 30, color="#888", alpha=0.10, zorder=0)
ax.text(20, 0.55, "Stable band $[10\\%,\\,30\\%]$",
        fontsize=8, color="#555", ha="center", va="bottom",
        style="italic", zorder=1)

# v1 (5 saturated benchmarks): all distinct colours retained; GSM8K and
# TruthfulQA are drawn thicker on top to surface the top-stratum K signal
# (monotonic rise in tau as k increases) without muting the others.
v1_specs = [
    # key,                label,             color,     ls,  marker, highlight
    ("arc_challenge_v1", "ARC-C (v1)",      "#1f77b4", "-",  "o", False),
    ("hellaswag_v1",     "HellaSwag (v1)",  "#2ca02c", "-",  "s", False),
    ("winogrande_v1",    "WinoGrande (v1)", "#34495e", "-",  "D", False),
    ("truthfulqa_v1",    "TruthfulQA (v1)", "#9467bd", "-",  "^", True),
    ("gsm8k_v1",         "GSM8K (v1)",      "#d62728", "-",  "v", True),
]
# v2 multi-point: dashed lines
v2_lines = [
    ("bbh_v2",    "BBH (v2)",    "#ff7f0e", "--", "o", False),
    ("ifeval_v2", "IFEval (v2)", "#f1c40f", "--", "s", False),
]
# v2 single-point: scatter markers
v2_markers = [
    ("mmlu_pro_v2",  "MMLU-PRO (v2)",  "#8c564b", "P"),
    ("math_lvl5_v2", "MATH-5 (v2)",    "#e377c2", "X"),
    ("gpqa_v2",      "GPQA (v2)",      "#7f7f7f", "*"),
]

for key, label, color, ls, marker, hi in v1_specs + v2_lines:
    ks, hls, lo, hi_ci = fetch(key)
    if len(ks) >= 2:
        lw_ = 2.6 if hi else 1.4
        ms_ = 7 if hi else 5
        zorder_ = 4 if hi else 3
        ax.plot(ks, hls, marker=marker, linestyle=ls, label=label,
                color=color, lw=lw_, markersize=ms_,
                markeredgecolor="white", markeredgewidth=0.5, zorder=zorder_)
        ax.fill_between(ks, lo, hi_ci, color=color,
                        alpha=0.18 if hi else 0.08, zorder=zorder_ - 1)

for key, label, color, marker in v2_markers:
    ks, hls, _, _ = fetch(key)
    if ks:
        ax.scatter([ks[0]], [hls[0]], marker=marker, s=110,
                   color=color, edgecolor="white", linewidth=1.0,
                   label=label + r" ($k{=}10$%)", zorder=4)

ax.set_xlabel("Top-$k$ stratum (%)", fontsize=11)
ax.set_ylabel("Discriminative half-life (months)", fontsize=11)
ax.set_xticks([5, 10, 15, 20, 25, 30, 40, 50])
ax.set_xlim(3, 53)
ax.set_yscale("log")
ax.set_ylim(0.5, 250)
ax.grid(True, alpha=0.25, which="both")
ax.legend(fontsize=7.5, loc="upper right", ncol=2,
          frameon=False,
          columnspacing=0.6, handletextpad=0.4)

plt.tight_layout()
plt.savefig(FIG / "fig_topk_sensitivity.pdf", bbox_inches="tight")
plt.savefig(FIG / "fig_topk_sensitivity.png", bbox_inches="tight", dpi=150)
print(f"Saved: {FIG / 'fig_topk_sensitivity.pdf'}")
plt.close()


# ============================================================
# Figure 3 (NEW): Stratification effect on GSM8K
# ============================================================
df = pd.read_csv(DATA / "leaderboard" / "v1_scores_full.csv", parse_dates=["eval_date"])
df = df.dropna(subset=["gsm8k", "eval_date"])
df["quarter"] = df["eval_date"].dt.to_period("Q")

quarters, full_v, top20_v = [], [], []
for q, g in df.groupby("quarter"):
    if len(g) < 5:
        continue
    s = g["gsm8k"].values
    quarters.append(str(q))
    full_v.append(s.var(ddof=1))
    thresh = np.percentile(s, 80)
    top = s[s >= thresh]
    top20_v.append(top.var(ddof=1) if len(top) >= 3 else np.nan)

months = list(range(0, 3 * len(quarters), 3))

fig, ax = plt.subplots(figsize=(5.5, 4.5))
FULL_COLOR = "#1f77b4"
TOP_COLOR = "#d62728"
line_full, = ax.plot(months, full_v, "o-", color=FULL_COLOR, lw=2.0, markersize=8,
                     markeredgecolor="white", markeredgewidth=0.6, zorder=3)
line_top, = ax.plot(months, top20_v, "s-", color=TOP_COLOR, lw=2.0, markersize=8,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)

# Quarter labels as x-axis ticks (one tick per data point, label below the symbol)
ax.set_xticks(months)
ax.set_xticklabels(quarters, fontsize=9)

# Inline series labels (replaces legend box) — colored text near each line
ax.text(months[-1] + 0.6, full_v[-1], "Full population",
        color=FULL_COLOR, fontsize=9, fontweight="bold",
        ha="left", va="center")
ax.text(months[-1] + 0.6,
        top20_v[-1] if not np.isnan(top20_v[-1]) else top20_v[-2],
        "Top-20% (frontier)",
        color=TOP_COLOR, fontsize=9, fontweight="bold",
        ha="left", va="center")

ax.set_xlabel("Quarter", fontsize=11)
ax.set_ylabel("GSM8K score variance", fontsize=11)
ax.set_yscale("log")
ax.grid(True, alpha=0.25, which="both")
ax.set_xlim(-1, max(months) + 7)

plt.tight_layout()
plt.savefig(FIG / "fig_stratification_effect.pdf", bbox_inches="tight")
plt.savefig(FIG / "fig_stratification_effect.png", bbox_inches="tight", dpi=150)
print(f"Saved: {FIG / 'fig_stratification_effect.pdf'}")
