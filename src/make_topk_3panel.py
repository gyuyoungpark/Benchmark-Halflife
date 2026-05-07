"""3-panel top-k sensitivity figure for §2 Framework.

Panel A: v1 saturated benchmarks — top-k sensitivity, recommended stable range
Panel B: v2 mixed-stage benchmarks — multi-point lines + single-point markers
Panel C: Stratification effect — GSM8K full-population vs top-20% variance over time
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

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


# Build figure
fig = plt.figure(figsize=(15, 5.2))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.38)

# ============================================================
# Panel A: v1 saturated benchmarks
# ============================================================
axA = fig.add_subplot(gs[0, 0])
axA.axvspan(10, 30, color="#888", alpha=0.10, zorder=0)
axA.text(20, 0.55, "$k{=}20\\%$ recommended",
         fontsize=8.5, color="#555", ha="center", va="bottom", style="italic", zorder=1)

v1_specs = [
    ("arc_challenge_v1", "ARC-Challenge", "#1f77b4", "o"),
    ("hellaswag_v1",     "HellaSwag",     "#2ca02c", "s"),
    ("winogrande_v1",    "WinoGrande",    "#17becf", "D"),
    ("truthfulqa_v1",    "TruthfulQA",    "#9467bd", "^"),
    ("gsm8k_v1",         "GSM8K",         "#d62728", "v"),
]
for key, label, color, marker in v1_specs:
    ks, hls, lo, hi = fetch(key)
    if len(ks) >= 2:
        axA.plot(ks, hls, marker=marker, linestyle="-", label=label,
                 color=color, lw=1.8, markersize=6,
                 markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        axA.fill_between(ks, lo, hi, color=color, alpha=0.10, zorder=2)

axA.set_xlabel("Top-$k$ stratum (%)", fontsize=11)
axA.set_ylabel("Discriminative half-life (months)", fontsize=11)
axA.set_title("(A) v1 saturated: stable range, stratum-sensitivity",
              fontsize=10.5, fontweight="bold", pad=8)
axA.set_xticks([5, 10, 15, 20, 25, 30, 40, 50])
axA.set_xlim(3, 53)
axA.set_yscale("log")
axA.set_ylim(0.5, 50)
axA.grid(True, alpha=0.25, which="both")
axA.legend(fontsize=8.5, loc="upper right", framealpha=0.95,
           edgecolor="#888", fancybox=True)

# ============================================================
# Panel B: v2 mixed-stage benchmarks
# ============================================================
axB = fig.add_subplot(gs[0, 1])
axB.axvspan(10, 30, color="#888", alpha=0.10, zorder=0)

v2_lines = [
    ("bbh_v2",    "BBH (collapsing)",    "#ff7f0e", "--", "o"),
    ("ifeval_v2", "IFEval (decaying)", "#bcbd22", "--", "s"),
]
v2_markers = [
    ("mmlu_pro_v2",  "MMLU-PRO (single $k$)",  "#8c564b", "P"),
    ("math_lvl5_v2", "MATH Lvl 5 (single $k$)", "#e377c2", "X"),
    ("gpqa_v2",      "GPQA (single $k$)",      "#7f7f7f", "*"),
]

for key, label, color, ls, marker in v2_lines:
    ks, hls, lo, hi = fetch(key)
    if len(ks) >= 2:
        axB.plot(ks, hls, marker=marker, linestyle=ls, label=label,
                 color=color, lw=1.8, markersize=6,
                 markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        axB.fill_between(ks, lo, hi, color=color, alpha=0.10, zorder=2)

for key, label, color, marker in v2_markers:
    ks, hls, _, _ = fetch(key)
    if ks:
        axB.scatter([ks[0]], [hls[0]], marker=marker, s=120,
                    color=color, edgecolor="white", linewidth=1.2,
                    label=label, zorder=4)

axB.text(0.02, 0.97,
         "Single-point markers: only $k{=}10\\%$ converges within\nthe 4-quarter v2 window — not yet decaying at larger $k$.",
         transform=axB.transAxes, fontsize=8, color="#555",
         va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaa", lw=0.5, alpha=0.9))

axB.set_xlabel("Top-$k$ stratum (%)", fontsize=11)
axB.set_ylabel("Discriminative half-life (months)", fontsize=11)
axB.set_title("(B) v2 mixed-stage: collapsing / decaying / honeymoon",
              fontsize=10.5, fontweight="bold", pad=8)
axB.set_xticks([5, 10, 15, 20, 25, 30, 40, 50])
axB.set_xlim(3, 53)
axB.set_yscale("log")
axB.set_ylim(2, 250)
axB.grid(True, alpha=0.25, which="both")
axB.legend(fontsize=8.5, loc="lower right", framealpha=0.95,
           edgecolor="#888", fancybox=True)

# ============================================================
# Panel C: Stratification effect on GSM8K
# ============================================================
axC = fig.add_subplot(gs[0, 2])

df = pd.read_csv(DATA / "leaderboard" / "v1_scores_full.csv", parse_dates=["eval_date"])
df = df.dropna(subset=["gsm8k", "eval_date"])
df["quarter"] = df["eval_date"].dt.to_period("Q")

quarters, full_v, top20_v, full_n = [], [], [], []
for q, g in df.groupby("quarter"):
    if len(g) < 5:
        continue
    s = g["gsm8k"].values
    quarters.append(str(q))
    full_v.append(s.var(ddof=1))
    full_n.append(len(s))
    thresh = np.percentile(s, 80)
    top = s[s >= thresh]
    top20_v.append(top.var(ddof=1) if len(top) >= 3 else np.nan)

# Months from start
months = [0, 3.0, 6.0, 9.0][:len(quarters)]

axC.plot(months, full_v, "o-", color="#1f77b4", lw=2.0, markersize=8,
         label="Full population (3{,}906 models)", markeredgecolor="white",
         markeredgewidth=0.6, zorder=3)
axC.plot(months, top20_v, "s-", color="#d62728", lw=2.0, markersize=8,
         label="Top-20% (frontier $\\approx 781$ models)",
         markeredgecolor="white", markeredgewidth=0.6, zorder=3)

# Annotate trajectories
for i, q in enumerate(quarters):
    axC.annotate(q.replace("Q", " Q"), (months[i], full_v[i]),
                 xytext=(0, 10), textcoords="offset points",
                 fontsize=7.5, color="#1f77b4", ha="center")

# Highlight divergence
axC.annotate("", xy=(8.5, full_v[-1]), xytext=(8.5, top20_v[-1] if not np.isnan(top20_v[-1]) else top20_v[-2]),
             arrowprops=dict(arrowstyle="<->", color="#444", lw=1.0))
axC.text(8.7, np.sqrt(full_v[-1] * (top20_v[-1] if not np.isnan(top20_v[-1]) else top20_v[-2])),
         f"Stratification\nrequired", fontsize=8.5, color="#444",
         va="center", ha="left", style="italic")

axC.set_xlabel("Months from 2023Q3", fontsize=11)
axC.set_ylabel("Score variance", fontsize=11)
axC.set_title("(C) GSM8K: stratification reveals hidden decay",
              fontsize=10.5, fontweight="bold", pad=8)
axC.set_yscale("log")
axC.grid(True, alpha=0.25, which="both")
axC.legend(fontsize=9, loc="lower left", framealpha=0.95,
           edgecolor="#888", fancybox=True)
axC.set_xlim(-1, 12)

plt.tight_layout()
plt.savefig(FIG / "fig_topk_sensitivity.pdf", bbox_inches="tight")
plt.savefig(FIG / "fig_topk_sensitivity.png", bbox_inches="tight", dpi=150)
print(f"Saved: {FIG / 'fig_topk_sensitivity.pdf'}")
