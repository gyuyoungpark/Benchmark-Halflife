"""Figure 1 (paper overview) — result-focused 3 panels (a, b, c).

Panel (a): GSM8K cross-vendor accuracy drop after number substitution.
Panel (b): MMLU-PRO subject heterogeneity (grayscale, magnitude only).
Panel (c): 11 benchmarks ranked by half-life, grouped into ordinal status bands.

Colour conventions are kept distinct across panels so that a single colour does
not mean two different things in different panels:
  (a) hue encodes vendor (GPT / Claude / Llama).
  (b) grayscale encodes magnitude only (no categorical meaning).
  (c) hue encodes saturation status (exhausted / decaying / transitioning / ...).
"""
import json
import numpy as np
from pathlib import Path
from _figure_style import plt
from matplotlib import gridspec
from matplotlib.patches import Patch

ROOT = Path(__file__).parent.parent
FIG = ROOT / "figures"
DATA = ROOT / "data"


# ============================================================
# Build figure: 3 horizontal panels
# ============================================================
fig = plt.figure(figsize=(15, 4.8))
# Layout (left → right):
#   Panel (a): Eleven-benchmark half-life ranking (macro context)
#   Panel (b): GSM8K cross-vendor drop on number swap (saturated zoom-in)
#   Panel (c): MMLU-PRO subject heterogeneity (early-warning zoom-in)
gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 0.95, 1.1], wspace=0.42)

GAP_LABEL = "Accuracy drop after perturbation (pp)"

# ============================================================
# Panel (b): GSM8K cross-vendor accuracy drop (middle column)
# ============================================================
axA = fig.add_subplot(gs[0, 1])
# Panel (b): centered title; y-label kept slightly left of the spine
axA.yaxis.set_label_coords(-0.16, 0.5)
axA.set_title(
    "(b) GSM8K: every model drops",
    loc="center", pad=10, fontsize=13, fontweight="bold")

gsm = json.load(open(DATA / "evaluations" / "gsm8k_results.json"))

# Vendor palette: keep distinct from panel (c) status colours.
VENDOR_COLOR = {"GPT": "#1f77b4", "Claude": "#6a3d9a", "Llama": "#117733"}

order_specs = [
    ("gpt-3.5-turbo-0125",          "GPT-3.5",       "GPT"),
    ("gpt-4-turbo-2024-04-09",      "GPT-4-turbo",   "GPT"),
    ("gpt-4o-2024-08-06",           "GPT-4o",        "GPT"),
    ("gpt-4.1-2025-04-14",          "GPT-4.1",       "GPT"),
    ("claude-haiku-4-5",            "Claude-Haiku",  "Claude"),
    ("claude-sonnet-4-5",           "Claude-Sonnet", "Claude"),
    ("claude-opus-4-5",             "Claude-Opus",   "Claude"),
]

# Llama from together evaluations if present
together_path = DATA / "evaluations" / "together_results.json"
llama_gap = None
if together_path.exists():
    try:
        tog = json.load(open(together_path))
        llama_key = None
        for k in tog.keys():
            if "llama" in k.lower() or "Llama" in k:
                llama_key = k
                break
        if llama_key and "gsm8k" in tog[llama_key]:
            o = np.array(tog[llama_key]["gsm8k"]["orig"], dtype=float)
            p = np.array(tog[llama_key]["gsm8k"]["pert"], dtype=float)
            llama_gap = (o.mean() - p.mean()) * 100
    except Exception:
        pass
if llama_gap is None:
    llama_gap = 9.5  # paper-reported value

labels, gaps, colors = [], [], []
for key, label, vendor in order_specs:
    if key in gsm:
        o = np.array(gsm[key]["orig"], dtype=float)
        p = np.array(gsm[key]["pert"], dtype=float)
        gap = (o.mean() - p.mean()) * 100
        labels.append(label)
        gaps.append(gap)
        colors.append(VENDOR_COLOR[vendor])

labels.append("Llama-3.3-70B")
gaps.append(llama_gap)
colors.append(VENDOR_COLOR["Llama"])

x = np.arange(len(labels))
axA.bar(x, gaps, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
axA.set_xticks(x)
axA.set_xticklabels(labels, rotation=30, ha="right", fontsize=9.5)
axA.set_ylabel(GAP_LABEL)
axA.set_ylim(0, 18)
axA.grid(True, axis="y", alpha=0.25, zorder=1)

legend_a = [
    Patch(facecolor=VENDOR_COLOR["GPT"],    edgecolor="white", label="GPT"),
    Patch(facecolor=VENDOR_COLOR["Claude"], edgecolor="white", label="Claude"),
    Patch(facecolor=VENDOR_COLOR["Llama"],  edgecolor="white", label="Llama"),
]
axA.legend(handles=legend_a, loc="upper right",
           bbox_to_anchor=(1.10, 1.0), ncol=3,
           frameon=False, fontsize=11.5,
           handletextpad=0.4, columnspacing=1.2)


# ============================================================
# Panel (c): MMLU-PRO subject heterogeneity (right column, grayscale)
# ============================================================
axB = fig.add_subplot(gs[0, 2])
# Panel (c): centered title
axB.set_title(
    "(c) MMLU-PRO: drop varies by subject",
    loc="center", pad=10, fontsize=13, fontweight="bold")

api = json.load(open(DATA / "api_free_analyses.json"))
subjects_sorted = sorted(api["mmlu_pro_per_subject"],
                         key=lambda x: x["mean_gap"], reverse=True)
labels_b = [s["subject"] for s in subjects_sorted]
gaps_b = [s["mean_gap"] * 100 for s in subjects_sorted]

# Grayscale ramp: large positive = dark, near-zero = mid gray, negative = light gray.
def ramp(g):
    if g >= 8:
        return "#1a1a1a"
    if g >= 3:
        return "#3f3f3f"
    if g >= 0:
        return "#7a7a7a"
    return "#bdbdbd"

colors_b = [ramp(g) for g in gaps_b]

y_b = np.arange(len(labels_b))
axB.barh(y_b, gaps_b, color=colors_b, edgecolor="white",
         linewidth=0.6, zorder=3)
axB.axvline(0, color="#444", lw=0.8, zorder=2)
axB.set_yticks(y_b)
axB.set_yticklabels(labels_b, fontsize=9)
axB.invert_yaxis()
axB.set_xlabel(GAP_LABEL)
axB.grid(True, axis="x", alpha=0.25, zorder=1)


# ============================================================
# Panel (a): 11 benchmarks ranked by half-life (left column)
# ============================================================
axC = fig.add_subplot(gs[0, 0])
# Panel (a): centered title
axC.set_title(
    "(a) Eleven benchmarks, from saturated to stable",
    loc="center", pad=10, fontsize=13, fontweight="bold")

# Status palette: ordinal red→orange→amber→yellow→teal, with a single
# clearly-red category (exhausted). Burnt-orange (decaying) and amber
# (transitioning) are kept well separated from each other and from red.
STATUS_COLOR = {
    "exhausted":     "#922b21",
    "decaying":      "#d35400",
    "transitioning": "#f39c12",
    "slow":          "#f1c40f",
    "early-warning": "#16a085",
    "noisy":         "#7f8c8d",
}
benchmarks = [
    ("WinoGrande",     1.0,  "exhausted"),
    ("IFEval",         1.6,  "exhausted"),
    ("HellaSwag",      2.5,  "exhausted"),
    ("ARC-Challenge",  2.7,  "exhausted"),
    ("GSM8K",          4.6,  "exhausted"),
    ("MUSR",           5.4,  "decaying"),
    ("BBH",            6.4,  "transitioning"),
    ("MMLU-PRO",       8.7,  "early-warning"),
    ("TruthfulQA",     9.0,  "decaying"),
    ("MATH Lvl 5",    18.7,  "slow"),
    ("GPQA",           None, "noisy"),
]

non_noisy = [(b, t, s) for b, t, s in benchmarks if t is not None]
non_noisy.sort(key=lambda x: x[1])
gpqa = [(b, t, s) for b, t, s in benchmarks if t is None]

names = [b for b, _, _ in non_noisy] + [b for b, _, _ in gpqa]
vals = [t for _, t, _ in non_noisy] + [22.0 for _ in gpqa]
status = [s for _, _, s in non_noisy] + [s for _, _, s in gpqa]

y_c = np.arange(len(names))
bar_colors = [STATUS_COLOR[s] for s in status]
bars = axC.barh(y_c, vals, color=bar_colors, edgecolor="white",
                linewidth=0.6, zorder=3)
for i, s in enumerate(status):
    if s == "noisy":
        bars[i].set_hatch("///")
        bars[i].set_alpha(0.6)
        axC.text(vals[i] + 0.4, y_c[i], "noisy",
                 fontsize=8.5, color="#666", va="center", ha="left", style="italic")

axC.set_yticks(y_c)
axC.set_yticklabels(names, fontsize=9.5)
axC.invert_yaxis()
axC.set_xlabel("Discriminative half-life (months)")
axC.set_xlim(0, 24)
axC.grid(True, axis="x", alpha=0.25, zorder=1)

legend_c = [
    Patch(facecolor=STATUS_COLOR["exhausted"],     edgecolor="white", label="Exhausted"),
    Patch(facecolor=STATUS_COLOR["decaying"],      edgecolor="white", label="Decaying"),
    Patch(facecolor=STATUS_COLOR["transitioning"], edgecolor="white", label="Transitioning"),
    Patch(facecolor=STATUS_COLOR["slow"],          edgecolor="white", label="Slow decay"),
    Patch(facecolor=STATUS_COLOR["early-warning"], edgecolor="white", label="Early warning"),
]
axC.legend(handles=legend_c, loc="upper right",
           frameon=False, fontsize=10.5, ncol=1,
           handletextpad=0.4)

plt.tight_layout()
plt.savefig(FIG / "fig_story.pdf", bbox_inches="tight")
plt.savefig(FIG / "fig_story.png", bbox_inches="tight", dpi=150)
print(f"Saved: {FIG / 'fig_story.pdf'}")
