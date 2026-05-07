"""Figure 3 in main: decomposition $G/K/F$ + per-model orig-pert gap.

Vertical layout:
  (a) top, full width — G/K/F decomposition as horizontal stacked strip
  (b) bottom, full width — per-model orig-pert gap, vendor-grouped
"""
import json
import numpy as np
from pathlib import Path
from _figure_style import plt
from matplotlib import gridspec

ROOT = Path(__file__).parent.parent
FIG = ROOT / "figures"
DATA = ROOT / "data"

# Per-benchmark colors. GSM8K moved off red so it doesn't clash with K (red)
# in the (a) decomposition panel.
benches = [
    ("mmlu",  "MMLU",  "#f39c12"),  # orange
    ("arc",   "ARC",   "#27ae60"),  # green
    ("gsm8k", "GSM8K", "#8e44ad"),  # purple
]
data_per_bench = {}
for slug, _, _ in benches:
    fn = DATA / "evaluations" / f"{slug}_results.json"
    r = json.load(open(fn))
    data_per_bench[slug] = {}
    for m in r.keys():
        o = [x for x in r[m]["orig"] if x is not None]
        p = [x for x in r[m]["pert"] if x is not None]
        if not o or not p:
            continue
        oa = np.array(o, dtype=float)
        pa = np.array(p, dtype=float)
        data_per_bench[slug][m] = (oa.mean(), pa.mean(), oa.mean() - pa.mean())

# Together AI provides Llama-3.3-70B on GSM8K (MMLU/ARC not run); we add it
# under a synthetic key so it appears as a 3rd-vendor column in (b).
LLAMA_KEY = "llama-3.3-70b"
together = json.load(open(DATA / "evaluations" / "together_results.json"))
llama_gsm = together.get("Llama-3.3-70B-Instruct-Turbo_gsm8k", {})
if llama_gsm.get("orig") and llama_gsm.get("pert"):
    o = [x for x in llama_gsm["orig"] if x is not None]
    p = [x for x in llama_gsm["pert"] if x is not None]
    oa, pa = np.array(o, float), np.array(p, float)
    data_per_bench["gsm8k"][LLAMA_KEY] = (oa.mean(), pa.mean(), oa.mean() - pa.mean())

# Vendor-grouped order: GPT (chronological) → Claude (size-ordered) → Llama (3rd vendor).
# Llama is included even though it appears only on GSM8K — its column will show
# only the GSM8K bar (MMLU/ARC slots stay empty), conveying the cross-vendor message.
gsm_models = set(data_per_bench["gsm8k"].keys())
mmlu_arc_models = set(data_per_bench["mmlu"].keys()) & set(data_per_bench["arc"].keys())
vendor_order = [
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-08-06",
    "gpt-4.1-2025-04-14",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    LLAMA_KEY,
]
common = [m for m in vendor_order if m in (mmlu_arc_models | gsm_models)]


# ============================================================
fig = plt.figure(figsize=(9.0, 5.0))
gs = gridspec.GridSpec(2, 1, height_ratios=[0.7, 1.6], hspace=0.95)

# ============================================================
# (a) TOP — G/K/F decomposition, horizontal stacked strip
# ============================================================
axA = fig.add_subplot(gs[0, 0])

# Mechanism colors (distinct from benchmark colors above)
G_color = "#3498db"   # blue
K_color = "#c0392b"   # red
F_color = "#7f8c8d"   # gray

attrib = {
    "MMLU":  {"G": 0.55, "K": 0.30, "F": 0.15},
    "ARC":   {"G": 0.20, "K": 0.10, "F": 0.70},
    "GSM8K": {"G": 0.20, "K": 0.65, "F": 0.15},
}
ordered = ["MMLU", "ARC", "GSM8K"]
y = np.arange(len(ordered))[::-1]
G_vals = np.array([attrib[b]["G"] for b in ordered])
K_vals = np.array([attrib[b]["K"] for b in ordered])
F_vals = np.array([attrib[b]["F"] for b in ordered])

axA.barh(y, G_vals, color=G_color, edgecolor="white", linewidth=0.7, zorder=3)
axA.barh(y, K_vals, left=G_vals, color=K_color, edgecolor="white", linewidth=0.7, zorder=3)
axA.barh(y, F_vals, left=G_vals + K_vals, color=F_color, edgecolor="white", linewidth=0.7, zorder=3)

# In-bar labels — convey colour key directly without a legend box
for i, b in enumerate(ordered):
    g, k, f = attrib[b]["G"], attrib[b]["K"], attrib[b]["F"]
    if g >= 0.10:
        axA.text(g / 2, y[i], f"$G$ {int(round(g*100))}%",
                 color="white", fontsize=9.5, ha="center", va="center_baseline", fontweight="bold")
    if k >= 0.10:
        axA.text(g + k / 2, y[i], f"$K$ {int(round(k*100))}%",
                 color="white", fontsize=9.5, ha="center", va="center_baseline", fontweight="bold")
    if f >= 0.10:
        axA.text(g + k + f / 2, y[i], f"$F$ {int(round(f*100))}%",
                 color="white", fontsize=9.5, ha="center", va="center_baseline", fontweight="bold")

axA.set_yticks(y)
axA.set_yticklabels(ordered, fontsize=10)
axA.set_xlim(0, 1.0)
axA.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axA.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
axA.set_xlabel("Share of discriminative-decay attributed to each mechanism", fontsize=10)
axA.set_title(
    "(a) What explains each benchmark's decay? "
    "$G$ genuine convergence, $K$ surface-form memorization, $F$ score ceiling",
    loc="left", pad=10, fontsize=10.5)
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)
axA.tick_params(axis="y", length=0)
axA.grid(False)


# ============================================================
# (b) BOTTOM — per-model orig-pert gap, vendor-grouped
# ============================================================
axB = fig.add_subplot(gs[1, 0])
n_models = len(common)
bar_w = 0.25
x = np.arange(n_models)

bar_handles = {}
for i, (slug, label, color) in enumerate(benches):
    gaps = [data_per_bench[slug].get(m, (0, 0, 0))[2] * 100 for m in common]
    axB.bar(x + (i - 1) * bar_w, gaps, width=bar_w, color=color,
            edgecolor="white", linewidth=0.8, zorder=3)
    bar_handles[slug] = (label, color)

axB.axhline(0, color="#444", lw=0.8, zorder=2)
# Headroom for the in-figure color key above the bars
y_top = max(max(data_per_bench[s].get(m, (0, 0, 0))[2] * 100 for m in common)
            for s, _, _ in benches)
axB.set_ylim(top=y_top * 1.30)
axB.set_xticks(x)
def _short(m):
    if m == LLAMA_KEY:
        return "Llama-3.3 70B"
    return (m.replace("claude-", "").replace("-2024", "").replace("-2025", "")
             .replace("-2023", "").replace("gpt-", "GPT-")
             .replace("turbo-04-09", "turbo")
             .replace("o-08-06", "o").replace("1-04-14", "1")
             .replace("-0125", "").replace("-4-5", " 4.5"))
short_names = [_short(m) for m in common]
axB.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9)
axB.set_ylabel("Accuracy drop after perturbation\n(original $-$ perturbed, pp)", fontsize=10)
axB.set_title(
    "(b) Every model loses accuracy when items are perturbed; the drop is largest on GSM8K",
    loc="left", pad=10, fontsize=10.5)
axB.grid(True, axis="y", alpha=0.25, zorder=1)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)

# Vendor bracket annotations under x tick labels (replaces vendor legend).
# Group ranges are computed dynamically from `common` so adding/removing a
# vendor only needs the prefix matcher below.
def _bracket(i_lo, i_hi, label):
    axB.annotate("", xy=(x[i_lo] - bar_w * 1.5, -0.42),
                 xytext=(x[i_hi] + bar_w * 1.5, -0.42),
                 xycoords=("data", "axes fraction"),
                 textcoords=("data", "axes fraction"),
                 arrowprops=dict(arrowstyle="-", color="#666", lw=0.8))
    axB.text((x[i_lo] + x[i_hi]) / 2, -0.46, label, ha="center", va="top",
             fontsize=9, color="#444", transform=axB.get_xaxis_transform())

vendor_groups = [
    ("OpenAI GPT",      [i for i, m in enumerate(common) if m.startswith("gpt-")]),
    ("Anthropic Claude", [i for i, m in enumerate(common) if m.startswith("claude-")]),
    ("Together / Llama", [i for i, m in enumerate(common) if m == LLAMA_KEY]),
]
for label, idxs in vendor_groups:
    if idxs:
        _bracket(idxs[0], idxs[-1], label)

# Inline color key (replaces legend box) — placed in axes-fraction coords near
# the top-left of the bar plot, well clear of the title and the bars.
key_specs = [
    (0.012, "MMLU",  benches[0][2]),
    (0.085, "ARC",   benches[1][2]),
    (0.150, "GSM8K", benches[2][2]),
]
for fx, label, color in key_specs:
    axB.text(fx, 0.94, "■ " + label, transform=axB.transAxes,
             fontsize=9, ha="left", va="top", color=color, fontweight="bold")

plt.savefig(FIG / "fig_decomposition.pdf", bbox_inches="tight")
plt.savefig(FIG / "fig_decomposition.png", bbox_inches="tight", dpi=150)
print(f"Saved: {FIG / 'fig_decomposition.pdf'}")
