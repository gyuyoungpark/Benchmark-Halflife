"""GSM8K per-item contamination heatmap.

Each row is one of 7 frontier models, each column is one GSM8K item, sorted
left-to-right by mean (orig - pert) gap across the 7 models. The "12% most
contaminated" annotation is placed outside the axes margin so it does not
collide with title or data.
"""
import json
import numpy as np
from pathlib import Path
from _figure_style import plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "evaluations"
FIG = ROOT / "figures"

# Match Section 6 model order; keep short labels for the y-axis
MODEL_ORDER = [
    ("gpt-3.5-turbo-0125",      "gpt-3.5"),
    ("gpt-4-turbo-2024-04-09",  "gpt-4-turbo"),
    ("gpt-4o-2024-08-06",       "gpt-4o"),
    ("gpt-4.1-2025-04-14",      "gpt-4.1"),
    ("claude-haiku-4-5",        "haiku"),
    ("claude-sonnet-4-5",       "sonnet"),
    ("claude-opus-4-5",         "opus"),
]

r = json.load(open(DATA / "gsm8k_results.json"))

# Build (n_models, n_items) gap matrix
gaps = []
labels = []
for key, label in MODEL_ORDER:
    if key not in r:
        continue
    o = np.array(r[key]["orig"], dtype=float)
    p = np.array(r[key]["pert"], dtype=float)
    gaps.append(o - p)
    labels.append(label)
gap = np.vstack(gaps)

# Sort columns by mean gap across models, descending
mean_gap = gap.mean(axis=0)
order = np.argsort(-mean_gap)
gap = gap[:, order]
n_items = gap.shape[1]
contam_cut = max(1, int(round(0.12 * n_items)))  # 12% most contaminated boundary

fig, ax = plt.subplots(figsize=(13.0, 3.4))

# Diverging cmap centred at zero — red high (memorisation), blue negative
cmap = LinearSegmentedColormap.from_list(
    "div_rb", [(0.0, "#1f3a93"), (0.5, "#f5f5f5"), (1.0, "#922b21")])
im = ax.imshow(gap, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
               interpolation="nearest")

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("GSM8K items (sorted by mean orig$-$pert gap)", fontsize=10)
ax.set_title("Per-item contamination gap across 7 models",
             fontsize=12, fontweight="bold")

# Vertical guide marking the 12% most contaminated cut
ax.axvline(contam_cut - 0.5, color="black", linestyle="--", lw=1.0, alpha=0.7)

# 12% annotation OUTSIDE the axes (above the plot, near the cut x position)
# x in data coords, y in axes-fraction so we sit just above the top spine.
ax.annotate(f"12% most contaminated  ($n{{=}}{contam_cut}$)",
            xy=(contam_cut - 0.5, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, color="#222",
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.7))

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Gap (orig $-$ pert)", fontsize=9)

# Make sure annotation outside the axes is preserved by bbox_inches="tight"
plt.savefig(FIG / "fig_contamination_heatmap.pdf", bbox_inches="tight")
plt.savefig(FIG / "fig_contamination_heatmap.png", bbox_inches="tight", dpi=150)
print(f"Saved: {FIG / 'fig_contamination_heatmap.pdf'}")
