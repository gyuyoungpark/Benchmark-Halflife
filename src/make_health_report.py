"""Generate Benchmark Health Report Card without 'Retire' wording.

Practitioner-facing card for 11 LLM benchmarks. Status labels and recommendations
follow the main-text Section 6 framing: 'pair with held-out items' rather than 'retire'.
The framing/disclaimer text lives in the LaTeX caption, not in the figure itself.
"""
from _figure_style import plt  # apply Helvetica + shared sizing
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).parent.parent / "figures"

# Per-benchmark data (half-life in months, orig-pert gap, fidelity r, status)
DATA = [
    # name,           tau_1/2,   gap,     fidelity, status,    recommendation
    ("WinoGrande",    1.0,       None,    None,     "degraded",
        "Pair with held-out items; not for frontier ranking"),
    ("HellaSwag",     2.5,       None,    None,     "degraded",
        "Pair with held-out items; not for frontier ranking"),
    ("ARC-Challenge", 2.7,       0.7,     0.89,     "degraded",
        "Ceiling-bound;\nuseful below frontier"),
    ("GSM8K",         4.6,       13.0,    0.27,     "degraded",
        "Surface-form memorization;\npair with verified holdouts"),
    ("TruthfulQA",    9.0,       3.1,     None,     "decaying",
        "Monitor; fidelity-unverified perturbation"),
    ("IFEval",        1.6,       None,    None,     "degraded",
        "Subsumed by RLHF/DPO\npost-training;\npair with held-outs"),
    ("MUSR",          5.4,       None,    None,     "decaying",
        "Monitor closely"),
    ("MMLU-PRO",      8.7,       2.8,     0.83,     "early-warning",
        "Healthy by leaderboard variance; subject-specific gap"),
    ("BBH",           6.4,       None,    None,     "transitioning",
        "Honeymoon-to-collapse transition observed"),
    ("MATH Lvl 5",    18.7,      None,    0.42,     "slow",
        "Slow decay; perturbation method unreliable"),
    ("GPQA",          None,      None,    None,     "noisy",
        "Insufficient signal in window"),
]

STATUS_INFO = {
    "degraded":      ("#922b21", "Top-stratum utility substantially degraded"),
    "decaying":      ("#d35400", "Decaying"),
    "transitioning": ("#f39c12", "Honeymoon-to-collapse transition"),
    "slow":          ("#f1c40f", "Slow decay or boundary case"),
    "early-warning": ("#16a085", "Healthy variance with early-warning gap"),
    "noisy":         ("#7f8c8d", "Insufficient signal"),
}

ORDER = ["degraded", "decaying", "transitioning", "slow", "early-warning", "noisy"]
groups = {s: [d for d in DATA if d[4] == s] for s in ORDER}

fig, ax = plt.subplots(figsize=(13, 11.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Title only — disclaimer/framing moved to LaTeX caption
ax.text(50, 97.5, "Benchmark Health Report Card",
        fontsize=20, fontweight="bold", ha="center", color="#2c3e50")

# Layout
x_left = 2
x_right = 98
y_top = 94
card_height = 11.0
gap_y = 1.2
label_band_height = 2.6
label_gap = 1.0

# Font sizes (bumped from previous version for readability)
FS_BAND = 11.5
FS_NAME = 12.5
FS_STATS = 10.0
FS_REC = 9.0

# Card area is slightly inset from the figure margins; status bands are
# aligned to the same horizontal extent as the cards below them.
card_area_left = x_left + 0.5
card_area_width = (x_right - x_left) - 1.0
card_area_right = card_area_left + card_area_width
card_spacing = 0.8

current_y = y_top
for status in ORDER:
    items = groups[status]
    if not items:
        continue
    color, label = STATUS_INFO[status]

    # Status label band — width matches the card area; same border weight
    # as the white cards below, with the edge colour tied to the band fill
    # so the band reads as a self-contained title block for its row.
    ax.add_patch(FancyBboxPatch(
        (card_area_left, current_y - label_band_height),
        card_area_width, label_band_height,
        boxstyle="round,pad=0.05,rounding_size=0.4",
        linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.9, zorder=2,
    ))
    ax.text(card_area_left + 1.2, current_y - label_band_height / 2, label,
            fontsize=FS_BAND, fontweight="bold", color="white",
            va="center_baseline", zorder=3)

    # Cards
    cards_y_top = current_y - label_band_height - label_gap
    n_cards = len(items)
    card_w = (card_area_width - (n_cards - 1) * card_spacing) / n_cards

    for i, (name, tau, gap, fid, _, rec) in enumerate(items):
        cx = card_area_left + i * (card_w + card_spacing)
        ax.add_patch(FancyBboxPatch(
            (cx, cards_y_top - card_height), card_w, card_height,
            boxstyle="round,pad=0.1,rounding_size=0.3",
            linewidth=1.2, edgecolor=color, facecolor="white", zorder=2,
        ))

        cx_centre = cx + card_w / 2
        stats = []
        if tau is not None:
            stats.append(f"$\\tau_{{1/2}}\\!=\\!{tau:g}$ mo")
        if gap is not None:
            stats.append(f"$+{gap:g}$pp")
        if fid is not None:
            stats.append(f"$r\\!=\\!{fid:g}$")
        # Split stats onto 2 lines when 3 are present (otherwise it overflows
        # narrow cards in the 5-per-row "degraded" band on benchmarks like
        # ARC-Challenge and GSM8K). Single/double-stat cards stay on 1 line.
        if not stats:
            stats_lines = ["no measurable signal"]
            stats_italic = True
        elif len(stats) >= 3:
            stats_lines = [stats[0], "  ".join(stats[1:])]
            stats_italic = False
        else:
            stats_lines = [" · ".join(stats)]
            stats_italic = False

        # Manual line breaks (\n) take precedence — used to control wrapping
        # for specific cards. Otherwise greedy-wrap to a per-card-width cap.
        import textwrap
        cap = max(int(card_w * 1.5), 10)
        if "\n" in rec:
            rec_lines = rec.split("\n")
        else:
            rec_lines = textwrap.wrap(rec, width=cap, break_long_words=False) or [rec]
        if len(rec_lines) > 3:
            rec_lines = rec_lines[:2] + [" ".join(rec_lines[2:])]

        # Vertical centring of the (name, stats..., rec...) stack within the
        # card. line_step is in axes units; with figsize=(13, 11.5) → 1 unit
        # ≈ 0.1in. Stats may now occupy 1 or 2 lines.
        n_text_rows = 1 + len(stats_lines) + len(rec_lines)
        line_step = 2.0
        block_height = (n_text_rows - 1) * line_step
        card_centre_y = cards_y_top - card_height / 2
        first_y = card_centre_y + block_height / 2

        ax.text(cx_centre, first_y, name,
                fontsize=FS_NAME, fontweight="bold", ha="center",
                va="center_baseline", color="#2c3e50", zorder=3)
        for s_idx, sline in enumerate(stats_lines):
            ax.text(cx_centre, first_y - (s_idx + 1) * line_step, sline,
                    fontsize=FS_STATS if not stats_italic else FS_STATS - 0.5,
                    ha="center", va="center_baseline",
                    color="#555" if not stats_italic else "#888",
                    style="normal" if not stats_italic else "italic", zorder=3)
        rec_start = 1 + len(stats_lines)
        for j, rline in enumerate(rec_lines):
            ax.text(cx_centre, first_y - (rec_start + j) * line_step, rline,
                    fontsize=FS_REC, ha="center", va="center_baseline",
                    color="#333", zorder=3)

    current_y -= (label_band_height + label_gap + card_height + gap_y)

plt.savefig(OUT / "fig_health_report.pdf", bbox_inches="tight", dpi=300)
plt.savefig(OUT / "fig_health_report.png", bbox_inches="tight", dpi=150)
print(f"Saved: {OUT / 'fig_health_report.pdf'}")
print(f"Saved: {OUT / 'fig_health_report.png'}")
