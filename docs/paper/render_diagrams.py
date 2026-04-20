"""
Publication-ready pipeline diagrams – matplotlib.

Design principles:
- Figure width = 6.5 in (standard paper text width). Fonts are 1:1 at natural
  insert size, so 9 pt in the file == 9 pt on the printed page.
- No per-step annotation text: the pipeline step names are self-explanatory at
  this level; detail lives in the methods text.  Keep the figure uncluttered.
- Visible gaps between every box; arrows clearly visible.
- 300 DPI PNG.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path(__file__).parent / "diagrams"
OUT_DIR.mkdir(exist_ok=True)

# ── global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})

BOX_LW      = 1.0
ARROW_LW    = 1.0
LABEL_PT    = 9.0      # box label – matches printed pt when fig is 6.5 in wide
DASH_LW     = 0.9
CAPTION_PT  = 9.5
NOTE_PT     = 8.0


def _box(ax, cx, cy, w, h, label):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="square,pad=0",
        facecolor="white", edgecolor="black", linewidth=BOX_LW, zorder=3,
    ))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=LABEL_PT, zorder=4, linespacing=1.35)


def _arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=ARROW_LW, mutation_scale=8),
                zorder=2)


def _dashed_rect(ax, x0, y0, x1, y1, label=""):
    ax.add_patch(plt.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        fill=False, edgecolor="black",
        linewidth=DASH_LW, linestyle=(0, (4, 3)), zorder=1,
    ))
    if label:
        ax.text(x0 + 0.010, y1 - 0.018, label,
                ha="left", va="top", fontsize=8.5, fontweight="bold", zorder=4)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1  –  Section 1.4.6.1  EEG Pre-processing
# ═════════════════════════════════════════════════════════════════════════════
# Layout: Raw EEG | [5 steps inside dashed border] | Clean EEG
# 7 boxes, uniform width, equal spacing.

def figure_1461(out_path: Path) -> None:
    FW, FH = 6.5, 2.4
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("auto"); ax.axis("off")

    Y     = 0.55
    BH    = 0.34
    BW    = 0.108
    GAP   = 0.032
    STEP  = BW + GAP

    # 7 box centres
    X = [0.054 + i * STEP for i in range(7)]
    xR, x1, x2, x3, x4, x5, xO = X

    # Dashed border around steps 1-5
    db_l = x1 - BW/2 - 0.014
    db_r = x5 + BW/2 + 0.014
    _dashed_rect(ax, db_l, Y - BH/2 - 0.12, db_r, Y + BH/2 + 0.12,
                 label="EEG Pre-Processing")

    # Boxes
    _box(ax, xR, Y, BW, BH, "Raw\nEEG")
    _box(ax, x1, Y, BW, BH, "Montage\nAssignment")
    _box(ax, x2, Y, BW, BH, "Bad Channel\nDetection")
    _box(ax, x3, Y, BW, BH, "Artefact\nRemoval")
    _box(ax, x4, Y, BW, BH, "Filter &\nRe-reference")
    _box(ax, x5, Y, BW, BH, "Epoching")
    _box(ax, xO, Y, BW, BH, "Clean EEG\n(.mat)")

    # Arrows
    for a, b in zip(X, X[1:]):
        _arrow(ax, a + BW/2 + 0.003, b - BW/2 - 0.003, Y)

    # Feedback arrow
    yf = Y - BH/2 - 0.08
    ax.plot([xR + BW/2, xR + BW/2], [Y - BH/2, yf], color="black", lw=ARROW_LW, zorder=2)
    ax.plot([x2 - BW/2, x2 - BW/2], [Y - BH/2, yf], color="black", lw=ARROW_LW, zorder=2)
    ax.annotate("", xy=(x2 - BW/2, yf), xytext=(xR + BW/2, yf),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=ARROW_LW, mutation_scale=7), zorder=2)
    ax.text((xR + BW/2 + x2 - BW/2)/2, yf - 0.015,
            "Original channel information",
            ha="center", va="top", fontsize=7.5, style="italic")

    # Caption
    ax.text(0.5, 0.03,
            "Figure 1.  EEG pre-processing pipeline (Section 1.4.6.1).",
            ha="center", va="bottom", fontsize=CAPTION_PT, style="italic")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2  –  Section 1.4.6.2  GEDAI Pipeline
# ═════════════════════════════════════════════════════════════════════════════
# Layout: Raw EEG | [6 steps inside dashed border] | Denoised EEG
# 8 boxes.  No per-step annotations — note line below summarises key points.

def figure_1462(out_path: Path) -> None:
    FW, FH = 6.5, 2.7
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("auto"); ax.axis("off")

    Y     = 0.58
    BH    = 0.30
    BW    = 0.094
    GAP   = 0.026
    STEP  = BW + GAP

    X = [0.047 + i * STEP for i in range(8)]
    xR, x1, x2, x3, x4, x5, x6, xO = X

    # Dashed border around steps 1-6
    db_l = x1 - BW/2 - 0.012
    db_r = x6 + BW/2 + 0.012
    _dashed_rect(ax, db_l, Y - BH/2 - 0.12, db_r, Y + BH/2 + 0.18,
                 label="GEDAI Denoising")

    # Boxes
    _box(ax, xR, Y, BW, BH, "Raw\nEEG")
    _box(ax, x1, Y, BW, BH, "Leadfield\nMatrix")
    _box(ax, x2, Y, BW, BH, "Reference\nCovariance")
    _box(ax, x3, Y, BW, BH, "Data\nCovariance")
    _box(ax, x4, Y, BW, BH, "GEV\nDecomposition")
    _box(ax, x5, Y, BW, BH, "SENSAI\nThreshold")
    _box(ax, x6, Y, BW, BH, "Project\nArtefacts")
    _box(ax, xO, Y, BW, BH, "Denoised\nEEG")

    # Arrows
    for a, b in zip(X, X[1:]):
        _arrow(ax, a + BW/2 + 0.002, b - BW/2 - 0.002, Y)

    # Single note below
    ax.text(0.5, 0.16,
            "Applied to broadband signal prior to filtering  \u2502  "
            "SENSAI selects threshold automatically  \u2502  "
            "Fallback: identity-based \u2192 data-driven reference",
            ha="center", va="top", fontsize=NOTE_PT, style="italic",
            color="#222222", zorder=4)

    # Caption
    ax.text(0.5, 0.03,
            "Figure 2.  GEDAI EEG pre-processing pipeline (Section 1.4.6.2).",
            ha="center", va="bottom", fontsize=CAPTION_PT, style="italic")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Rendering Figure 1 – Section 1.4.6.1 ...")
    figure_1461(OUT_DIR / "fig1_eeg_preprocessing.png")
    print("Rendering Figure 2 – Section 1.4.6.2 ...")
    figure_1462(OUT_DIR / "fig2_gedai_pipeline.png")
    print("Done.")
