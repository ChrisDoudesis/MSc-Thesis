"""
Figure: Random Forest vs. XGBoost -- ensemble mechanisms (Section 2.3).

Matplotlib re-creation of the TikZ figure: two panels comparing the
Random Forest (parallel trees + majority vote) and XGBoost (sequential
trees + shrinking residual error) ensemble mechanisms.

Run:  python rf_vs_xgb_concept_figure.py
Output: ../results/rf_vs_xgb_concept.png (and .pdf)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrow
import matplotlib.colors as mcolors
import os

# ============================================================
# >>> ALL COLORS ARE DEFINED HERE -- edit only this block <<<
# ============================================================
RF_MAIN   = "#3D7A6B"   # Random Forest: dark muted teal-green
RF_LIGHT  = "#EBF4F1"   # Random Forest: light fill
XGB_MAIN  = "#2B5D8A"   # XGBoost: dark muted blue
XGB_LIGHT = "#EAF1F8"   # XGBoost: light fill
BAR_EMPTY = "#DDE4EA"   # vote bar, unfilled part
TXT_GRAY  = "#444444"   # secondary text


def lighten(color, factor):
    """Blend `color` toward white; factor=1 -> original, 0 -> white.
    Mimics TikZ's `rfMain!35` syntax (35 = factor 0.35)."""
    rgb = mcolors.to_rgb(color)
    return tuple(1 - factor * (1 - c) for c in rgb)


def draw_tree(ax, x, y, color, label):
    """Small stylized evergreen-tree glyph with a label underneath,
    equivalent to the TikZ \\evtree macro."""
    # lower triangle
    ax.add_patch(Polygon([(x - 0.28, y), (x, y + 0.42), (x + 0.28, y)],
                         closed=True, facecolor=color, edgecolor="none"))
    # upper triangle
    ax.add_patch(Polygon([(x - 0.22, y + 0.22), (x, y + 0.58),
                          (x + 0.22, y + 0.22)],
                         closed=True, facecolor=color, edgecolor="none"))
    # trunk
    ax.add_patch(Rectangle((x - 0.05, y - 0.14), 0.10, 0.14,
                           facecolor=color, edgecolor="none"))
    ax.text(x, y - 0.22, label, ha="center", va="top",
            color=color, fontsize=6.5)


def draw_panel(ax, x0, y0, w, h, color):
    """Rounded rectangle panel with a light tinted fill."""
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0,rounding_size=0.15",
        facecolor=lighten(color, 0.04), edgecolor=color, linewidth=1.4))


def main():
    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    ax.set_xlim(-0.2, 15.6)
    ax.set_ylim(-0.2, 6.6)
    ax.set_aspect("equal")
    ax.axis("off")

    rf_faded = lighten(RF_MAIN, 0.35)   # "no EV" trees (rfMain!35)

    # ============================================================
    # LEFT PANEL: Random Forest
    # ============================================================
    draw_panel(ax, 0, 0, 7.2, 6.4, RF_MAIN)
    ax.text(3.6, 6.15, "Random Forest:\nparallel trees, majority vote",
            ha="center", va="top", color=RF_MAIN,
            fontsize=10, fontweight="bold")

    # row 1 of trees
    row1 = [(0.9, "EV", RF_MAIN), (2.25, "EV", RF_MAIN),
            (3.6, "no EV", rf_faded), (4.95, "EV", RF_MAIN),
            (6.3, "EV", RF_MAIN)]
    # row 2 of trees
    row2 = [(0.9, "no EV", rf_faded), (2.25, "EV", RF_MAIN),
            (3.6, "EV", RF_MAIN), (4.95, "EV", RF_MAIN),
            (6.3, "no EV", rf_faded)]
    for x, label, c in row1:
        draw_tree(ax, x, 4.5, c, label)
    for x, label, c in row2:
        draw_tree(ax, x, 3.15, c, label)

    # vote bar (7 of 10)
    ax.text(3.6, 2.25, '7 of 10 trees vote "EV"',
            ha="center", va="center", color=TXT_GRAY,
            fontsize=9, fontweight="bold")
    bar_x, bar_y, bar_w, bar_h = 0.7, 1.5, 5.8, 0.4
    ax.add_patch(FancyBboxPatch(
        (bar_x, bar_y), bar_w, bar_h,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=BAR_EMPTY, edgecolor="none"))
    ax.add_patch(FancyBboxPatch(
        (bar_x, bar_y), 0.7 * bar_w, bar_h,          # 70 % filled
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=RF_MAIN, alpha=0.9, edgecolor="none"))
    ax.text(3.6, 0.8,
            "Averaged vote: $\\hat{p} = 0.7$\n"
            "$\\rightarrow$ EV charger predicted",
            ha="center", va="center", color=TXT_GRAY,
            fontsize=9, fontstyle="italic")

    # ============================================================
    # RIGHT PANEL: XGBoost
    # ============================================================
    draw_panel(ax, 8.2, 0, 7.2, 6.4, XGB_MAIN)
    ax.text(11.8, 6.15,
            "XGBoost:\nsequential trees, each correcting errors",
            ha="center", va="top", color=XGB_MAIN,
            fontsize=10, fontweight="bold")

    ax.text(11.8, 5.0, "Residual error shrinks with every added tree",
            ha="center", va="center", color=TXT_GRAY,
            fontsize=9, fontweight="bold")

    # shrinking error bars, baseline at y=2.2
    bars = [(9.0, 4.4, 0.35), (10.6, 3.75, 0.50),
            (12.2, 3.2, 0.65), (13.8, 2.7, 0.95)]
    for bx, btop, alpha in bars:
        ax.add_patch(Rectangle((bx, 2.2), 1.0, btop - 2.2,
                               facecolor=XGB_MAIN, alpha=alpha,
                               edgecolor="none"))

    # flow arrows between the bars
    for ax0 in (10.1, 11.7):
        ax.add_patch(FancyArrow(ax0, 3.0, 0.4, 0,
                                width=0.015, head_width=0.14,
                                head_length=0.18,
                                length_includes_head=True,
                                facecolor=XGB_MAIN, edgecolor=XGB_MAIN))
    ax.text(13.5, 3.0, "$\\cdots$", ha="center", va="center",
            color=XGB_MAIN, fontsize=12)

    # bar labels
    bar_labels = [(9.5, "Tree 1\nlarge error"),
                  (11.1, "Tree 2\nsmaller error"),
                  (12.7, "Tree 3\nsmaller still"),
                  (14.3, "Tree $M$\nminimal error")]
    for bx, text in bar_labels:
        ax.text(bx, 1.85, text, ha="center", va="top",
                color=TXT_GRAY, fontsize=6.5)

    ax.text(11.8, 0.8,
            "Final prediction: sum of all trees' contributions\n"
            "$\\hat{y} = \\sum_{m=1}^{M} f_m(\\mathbf{x})$",
            ha="center", va="center", color=TXT_GRAY,
            fontsize=9, fontstyle="italic")

    # ------------------------------------------------------------
    # save
    # ------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"rf_vs_xgb_concept.{ext}"),
                    dpi=300, bbox_inches="tight", transparent=False,
                    facecolor="white")
    print(f"Saved rf_vs_xgb_concept.png / .pdf to {os.path.abspath(out_dir)}")
    plt.show()


if __name__ == "__main__":
    main()
