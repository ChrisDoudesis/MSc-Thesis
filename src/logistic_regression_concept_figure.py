"""
Figure: Logistic Regression -- sigmoid function and linear decision
boundary (Section 2.3.3).

Matplotlib re-creation of the TikZ/pgfplots figure: panel (a) shows the
sigmoid function with the 0.5 decision threshold, panel (b) shows two
synthetic classes in feature space separated by a linear decision
boundary.

Run:  python logistic_regression_concept_figure.py
Output: ../results/logistic_regression_concept.png (and .pdf)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os

# ============================================================
# >>> ALL COLORS ARE DEFINED HERE -- edit only this block <<<
# Same muted palette as the other Chapter 2 figures.
# ============================================================
LR_MAIN       = "#2B5D8A"   # curve / EV class: dark muted blue
LR_NEG        = "#3D7A6B"   # no-EV class: muted teal-green
LR_GUIDE      = "#888888"   # dashed guide lines
LR_REGION_POS = "#EAF1F8"   # shaded region, EV side
LR_REGION_NEG = "#EBF4F1"   # shaded region, no-EV side


# same synthetic points as the TikZ version
NO_EV_POINTS = [
    (3.35, 2.10), (3.86, 4.58), (0.76, 1.77), (3.15, 3.00), (2.98, 2.33),
    (4.01, 4.37), (3.08, 4.81), (3.54, 2.33), (3.42, 2.20), (4.01, 3.34),
    (2.79, 2.55), (4.41, 3.21), (2.51, 2.96), (3.61, 3.86), (3.47, 3.94),
    (5.46, 2.89), (2.41, 2.38), (3.71, 4.81), (2.87, 2.35), (2.05, 4.21),
    (3.85, 4.08), (2.23, 3.69), (3.13, 3.67), (4.00, 3.68), (3.78, 3.48),
    (3.33, 4.19),
]
EV_POINTS = [
    (5.12, 6.20), (6.26, 5.80), (6.48, 8.47), (5.80, 7.81), (4.86, 6.18),
    (6.99, 7.33), (7.62, 7.59), (6.40, 6.02), (7.79, 6.36), (5.33, 5.18),
    (5.74, 7.22), (6.96, 7.46), (6.31, 6.80), (7.52, 6.21), (7.33, 5.77),
    (6.38, 6.12), (5.42, 7.21), (6.26, 6.62), (7.35, 7.16), (7.57, 6.48),
    (6.31, 6.50), (4.86, 4.79), (5.28, 5.35), (7.26, 5.47), (6.37, 8.22),
    (6.39, 7.52),
]


def style_axes(ax):
    """pgfplots `axis lines=left` look: only left and bottom spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    # ============================================================
    # (a) sigmoid function
    # ============================================================
    z = np.linspace(-6, 6, 200)
    sigma = 1.0 / (1.0 + np.exp(-z))

    # guide lines: threshold 0.5 at z = 0, asymptote at 1
    ax1.axhline(0.5, color=LR_GUIDE, linestyle="--", linewidth=0.8)
    ax1.axvline(0.0, color=LR_GUIDE, linestyle="--", linewidth=0.8)
    ax1.axhline(1.0, color=LR_GUIDE, linestyle=":", linewidth=0.8)

    ax1.plot(z, sigma, color=LR_MAIN, linewidth=1.8)

    ax1.text(-5.8, 0.53, "decision threshold $0.5$",
             color=LR_GUIDE, fontsize=8, ha="left", va="bottom")
    ax1.text(1.1, 0.88, "predict EV",
             color=LR_MAIN, fontsize=8, ha="left", va="center")
    ax1.text(-1.1, 0.12, "predict no EV",
             color=LR_NEG, fontsize=8, ha="right", va="center")

    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_xlabel(r"$z = \mathbf{w}^{\top}\mathbf{x} + b$", fontsize=9)
    ax1.set_ylabel(r"$\sigma(z) = P(y=1 \mid \mathbf{x})$", fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.set_title("(a) Logistic (sigmoid) function",
                  fontsize=9, fontweight="bold")
    style_axes(ax1)

    # ============================================================
    # (b) linear decision boundary
    # ============================================================
    # shaded half-planes (drawn first, so points sit on top)
    ax2.add_patch(Polygon([(0, 0), (0, 9.5), (9.66, 0)],
                          closed=True, facecolor=LR_REGION_NEG,
                          edgecolor="none", zorder=0))
    ax2.add_patch(Polygon([(0, 9.5), (0, 10), (10, 10), (10, 0), (9.66, 0)],
                          closed=True, facecolor=LR_REGION_POS,
                          edgecolor="none", zorder=0))

    # no-EV households (teal open circles)
    x0, y0 = zip(*NO_EV_POINTS)
    ax2.scatter(x0, y0, s=22, facecolors="none", edgecolors=LR_NEG,
                linewidths=1.0, zorder=3,
                label="no EV charger ($y=0$)")

    # EV households (blue filled triangles)
    x1, y1 = zip(*EV_POINTS)
    ax2.scatter(x1, y1, s=28, marker="^", color=LR_MAIN, zorder=3,
                label="EV charger ($y=1$)")

    # decision boundary: w'x + b = 0
    ax2.plot([0.3, 9.66], [9.21, 0], color="0.4", linestyle="--",
             linewidth=1.2, zorder=2,
             label=r"$\mathbf{w}^{\top}\mathbf{x} + b = 0$")

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("feature $x_1$", fontsize=9)
    ax2.set_ylabel("feature $x_2$", fontsize=9)
    ax2.set_title("(b) Linear decision boundary",
                  fontsize=9, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8, frameon=False)
    style_axes(ax2)

    fig.tight_layout(w_pad=3.0)

    # ------------------------------------------------------------
    # save
    # ------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir,
                                 f"logistic_regression_concept.{ext}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved logistic_regression_concept.png / .pdf to "
          f"{os.path.abspath(out_dir)}")
    plt.show()


if __name__ == "__main__":
    main()
