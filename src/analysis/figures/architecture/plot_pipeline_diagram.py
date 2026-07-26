from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT    = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "reports" / "figures" / "architectures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG       = "#c9a98a"
C_DATA   = "#4a72b8"
C_GRU    = "#7b3060"
C_GP     = "#2d7a4f"
C_UNSEEN = "#CC6600"
TEXT     = "#f0f0f0"

BW, BH = 2.3, 3.0
Y       = 5.0
_margin = 0.4
_gap    = (16 - 2 * _margin - 4 * BW) / 3
X0 = _margin + BW / 2
X1 = X0 + BW + _gap
X2 = X1 + BW + _gap
X3 = X2 + BW + _gap

G_ARROW = 0.22
FS_BOX  = 25


def box(ax, cx, cy, w, h, color, text):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04",
        facecolor=color, edgecolor="#555555", linewidth=1.5,
        transform=ax.transData, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=FS_BOX, fontweight="bold", color=TEXT, zorder=4,
            multialignment="center", linespacing=1.9)


def arrow(ax, x0, x1, y, label, fontsize=19):
    ax.annotate(
        "", xy=(x1 - G_ARROW, y), xytext=(x0 + G_ARROW, y),
        arrowprops=dict(
            arrowstyle="-|>,head_width=1.10,head_length=0.80",
            color="#333333", lw=3.0,
        ),
        zorder=4,
    )
    ax.text((x0 + x1) / 2, y + 0.32, label,
            ha="center", va="bottom", fontsize=fontsize,
            fontweight="bold", color="#333333", zorder=5,
            multialignment="center", linespacing=1.5)


def draw_boxes_and_arrows(ax):
    box(ax, X0, Y, BW, BH, C_DATA,   "Training\nwells\n(spatial\ntraining\nset)")
    box(ax, X1, Y, BW, BH, C_GRU,    "Temporal\nForecasts\n(GRU)")
    box(ax, X2, Y, BW, BH, C_GP,     "Spatial\nInter-\npolation\n(GP)")
    box(ax, X3, Y, BW, BH, C_UNSEEN, "Holdout\nwells\n(unseen\nduring\ntraining)")

    arrow(ax, X0 + BW/2, X1 - BW/2, Y, "history +\nfeatures")
    arrow(ax, X1 + BW/2, X2 - BW/2, Y, "$\\hat{y}$ at training\nlocations")
    arrow(ax, X2 + BW/2, X3 - BW/2, Y, "interpo-\nlated $\\hat{y}$")


def make_fig():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(2.0, 8.0)
    ax.axis("off")
    return fig, ax


def draw_oracle(ax):
    box(ax, X0, Y, BW, BH, C_DATA,   "Training\nwells\n(spatial\ntraining\nset)")
    box(ax, X1, Y, BW, BH, C_DATA,   "True\nobser-\nvations")
    box(ax, X2, Y, BW, BH, C_GP,     "Spatial\nInter-\npolation\n(GP)")
    box(ax, X3, Y, BW, BH, C_UNSEEN, "Holdout\nwells\n(unseen\nduring\ntraining)")

    arrow(ax, X0 + BW/2, X1 - BW/2, Y, "history +\nfeatures")
    arrow(ax, X1 + BW/2, X2 - BW/2, Y, "true $y$ at\ntrain wells")
    arrow(ax, X2 + BW/2, X3 - BW/2, Y, "interpo-\nlated $\\hat{y}$")


def main():
    fig, ax = make_fig()
    draw_boxes_and_arrows(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR / "pipeline_diagram.png", dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("saved pipeline_diagram.png")

    fig, ax = make_fig()
    draw_boxes_and_arrows(ax)

    pad = 0.45
    jx = X1 - BW/2 - pad
    jy = Y  - BH/2 - pad
    jw = (X2 + BW/2 + pad) - jx
    jh = BH + 2 * pad
    joint_rect = FancyBboxPatch(
        (jx, jy), jw, jh,
        boxstyle="round,pad=0.0",
        facecolor="none", edgecolor="#333333", linewidth=2.2,
        linestyle="--", zorder=5,
    )
    ax.add_patch(joint_rect)
    ax.text((X1 + X2) / 2, jy - 0.25, "End-to-end\nspatiotemporal model",
            ha="center", va="top", fontsize=19, fontweight="bold",
            color="#333333", zorder=6, multialignment="center", linespacing=1.5)

    arc_x0 = X2 - BW/2 - G_ARROW
    arc_x1 = X1 + BW/2 + G_ARROW
    arc_y  = Y - 0.7
    ax.annotate(
        "",
        xy=(arc_x1, arc_y),
        xytext=(arc_x0, arc_y),
        arrowprops=dict(
            arrowstyle="-|>,head_width=1.00,head_length=0.75",
            color="#333333", lw=2.5,
            connectionstyle="arc3,rad=-0.4",
        ),
        zorder=6,
    )
    ax.text((arc_x0 + arc_x1) / 2, arc_y - 0.45, "joint loss",
            ha="center", va="top", fontsize=19, fontweight="bold",
            color="#333333", zorder=6)

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR / "pipeline_diagram_joint.png", dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("saved pipeline_diagram_joint.png")

    fig, ax = make_fig()
    draw_oracle(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR / "pipeline_diagram_oracle.png", dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("saved pipeline_diagram_oracle.png")


if __name__ == "__main__":
    main()
