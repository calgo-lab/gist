import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

# ── Global typography ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       9,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor":   "#FFFFFF",
})

# ── Color tokens  (fill, border, text) ───────────────────────────────────────
COL = {
    "data":    ("#DBEAFE", "#2563EB", "#1E40AF"),  # blue
    "gru":     ("#FEE2E2", "#DC2626", "#991B1B"),  # red
    "gp":      ("#DCFCE7", "#16A34A", "#14532D"),  # green
    "joint":   ("#FEF3C7", "#D97706", "#92400E"),  # amber
    "split":   ("#EDE9FE", "#7C3AED", "#4C1D95"),  # violet
    "loss":    ("#FFEDD5", "#EA580C", "#7C2D12"),  # orange
    "neutral": ("#F1F5F9", "#94A3B8", "#334155"),  # slate
    "head":    ("#1E3A5F", "#1E3A5F", "#FFFFFF"),  # dark header
}
ARROW_C = "#475569"
TEXT_C  = "#1E293B"


# ── Helpers ───────────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, text, key, fs=9.5, bold=True, zorder=3):
    fill, border, tc = COL[key]
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.07", linewidth=1.3,
        facecolor=fill, edgecolor=border, zorder=zorder,
    ))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
            zorder=zorder+1, multialignment="center", linespacing=1.45)


def hbox(ax, cx, cy, w, h, text, fs=10.5, zorder=3):
    """Dark header box."""
    fill, border, tc = COL["head"]
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.07", linewidth=1.3,
        facecolor=fill, edgecolor=border, zorder=zorder,
    ))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold", zorder=zorder+1)


def arr(ax, x1, y1, x2, y2, lw=1.6, c=None, label="", lfs=8):
    c = c or ARROW_C
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                mutation_scale=11), zorder=6)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.06, my, label, fontsize=lfs, color=c,
                ha="left", va="center", style="italic", zorder=7)


def bracket(ax, x, y_bot, y_top, label="for each training epoch", fs=7.5):
    """Left bracket marking a loop."""
    ax.annotate("", xy=(x, y_top), xytext=(x, y_bot),
                arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=1.0,
                                connectionstyle="arc3,rad=0.0"))
    ax.text(x - 0.12, (y_bot + y_top) / 2, label,
            rotation=90, ha="center", va="center",
            fontsize=fs, color="#94A3B8", style="italic")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  –  Architecture Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_architectures():
    W, H = 14, 9.5
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # Title
    ax.text(W/2, 9.15, "Spatial Interpolation Pipelines: Decoupled vs. Joint",
            ha="center", va="center", fontsize=14, fontweight="bold", color=TEXT_C)

    # ── Shared input ──────────────────────────────────────────────────────────
    box(ax, W/2, 8.5, 11.5, 0.62,
        "1 040 monitoring wells   ·   Weekly groundwater-level observations   ·   1994 – 2024",
        "data", fs=10)

    # Fork arrows
    arr(ax, 3.85, 8.19, 3.5, 7.72)
    arr(ax, 10.15, 8.19, 10.5, 7.72)

    # ── Column headers ────────────────────────────────────────────────────────
    hbox(ax, 3.5, 7.4, 5.8, 0.56, "Decoupled  GRU → GP")
    hbox(ax, 10.5, 7.4, 5.8, 0.56, "Joint  GRU + GP")

    # ── Vertical divider ──────────────────────────────────────────────────────
    ax.axvline(7.0, ymin=0.03, ymax=0.88, color="#CBD5E1", lw=1.0, ls="--")

    # ══════════  LEFT COLUMN: DECOUPLED  ══════════════════════════════════════
    CL = 3.5   # center x

    # Split
    box(ax, CL, 6.75, 5.4, 0.62,
        "2-way spatial split\nTrain wells  (95%)   ·   Test wells  (5%)",
        "split", fs=9)
    arr(ax, CL, 7.12, CL, 7.07)

    # Stage 1 header
    ax.text(CL, 6.22, "Stage  ①  —  GRU Training", ha="center", va="center",
            fontsize=9, color=COL["gru"][1], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFF5F5",
                      ec=COL["gru"][1], lw=0.9))
    arr(ax, CL, 6.44, CL, 6.35)

    box(ax, CL, 5.45, 5.4, 0.92,
        "Temporal encoder-decoder  (GRU)\n"
        "Input: 52-week history + static well features\n"
        "Output: 16-week forecast per train well\n"
        "Loss:  MSE  (normalized space)",
        "gru", fs=9)
    arr(ax, CL, 4.99, CL, 4.52,
        label="GRU predictions at\ntrain well locations", lfs=7.8)

    # Stage 2 header
    ax.text(CL, 4.17, "Stage  ②  —  GP Fitting", ha="center", va="center",
            fontsize=9, color=COL["gp"][1], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F0FFF4",
                      ec=COL["gp"][1], lw=0.9))
    arr(ax, CL, 4.36, CL, 4.28)

    box(ax, CL, 3.48, 5.4, 0.98,
        "Gaussian Process  (spatial model)\n"
        "Input: GRU outputs at training locations\n"
        "Kernel: Matérn-3/2  ·  ARD lengthscales\n"
        "64 inducing points  ·  Coordinates + covariates",
        "gp", fs=9)
    arr(ax, CL, 2.99, CL, 2.52,
        label="Predict at unseen\ntest well locations", lfs=7.8)

    box(ax, CL, 2.15, 5.4, 0.62,
        "GP posterior prediction at test wells\n"
        "Predictive mean  ±  predictive std",
        "gp", fs=9)
    arr(ax, CL, 1.84, CL, 1.42)

    # Result
    box(ax, CL, 1.15, 5.4, 0.46,
        "nRMSE_pw = 1.96   (52 test wells)",
        "neutral", fs=9.5, bold=False)

    # ══════════  RIGHT COLUMN: JOINT  ═════════════════════════════════════════
    CR = 10.5

    # Split
    box(ax, CR, 6.75, 5.4, 0.62,
        "3-way spatial split\nTrain (~85%)   ·   Val (5%)   ·   Test (10%)",
        "split", fs=9)
    arr(ax, CR, 7.12, CR, 7.07)

    # Loop bracket
    bracket(ax, 7.55, 3.45, 6.35, label="for each\ntraining epoch")

    # Joint loop box
    arr(ax, CR, 6.44, CR, 6.07)
    box(ax, CR, 5.3, 5.4, 1.5,
        "①  GRU forward pass on train wells\n"
        "         → 16-week predictions (normalized)\n\n"
        "②  Denormalize  →  GP forward on val wells\n"
        "         → spatial correction at val locations\n\n"
        "③  Joint loss:     L  =  L_GRU  +  λ · L_GP\n\n"
        "④  Backpropagate through GRU and GP jointly",
        "joint", fs=9)

    # Early stopping note
    arr(ax, CR, 4.55, CR, 4.17)
    box(ax, CR, 3.9, 5.4, 0.46,
        "Early stopping on val loss   ·   Patience = 5 epochs",
        "neutral", fs=8.5, bold=False)

    arr(ax, CR, 3.67, CR, 3.22,
        label="Evaluate on\ntest locations", lfs=7.8)

    box(ax, CR, 2.88, 5.4, 0.62,
        "Joint GRU + GP prediction at test wells\n"
        "Predictive mean  ±  predictive std",
        "joint", fs=9)

    # Draw a curved arrow from GP inside the joint box back up (backprop visual)
    ax.annotate("", xy=(12.7, 5.68), xytext=(12.7, 4.80),
                arrowprops=dict(arrowstyle="-|>",
                                connectionstyle="arc3,rad=-0.5",
                                color=COL["loss"][1], lw=1.3,
                                mutation_scale=10), zorder=6)
    ax.text(13.15, 5.22, "gradients\nflow back", ha="center", va="center",
            fontsize=7.5, color=COL["loss"][1], style="italic")

    arr(ax, CR, 2.57, CR, 1.42)
    box(ax, CR, 1.15, 5.4, 0.46,
        "nRMSE_pw = 2.15   (104 test wells)",
        "neutral", fs=9.5, bold=False)

    # ── Shared evaluation note ────────────────────────────────────────────────
    ax.text(W/2, 0.45,
            "Evaluation metric:  nRMSE_pw  =  median over test wells of  ( RMSE_well / IQR_well )\n"
            "Both pipelines evaluated on wells fully excluded from training — no temporal or spatial leakage",
            ha="center", va="center", fontsize=8.5, color="#64748B",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F8FAFC",
                      ec="#CBD5E1", lw=0.8))

    # ── Key difference callout ────────────────────────────────────────────────
    ax.text(W/2, 7.4,
            "Key difference: gradient flow",
            ha="center", va="center", fontsize=8.5, color="#475569",
            style="italic")

    fig.tight_layout(pad=0.3)
    out = OUT / "fig1_architectures.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2  –  Data Flow & Spatial Splits
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_data_splits():
    W, H = 14, 7.0
    fig, axes = plt.subplots(1, 2, figsize=(W, H),
                             gridspec_kw={"width_ratios": [1, 1.05]})
    fig.suptitle("Dataset Structure & Spatial Splits",
                 fontsize=13, fontweight="bold", color=TEXT_C, y=0.98)

    # ── LEFT: data flow timeline ───────────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 7); ax.set_ylim(0, 7); ax.axis("off")

    ax.text(3.5, 6.6, "Temporal structure", ha="center", va="center",
            fontsize=11, fontweight="bold", color=TEXT_C)

    # Dataset box
    box(ax, 3.5, 6.0, 6.5, 0.58,
        "full_merged dataset   ·   1 040 wells   ·   weekly, ~30 years",
        "data", fs=9.5)
    arr(ax, 3.5, 5.71, 3.5, 5.3)

    # Normalization box
    box(ax, 3.5, 5.05, 6.5, 0.42,
        "Per-well normalization  (mean / std)  —  fitted on TRAIN data only",
        "neutral", fs=8.5, bold=False)
    arr(ax, 3.5, 4.84, 3.5, 4.45)

    # Sliding window box
    box(ax, 3.5, 4.2, 6.5, 0.48,
        "Sliding windows  ·  52-week input  →  16-week forecast horizon",
        "gru", fs=9)
    arr(ax, 3.5, 3.96, 3.5, 3.6)

    # Timeline bar
    bar_y = 3.3
    segs = [
        (0.25, 2.85, "#2563EB", "TRAIN\n(before 2016)"),
        (2.85, 4.85, "#D97706", "VALIDATION\n(2016 – 2020)"),
        (4.85, 6.75, "#DC2626", "TEST\n(after 2020)"),
    ]
    for x0, x1, color, lbl in segs:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, bar_y - 0.32), x1 - x0, 0.64,
            boxstyle="square,pad=0.0",
            facecolor=color, edgecolor="#FFFFFF", lw=1.5, zorder=3
        ))
        ax.text((x0+x1)/2, bar_y, lbl, ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold",
                zorder=4, multialignment="center")
    # year ticks
    ticks = [(0.25, "1994"), (2.85, "2016"), (4.85, "2020"), (6.75, "2024")]
    for x, yr in ticks:
        ax.plot([x, x], [bar_y - 0.47, bar_y - 0.32],
                color=TEXT_C, lw=1.0)
        ax.text(x, bar_y - 0.6, yr, ha="center", fontsize=8,
                color=TEXT_C, fontweight="bold")

    arr(ax, 3.5, 2.68, 3.5, 2.3)

    # Spatial filter
    box(ax, 3.5, 2.07, 6.5, 0.42,
        "Spatial filter: windows from train wells only  →  held-out wells excluded from training",
        "split", fs=8.5)
    arr(ax, 3.5, 1.86, 3.5, 1.47)

    # Three feature types
    feat_boxes = [
        (1.1, "Dynamic  (52 wk)\nGWS + 5 covariates\n(temp, humidity,\nprecip, doy sin/cos)",  "gru"),
        (3.5, "Static  (27 dims)\nSoil, geology,\nland cover, TWI,\nbio-climate vars",         "neutral"),
        (5.9, "Spatial  (GP)\nCoordinates\n+ optional:\ngok, aquifer type",                     "gp"),
    ]
    for cx, lbl, key in feat_boxes:
        box(ax, cx, 0.7, 2.0, 1.0, lbl, key, fs=7.8)
        arr(ax, cx, 1.47, cx, 1.2)

    # ── RIGHT: spatial splits table ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF"); ax2.axis("off")

    ax2.text(0.5, 0.97, "Spatial split configurations",
             transform=ax2.transAxes, ha="center", va="top",
             fontsize=11, fontweight="bold", color=TEXT_C)

    # Column specs: (x_left fraction, width fraction, label)
    cols = [
        (0.02, 0.27, "Split"),
        (0.30, 0.20, "Method"),
        (0.51, 0.12, "Train"),
        (0.64, 0.12, "Test"),
        (0.77, 0.21, "Used for"),
    ]
    row_h = 0.099
    hdr_y = 0.88

    # Header row
    for x, w, lbl in cols:
        ax2.add_patch(FancyBboxPatch(
            (x, hdr_y), w - 0.01, row_h,
            boxstyle="square,pad=0.0", transform=ax2.transAxes,
            facecolor=COL["head"][0], edgecolor="#FFFFFF", lw=1.0,
            clip_on=False, zorder=3
        ))
        ax2.text(x + (w-0.01)/2, hdr_y + row_h/2, lbl,
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=8.5, color="white", fontweight="bold", zorder=4)

    rows = [
        ("spf0p8",     "K-means\n20 clusters",  "841\n(81%)",  "199\n(19%)", "Feature ablation\nhyperparameter search"),
        ("spf0p9",     "K-means\n20 clusters",  "945\n(91%)",  "95\n(9%)",   "Alternative\ncluster split"),
        ("random_90",  "Purely\nrandom",         "936\n(90%)",  "104\n(10%)", "Random baseline"),
        ("rmd90",      "Max-distance\n≥ P90",    "936\n(90%)",  "104\n(10%)", "Joint GRU+GP\nevaluation"),
        ("rmd90_test52","rmd90 merged\n95/5",    "988\n(95%)",  "52\n(5%)",   "Decoupled GRU→GP\n(fair comparison)"),
        ("rmd5",       "Densest\nclusters",      "988\n(95%)",  "52\n(5%)",   "Most challenging\nspatial case"),
    ]

    bg_alt = ["#F8FAFC", "#EFF6FF"]
    for i, row in enumerate(rows):
        y0 = hdr_y - (i + 1) * row_h
        bg = bg_alt[i % 2]
        ax2.add_patch(FancyBboxPatch(
            (cols[0][0], y0), sum(w for _, w, _ in cols) - 0.01, row_h,
            boxstyle="square,pad=0.0", transform=ax2.transAxes,
            facecolor=bg, edgecolor="#E2E8F0", lw=0.5,
            clip_on=False, zorder=2
        ))
        for (x, w, _), val in zip(cols, row):
            ax2.text(x + (w-0.01)/2, y0 + row_h/2, val,
                     transform=ax2.transAxes, ha="center", va="center",
                     fontsize=8, color=TEXT_C, multialignment="center",
                     linespacing=1.3)

    # 3-way split note
    note_y = hdr_y - len(rows) * row_h - 0.03
    ax2.add_patch(FancyBboxPatch(
        (0.02, note_y - 0.10), 0.96, 0.10,
        boxstyle="round,pad=0.015", transform=ax2.transAxes,
        facecolor="#FFFBEB", edgecolor=COL["joint"][1], lw=0.9,
        clip_on=False, zorder=3
    ))
    ax2.text(0.5, note_y - 0.05,
             "Joint pipeline uses a 3-way split: val wells drawn from train set\n"
             "(val_from_train_fraction = 5%)  —  test set remains fully held out",
             transform=ax2.transAxes, ha="center", va="center",
             fontsize=8, color=COL["joint"][2], style="italic", multialignment="center")

    fig.tight_layout(pad=0.5)
    out = OUT / "fig2_data_splits.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3  –  GRU Encoder-Decoder Architecture
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_gru_architecture():
    W, H = 12, 8.5
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(W/2, 8.15, "GRU Encoder-Decoder Architecture",
            ha="center", fontsize=13, fontweight="bold", color=TEXT_C)

    # ── Input column ──────────────────────────────────────────────────────────
    # Dynamic input boxes
    box(ax, 2.0, 7.2, 3.2, 0.62,
        "Past GWS  (52 weeks)\nnormalized per well",
        "data", fs=9)
    box(ax, 2.0, 6.3, 3.2, 0.62,
        "Climate covariates  (52 × 5)\ntemp · humidity · precip\nday-of-year (sin/cos)",
        "data", fs=9)
    # Merge annotation
    ax.text(2.0, 5.8, "concatenated →", ha="center", fontsize=8,
            color="#94A3B8", style="italic")
    # Static box
    box(ax, 2.0, 4.6, 3.2, 1.1,
        "Static well features  (27 dims)\n─────────────────────\nbio-climate  ·  soil type\ngeology  ·  land cover\nTWI  (+ gok for best config)",
        "neutral", fs=8.5, bold=False)

    # ── Encoder ───────────────────────────────────────────────────────────────
    arr(ax, 3.6,  7.2, 5.05, 6.72)
    arr(ax, 3.6,  6.3, 5.05, 6.52)
    arr(ax, 3.6,  4.6, 5.0,  5.48)  # static to static-proj

    box(ax, 5.8, 5.25, 2.6, 0.52,
        "Static projection  →  h₀",
        "neutral", fs=8.5, bold=False)

    box(ax, 5.8, 6.58, 2.6, 0.78,
        "Encoder  GRU\n4 layers  ·  hidden = 256\ndropout = 0.2",
        "gru", fs=9.5)

    arr(ax, 5.8, 6.19, 5.8, 5.52)   # static proj → encoder h0
    arr(ax, 5.8, 5.99, 5.8, 5.55, lw=0.1)  # skip (already covered)

    # hidden state arrow
    arr(ax, 5.8, 5.9, 7.7, 5.9, label=" h_enc", lfs=8)  # encoder → decoder

    # ── Future covariates ─────────────────────────────────────────────────────
    box(ax, 10.0, 6.58, 3.2, 0.62,
        "Future covariates  (16 × 5)\ntemp · humidity · precip\nday-of-year (sin/cos)",
        "data", fs=9)
    arr(ax, 8.62, 6.58, 9.38, 6.58)

    # ── Decoder ───────────────────────────────────────────────────────────────
    box(ax, 8.0, 5.58, 2.6, 0.62,
        "Decoder  GRU\n4 layers  ·  hidden = 256",
        "gru", fs=9.5)
    arr(ax, 10.0, 6.27, 8.55, 5.9)  # future cov → decoder

    # ── Output head ───────────────────────────────────────────────────────────
    arr(ax, 8.0, 5.27, 8.0, 4.77)
    box(ax, 8.0, 4.5, 2.6, 0.52,
        "Linear head  →  16 × 1\n(normalized forecast)",
        "gru", fs=9)
    arr(ax, 8.0, 4.24, 8.0, 3.77)

    box(ax, 8.0, 3.52, 2.6, 0.5,
        "Denormalize\nŷ  =  ŷ_norm · σ  +  μ",
        "neutral", fs=9, bold=False)

    # ── Loss & training ───────────────────────────────────────────────────────
    arr(ax, 8.0, 3.27, 8.0, 2.77)
    box(ax, 8.0, 2.52, 2.6, 0.5,
        "Loss:   MSE( ŷ_norm ,  y_norm )",
        "loss", fs=9.5)
    arr(ax, 8.0, 2.27, 8.0, 1.77)

    box(ax, 8.0, 1.52, 2.6, 0.5,
        "Adam optimizer  ·  50 epochs\nEarly stopping  (patience = 5)",
        "neutral", fs=8.5, bold=False)

    # ── Output annotation (test-time) ─────────────────────────────────────────
    arr(ax, 8.0, 1.27, 8.0, 0.82)
    box(ax, 8.0, 0.57, 2.6, 0.5,
        "16-week GWS forecast per well",
        "gp", fs=9)

    # ── Legend / caption ─────────────────────────────────────────────────────
    ax.text(3.2, 0.65,
            "The GRU is shared across both pipeline variants.\n"
            "In the joint pipeline, gradients also flow back\n"
            "through the GP loss during training.",
            ha="left", va="center", fontsize=8.5, color="#64748B",
            style="italic", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F8FAFC",
                      ec="#CBD5E1", lw=0.8))

    fig.tight_layout(pad=0.3)
    out = OUT / "fig3_gru_architecture.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4  –  Evaluation, Metrics & Results
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_metrics_results():
    W, H = 14, 7.5
    fig, axes = plt.subplots(1, 2, figsize=(W, H),
                             gridspec_kw={"width_ratios": [1, 1]})
    fig.suptitle("Evaluation Setup & Reporting Metrics",
                 fontsize=13, fontweight="bold", color=TEXT_C, y=0.98)

    # ── LEFT: evaluation setup ────────────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 7); ax.set_ylim(0, 7.5); ax.axis("off")

    ax.text(3.5, 7.1, "Metric definitions", ha="center", va="center",
            fontsize=11, fontweight="bold", color=TEXT_C)

    # Metric boxes
    defs = [
        ("nRMSE",
         "RMSE_well\n────────────\nIQR_well",
         "Primary metric.  Scale-invariant.\nRobust to well-level amplitude differences.",
         "loss"),
        ("NSE",
         "        Σ (ŷ − y)²\n1  −  ─────────────\n        Σ (y − ȳ)²",
         "Nash-Sutcliffe efficiency.\n0 = mean baseline,   1 = perfect fit.",
         "gru"),
        ("RMSE",
         "√  mean( (ŷ − y)² )",
         "Root mean squared error in original GWS units.",
         "gp"),
    ]
    y0 = 6.5
    for name, formula, desc, key in defs:
        fill, border, tc = COL[key]
        ax.add_patch(FancyBboxPatch(
            (0.2, y0 - 1.12), 6.6, 1.0,
            boxstyle="round,pad=0.07", linewidth=1.2,
            facecolor=fill, edgecolor=border, zorder=3
        ))
        ax.text(1.05, y0 - 0.62, name, ha="center", va="center",
                fontsize=13, color=border, fontweight="bold", zorder=4)
        ax.text(2.4, y0 - 0.52, formula, ha="center", va="center",
                fontsize=9, color=tc, fontweight="bold", zorder=4,
                family="monospace", linespacing=1.3)
        ax.text(5.1, y0 - 0.62, desc, ha="center", va="center",
                fontsize=8.2, color=tc, zorder=4, linespacing=1.35,
                multialignment="center")
        ax.axvline(1.7, ymin=(y0-1.12)/7.5, ymax=(y0-0.12)/7.5,
                   color=border, lw=0.7, ls="--", alpha=0.4)
        ax.axvline(3.2, ymin=(y0-1.12)/7.5, ymax=(y0-0.12)/7.5,
                   color=border, lw=0.7, ls="--", alpha=0.4)
        y0 -= 1.18

    # Aggregation
    ax.text(3.5, 3.2, "Aggregation strategy", ha="center", va="center",
            fontsize=10, fontweight="bold", color=TEXT_C)
    agg_items = [
        "Per-well:       nRMSE and NSE computed for each well independently over time",
        "nRMSE_pw:    median of per-well nRMSE  ←  headline metric",
        "Per-horizon:  metrics broken down by forecast week  h = 1 … 16",
        "Clipping:       outlier wells capped at nRMSE = 5  before median",
    ]
    for i, txt in enumerate(agg_items):
        ax.text(0.4, 2.82 - i * 0.42, f"•  {txt}", fontsize=8.5,
                va="center", color=TEXT_C)

    # ── RIGHT: benchmark results ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF"); ax2.axis("off")

    ax2.text(0.5, 0.97, "Benchmark results  (test wells, 2020 – 2024)",
             transform=ax2.transAxes, ha="center", va="top",
             fontsize=11, fontweight="bold", color=TEXT_C)

    # Sub-header
    ax2.text(0.5, 0.915,
             "52 held-out test wells  ·  rmd90_test52 split",
             transform=ax2.transAxes, ha="center", va="center",
             fontsize=8.5, color="#64748B", style="italic")

    # Table
    cols = [
        (0.02, 0.40, "Pipeline"),
        (0.43, 0.18, "nRMSE_pw"),
        (0.62, 0.19, "nRMSE_pw\n(clipped)"),
        (0.82, 0.16, "% clipped"),
    ]
    row_h = 0.115
    hdr_y = 0.86

    for x, w, lbl in cols:
        ax2.add_patch(FancyBboxPatch(
            (x, hdr_y), w - 0.01, row_h,
            boxstyle="square,pad=0.0", transform=ax2.transAxes,
            facecolor=COL["head"][0], edgecolor="#FFFFFF", lw=1.0,
            clip_on=False, zorder=3
        ))
        ax2.text(x + (w-0.01)/2, hdr_y + row_h/2, lbl,
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=8.5, color="white", fontweight="bold", zorder=4,
                 multialignment="center")

    results = [
        ("GRU all_train\n(in-sample ceiling)",                     "0.313",  "—",     "—",    "#FFF5F5", COL["gru"][1]),
        ("Oracle GP\n(true GWS as GP input)",                      "1.666",  "1.666", "0.1%", "#F0FFF4", COL["gp"][1]),
        ("Decoupled  GRU → GP\n(95 / 5 split, rmd90_test52)",      "1.975",  "1.959", "5.5%", "#EFF6FF", COL["data"][1]),
        ("Joint  GRU + GP  (λ = 0.001)\n(90 / 10 split, rmd90)",   "2.281",  "2.146", "7.7%", "#FFFBEB", COL["joint"][1]),
    ]

    for i, (name, v1, v2, v3, bg, border_c) in enumerate(results):
        y0 = hdr_y - (i + 1) * row_h
        ax2.add_patch(FancyBboxPatch(
            (cols[0][0], y0), sum(w for _, w, _ in cols) - 0.01, row_h,
            boxstyle="square,pad=0.0", transform=ax2.transAxes,
            facecolor=bg, edgecolor="#E2E8F0", lw=0.5,
            clip_on=False, zorder=2
        ))
        vals = [name, v1, v2, v3]
        for (x, w, _), val in zip(cols, vals):
            ax2.text(x + (w-0.01)/2, y0 + row_h/2, val,
                     transform=ax2.transAxes, ha="center", va="center",
                     fontsize=8.5, color=TEXT_C, multialignment="center",
                     linespacing=1.3)

    # Notes
    note_y = hdr_y - len(results) * row_h - 0.04
    notes = [
        "★  nRMSE_pw (clipped) = headline metric after capping outlier wells at nRMSE = 5",
        "★  GRU all_train is evaluated in-sample — not a fair spatial-generalization baseline",
        "★  Oracle GP sets the upper bound for GP-based spatial correction",
    ]
    for j, n in enumerate(notes):
        ax2.text(0.04, note_y - j * 0.07, n,
                 transform=ax2.transAxes, ha="left", va="top",
                 fontsize=8, color="#64748B", style="italic")

    # Bar chart: visual comparison of the 3 meaningful numbers
    bar_ax = fig.add_axes([0.59, 0.07, 0.38, 0.25])
    bar_ax.set_facecolor("#F8FAFC")
    labels_bar = ["Oracle\nGP", "Decoupled\nGRU→GP", "Joint\nGRU+GP"]
    vals_bar   = [1.666, 1.959, 2.146]
    colors_bar = [COL["gp"][1], COL["data"][1], COL["joint"][1]]
    bars = bar_ax.barh(range(3), vals_bar, color=colors_bar,
                       edgecolor="#FFFFFF", lw=1.2, height=0.55)
    for b, v in zip(bars, vals_bar):
        bar_ax.text(v + 0.03, b.get_y() + b.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=8.5,
                    color=TEXT_C, fontweight="bold")
    bar_ax.set_yticks(range(3))
    bar_ax.set_yticklabels(labels_bar, fontsize=8)
    bar_ax.set_xlabel("nRMSE_pw  (clipped)", fontsize=8)
    bar_ax.set_xlim(0, 2.6)
    bar_ax.spines[["top", "right"]].set_visible(False)
    bar_ax.tick_params(labelsize=8)
    bar_ax.axvline(1.666, color=COL["gp"][1], lw=0.8, ls="--", alpha=0.5)
    bar_ax.set_title("nRMSE_pw comparison", fontsize=8.5,
                     color=TEXT_C, pad=4)

    fig.tight_layout(pad=0.5)
    out = OUT / "fig4_metrics_results.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig1_architectures()
    fig2_data_splits()
    fig3_gru_architecture()
    fig4_metrics_results()
    print("\nDone —", OUT)
