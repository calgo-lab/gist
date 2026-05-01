import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── colours  (3-family palette) ───────────────────────────────────────────────
C_DATA    = "#DBEAFE"; C_DATA_B    = "#2563EB"   # blue  — data / tensors / files
C_OP      = "#FEF9C3"; C_OP_B      = "#CA8A04"   # amber — operations / processing
C_GRU     = "#DCFCE7"; C_GRU_B     = "#16A34A"   # green — all neural-network components

# hidden-state family — light red fill, same red border as bridge arrows
C_HIDDEN  = "#FFE4E6"; C_HIDDEN_B  = "#DC2626"
C_H0_FC   = "#FFE4E6"; C_H0_EC     = "#DC2626"

C_OUT     = C_DATA;   C_OUT_B     = C_DATA_B
C_STEP    = C_GRU;    C_STEP_EC   = C_GRU_B

C_GP      = "#F3E8FF"; C_GP_B      = "#7C3AED"   # violet — GP model components
C_LOSS    = "#FFF7ED"; C_LOSS_B    = "#EA580C"   # orange — loss / objectives

C_ARROW   = "#475569"
C_BRIDGE  = "#DC2626"   # red — h_enc transfer arrows (not a box colour)

# ── gap constant ──────────────────────────────────────────────────────────────
G = 0.07   # gap between arrowhead/tail and box edge

# ── helpers ───────────────────────────────────────────────────────────────────
def rbox(ax, cx, cy, w, h, text, fc, ec, fs=8.0, bold=False, z=3, lw=1.4):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle="round,pad=0.04",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal",
            multialignment="center", zorder=z+1)

def arr(ax, x0, y0, x1, y1, c=C_ARROW, lw=1.4, rad=0.0, hw=0.60, hl=0.45):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=f"-|>,head_width={hw},head_length={hl}",
                    color=c, lw=lw, connectionstyle=f"arc3,rad={rad}"),
                zorder=5)

def htick(ax, x0, x1, y, c=C_ARROW, lw=1.1, hw=0.26, hl=0.18):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle=f"-|>,head_width={hw},head_length={hl}",
                                color=c, lw=lw, connectionstyle="arc3,rad=0"),
                zorder=5)

def vtick(ax, x, y0, y1, c=C_ARROW, lw=1.3, hw=0.60, hl=0.45):
    """Upward arrow; caller is responsible for gap offsets."""
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle=f"-|>,head_width={hw},head_length={hl}",
                                color=c, lw=lw, connectionstyle="arc3,rad=0"),
                zorder=6)

def sec(ax, W, y, text):
    ax.text(0.25, y, text, fontsize=10, fontweight="bold", color="#1e293b",
            va="center", ha="left", zorder=6)
    ax.plot([0.15, W-0.15], [y-0.27, y-0.27], color="#cbd5e1", lw=0.7)

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
N_LAYERS   = 4
LAYER_H    = 0.88
LAYER_STEP = 1.52

L1_Y = 8.0
LY   = [L1_Y + i * LAYER_STEP for i in range(N_LAYERS)]

ENC_CX = 7.0;  ENC_W = 11.2
DEC_CX = 24.4; DEC_W = 11.2
ENC_L = ENC_CX - ENC_W/2  # 1.4
ENC_R = ENC_CX + ENC_W/2  # 12.6
DEC_L = DEC_CX - DEC_W/2  # 18.8
DEC_R = DEC_CX + DEC_W/2  # 30.0

CW = 0.58; CH = 0.62

# encoder cell x-centres
EH0_CX  = ENC_L + 0.55
ET1_CX  = EH0_CX + CW + 0.22
ET2_CX  = ET1_CX + CW + 0.18
ET52_CX = ENC_R  - 0.55

# decoder cell x-centres
DH0_CX  = DEC_L + 0.55
DT1_CX  = DH0_CX + CW + 0.22
DT2_CX  = DT1_CX + CW + 0.18
DT16_CX = DEC_R  - 0.55

VGAP = 0.10   # gap for inter-layer vertical arrows

# ─────────────────────────────────────────────────────────────────────────────
# Y POSITIONS
# ─────────────────────────────────────────────────────────────────────────────
ENC_TOP = LY[N_LAYERS-1] + LAYER_H/2
ENC_BOT = LY[0]          - LAYER_H/2

S45_Y   = ENC_TOP + 0.70
H0_Y    = S45_Y   + 1.65
SPROJ_Y = H0_Y    + 1.25
S3_Y    = SPROJ_Y + 1.05
TENS_Y  = S3_Y    + 1.45
S2_Y    = TENS_Y  + 1.05
WIN_Y   = S2_Y    + 1.15
NORM_Y  = WIN_Y   + 1.25
TEMP_SPLIT_Y = NORM_Y  + 1.20
SPLIT_Y      = TEMP_SPLIT_Y + 1.10
WELLS_Y      = SPLIT_Y + 1.10
S1_Y    = WELLS_Y + 0.72
TITLE_Y = S1_Y    + 0.72

INPUT_Y  = ENC_BOT - 0.68
S6_Y     = INPUT_Y  - 0.68
HEAD_Y   = S6_Y     - 0.78
FINAL_PRED_Y = HEAD_Y       - 1.10   # final predictions on test set
DENORM_Y     = FINAL_PRED_Y - 1.05   # denormalisation step
OUT_Y        = DENORM_Y     - 1.05   # forecast GWL
PRED_Y       = OUT_Y        - 1.00   # pred.parquet
LOSS_Y   = HEAD_Y   - 1.80   # MSE Loss (right column, below y_future at HEAD_Y)

GP_EXTRA       = 6.5    # extra data-units below y=0 for section ⑦
S7_Y           = OUT_Y         - 1.30
WATT_Y         = S7_Y          - 0.80   # well attributes box, below section ⑦ separator
GP_PRETRAIN_Y  = S7_Y          - 1.60   # GP kernel pretraining box
GP_KERN_Y      = S7_Y          - 3.00   # GP kernel box
GP_OUT_Y       = GP_KERN_Y     - 1.50   # GP posterior output
EVAL_Y         = GP_OUT_Y      - 1.30   # evaluation box

W = 31.5
H = TITLE_Y + 0.55

fig, ax = plt.subplots(figsize=(W, H + GP_EXTRA))
ax.set_xlim(0, W); ax.set_ylim(-GP_EXTRA, H); ax.axis("off")
fig.patch.set_facecolor("white")

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
ax.text(W/2, TITLE_Y,
        "GRU + GP Decoupled Training — Groundwater Level Forecasting Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#0f172a")

# ─────────────────────────────────────────────────────────────────────────────
# ① DATA PREPARATION
sec(ax, W, S1_Y, "①  Data Preparation")
DW = 17.0
rbox(ax, W/2, WELLS_Y, DW, 0.70,
     "1 040 wells  ·  weekly GWL  ·  5 weather covariates  ·  27 static features",
     C_DATA, C_DATA_B, fs=11, bold=True)
rbox(ax, W/2, SPLIT_Y, DW, 0.70,
     "Spatial split   ·   spatial_train  /  spatial_test   (held out entirely from GRU training and evaluation)",
     C_OP, C_OP_B, fs=10, bold=True)
rbox(ax, W/2, TEMP_SPLIT_Y, DW, 0.70,
     "Temporal split   ·   train (≤ 2016)  /  val (2016–2020)  /  test (> 2020)   ·   applied to spatial_train wells",
     C_OP, C_OP_B, fs=10, bold=True)
rbox(ax, W/2, NORM_Y, DW, 0.70,
     "Normalisation:  per-well z-score on GWL  ·  global StandardScaler on weather & static features",
     C_OP, C_OP_B, fs=10, bold=True)
rbox(ax, W/2, WIN_Y, DW, 0.70,
     "Sliding window builder   stride = 1 week  ·  52-step past window  +  16-step future window + static features per well",
     C_OP, C_OP_B, fs=10, bold=True)
arr(ax, W/2, WELLS_Y      - 0.35 - G, W/2, SPLIT_Y      + 0.35 + G)
arr(ax, W/2, SPLIT_Y      - 0.35 - G, W/2, TEMP_SPLIT_Y + 0.35 + G)
arr(ax, W/2, TEMP_SPLIT_Y - 0.35 - G, W/2, NORM_Y       + 0.44 + G)
arr(ax, W/2, NORM_Y       - 0.44 - G, W/2, WIN_Y        + 0.35 + G)

# ─────────────────────────────────────────────────────────────────────────────
# ② MODEL INPUTS
sec(ax, W, S2_Y, "②  Model Inputs  (batch size B)")
TH = 0.95; TW = 6.0
CXS = [4.0, 10.5, 20.5, 28.0]
T_TEXTS = [
    "x_past   (B, 52, 6)\ngwl + 5 weather columns",
    "x_future   (B, 16, 5)\nscaled future weather",
    "x_static   (B, 27)\nscaled static well attributes",
    "y_future   (B, 16)   ★\nnorm. GWL targets",
]
T_COLORS = [(C_DATA, C_DATA_B), (C_DATA, C_DATA_B), (C_DATA, C_DATA_B), (C_OUT, C_OUT_B)]
for cx, txt, (fc, ec) in zip(CXS, T_TEXTS, T_COLORS):
    rbox(ax, cx, TENS_Y, TW, TH, txt, fc, ec, fs=9.5, bold=True)
arr(ax, W/2, WIN_Y - 0.35 - G, CXS[0], TENS_Y + TH/2 + G, rad=0)
arr(ax, W/2, WIN_Y - 0.35 - G, CXS[1], TENS_Y + TH/2 + G, rad=0)
arr(ax, W/2, WIN_Y - 0.35 - G, CXS[2], TENS_Y + TH/2 + G, rad=0)
arr(ax, W/2, WIN_Y - 0.35 - G, CXS[3], TENS_Y + TH/2 + G, rad=0)

# ─────────────────────────────────────────────────────────────────────────────
# ③ STATIC PROJECTION
sec(ax, W, S3_Y, "③  Static Projection  →  Encoder Initial State  h₀")
rbox(ax, ENC_CX, SPROJ_Y, 8.5, 0.70,
     "static_proj     Linear( 27  →  256 × 4 = 1 024 )",
     C_OP, C_OP_B, fs=9, bold=True)
rbox(ax, ENC_CX, H0_Y, 8.5, 0.70,
     "h₀  —  one 256-dim vector per encoder layer\nconditions each layer on well identity before seeing any data",
     C_H0_FC, C_H0_EC, fs=8.5, bold=True)
arr(ax, CXS[2], TENS_Y - TH/2 - G, ENC_CX + 4.25 - G, SPROJ_Y, c=C_H0_EC, lw=1.3, rad=-0.12)
arr(ax, ENC_CX, SPROJ_Y - 0.35 - G, ENC_CX, H0_Y + 0.41 + G, c=C_H0_EC)

# h₀ → all 4 encoder layers via spine on the left
SPINE_X = ENC_L - 0.60
ax.annotate("", xy=(SPINE_X, LY[N_LAYERS-1] + 0.2),
            xytext=(ENC_CX - 4.25, H0_Y - 0.41),
            arrowprops=dict(arrowstyle="-", color=C_H0_EC, lw=1.3,
                            linestyle="dashed"), zorder=4)
ax.plot([SPINE_X, SPINE_X], [LY[0], LY[N_LAYERS-1] + 0.2],
        color=C_H0_EC, lw=1.3, linestyle="dashed", zorder=4)
for i in range(N_LAYERS):
    ax.annotate("", xy=(ENC_L - 0.25, LY[i]),
                xytext=(SPINE_X, LY[i]),
                arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.32",
                                color=C_H0_EC, lw=1.1, linestyle="dashed"), zorder=4)

# ─────────────────────────────────────────────────────────────────────────────
# ④ ENCODER  +  ⑤ DECODER
sec(ax, W, S45_Y, "④  Encoder GRU   (52 steps)")
ax.text(DEC_L, S45_Y, "⑤  Decoder GRU   (16 steps)",
        fontsize=10, fontweight="bold", color="#1e293b", va="center", ha="left", zorder=6)
ax.text(ENC_L, S45_Y - 0.12, "4 layers × 256 nodes  ·  dropout 0.2",
        ha="left", va="top", fontsize=8.8, fontweight="bold", color=C_GRU_B)
ax.text(DEC_L, S45_Y - 0.12, "4 layers × 256 nodes  ·  dropout 0.2",
        ha="left", va="top", fontsize=8.8, fontweight="bold", color=C_GRU_B)

MID_X = (ENC_R + 0.12 + DEC_L - 0.42) / 2

for i, y in enumerate(LY):
    lnum = i + 1

    # background row boxes
    rbox(ax, ENC_CX, y, ENC_W, LAYER_H, "", C_GRU, C_GRU_B, z=2, lw=1.1)
    rbox(ax, DEC_CX, y, DEC_W, LAYER_H, "", C_GRU, C_GRU_B, z=2, lw=1.1)

    # layer labels — between h₀/h_enc arrowheads and respective box edges
    ax.text(ENC_L - 0.11, y, f"L{lnum}", ha="center", va="center",
            fontsize=8, fontweight="bold", color=C_GRU_B, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.text(DEC_L - 0.18, y, f"L{lnum}", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C_GRU_B, zorder=7,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    # ── ENCODER cells ──────────────────────────────────────────────────────
    rbox(ax, EH0_CX,  y, CW, CH, f"h₀{lnum}", C_H0_FC, C_H0_EC,   fs=7.0, bold=True, z=4, lw=1.8)
    rbox(ax, ET1_CX,  y, CW, CH,  "t=1",       C_STEP,  C_STEP_EC, fs=7.5, bold=True, z=4, lw=0.9)
    rbox(ax, ET2_CX,  y, CW, CH,  "t=2",       C_STEP,  C_STEP_EC, fs=7.5, bold=True, z=4, lw=0.9)
    ax.text((ET2_CX + ET52_CX)/2, y, "···",
            ha="center", va="center", fontsize=12, fontweight="bold", color=C_STEP_EC, zorder=5)
    rbox(ax, ET52_CX, y, CW, CH, "t=52", C_HIDDEN, C_HIDDEN_B, fs=7.0, bold=True, z=4, lw=1.8)

    GC = 0.0  # no gap for within-cell arrows — needed for larger arrowheads to fit
    htick(ax, EH0_CX + CW/2 + GC, ET1_CX  - CW/2 - GC, y, c=C_STEP_EC)
    htick(ax, ET1_CX  + CW/2 + GC, ET2_CX  - CW/2 - GC, y, c=C_STEP_EC)
    htick(ax, ET2_CX  + CW/2, ET2_CX  + 0.52, y, c=C_STEP_EC, hw=0, hl=0)
    htick(ax, ET52_CX - CW/2 - 0.52, ET52_CX - CW/2 - 0.08, y, c=C_STEP_EC)

    # ── DECODER cells ──────────────────────────────────────────────────────
    rbox(ax, DH0_CX,  y, CW, CH, f"h_enc{lnum}", C_HIDDEN, C_HIDDEN_B, fs=6.5, bold=True, z=4, lw=1.8)
    rbox(ax, DT1_CX,  y, CW, CH,  "t=1",          C_STEP,  C_STEP_EC,  fs=7.5, bold=True, z=4, lw=0.9)
    rbox(ax, DT2_CX,  y, CW, CH,  "t=2",          C_STEP,  C_STEP_EC,  fs=7.5, bold=True, z=4, lw=0.9)
    ax.text((DT2_CX + DT16_CX)/2, y, "···",
            ha="center", va="center", fontsize=12, fontweight="bold", color=C_STEP_EC, zorder=5)
    if i == N_LAYERS - 1:
        rbox(ax, DT16_CX, y, CW, CH, "t=16", C_HIDDEN, C_HIDDEN_B, fs=7.0, bold=True, z=4, lw=1.8)
    else:
        rbox(ax, DT16_CX, y, CW, CH, "t=16", "#fffbeb", C_OP_B,     fs=7.0, bold=True, z=4, lw=0.9)

    htick(ax, DH0_CX  + CW/2 + GC, DT1_CX  - CW/2 - GC, y, c=C_STEP_EC)
    htick(ax, DT1_CX  + CW/2 + GC, DT2_CX  - CW/2 - GC, y, c=C_STEP_EC)
    htick(ax, DT2_CX  + CW/2, DT2_CX  + 0.52, y, c=C_STEP_EC, hw=0, hl=0)
    htick(ax, DT16_CX - CW/2 - 0.52, DT16_CX - CW/2 - 0.08, y, c=C_STEP_EC)

    # ── VERTICAL inter-layer arrows
    if i < N_LAYERS - 1:
        y_next  = LY[i + 1]
        gap_bot = y      + LAYER_H/2 + VGAP
        gap_top = y_next - LAYER_H/2 - VGAP
        for xp in (ET1_CX, ET2_CX, ET52_CX):
            vtick(ax, xp, gap_bot, gap_top, c=C_GRU_B)
        for xp in (DT1_CX, DT2_CX, DT16_CX):
            vtick(ax, xp, gap_bot, gap_top, c=C_GRU_B)

    # ── h_enc TRANSFER ARROWS — fully outside both blocks
    arr_x0 = ENC_R + 0.12
    arr_x1 = DEC_L - 0.42
    arr(ax, arr_x0, y, arr_x1, y, c=C_BRIDGE, lw=2.2, hw=0.65, hl=0.48)
    ax.text(MID_X, y + 0.26, f"h_enc{lnum}",
            ha="center", va="bottom", fontsize=7.5,
            color=C_BRIDGE, fontweight="bold", zorder=7)

# x_past / x_future → encoder / decoder L1 via separate bus routing
# Two independent buses at different y-levels; vtick tails start AT the bus for visual continuity.
ENC_BUS_Y = LY[0] - LAYER_H/2 - 0.40   # encoder bus (higher)
DEC_BUS_Y = LY[0] - LAYER_H/2 - 0.78   # decoder bus (lower, clearly separate)
CELL_BOT  = LY[0] - LAYER_H/2 - G       # bottom edge of L1 background box (arrowhead target)

# encoder: vertical trunk from x_past to enc bus, horizontal bus, arrowheads up into cells
ax.plot([CXS[0], CXS[0]], [TENS_Y - TH/2, ENC_BUS_Y], color=C_DATA_B, lw=1.0, zorder=5)
ax.plot([ET1_CX - 0.08, ET52_CX + 0.08], [ENC_BUS_Y, ENC_BUS_Y], color=C_DATA_B, lw=1.0, zorder=5)
for xt in [ET1_CX, ET2_CX, ET52_CX]:
    vtick(ax, xt, ENC_BUS_Y, CELL_BOT, c=C_DATA_B, lw=1.0, hw=0.32, hl=0.22)

# decoder: L-shaped trunk (x_future down then right to decoder region), decoder bus, arrowheads
ax.plot([CXS[1], CXS[1], DT16_CX + 0.08], [TENS_Y - TH/2, DEC_BUS_Y, DEC_BUS_Y],
        color=C_DATA_B, lw=1.0, zorder=5)
for xt in [DT1_CX, DT2_CX, DT16_CX]:
    vtick(ax, xt, DEC_BUS_Y, CELL_BOT, c=C_DATA_B, lw=1.0, hw=0.32, hl=0.22)

# ─────────────────────────────────────────────────────────────────────────────
# ⑥ OUTPUT HEAD & PREDICTIONS
sec(ax, W, S6_Y, "⑥  Output Head,  Denormalisation  &  Saved Predictions")
OW = 14.0; OCX = 13.0

# ── Left column: linear head → denorm → forecast GWL → saved ─────────────────
rbox(ax, OCX, HEAD_Y, OW, 0.62,
     "Linear( 256 → 1 )  ·  applied at each of the 16 decoder timesteps  ·  shared weights",
     C_OP, C_OP_B, fs=9.5, bold=True)
rbox(ax, OCX, FINAL_PRED_Y, OW, 0.62,
     "final model  ·  final predictions on temporal test set",
     C_DATA, C_DATA_B, fs=9.5, bold=True)
rbox(ax, OCX, DENORM_Y, OW, 0.60,
     "denormalising predictions to absolute values",
     C_OP, C_OP_B, fs=9.5, bold=True)
rbox(ax, OCX, OUT_Y, OW, 0.72,
     "forecast GWL in metres a.m.s.l.\none value per well per horizon  (16 horizons)",
     C_OUT, C_OUT_B, fs=10, bold=True)
# decoder L4 t=16 → top of linear head box (arrow starts inside the cell)
arr(ax, DT16_CX, LY[N_LAYERS-1] - CH/2,
    OCX + OW/2, HEAD_Y + 0.31 + G,
    c=C_HIDDEN_B, lw=2.0, hw=0.60, hl=0.45, rad=0.08)

# left-column vertical arrows
arr(ax, OCX, HEAD_Y       - 0.31 - G, OCX, FINAL_PRED_Y + 0.31 + G)
arr(ax, OCX, FINAL_PRED_Y - 0.31 - G, OCX, DENORM_Y     + 0.30 + G)
arr(ax, OCX, DENORM_Y     - 0.30 - G, OCX, OUT_Y        + 0.36 + G)

# temporal test → final predictions (dashed arrow from right side of temporal split box)
ax.annotate("", xy=(OCX + OW/2, FINAL_PRED_Y),
            xytext=(W/2 + DW/2, TEMP_SPLIT_Y - 0.35),
            arrowprops=dict(arrowstyle="-|>,head_width=0.60,head_length=0.45",
                            color=C_ARROW, lw=1.0, linestyle="dashed",
                            connectionstyle="arc3,rad=-0.25"),
            zorder=5)

# ── Right column: MSE Loss ────────────────────────────────────────────────────
rbox(ax, W - 4.5, LOSS_Y, 5.5, 0.90,
     "MSE Loss\n= mean( (ŷ − y_future)² )\n→ backprop  →  optimizer.step()\n→ update weights",
     C_LOSS, C_LOSS_B, fs=8.0, bold=True)

# linear head → MSE Loss
arr(ax, OCX + OW/2 + G, HEAD_Y,
    W - 4.5 - 5.5/2 - G, LOSS_Y,
    c=C_OP_B, lw=1.4, rad=0.22)
# y_future (section ②) → MSE Loss (straight)
arr(ax, CXS[3], TENS_Y - TH/2 - G, W - 4.5, LOSS_Y + 0.45 + G, c=C_OUT_B, lw=1.0, rad=0.0)

# ─────────────────────────────────────────────────────────────────────────────
# ⑦ GP SPATIAL INTERPOLATION
sec(ax, W, S7_Y, "⑦  GP Spatial Interpolation")

CX_GP   = 13.0   # main-column x (aligns with section ⑥)
GW_GP   = 17.0   # main-column box width
CX_FEAT = 26.5   # feature-panel x (right side)
W_FEAT  = 8.5    # feature-panel width
H_KERN  = 1.10   # GP kernel box height

# ── Well attributes box (below section separator, right column) ───────────────
rbox(ax, CX_FEAT, WATT_Y, W_FEAT, 0.80,
     "3 static features  +  coordinates\nsiwa_verweilzeit_j  ·  hydroraum  ·  gw_gespannt\nx_coord  ·  y_coord",
     C_DATA, C_DATA_B, fs=9.5, bold=True)

# ── GP kernel pretraining box ─────────────────────────────────────────────────
rbox(ax, CX_GP, GP_PRETRAIN_Y, GW_GP, 0.62,
     "GP kernel pretraining",
     C_GP, C_GP_B, fs=9.5, bold=True)

# ── GP kernel box (main column) ───────────────────────────────────────────────
rbox(ax, CX_GP, GP_KERN_Y, GW_GP, H_KERN,
     "Spatial GP  —  predicts GWL at spatial_test wells\n"
     "fitted separately per forecast horizon  (1 … 16 weeks ahead)",
     C_GP, C_GP_B, fs=9.5, bold=True)

# ── GP posterior output ───────────────────────────────────────────────────────
rbox(ax, CX_GP, GP_OUT_Y, GW_GP, 0.62,
     "GP posterior  ·  predicted GWL at 52 spatial_test wells",
     C_DATA, C_DATA_B, fs=10, bold=True)

# ── Arrows ────────────────────────────────────────────────────────────────────
# GRU forecast GWL → GP kernel pretraining
arr(ax, OCX, OUT_Y - 0.36 - G, CX_GP, GP_PRETRAIN_Y + 0.31 + G)
# GP kernel pretraining → GP kernel
arr(ax, CX_GP, GP_PRETRAIN_Y - 0.31 - G, CX_GP, GP_KERN_Y + H_KERN/2 + G)
# well attributes → GP kernel (diagonal from right side)
arr(ax, CX_FEAT, WATT_Y - 0.40 - G, CX_GP + GW_GP/2, GP_KERN_Y,
    c=C_DATA_B, lw=1.3, rad=-0.15)
# GP kernel → output
arr(ax, CX_GP, GP_KERN_Y - H_KERN/2 - G, CX_GP, GP_OUT_Y + 0.31 + G)
# WELLS box → well attributes (27 static features, 3 used by GP)
arr(ax, W/2 + DW/2, WELLS_Y, CX_FEAT + 3.5, WATT_Y + 0.40 + G, c=C_DATA_B, lw=1.1, rad=-0.3)
# spatial split → GP posterior (52 holdout wells are what the GP predicts at)
ax.annotate("", xy=(CX_GP - GW_GP/2 + 1.0, GP_KERN_Y),
            xytext=(W/2 - DW/2 + 1.0, SPLIT_Y),
            arrowprops=dict(arrowstyle="-|>,head_width=0.60,head_length=0.45",
                           color=C_ARROW, lw=1.0, linestyle="dashed",
                           connectionstyle="arc3,rad=0.5"),
            zorder=5)

# ── Evaluation box ────────────────────────────────────────────────────────────
rbox(ax, CX_GP, EVAL_Y, GW_GP, 0.62,
     "Evaluation against true observations at spatial_test wells",
     C_OP, C_OP_B, fs=10, bold=True)
arr(ax, CX_GP, GP_OUT_Y - 0.31 - G, CX_GP, EVAL_Y + 0.31 + G)

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
items = [
    (C_DATA,    C_DATA_B,    "Data"),
    (C_OP,      C_OP_B,      "Processing"),
    (C_GRU,     C_GRU_B,     "GRU component"),
    (C_HIDDEN,  C_HIDDEN_B,  "Transferred hidden state"),
    (C_GP,      C_GP_B,      "GP component"),
    (C_LOSS,    C_LOSS_B,    "Loss / objective"),
]
lx = W - 5.5; ly = TITLE_Y - 0.15
ax.text(lx + 0.25, ly + 0.40, "Legend", fontsize=9, fontweight="bold",
        ha="center", va="center")
for k, (fc, ec, lab) in enumerate(items):
    r = FancyBboxPatch((lx, ly - k*0.62 - 0.22), 0.44, 0.36,
                       boxstyle="round,pad=0.03",
                       facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=7)
    ax.add_patch(r)
    ax.text(lx + 0.60, ly - k*0.62, lab, fontsize=8, va="center", zorder=7)

# ─────────────────────────────────────────────────────────────────────────────
OUT = "reports/figures/decoupled_flowchart.svg"
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")
plt.close(fig)
