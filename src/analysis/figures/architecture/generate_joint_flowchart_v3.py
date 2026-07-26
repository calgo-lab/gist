from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

C_DATA    = "#DBEAFE"; C_DATA_B    = "#2563EB"
C_OP      = "#FEF9C3"; C_OP_B      = "#CA8A04"
C_GRU     = "#DCFCE7"; C_GRU_B     = "#16A34A"
C_HIDDEN  = "#FFE4E6"; C_HIDDEN_B  = "#DC2626"
C_H0_FC   = "#FFE4E6"; C_H0_EC     = "#DC2626"
C_OUT     = C_DATA;    C_OUT_B     = C_DATA_B
C_STEP    = C_GRU;     C_STEP_EC   = C_GRU_B
C_GP      = "#F3E8FF"; C_GP_B      = "#7C3AED"
C_LOSS    = "#FFF7ED"; C_LOSS_B    = "#EA580C"
C_JOINT_B = "#FF6600"
C_ARROW   = "#475569"
C_BRIDGE  = "#DC2626"

FS = 2.5
G  = 0.07
FB = 9

def rbox(ax, cx, cy, w, h, text, fc, ec, fs=FB, bold=False, z=3, lw=1.4):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.04", facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs*FS,
            fontweight="bold" if bold else "normal", multialignment="center", zorder=z+1)

def arr(ax, x0, y0, x1, y1, c=C_ARROW, lw=3.5, rad=0.0, hw=0.90, hl=0.65):
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
        arrowprops=dict(arrowstyle=f"-|>,head_width={hw},head_length={hl}",
                        color=c, lw=lw, connectionstyle=f"arc3,rad={rad}"), zorder=5)

def htick(ax, x0, x1, y, c=C_ARROW, lw=2.5, hw=0.45, hl=0.30):
    ax.annotate("", xy=(x1,y), xytext=(x0,y),
        arrowprops=dict(arrowstyle=f"-|>,head_width={hw},head_length={hl}",
                        color=c, lw=lw, connectionstyle="arc3,rad=0"), zorder=5)

def vtick(ax, x, y0, y1, c=C_ARROW, lw=3.0, hw=0.90, hl=0.65):
    ax.annotate("", xy=(x,y1), xytext=(x,y0),
        arrowprops=dict(arrowstyle=f"-|>,head_width={hw},head_length={hl}",
                        color=c, lw=lw, connectionstyle="arc3,rad=0"), zorder=6)

def sec(ax, W, y, text):
    ax.text(0.25, y, text, fontsize=10*FS, fontweight="bold", color="#1e293b",
            va="center", ha="left", zorder=6)
    ax.plot([0.15, W-0.15], [y-0.35, y-0.35], color="#cbd5e1", lw=0.8)

N_LAYERS   = 4
LAYER_H    = 1.15
LAYER_STEP = 1.65
L1_Y = 8.0
LY   = [L1_Y + i*LAYER_STEP for i in range(N_LAYERS)]

ENC_CX = 7.0;  ENC_W = 11.2
DEC_CX = 24.4; DEC_W = 11.2
ENC_L = ENC_CX - ENC_W/2
ENC_R = ENC_CX + ENC_W/2
DEC_L = DEC_CX - DEC_W/2
DEC_R = DEC_CX + DEC_W/2

CW = 1.05; CH = 0.90
EH0_CX  = ENC_L + 0.55
ET1_CX  = EH0_CX + CW + 0.38
ET2_CX  = ET1_CX + CW + 0.35
ET52_CX = ENC_R  - 0.55
DH0_CX  = DEC_L  + 0.55
DT1_CX  = DH0_CX + CW + 0.38
DT2_CX  = DT1_CX + CW + 0.35
DT16_CX = DEC_R  - 0.55
VGAP = 0.10

ENC_TOP = LY[N_LAYERS-1] + LAYER_H/2
ENC_BOT = LY[0]          - LAYER_H/2

S45_Y        = ENC_TOP + 1.10
H0_Y         = S45_Y   + 1.65
SPROJ_Y      = H0_Y    + 1.35
S3_Y         = SPROJ_Y + 1.10
TENS_Y       = S3_Y    + 1.55
S2_Y         = TENS_Y  + 1.10
WIN_Y        = S2_Y    + 1.20
NORM_Y       = WIN_Y   + 1.45
TEMP_SPLIT_Y = NORM_Y  + 1.40
SPLIT_Y      = TEMP_SPLIT_Y + 1.30
WELLS_Y      = SPLIT_Y + 1.30
S1_Y         = WELLS_Y + 0.80
TITLE_Y      = S1_Y    + 1.20

S6_Y         = ENC_BOT - 1.50
HEAD_Y       = S6_Y    - 1.00
FINAL_PRED_Y = HEAD_Y  - 1.50
DENORM_Y     = FINAL_PRED_Y - 1.30
OUT_Y        = DENORM_Y     - 1.30
LOSS_Y       = HEAD_Y  - 2.20

LOSS_CX = 26.5; LOSS_W = 9.0; LOSS_H = 3.0

HB = 0.85; HT = 1.10; HG = 0.80; H_KERN = 1.30

CX_GP  = 13.0; GW_GP  = 17.0
CX_FEAT = 26.5; W_FEAT = 9.0

S7_Y          = OUT_Y  - 1.30
WATT_Y        = S7_Y   - 0.90
GP_PRETRAIN_Y = S7_Y   - 1.80
GP_KERN_Y     = S7_Y   - 3.30
GP_OUT_Y      = GP_KERN_Y - 1.60
EVAL_Y        = GP_OUT_Y  - 1.40

S8_Y       = EVAL_Y   - 0.90
L_GP_Y     = S8_Y     - 1.05
L_JOINT_Y  = L_GP_Y   - 1.50

W = 31.5
H = TITLE_Y + 0.55
Y_BOTTOM = L_JOINT_Y - 1.30/2 - 1.10

fig, ax = plt.subplots(figsize=(W, H - Y_BOTTOM))
ax.set_xlim(0, W); ax.set_ylim(Y_BOTTOM, H); ax.axis("off")
fig.patch.set_facecolor("white")

sec(ax, W, S1_Y, "①  Data Preparation")
DW = 17.0
rbox(ax, W/2, WELLS_Y, DW, HB,
     "1 040 wells  ·  weekly GWL  ·  5 weather covariates  ·  35 static features",
     C_DATA, C_DATA_B, bold=True)
rbox(ax, W/2, SPLIT_Y, DW, HB,
     "Three-way spatial split:   train  /  val  /  test    ·    test held out entirely",
     C_OP, C_JOINT_B, bold=True, lw=7.5)
rbox(ax, W/2, TEMP_SPLIT_Y, DW, HB,
     "Temporal split:   train  /  val  /  test    ·    on spatial train wells",
     C_OP, C_OP_B, bold=True)
rbox(ax, W/2, NORM_Y, DW, HB,
     "Normalization:   per-well on GWL   ·   global on input features",
     C_OP, C_OP_B, bold=True)
rbox(ax, W/2, WIN_Y, DW, HB,
     "Sliding windows:   52-week input  ·  16-week output  ·  static feat. per well",
     C_OP, C_OP_B, bold=True)
arr(ax, W/2, WELLS_Y - HB/2 - G, W/2, SPLIT_Y      + HB/2 + G)
arr(ax, W/2, SPLIT_Y - HB/2 - G, W/2, TEMP_SPLIT_Y + HB/2 + G)
arr(ax, W/2, TEMP_SPLIT_Y - HB/2 - G, W/2, NORM_Y  + HB/2 + G)
arr(ax, W/2, NORM_Y - HB/2 - G, W/2, WIN_Y         + HB/2 + G)

sec(ax, W, S2_Y, "②  Model Inputs  (batch size B)")
TH = 1.40
TWS = [7.2, 7.2, 6.5, 6.5]
CXS = [5.2, 12.7, 20.5, 27.5]
T_TEXTS = [
    "x_past   (B, 52, 6)\ngwl + weather covariates",
    "x_future   (B, 16, 5)\nfuture weather covariates",
    "x_static   (B, 35)\nstatic well attributes",
    "y_future   (B, 16)   ★\nnorm. GWL targets",
]
T_COLORS = [(C_DATA, C_DATA_B)] * 3 + [(C_OUT, C_OUT_B)]
for cx, txt, (fc, ec), tw in zip(CXS, T_TEXTS, T_COLORS, TWS):
    rbox(ax, cx, TENS_Y, tw, TH, txt, fc, ec, fs=10.5, bold=True)
for cx in CXS:
    arr(ax, W/2, WIN_Y - HB/2 - G, cx, TENS_Y + TH/2 + G, rad=0)

sec(ax, W, S3_Y, "③  Static Projection  →  Encoder Initial State  h₀")
SPROJ_W = 12.0
rbox(ax, ENC_CX, SPROJ_Y, SPROJ_W, HB,
     "static_proj     Linear( 35  →  256 × 4 = 1 024 )",
     C_OP, C_OP_B, bold=True)
rbox(ax, ENC_CX, H0_Y, SPROJ_W, HT,
     "h₀  —  one 256-dim vector per encoder layer\ninitializes each layer with well identity",
     C_H0_FC, C_H0_EC, bold=True)
arr(ax, CXS[2], TENS_Y-TH/2-G, ENC_CX+SPROJ_W/2-G, SPROJ_Y, c=C_H0_EC, lw=3.0, rad=-0.12)
arr(ax, ENC_CX, SPROJ_Y-HB/2-G, ENC_CX, H0_Y+HT/2+G, c=C_H0_EC)
SPINE_X = ENC_L - 0.65
ax.annotate("", xy=(SPINE_X, LY[N_LAYERS-1]+0.25), xytext=(ENC_CX-4.5, H0_Y-HT/2),
    arrowprops=dict(arrowstyle="-", color=C_H0_EC, lw=2.5, linestyle="dashed"), zorder=4)
ax.plot([SPINE_X, SPINE_X], [LY[0], LY[N_LAYERS-1]+0.25],
        color=C_H0_EC, lw=2.5, linestyle="dashed", zorder=4)
for i in range(N_LAYERS):
    ax.annotate("", xy=(ENC_L-0.25, LY[i]), xytext=(SPINE_X, LY[i]),
        arrowprops=dict(arrowstyle="-|>,head_width=0.55,head_length=0.40",
                        color=C_H0_EC, lw=2.5, linestyle="dashed"), zorder=4)

sec(ax, W, S45_Y, "④  Encoder GRU   (52 steps)")
ax.text(DEC_L, S45_Y, "⑤  Decoder GRU   (16 steps)",
        fontsize=10*FS, fontweight="bold", color="#1e293b", va="center", ha="left", zorder=6)
ax.text(ENC_L, S45_Y-0.42, "4 layers × 256 nodes  ·  dropout 0.2",
        ha="left", va="top", fontsize=8.5*FS, fontweight="bold", color=C_GRU_B)
ax.text(DEC_L, S45_Y-0.42, "4 layers × 256 nodes  ·  dropout 0.2",
        ha="left", va="top", fontsize=8.5*FS, fontweight="bold", color=C_GRU_B)

MID_X = (ENC_R + 0.12 + DEC_L - 0.45) / 2

for i, y in enumerate(LY):
    lnum = i + 1
    rbox(ax, ENC_CX, y, ENC_W, LAYER_H, "", C_GRU, C_GRU_B, z=2, lw=1.3)
    rbox(ax, DEC_CX, y, DEC_W, LAYER_H, "", C_GRU, C_GRU_B, z=2, lw=1.3)
    ax.text(ENC_L-0.50, y, f"L{lnum}", ha="center", va="center",
            fontsize=8*FS, fontweight="bold", color=C_GRU_B, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.text(DEC_L-0.55, y, f"L{lnum}", ha="center", va="center",
            fontsize=8*FS, fontweight="bold", color=C_GRU_B, zorder=7,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    rbox(ax, EH0_CX,  y, CW, CH, f"h₀{lnum}",    C_H0_FC,  C_H0_EC,   fs=9.0, bold=True, z=4, lw=2.0)
    rbox(ax, ET1_CX,  y, CW, CH, "t=1",           C_STEP,   C_STEP_EC, fs=9.5, bold=True, z=4, lw=1.1)
    rbox(ax, ET2_CX,  y, CW, CH, "t=2",           C_STEP,   C_STEP_EC, fs=9.5, bold=True, z=4, lw=1.1)
    ax.text((ET2_CX+ET52_CX)/2, y, "···", ha="center", va="center",
            fontsize=13*FS, fontweight="bold", color=C_STEP_EC, zorder=5)
    rbox(ax, ET52_CX, y, CW, CH, "t=52",          C_HIDDEN, C_HIDDEN_B, fs=9.0, bold=True, z=4, lw=2.0)
    htick(ax, EH0_CX+CW/2, ET1_CX-CW/2,  y, c=C_STEP_EC)
    htick(ax, ET1_CX+CW/2, ET2_CX-CW/2,  y, c=C_STEP_EC)
    htick(ax, ET2_CX+CW/2, ET2_CX+0.55,  y, c=C_STEP_EC, hw=0, hl=0)
    htick(ax, ET52_CX-CW/2-0.55, ET52_CX-CW/2-0.08, y, c=C_STEP_EC)

    rbox(ax, DH0_CX,  y, CW, CH, f"h_enc{lnum}", C_HIDDEN, C_HIDDEN_B, fs=6.5, bold=True, z=4, lw=2.0)
    rbox(ax, DT1_CX,  y, CW, CH, "t=1",          C_STEP,   C_STEP_EC,  fs=9.5, bold=True, z=4, lw=1.1)
    rbox(ax, DT2_CX,  y, CW, CH, "t=2",          C_STEP,   C_STEP_EC,  fs=9.5, bold=True, z=4, lw=1.1)
    ax.text((DT2_CX+DT16_CX)/2, y, "···", ha="center", va="center",
            fontsize=13*FS, fontweight="bold", color=C_STEP_EC, zorder=5)
    if i == N_LAYERS - 1:
        rbox(ax, DT16_CX, y, CW, CH, "t=16", C_HIDDEN, C_HIDDEN_B, fs=9.0, bold=True, z=4, lw=2.0)
    else:
        rbox(ax, DT16_CX, y, CW, CH, "t=16", "#fffbeb", C_OP_B,     fs=9.0, bold=True, z=4, lw=1.1)
    htick(ax, DH0_CX+CW/2,  DT1_CX-CW/2,  y, c=C_STEP_EC)
    htick(ax, DT1_CX+CW/2,  DT2_CX-CW/2,  y, c=C_STEP_EC)
    htick(ax, DT2_CX+CW/2,  DT2_CX+0.55,  y, c=C_STEP_EC, hw=0, hl=0)
    htick(ax, DT16_CX-CW/2-0.55, DT16_CX-CW/2-0.08, y, c=C_STEP_EC)

    if i < N_LAYERS - 1:
        gap_bot = y + LAYER_H/2 + VGAP
        gap_top = LY[i+1] - LAYER_H/2 - VGAP
        for xp in (ET1_CX, ET2_CX, ET52_CX):
            vtick(ax, xp, gap_bot, gap_top, c=C_GRU_B)
        for xp in (DT1_CX, DT2_CX, DT16_CX):
            vtick(ax, xp, gap_bot, gap_top, c=C_GRU_B)

    _tip = DEC_L - 1.1
    ax.plot([ENC_R+0.12, _tip-0.22], [y, y], color=C_BRIDGE, lw=3.5, solid_capstyle='butt', zorder=5)
    ax.add_patch(Polygon([[_tip-0.22, y-0.13], [_tip-0.22, y+0.13], [_tip, y]],
                         closed=True, fc=C_BRIDGE, ec=C_BRIDGE, zorder=8))
    ax.text(MID_X, y+0.30, f"h_enc{lnum}", ha="center", va="bottom",
            fontsize=7.5*FS, color=C_BRIDGE, fontweight="bold", zorder=7)

ENC_BUS_Y = LY[0] - LAYER_H/2 - 0.45
DEC_BUS_Y = LY[0] - LAYER_H/2 - 0.88
CELL_BOT  = LY[0] - LAYER_H/2 - G

ax.plot([5.55, 5.55], [TENS_Y-TH/2, ENC_BUS_Y], color=C_DATA_B, lw=3.0, zorder=5)
ax.plot([ET1_CX, ET52_CX+0.10], [ENC_BUS_Y, ENC_BUS_Y], color=C_DATA_B, lw=3.0, zorder=5)
for xt in [ET1_CX, ET2_CX, ET52_CX]:
    vtick(ax, xt, ENC_BUS_Y, CELL_BOT, c=C_DATA_B, lw=3.0, hw=0.55, hl=0.40)

XFUT_DROP_X = CXS[1] + 1.3
ax.plot([XFUT_DROP_X, XFUT_DROP_X, DT16_CX+0.10], [TENS_Y-TH/2, DEC_BUS_Y, DEC_BUS_Y],
        color=C_DATA_B, lw=3.0, zorder=5)
for xt in [DT1_CX, DT2_CX, DT16_CX]:
    vtick(ax, xt, DEC_BUS_Y, CELL_BOT, c=C_DATA_B, lw=3.0, hw=0.55, hl=0.40)

sec(ax, W, S6_Y, "⑥  Output Head,  Denormalization  &  Predictions")
OW = 16.0; OCX = 13.0
rbox(ax, OCX, HEAD_Y, OW, HT,
     "Output head:  Linear( 256 → 1 )\napplied per decoder timestep  →  16 normalized GWL values",
     C_OP, C_OP_B, bold=True)
rbox(ax, OCX, FINAL_PRED_Y, OW, HB,
     "final predictions on temporal test set",
     C_DATA, C_DATA_B, bold=True)
rbox(ax, OCX, DENORM_Y, OW, HB,
     "denormalize with per-well z-score   →   absolute GWL values",
     C_OP, C_OP_B, bold=True)
rbox(ax, OCX, OUT_Y, OW, HB,
     "forecast GWL  (m a.m.s.l.)",
     C_OUT, C_OUT_B, bold=True)

arr(ax, DT16_CX, LY[N_LAYERS-1]-CH/2, OCX+OW/2, HEAD_Y+HT/2+G,
    c=C_HIDDEN_B, lw=3.5, hw=0.90, hl=0.65, rad=0.08)
arr(ax, OCX, HEAD_Y       - HT/2 - G, OCX, FINAL_PRED_Y + HB/2 + G)
arr(ax, OCX, FINAL_PRED_Y - HB/2 - G, OCX, DENORM_Y     + HB/2 + G)
arr(ax, OCX, DENORM_Y     - HB/2 - G, OCX, OUT_Y        + HB/2 + G)

ax.annotate("", xy=(OCX+OW/2, FINAL_PRED_Y),
    xytext=(W/2+DW/2, TEMP_SPLIT_Y-HB/2),
    arrowprops=dict(arrowstyle="-|>,head_width=0.90,head_length=0.65",
                    color=C_ARROW, lw=2.5, linestyle="dashed",
                    connectionstyle="arc3,rad=-0.25"), zorder=5)

rbox(ax, LOSS_CX, LOSS_Y, LOSS_W, LOSS_H,
     "MSE Loss\n= mean( (ŷ − y_future)² )\n→ backprop  →  optimizer.step()\n→ update GRU weights",
     C_LOSS, C_LOSS_B, bold=True)
arr(ax, OCX+OW/2+G, HEAD_Y, LOSS_CX-LOSS_W/2-G, LOSS_Y, c=C_OP_B, lw=3.5, rad=0.18)
arr(ax, CXS[3], TENS_Y-TH/2-G, CXS[3], LOSS_Y+LOSS_H/2+G, c=C_OUT_B, lw=3.0, rad=0.0)

sec(ax, W, S7_Y, "⑦  GP Spatial Interpolation  (spatial_val wells)")

rbox(ax, CX_FEAT, WATT_Y, W_FEAT, HT,
     "coordinates  +  hydro-zone\n(as GP input)",
     C_DATA, C_DATA_B, bold=True)
rbox(ax, CX_GP, GP_PRETRAIN_Y, GW_GP, HG,
     "GP kernel pretraining",
     C_GP, C_GP_B, bold=True)
rbox(ax, CX_GP, GP_KERN_Y, GW_GP, H_KERN,
     "Spatial GP  —  predicts GWL at spatial_val wells\n"
     "fitted separately per forecast horizon  (1 … 16 weeks ahead)",
     C_GP, C_GP_B, bold=True)
rbox(ax, CX_GP, GP_OUT_Y, GW_GP, HG,
     "GP posterior  ·  predicted GWL at spatial_val wells",
     C_DATA, C_DATA_B, bold=True)
rbox(ax, CX_GP, EVAL_Y, GW_GP, HG,
     "Evaluation against true observations at spatial_val wells",
     C_OP, C_OP_B, bold=True)

CX_FINAL_GP = 26.5; W_FINAL_GP = 9.0; H_FINAL_GP = 1.4
rbox(ax, CX_FINAL_GP, EVAL_Y, W_FINAL_GP, H_FINAL_GP,
     "final model\nfinal predictions on spatial test set",
     C_DATA, C_JOINT_B, bold=True, lw=7.5)

arr(ax, OCX, OUT_Y-HB/2-G, CX_GP, GP_PRETRAIN_Y+HG/2+G)
arr(ax, CX_GP, GP_PRETRAIN_Y-HG/2-G, CX_GP, GP_KERN_Y+H_KERN/2+G)
arr(ax, CX_FEAT, WATT_Y-HT/2-G, CX_GP+GW_GP/2, GP_KERN_Y, c=C_DATA_B, lw=3.0, rad=-0.15)
arr(ax, CX_GP, GP_KERN_Y-H_KERN/2-G, CX_GP, GP_OUT_Y+HG/2+G)
arr(ax, CX_GP, GP_OUT_Y-HG/2-G, CX_GP, EVAL_Y+HG/2+G)
arr(ax, W/2+DW/2, WELLS_Y, CX_FEAT+3.8, WATT_Y+HT/2+G, c=C_DATA_B, lw=3.0, rad=-0.21)
ax.annotate("", xy=(CX_GP-GW_GP/2+1.0, GP_KERN_Y),
    xytext=(W/2-DW/2+1.0, SPLIT_Y),
    arrowprops=dict(arrowstyle="-|>,head_width=0.90,head_length=0.65",
                    color=C_ARROW, lw=2.5, linestyle="dashed",
                    connectionstyle="arc3,rad=0.5"), zorder=5)
arr(ax, CX_GP+GW_GP/2+G, GP_OUT_Y, CX_FINAL_GP-1.5, EVAL_Y+H_FINAL_GP/2+G, c=C_DATA_B, lw=3.0)
ax.annotate("", xy=(CX_FINAL_GP-2.0, EVAL_Y+H_FINAL_GP/2+G),
    xytext=(W/2+DW/2, SPLIT_Y-HB/2),
    arrowprops=dict(arrowstyle="-|>,head_width=0.90,head_length=0.65",
                    color=C_ARROW, lw=2.5, linestyle="dashed",
                    connectionstyle="arc3,rad=-0.3"), zorder=5)

sec(ax, W, S8_Y, "⑧  Joint Objective")
L_GP_W = GW_GP
rbox(ax, CX_GP, L_GP_Y, L_GP_W, HT,
     "L_GP  =  mean_h [ MSE( ŷ_gp_h ,  y_val_true_h ) ]\nspatial accuracy at spatial_val wells",
     C_LOSS, C_JOINT_B, bold=True, lw=7.5)
arr(ax, CX_GP, EVAL_Y-HG/2-G, CX_GP, L_GP_Y+HT/2+G)

JOINT_CX = W/2; JOINT_W = 16.0; JOINT_H = 1.30
rbox(ax, JOINT_CX, L_JOINT_Y, JOINT_W, JOINT_H,
     "L  =  L_GRU  +  λ · L_GP,   λ = 0.0001\n→ backprop → updates GRU weights and GP kernels",
     C_LOSS, C_JOINT_B, bold=True, lw=7.5)
arr(ax, CX_GP, L_GP_Y-HT/2-G, JOINT_CX-2.0, L_JOINT_Y+JOINT_H/2+G, rad=-0.12)
arr(ax, LOSS_CX, LOSS_Y-LOSS_H/2-G, JOINT_CX+2.5, L_JOINT_Y+JOINT_H/2+G,
    c=C_OP_B, lw=3.5, rad=0.28)

C_GRAD  = "#EF4444"
GRAD_LS = '--'
GRAD_X  = LOSS_CX + LOSS_W / 2
OFF     = 0.12
H0_END_X = ENC_CX + SPROJ_W / 2
for sign in (-1, +1):
    ax.plot([GRAD_X + sign*OFF, GRAD_X + sign*OFF, H0_END_X + 0.8],
            [LOSS_Y + LOSS_H/2, H0_Y + sign*OFF, H0_Y + sign*OFF],
            color=C_GRAD, lw=2.0, linestyle=GRAD_LS, zorder=6, solid_capstyle='round')
ax.annotate("", xy=(H0_END_X, H0_Y), xytext=(H0_END_X + 0.8, H0_Y),
    arrowprops=dict(arrowstyle="-|>,head_width=0.70,head_length=0.55",
                    color=C_GRAD, lw=2.5, connectionstyle="arc3,rad=0.0"), zorder=7)

box_items = [(C_DATA,C_DATA_B,"Data"),(C_OP,C_OP_B,"Processing"),
             (C_GRU,C_GRU_B,"GRU component"),(C_HIDDEN,C_HIDDEN_B,"Transferred hidden state"),
             (C_GP,C_GP_B,"GP component"),(C_LOSS,C_LOSS_B,"Loss / objective"),
             ("#FFFFFF",C_JOINT_B,"Joint training only")]
LEG_R  = W - 0.30
SW_W   = 0.55
SW_H   = 0.44
SW_GAP = 0.18
ly = WELLS_Y + 0.20
ax.text(LEG_R, ly + 0.50, "Legend",
        fontsize=9*FS, fontweight="bold", ha="right", va="center", zorder=8)
for k, (fc, ec, lab) in enumerate(box_items):
    lw_leg = 7.5 if ec == C_JOINT_B else 1.4
    iy = ly - k * 0.90
    ax.add_patch(FancyBboxPatch((LEG_R-SW_W, iy-SW_H/2), SW_W, SW_H,
        boxstyle="round,pad=0.03", facecolor=fc, edgecolor=ec, linewidth=lw_leg, zorder=7))
    ax.text(LEG_R-SW_W-SW_GAP, iy, lab, fontsize=8*FS, va="center", ha="right", zorder=7)
grad_leg_y = ly - len(box_items) * 0.90
for sign in (-1, +1):
    ax.plot([LEG_R-SW_W, LEG_R], [grad_leg_y + sign*0.08, grad_leg_y + sign*0.08],
            color=C_GRAD, lw=2.0, linestyle=GRAD_LS, zorder=7)
ax.text(LEG_R-SW_W-SW_GAP, grad_leg_y, "Gradient backprop",
        fontsize=8*FS, va="center", ha="right", zorder=7)

OUT = str(Path(__file__).resolve().parents[4] / "reports/figures/architectures/joint_flowchart_v3.svg")
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
fig.savefig(OUT.replace(".svg", ".pdf"), bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")
plt.close(fig)
