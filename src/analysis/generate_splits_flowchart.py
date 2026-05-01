import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── colours (same 3-family palette as GRU/joint charts) ──────────────────────
C_DATA   = "#DBEAFE"; C_DATA_B   = "#2563EB"   # blue   — data / input / output
C_OP     = "#FEF9C3"; C_OP_B     = "#CA8A04"   # amber  — operations / processing
C_GRU    = "#DCFCE7"; C_GRU_B    = "#16A34A"   # green  — split result
C_GP     = "#F3E8FF"; C_GP_B     = "#7C3AED"   # violet — clustering / features
C_HIDDEN = "#FFE4E6"; C_HIDDEN_B = "#DC2626"   # red    — spatial eligibility constraint
C_ARROW  = "#475569"

G = 0.07   # gap between arrowhead and box edge


def rbox(ax, cx, cy, w, h, text, fc, ec, fs=9.0, bold=False, z=3, lw=1.4):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(p)
    ax.text(
        cx, cy, text, ha="center", va="center", fontsize=fs,
        fontweight="bold" if bold else "normal",
        multialignment="center", zorder=z + 1,
    )


def arr(ax, x0, y0, x1, y1, c=C_ARROW, lw=1.4, hw=0.55, hl=0.40):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=f"-|>,head_width={hw},head_length={hl}",
            color=c, lw=lw, connectionstyle="arc3,rad=0",
        ),
        zorder=5,
    )


# ── canvas ────────────────────────────────────────────────────────────────────
W = 30; H = 21; YMIN = 3.5
fig, ax = plt.subplots(figsize=(W, H - YMIN))
ax.set_xlim(0, W)
ax.set_ylim(YMIN, H)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── main title ────────────────────────────────────────────────────────────────
TITLE_Y = H - 0.55
ax.text(
    W / 2, TITLE_Y,
    "Spatial Split Algorithms — Three Variants",
    ha="center", va="center", fontsize=14, fontweight="bold", color="#0f172a",
)
ax.plot([1.0, W - 1.0], [TITLE_Y - 0.38, TITLE_Y - 0.38], color="#cbd5e1", lw=0.8)

# ── column separator lines ────────────────────────────────────────────────────
for xsep in [10.0, 20.0]:
    ax.plot([xsep, xsep], [YMIN + 0.2, H - 1.1], color="#e2e8f0", lw=0.9, zorder=1)

# ── column geometry ───────────────────────────────────────────────────────────
CX = [5.0, 15.0, 25.0]   # column centre x-values
BW = 8.2                   # standard box width
BH = 0.82                  # standard box height
CT_Y = H - 1.22            # column title y (19.78)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN 1 — RANDOM SPLIT
# ─────────────────────────────────────────────────────────────────────────────
cx = CX[0]

ax.text(cx, CT_Y, "① Random Split",
        ha="center", va="center", fontsize=11, fontweight="bold", color=C_DATA_B)

B1 = 17.60   # input
B2 = 14.33   # shuffle  (evenly spaced so result aligns with col 3)
B3 = 11.07   # take 90 %
BR =  7.80   # result

rbox(ax, cx, B1, BW, BH,
     "1 040 wells",
     C_DATA, C_DATA_B, fs=9.5, bold=True)
arr(ax, cx, B1 - BH / 2 - G, cx, B2 + BH / 2 + G)

rbox(ax, cx, B2, BW, BH,
     "random shuffle of well IDs",
     C_OP, C_OP_B, fs=8.5, bold=True)
arr(ax, cx, B2 - BH / 2 - G, cx, B3 + BH / 2 + G)

rbox(ax, cx, B3, BW, BH,
     "Take ⌈n × 0.95⌉ IDs → train\nremaining IDs → holdout",
     C_OP, C_OP_B, fs=8.5, bold=True)
arr(ax, cx, B3 - BH / 2 - G, cx, BR + BH / 2 + G)

rbox(ax, cx, BR, BW, BH,
     "spatial_train:  988  wells\nspatial_holdout:  52  wells",
     C_GRU, C_GRU_B, fs=9.0, bold=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN 2 — K-MEANS STRATIFIED SPLIT
# ─────────────────────────────────────────────────────────────────────────────
cx = CX[1]

ax.text(cx, CT_Y, "② K-means Stratified Split",
        ha="center", va="center", fontsize=11, fontweight="bold", color=C_GP_B)

K1   = 17.60   # input
K2   = 15.29   # scaler     (evenly spaced so result aligns with col 3)
K3   = 12.87   # kmeans    (BH_K3 = 1.05)
K4   = 10.28   # per-cluster  (BH_K4 = 1.15)
KR   =  7.80   # result

BH_K3 = 1.05
BH_K4 = 1.15

rbox(ax, cx, K1, BW, BH,
     "1 040 wells  +  27 static features",
     C_DATA, C_DATA_B, fs=9.5, bold=True)
arr(ax, cx, K1 - BH / 2 - G, cx, K2 + BH / 2 + G)

rbox(ax, cx, K2, BW, BH,
     "StandardScaler\nper-feature z-score normalisation",
     C_OP, C_OP_B, fs=8.5, bold=True)
arr(ax, cx, K2 - BH / 2 - G, cx, K3 + BH_K3 / 2 + G)

rbox(ax, cx, K3, BW, BH_K3,
     "KMeans  (k = 20 clusters,  seed = 42)\ngroups wells by feature similarity",
     C_GP, C_GP_B, fs=8.5, bold=True)
arr(ax, cx, K3 - BH_K3 / 2 - G, cx, K4 + BH_K4 / 2 + G)

rbox(ax, cx, K4, BW, BH_K4,
     "Per cluster:  take ⌈n × 0.95⌉ → train\nremaining IDs per cluster → holdout\n(preserves feature distribution across splits)",
     C_OP, C_OP_B, fs=8.2, bold=True)
arr(ax, cx, K4 - BH_K4 / 2 - G, cx, KR + BH / 2 + G)

rbox(ax, cx, KR, BW, BH,
     "spatial_train:  988  wells\nspatial_holdout:  52  wells",
     C_GRU, C_GRU_B, fs=9.0, bold=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN 3 — RANDOM MAX-DIST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
cx = CX[2]

ax.text(cx, CT_Y, "③ Random Max-Dist Split",
        ha="center", va="center", fontsize=11, fontweight="bold", color=C_HIDDEN_B)

M1  = 17.60   # input
M2  = 15.62   # k-d tree     (BH_M2 = 1.05)  — naturally tight
M3  = 13.63   # d_thresh
M4  = 11.65   # eligible     (BH_M4 = 1.05)
M5  =  9.67   # sample
MR  =  7.80   # result

BH_M2 = 1.05
BH_M4 = 1.05

rbox(ax, cx, M1, BW, BH,
     "1 040 wells  +  coordinates",
     C_DATA, C_DATA_B, fs=9.5, bold=True)
arr(ax, cx, M1 - BH / 2 - G, cx, M2 + BH_M2 / 2 + G)

rbox(ax, cx, M2, BW, BH_M2,
     "compute mean distance per well\nfor k nearest neighbours  (k = 3)",
     C_OP, C_OP_B, fs=8.5, bold=True)
arr(ax, cx, M2 - BH_M2 / 2 - G, cx, M3 + BH / 2 + G)

rbox(ax, cx, M3, BW, BH,
     "threshold:  xth percentile  of  mean distance  (x = 10)",
     C_OP, C_OP_B, fs=8.2, bold=True)
arr(ax, cx, M3 - BH / 2 - G, cx, M4 + BH_M4 / 2 + G)

rbox(ax, cx, M4, BW, BH_M4,
     "eligible for holdout:  mean distance  ≤  threshold",
     C_HIDDEN, C_HIDDEN_B, fs=8.5, bold=True)
arr(ax, cx, M4 - BH_M4 / 2 - G, cx, M5 + BH / 2 + G)

rbox(ax, cx, M5, BW, BH,
     "n × 0.5 IDs from eligible wells  →  spatial holdout\nremaining IDs  →  spatial train",
     C_OP, C_OP_B, fs=8.5, bold=True)
arr(ax, cx, M5 - BH / 2 - G, cx, MR + BH / 2 + G)

rbox(ax, cx, MR, BW, BH,
     "spatial_train:  988  wells\nspatial_holdout:  52  wells",
     C_GRU, C_GRU_B, fs=9.0, bold=True)

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND (horizontal strip, bottom-centred)
# ─────────────────────────────────────────────────────────────────────────────
items = [
    (C_DATA,   C_DATA_B,   "Input data"),
    (C_OP,     C_OP_B,     "Processing step"),
    (C_GP,     C_GP_B,     "Clustering"),
    (C_HIDDEN, C_HIDDEN_B, "Spatial constraint"),
    (C_GRU,    C_GRU_B,    "Split result"),
]
LEG_Y  = 4.5
LEG_W  = 4.0   # width per legend item slot
n_items = len(items)
start_x = W / 2 - (n_items * LEG_W) / 2

ax.text(W / 2, LEG_Y + 0.55, "Legend",
        ha="center", va="center", fontsize=9, fontweight="bold", color="#1e293b")

for k, (fc, ec, lab) in enumerate(items):
    lx = start_x + k * LEG_W + 0.20
    r = FancyBboxPatch(
        (lx, LEG_Y - 0.20), 0.44, 0.36,
        boxstyle="round,pad=0.03",
        facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=7,
    )
    ax.add_patch(r)
    ax.text(lx + 0.60, LEG_Y - 0.02, lab,
            fontsize=8.0, va="center", zorder=7)

# ─────────────────────────────────────────────────────────────────────────────
OUT = "reports/figures/splits_flowchart.svg"
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")
plt.close(fig)
