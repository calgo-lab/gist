import os
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

from pathlib import Path

GIST = str(Path(__file__).resolve().parents[4])
GP_BASE = os.path.join(GIST, "outputs", "gp")
JT_BASE = os.path.join(GIST, "outputs", "main_runs", "joint")
OUT_DIR = os.path.join(GIST, "reports", "figures", "joint_vs_two_stage")

SPLIT_SEEDS = list(range(42, 82))
DEC_PATTERN = (
    "GRU_FCOV_in52_out16_ep50_bs4096_seed47_coloc_rand90_f90_ss{ss}"
    "__dec_prod_hr_gps7__predobstrain"
)

CLIP = 5.0
COLOR = "#4C72B0"

print("Mapping joint model directories …")
joint_ss_to_dir = {}
for d in os.listdir(JT_BASE):
    meta_path = os.path.join(JT_BASE, d, "meta.yaml")
    if not os.path.exists(meta_path):
        continue
    with open(meta_path) as f:
        m = yaml.safe_load(f)
    sig = m.get("run_sig", "")
    if "seed44" not in sig or "rand90" not in sig or "f90" not in sig or "valcheck" in sig:
        continue
    match = re.search(r"ss(\d+)$", sig)
    if match:
        joint_ss_to_dir[int(match.group(1))] = os.path.join(JT_BASE, d)

print(f"  Found joint runs for seeds: {sorted(joint_ss_to_dir)}")


def pw_rmse_h16(pred_path: str) -> pd.Series:
    df = pd.read_parquet(pred_path, columns=["id", "horizon", "gws_true", "gws_forecast"])
    df = df[df["horizon"] == 16]
    return (
        df.assign(sq=(df["gws_true"] - df["gws_forecast"]) ** 2)
        .groupby("id")["sq"].mean().apply(np.sqrt)
    )


dec_vals, jt_vals = [], []

for ss in SPLIT_SEEDS:
    dec_path = os.path.join(GP_BASE, DEC_PATTERN.format(ss=ss), "gp_pred.parquet")

    if ss not in joint_ss_to_dir:
        print(f"  ss={ss}: no joint dir, skipping")
        continue
    jt_path = os.path.join(joint_ss_to_dir[ss], "eval", "test", "gp_pred.parquet")

    if not os.path.exists(dec_path):
        print(f"  ss={ss}: missing decoupled pred, skipping")
        continue
    if not os.path.exists(jt_path):
        print(f"  ss={ss}: missing joint pred, skipping")
        continue

    dec_rmse = pw_rmse_h16(dec_path)
    jt_rmse  = pw_rmse_h16(jt_path)

    common = sorted(set(dec_rmse.index) & set(jt_rmse.index))
    dec_vals.extend(dec_rmse.loc[common].values.tolist())
    jt_vals.extend(jt_rmse.loc[common].values.tolist())

    if ss % 10 == 2:
        print(f"  processed ss={ss} … {len(dec_vals)} pairs so far")

xv = np.array(dec_vals)
yv = np.array(jt_vals)
print(f"\nTotal (well, seed) pairs: {len(xv)}")


xc = np.clip(xv, 0, CLIP)
yc = np.clip(yv, 0, CLIP)
n_clipped = int(np.sum((xv > CLIP) | (yv > CLIP)))
x_better  = int(np.sum(xv < yv))
med_x = float(np.median(xv))
med_y = float(np.median(yv))

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(xc, yc, s=28, alpha=0.65, color=COLOR, edgecolors="none", zorder=3)
ax.plot([0, CLIP], [0, CLIP], "k--", lw=1.2, alpha=0.5, zorder=2)
ax.axvline(min(med_x, CLIP), color="#C44E52", lw=1.2, ls="--", alpha=0.7, zorder=2)
ax.axhline(min(med_y, CLIP), color="#55A868", lw=1.2, ls="--", alpha=0.7, zorder=2)

ax.set_xlim(0, CLIP)
ax.set_ylim(0, CLIP)
ax.set_xlabel("Two-stage pipeline (GRU→GP) — per-well RMSE [m] at h=16", fontsize=10)
ax.set_ylabel("Jointly trained (GRU+GP) — per-well RMSE [m] at h=16", fontsize=10)
ax.set_title(
    "Jointly trained vs two-stage pipeline\n"
    "(Random 90/10 · 40 seeds · 104 test wells/seed)",
    fontsize=10,
)
ax.grid(True, alpha=0.25)

line1 = f"n={len(xv)}   x better: {x_better}/{len(xv)}"
if n_clipped:
    line1 += f"   {n_clipped} clipped at {CLIP:g} m"
line2 = f"median x: {med_x:.3f} m   median y: {med_y:.3f} m"
txt1 = TextArea(line1, textprops=dict(fontsize=8))
txt2 = TextArea(line2, textprops=dict(fontsize=8, fontweight="bold"))
packed = VPacker(children=[txt1, txt2], pad=0, sep=2)
ab = AnchoredOffsetbox(
    loc="upper left", child=packed, pad=0.3, frameon=True,
    bbox_to_anchor=(0.03, 0.97), bbox_transform=ax.transAxes, borderpad=0.4,
)
ab.patch.set(facecolor="white", alpha=0.8)
ax.add_artist(ab)

fig.tight_layout()
out_path = os.path.join(OUT_DIR, "perwell_rmse_scatter.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved: {out_path}")
