import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

REPO      = Path(__file__).resolve().parents[4]
DEC_DIR   = REPO / "outputs/gp"
JOINT_DIR = REPO / "outputs/01_final_results/GRU_GP_JOINT"
OUT_DIR   = REPO / "reports/figures/splits"

SEEDS = list(range(42, 52))
CLIP_DEFAULT = 5.0
CLIP_BY_SPLIT = {"far80": 10.0}

JOINT_RUN_IDS = {
    "md50": {42: 1402, 43: 1399, 44: 1405, 45: 1407, 46: 1406,
             47: 1408, 48: 1411, 49: 1410, 50: 1409, 51: 1412},
    "km":   {42: 1416, 43: 1417, 44: 1419, 45: 1413, 46: 1414,
             47: 1415, 48: 1418, 49: 1420, 50: 1421, 51: 1422},
    "far80":{42: 1503, 43: 1504, 44: 1508, 45: 1506, 46: 1507,
             47: 1509, 48: 1505, 49: 1510, 50: 1511, 51: 1512},
}

SPLIT_TITLE = {
    "md50":  "Max-Distance 50%",
    "far80": "Far-80%",
    "km":    "K-Means",
}


def rmse_from_pred(pred_path: Path) -> dict[str, float]:
    pred = pq.read_table(pred_path).to_pandas()
    err = pred.groupby("id").apply(
        lambda g: np.sqrt(((g["gws_true"] - g["gws_forecast"]) ** 2).mean()),
        include_groups=False,
    )
    return err.to_dict()


def collect(split_key: str):
    dec_all, jnt_all = [], []

    for ss in SEEDS:
        dec_name = (
            f"GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_{split_key}_f90_ss{ss}"
            f"__dec_prod_hydroraum_gps1__predobstrain"
        )
        dec_csv = DEC_DIR / dec_name / "gp_metrics_by_id.csv"
        if not dec_csv.exists():
            print(f"  MISSING dec: {dec_csv}", file=sys.stderr)
            continue
        dec_df = pd.read_csv(dec_csv, usecols=["id", "RMSE_over_time"]).set_index("id")

        rid = JOINT_RUN_IDS[split_key][ss]
        jnt_pred = JOINT_DIR / f"GRU_GP_JOINT_{rid}" / "eval" / "test" / "gp_pred.parquet"
        if not jnt_pred.exists():
            print(f"  MISSING joint pred: {jnt_pred}", file=sys.stderr)
            continue
        jnt_rmse = rmse_from_pred(jnt_pred)

        common = set(dec_df.index) & set(jnt_rmse)
        for wid in sorted(common):
            dec_all.append(dec_df.loc[wid, "RMSE_over_time"])
            jnt_all.append(jnt_rmse[wid])

    return np.array(dec_all), np.array(jnt_all)


def make_scatter(xv: np.ndarray, yv: np.ndarray, split_key: str, out_path: Path):
    clip = CLIP_BY_SPLIT.get(split_key, CLIP_DEFAULT)
    xc = np.clip(xv, 0, clip)
    yc = np.clip(yv, 0, clip)
    n         = len(xv)
    x_better  = int(np.sum(xv < yv))
    n_clipped = int(np.sum((xv > clip) | (yv > clip)))
    med_x     = float(np.median(xv))
    med_y     = float(np.median(yv))

    n_seeds = 10
    title = (
        f"Two-stage vs end-to-end spatiotemporal model\n"
        f"({SPLIT_TITLE[split_key]} split · 90/10 · {n_seeds} seeds)"
    )

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(xc, yc, s=28, alpha=0.65, color="#4C72B0", edgecolors="none", zorder=3)
    ax.plot([0, clip], [0, clip], "k--", lw=1.2, alpha=0.5, zorder=2)
    ax.axvline(min(med_x, clip), color="#C44E52", lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.axhline(min(med_y, clip), color="#55A868", lw=1.2, ls="--", alpha=0.7, zorder=2)

    ax.set_xlim(0, clip)
    ax.set_ylim(0, clip)
    ax.set_xlabel("Two-stage model (GRU → GP) — per-well RMSE [m]", fontsize=10)
    ax.set_ylabel("End-to-end model (GRU + GP) — per-well RMSE [m]", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)

    line1 = f"n={n}   x better: {x_better}/{n}"
    if n_clipped:
        line1 += f"   {n_clipped} clipped at {clip:g} m"
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
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_key in ("md50", "far80", "km"):
        print(f"\nCollecting {split_key} …")
        xv, yv = collect(split_key)
        print(f"  {len(xv)} (well, seed) pairs")
        if len(xv) == 0:
            print("  SKIPPED"); continue
        make_scatter(xv, yv, split_key, OUT_DIR / f"scatter_splits_{split_key}.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
