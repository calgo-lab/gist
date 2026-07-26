import re, sys, yaml
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from scipy.spatial import cKDTree

REPO      = Path(__file__).resolve().parents[4]
META_PATH = REPO.parent / "data" / "merged.parquet"
MAIN      = REPO / "outputs/main_runs"
DEC_BASE  = MAIN / "decoupled"
JNT_BASE  = MAIN / "joint"
OUT_DIR   = REPO / "reports/figures/joint_vs_two_stage"

CLIP_DEFAULT = 5.0


def load_coords():
    t = pq.read_table(META_PATH, columns=["id", "x_25833", "y_25833"])
    return t.to_pandas().drop_duplicates("id").set_index("id")


def rmse_from_pred(pred_path: Path) -> dict:
    pred = pq.read_table(pred_path).to_pandas()
    err = pred.groupby("id").apply(
        lambda g: np.sqrt(((g["gws_true"] - g["gws_forecast"]) ** 2).mean()),
        include_groups=False,
    )
    return err.to_dict()


def build_joint_map() -> dict:
    mapping = {}
    for d in sorted(JNT_BASE.iterdir()):
        meta = d / "meta.yaml"
        if not meta.exists():
            continue
        with open(meta) as f:
            m = yaml.safe_load(f)
        sig = m.get("run_sig", "")
        match = re.search(r"_ss(\d+)$", sig)
        if match:
            ss = int(match.group(1))
            mapping[ss] = d
    return mapping


def collect_all(coords: pd.DataFrame) -> pd.DataFrame:
    joint_map = build_joint_map()
    rows = []

    for ss, jnt_dir in sorted(joint_map.items()):
        split_df = pd.read_csv(jnt_dir / "split_info.csv")
        train_ids = set(split_df.loc[split_df["spatial_split"] == "spatial_train", "id"])
        test_ids  = set(split_df.loc[split_df["spatial_split"] == "spatial_test",  "id"])

        train_coords = coords.loc[coords.index.isin(train_ids), ["x_25833","y_25833"]].dropna()
        test_coords  = coords.loc[coords.index.isin(test_ids),  ["x_25833","y_25833"]].dropna()
        if len(train_coords) == 0 or len(test_coords) == 0:
            continue
        tree = cKDTree(train_coords.values)
        dists_m, _ = tree.query(test_coords.values, k=1)

        dec_pattern = f"*_coloc_rand90_f90_ss{ss}__dec_prod_hr_gps7__predobstrain"
        dec_matches = list(DEC_BASE.glob(dec_pattern))
        if not dec_matches:
            print(f"  MISSING dec ss={ss}", file=sys.stderr)
            continue
        dec_csv = dec_matches[0] / "gp_metrics_by_id.csv"
        if not dec_csv.exists():
            print(f"  MISSING dec csv ss={ss}", file=sys.stderr)
            continue
        dec_df = pd.read_csv(dec_csv, usecols=["id", "RMSE_over_time"]).set_index("id")

        jnt_pred = jnt_dir / "eval" / "test" / "gp_pred.parquet"
        if not jnt_pred.exists():
            print(f"  MISSING joint pred ss={ss}", file=sys.stderr)
            continue
        jnt_rmse = rmse_from_pred(jnt_pred)

        for wid, dist_m in zip(test_coords.index, dists_m):
            if wid not in dec_df.index or wid not in jnt_rmse:
                continue
            rows.append({
                "id":      wid,
                "seed":    ss,
                "ntn_km":  dist_m / 1000.0,
                "dec_rmse": dec_df.loc[wid, "RMSE_over_time"],
                "jnt_rmse": jnt_rmse[wid],
            })

    return pd.DataFrame(rows)


def make_scatter(df: pd.DataFrame, title: str, subtitle: str, out_path: Path,
                 clip: float = CLIP_DEFAULT):
    xv = df["dec_rmse"].values
    yv = df["jnt_rmse"].values
    xc = np.clip(xv, 0, clip)
    yc = np.clip(yv, 0, clip)
    n         = len(xv)
    x_better  = int(np.sum(xv < yv))
    n_clipped = int(np.sum((xv > clip) | (yv > clip)))
    med_x     = float(np.median(xv))
    med_y     = float(np.median(yv))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(xc, yc, s=28, alpha=0.65, color="#4C72B0", edgecolors="none", zorder=3)
    ax.plot([0, clip], [0, clip], "k--", lw=1.2, alpha=0.5, zorder=2)
    ax.axvline(min(med_x, clip), color="#C44E52", lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.axhline(min(med_y, clip), color="#55A868", lw=1.2, ls="--", alpha=0.7, zorder=2)

    ax.set_xlim(0, clip)
    ax.set_ylim(0, clip)
    ax.set_xlabel("Two-stage model (GRU → GP) — per-well RMSE [m]", fontsize=10)
    ax.set_ylabel("End-to-end model (GRU + GP) — per-well RMSE [m]", fontsize=10)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
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

    print("Loading coordinates …")
    coords = load_coords()

    print("Collecting all 40 seeds …")
    df = collect_all(coords)
    print(f"  {len(df)} (well, seed) pairs total")

    p10 = np.percentile(df["ntn_km"], 10)
    p90 = np.percentile(df["ntn_km"], 90)
    print(f"  NTN 10th percentile: {p10:.3f} km, 90th: {p90:.3f} km")

    close = df[df["ntn_km"] <= p10]
    far   = df[df["ntn_km"] >= p90]
    print(f"  Close (≤P10): {len(close)} pairs, Far (≥P90): {len(far)} pairs")

    n_seeds = df["seed"].nunique()
    subtitle = f"Random 90/10 · {n_seeds} seeds"

    make_scatter(
        close,
        title="Two-stage vs end-to-end: closest 10% of test wells (by NTN)",
        subtitle=subtitle,
        out_path=OUT_DIR / "scatter_main_close.png",
        clip=10.0,
    )
    make_scatter(
        far,
        title="Two-stage vs end-to-end: farthest 10% of test wells (by NTN)",
        subtitle=subtitle,
        out_path=OUT_DIR / "scatter_main_far.png",
        clip=10.0,
    )
    print("Done.")


if __name__ == "__main__":
    main()
