from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

ROOT     = Path(__file__).resolve().parents[3]
GP_DIR   = ROOT / "outputs" / "gp"
JT_DIR   = ROOT / "outputs" / "01_final_results" / "GRU_GP_JOINT"
SPLIT_DIR = ROOT / "splits"
OUT_DIR  = ROOT / "reports" / "figures" / "slides"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_CUTOFF = pd.Timestamp("2020-01-01")

JOINT_RAND90_F90_BY_SS = {42: 1485, 43: 1490, 44: 1483, 45: 1492, 46: 1488,
                           47: 1484, 48: 1491, 49: 1486, 50: 1487, 51: 1489}

meta = (
    pq.read_table(ROOT.parent / "data" / "merged.parquet",
                  columns=["id", "x_25833", "y_25833"])
    .to_pandas()
    .groupby("id")
    .first()
    .reset_index()
)
coord_by_id = meta.set_index("id")[["x_25833", "y_25833"]]


def _rmse_per_well(df: pd.DataFrame) -> pd.Series:
    df = df[df["datum"] > TEST_CUTOFF].dropna(subset=["gws_true", "gws_forecast"])
    return (
        df.groupby("id")
        .apply(lambda g: np.sqrt(np.mean((g["gws_true"] - g["gws_forecast"]) ** 2)),
               include_groups=False)
        .rename("rmse")
    )


def _dist_to_nearest_train(holdout_ids, train_ids):
    h_coords = coord_by_id.loc[list(holdout_ids)].values
    t_coords = coord_by_id.loc[list(train_ids)].values
    dists, _ = cKDTree(t_coords).query(h_coords, k=1)
    return pd.Series(dists, index=list(holdout_ids))


def main():
    dec_easy, dec_hard = [], []
    jt_easy,  jt_hard  = [], []

    for ss in range(42, 52):
        split = pd.read_csv(
            SPLIT_DIR / f"spatial_split_full_merged_coloc_rand90_f90_ss{ss}.csv"
        )
        holdout_ids = set(split.loc[
            split.spatial_split.isin(["spatial_holdout", "spatial_test"]), "id"
        ])
        train_ids = set(split.loc[split.spatial_split == "spatial_train", "id"])

        dist = _dist_to_nearest_train(holdout_ids, train_ids)
        easy_ids = set(dist[dist <= dist.quantile(0.10)].index)
        hard_ids = set(dist[dist >= dist.quantile(0.90)].index)

        p = (GP_DIR /
             f"GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_rand90_f90_ss{ss}"
             f"__dec_prod_hydroraum_gps1__predobstrain" / "gp_pred.parquet")
        df = pq.read_table(p, columns=["id","datum","gws_true","gws_forecast"]).to_pandas()
        df["datum"] = pd.to_datetime(df["datum"])
        rmse_dec = _rmse_per_well(df)
        dec_easy.append(float(np.median(rmse_dec.loc[rmse_dec.index.isin(easy_ids)].values)))
        dec_hard.append(float(np.median(rmse_dec.loc[rmse_dec.index.isin(hard_ids)].values)))

        run_id = JOINT_RAND90_F90_BY_SS[ss]
        p = JT_DIR / f"GRU_GP_JOINT_{run_id}" / "eval" / "test" / "gp_pred.parquet"
        df = pq.read_table(p, columns=["id","datum","gws_true","gws_forecast"]).to_pandas()
        df["datum"] = pd.to_datetime(df["datum"])
        rmse_jt = _rmse_per_well(df)
        jt_easy.append(float(np.median(rmse_jt.loc[rmse_jt.index.isin(easy_ids)].values)))
        jt_hard.append(float(np.median(rmse_jt.loc[rmse_jt.index.isin(hard_ids)].values)))

        print(f"  ss{ss}: dec easy={dec_easy[-1]:.3f} hard={dec_hard[-1]:.3f} | "
              f"jt easy={jt_easy[-1]:.3f} hard={jt_hard[-1]:.3f}")

    vals = {
        ("Easy wells\n(closest 10%)", "Two-stage\n(GRU → GP)"):  np.mean(dec_easy),
        ("Easy wells\n(closest 10%)", "End-to-end\n(GRU + GP)"): np.mean(jt_easy),
        ("Hard wells\n(farthest 10%)", "Two-stage\n(GRU → GP)"):  np.mean(dec_hard),
        ("Hard wells\n(farthest 10%)", "End-to-end\n(GRU + GP)"): np.mean(jt_hard),
    }

    rows = ["Easy wells\n(closest 10%)", "Hard wells\n(farthest 10%)"]
    cols = ["Two-stage\n(GRU → GP)", "End-to-end\n(GRU + GP)"]

    print("\nSummary (mean-of-per-seed medians):")
    for r in rows:
        for c in cols:
            print(f"  {r.split(chr(10))[0]} | {c.split(chr(10))[0]}: {vals[(r,c)]:.3f} m")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")

    cell_text = [[f"{vals[(r,c)]:.2f} m" for c in cols] for r in rows]

    cell_colors = []
    for r in rows:
        row_vals = [vals[(r, c)] for c in cols]
        winner = min(range(len(cols)), key=lambda i: row_vals[i])
        cell_colors.append([
            "#d4edda" if i == winner else "#f8d7da"
            for i in range(len(cols))
        ])

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=cols,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(2.2, 2.8)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor("#343a40")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#aaaaaa")

    ax.set_title(
        "Median per-well RMSE by difficulty — random 90/10 split (10 seeds, bottom/top 10%)",
        fontsize=11, pad=14
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)
    out = OUT_DIR / "table_easy_hard_rand_f90.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → {out.name}")


if __name__ == "__main__":
    main()
