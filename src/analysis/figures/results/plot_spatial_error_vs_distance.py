import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyproj
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT   = Path(__file__).resolve().parents[4]
BASE   = ROOT / "outputs/gp"
SPLITS = ROOT / "splits"
META   = ROOT.parent / "data" / "merged.parquet"
BND    = ROOT / "data/boundaries/geoBoundaries-DEU-ADM1_simplified.geojson"
SEEDS  = list(range(42, 52))
CMAP   = plt.cm.YlOrRd


def load_boundary_rings():
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
    with open(BND) as f:
        gj = json.load(f)
    feat = next(f for f in gj["features"] if f["properties"].get("shapeName") == "Brandenburg")

    def xform(coords):
        return transformer.transform([c[0] for c in coords], [c[1] for c in coords])

    rings = []
    geom = feat["geometry"]
    if geom["type"] == "Polygon":
        for ring in geom["coordinates"]:
            rings.append(xform(ring))
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            for ring in poly:
                rings.append(xform(ring))
    return rings


def draw_boundary(ax, rings):
    for rx, ry in rings:
        ax.plot(rx, ry, "k-", lw=0.7, zorder=2)


def load_data():
    all_meta = (
        pq.read_table(META, columns=["id", "x_25833", "y_25833"])
        .to_pandas()
        .drop_duplicates("id")
        .set_index("id")
    )

    rmse_by_well  = {}
    dist1_by_well = {}
    dist3_by_well = {}
    coords_df     = {}
    hr_by_well    = {}

    for ss in SEEDS:
        pred_dir = (
            BASE
            / f"GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_rand90_f90_ss{ss}"
              f"__dec_prod_hydroraum_gps1__predobstrain"
        )
        pred = pq.read_table(pred_dir / "gp_pred.parquet").to_pandas()

        err = (
            pred.groupby("id")
            .apply(lambda g: np.sqrt(((g["gws_true"] - g["gws_forecast"]) ** 2).mean()),
                   include_groups=False)
        )
        for wid, v in err.items():
            rmse_by_well.setdefault(wid, []).append(v)

        wcoords = pred.groupby("id")[["x_25833", "y_25833"]].first()
        for wid, row in wcoords.iterrows():
            coords_df.setdefault(wid, (row["x_25833"], row["y_25833"]))

        if "hydroraum_Speisungsgebiete" in pred.columns:
            hr = pred.groupby("id")[["hydroraum_Speisungsgebiete",
                                     "hydroraum_Transitgebiete"]].first()
            for wid, row in hr.iterrows():
                if wid not in hr_by_well:
                    if row["hydroraum_Speisungsgebiete"] == 1:
                        hr_by_well[wid] = "Recharge"
                    elif row["hydroraum_Transitgebiete"] == 1:
                        hr_by_well[wid] = "Transit"
                    else:
                        hr_by_well[wid] = "Discharge"

        split = pd.read_csv(
            SPLITS / f"spatial_split_full_merged_coloc_rand90_f90_ss{ss}.csv"
        )
        train_ids = set(split.loc[split["spatial_split"] == "spatial_train", "id"])
        train_meta = all_meta.loc[all_meta.index.isin(train_ids)]
        tree = cKDTree(train_meta[["x_25833", "y_25833"]].values)

        dists1, _ = tree.query(wcoords.values, k=1)
        dists3, _ = tree.query(wcoords.values, k=3)
        for wid, d1, d3row in zip(wcoords.index, dists1.flatten(), dists3):
            dist1_by_well.setdefault(wid, []).append(d1 / 1000.0)
            dist3_by_well.setdefault(wid, []).append(d3row.mean() / 1000.0)

    wells = sorted(set(rmse_by_well) & set(dist1_by_well))
    df = pd.DataFrame({
        "id":       wells,
        "x":        [coords_df[w][0] for w in wells],
        "y":        [coords_df[w][1] for w in wells],
        "med_rmse": [np.median(rmse_by_well[w])  for w in wells],
        "med_dist1":[np.median(dist1_by_well[w]) for w in wells],
        "med_dist3":[np.median(dist3_by_well[w]) for w in wells],
        "hydroraum":[hr_by_well.get(w, "Unknown") for w in wells],
    })
    return df


def rmse_norm(med_rmse):
    log = np.log10(np.clip(med_rmse, 1e-3, None))
    return log, mcolors.Normalize(vmin=log.min(), vmax=np.percentile(log, 98))


def rmse_ticks():
    return np.log10([0.1, 0.3, 1, 3, 10, 30]), ["0.1", "0.3", "1", "3", "10", "30"]


def scatter_panel(ax, fig, xs, ys, vals, norm, label, ticks=None, ticklabels=None, rings=None,
                  label_fontsize=9, tick_fontsize=8):
    if rings:
        draw_boundary(ax, rings)
    sc = ax.scatter(xs, ys, c=vals, cmap=CMAP, norm=norm, s=30, zorder=3, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label(label, fontsize=label_fontsize)
    cb.ax.tick_params(labelsize=tick_fontsize)
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklabels)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_k1_uncapped(df, rings, out_dir):
    rmse_log, norm_r = rmse_norm(df["med_rmse"].values)
    norm_d = mcolors.Normalize(vmin=0, vmax=np.percentile(df["med_dist1"], 98))
    rho, pval = spearmanr(df["med_dist1"], df["med_rmse"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter_panel(axes[0], fig, df["x"], df["y"], rmse_log, norm_r,
                  "Median per-well RMSE (m, log scale)", *rmse_ticks(), rings)
    scatter_panel(axes[1], fig, df["x"], df["y"], df["med_dist1"], norm_d,
                  "Median distance to nearest training well (km)", rings=rings)
    axes[0].set_title("Per-well RMSE", fontsize=11)
    axes[1].set_title("Distance to nearest training well (k=1)", fontsize=11)
    fig.suptitle(
        f"Spatial error vs. distance to nearest training well  |  "
        f"{len(df)} holdout wells, median over 10 seeds (ss42–51)  |  "
        f"Spearman ρ={rho:.3f}, p={pval:.4f}",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out = out_dir / "spatial_error_vs_distance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_k1_cap4km(df, rings, out_dir, show_title=True):
    DIST_MAX = 6.0
    rmse_log, norm_r = rmse_norm(df["med_rmse"].values)
    dist_disp = np.clip(df["med_dist1"].values, 0, DIST_MAX)
    norm_d = mcolors.Normalize(vmin=0, vmax=DIST_MAX)
    rho, pval = spearmanr(df["med_dist1"], df["med_rmse"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter_panel(axes[0], fig, df["x"], df["y"], rmse_log, norm_r,
                  "Median per-well RMSE (m, log scale)", *rmse_ticks(), rings,
                  label_fontsize=14, tick_fontsize=13)
    scatter_panel(axes[1], fig, df["x"], df["y"], dist_disp, norm_d,
                  "Median distance to nearest training well (km)",
                  rings=rings, label_fontsize=14, tick_fontsize=13)
    axes[0].set_title("Per-well RMSE", fontsize=16)
    axes[1].set_title("Distance to nearest training well", fontsize=16)
    if show_title:
        fig.suptitle(
            f"Spatial error vs. distance  |  Spearman ρ={rho:.3f}",
            fontsize=19, y=1.01,
        )
    fig.tight_layout()
    suffix = "" if show_title else "_notitle"
    out = out_dir / f"spatial_error_vs_distance_cap4km{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_k3_cap4km(df, rings, out_dir):
    K = 3
    DIST_MAX = 4.0
    rmse_log, norm_r = rmse_norm(df["med_rmse"].values)
    dist_disp = np.clip(df["med_dist3"].values, 0, DIST_MAX)
    norm_d = mcolors.Normalize(vmin=0, vmax=DIST_MAX)
    rho, pval = spearmanr(df["med_dist3"], df["med_rmse"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter_panel(axes[0], fig, df["x"], df["y"], rmse_log, norm_r,
                  "Median per-well RMSE (m, log scale)", *rmse_ticks(), rings)
    scatter_panel(axes[1], fig, df["x"], df["y"], dist_disp, norm_d,
                  f"Median avg. distance to {K} nearest training wells (km)\n[capped at {DIST_MAX} km]",
                  rings=rings)
    axes[0].set_title("Per-well RMSE", fontsize=11)
    axes[1].set_title(f"Avg. distance to {K} nearest training wells", fontsize=11)
    fig.suptitle(
        f"Spatial error vs. avg. distance to {K} nearest training wells  |  "
        f"{len(df)} holdout wells, median over 10 seeds (ss42–51)  |  "
        f"Spearman ρ={rho:.3f}, p={pval:.4f}",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out = out_dir / "spatial_error_vs_distance_k3.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_hydroraum(df, rings, out_dir):
    rmse_log, norm_r = rmse_norm(df["med_rmse"].values)
    DIST_MAX = 4.0
    dist_disp = np.clip(df["med_dist1"].values, 0, DIST_MAX)
    norm_d = mcolors.Normalize(vmin=0, vmax=DIST_MAX)

    hr_colors = {"Recharge": "tab:red", "Transit": "tab:orange", "Discharge": "tab:blue"}
    hr_vals = np.array([{"Recharge": 2, "Transit": 1, "Discharge": 0}.get(h, -1)
                        for h in df["hydroraum"]])
    hr_cmap = mcolors.ListedColormap(["tab:blue", "tab:orange", "tab:red"])
    hr_norm = mcolors.BoundaryNorm([-.5, .5, 1.5, 2.5], hr_cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    scatter_panel(axes[0], fig, df["x"], df["y"], rmse_log, norm_r,
                  "Median per-well RMSE (m, log scale)", *rmse_ticks(), rings)
    scatter_panel(axes[1], fig, df["x"], df["y"], dist_disp, norm_d,
                  f"Median distance to nearest training well (km)\n[capped at {DIST_MAX} km]",
                  rings=rings)

    draw_boundary(axes[2], rings)
    sc = axes[2].scatter(df["x"], df["y"], c=hr_vals, cmap=hr_cmap, norm=hr_norm,
                         s=30, zorder=3, edgecolors="none")
    cb = fig.colorbar(sc, ax=axes[2], shrink=0.7, pad=0.02, ticks=[0, 1, 2])
    cb.set_ticklabels(["Discharge", "Transit", "Recharge"])
    cb.set_label("Hydrological zone", fontsize=9)
    axes[2].set_aspect("equal"); axes[2].axis("off")

    axes[0].set_title("Per-well RMSE", fontsize=11)
    axes[1].set_title("Distance to nearest training well", fontsize=11)
    axes[2].set_title("Hydrological zone (hydroraum)", fontsize=11)

    rho, pval = spearmanr(df["med_dist1"], df["med_rmse"])
    fig.suptitle(
        f"Spatial error, distance, and hydroraum  |  {len(df)} holdout wells, "
        f"median over 10 seeds (ss42–51)  |  Spearman ρ={rho:.3f}, p={pval:.4f}",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out = out_dir / "spatial_error_vs_distance_hydroraum.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_hydroraum_recharge(df, rings, out_dir):
    sub = df[df["hydroraum"] == "Recharge"].copy()
    rmse_log_all, norm_r = rmse_norm(df["med_rmse"].values)
    rmse_log_sub = np.log10(np.clip(sub["med_rmse"].values, 1e-3, None))
    DIST_MAX = 8.0
    dist_disp_all = np.clip(df["med_dist1"].values, 0, DIST_MAX)
    dist_disp_sub = np.clip(sub["med_dist1"].values, 0, DIST_MAX)
    norm_d = mcolors.Normalize(vmin=0, vmax=DIST_MAX)
    rho, pval = spearmanr(sub["med_dist1"], sub["med_rmse"])

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    scatter_panel(axes[0], fig, sub["x"], sub["y"], rmse_log_sub, norm_r,
                  "Median per-well RMSE (m, log scale)", *rmse_ticks(), rings)
    scatter_panel(axes[1], fig, sub["x"], sub["y"], dist_disp_sub, norm_d,
                  f"Median distance to nearest training well (km)\n[capped at {DIST_MAX} km]",
                  rings=rings)

    draw_boundary(axes[2], rings)
    axes[2].scatter(df["x"], df["y"], c="lightgrey", s=15, zorder=2, edgecolors="none")
    axes[2].scatter(sub["x"], sub["y"], c="tab:red", s=30, zorder=3, edgecolors="none",
                    label=f"Recharge (n={len(sub)})")
    axes[2].legend(loc="lower right", fontsize=9)
    axes[2].set_aspect("equal"); axes[2].axis("off")

    axes[0].set_title("Per-well RMSE (recharge only)", fontsize=11)
    axes[1].set_title("Distance to nearest training well (recharge only)", fontsize=11)
    axes[2].set_title("Recharge wells in context", fontsize=11)
    fig.suptitle(
        f"Recharge (Speisung) wells only  |  n={len(sub)}  |  "
        f"Spearman ρ={rho:.3f}, p={pval:.4f}",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out = out_dir / "spatial_error_vs_distance_hydroraum_recharge.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_recharge_transit(df, rings, out_dir, dist_max=8.0):
    sub = df[df["hydroraum"].isin(["Recharge", "Transit"])].copy()
    rho, pval = spearmanr(sub["med_dist1"], sub["med_rmse"])

    rmse_log_all, norm_r = rmse_norm(df["med_rmse"].values)
    rmse_log_sub = np.log10(np.clip(sub["med_rmse"].values, 1e-3, None))
    norm_d = mcolors.Normalize(vmin=0, vmax=dist_max)
    dist_disp = np.clip(sub["med_dist1"].values, 0, dist_max)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    scatter_panel(axes[0], fig, sub["x"], sub["y"], rmse_log_sub, norm_r,
                  "Median per-well RMSE (m, log scale)", *rmse_ticks(), rings)
    scatter_panel(axes[1], fig, sub["x"], sub["y"], dist_disp, norm_d,
                  f"Median distance to nearest training well (km)\n[capped at {dist_max} km]",
                  rings=rings)

    draw_boundary(axes[2], rings)
    axes[2].scatter(df["x"], df["y"], c="lightgrey", s=15, zorder=2, edgecolors="none")
    for zone, color in [("Recharge", "tab:red"), ("Transit", "tab:orange")]:
        mask = sub["hydroraum"] == zone
        axes[2].scatter(sub.loc[mask, "x"], sub.loc[mask, "y"], c=color, s=30,
                        zorder=3, edgecolors="none", label=f"{zone} (n={mask.sum()})")
    axes[2].legend(loc="lower right", fontsize=9)
    axes[2].set_aspect("equal"); axes[2].axis("off")

    axes[0].set_title("Per-well RMSE (recharge + transit)", fontsize=11)
    axes[1].set_title("Distance to nearest training well", fontsize=11)
    axes[2].set_title("Recharge & transit wells in context", fontsize=11)
    fig.suptitle(
        f"Recharge + transit wells (discharge excluded)  |  n={len(sub)}  |  "
        f"Spearman ρ={rho:.3f}, p={pval:.4f}",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out = out_dir / "spatial_error_vs_distance_recharge_transit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",
                        default=str(Path(__file__).resolve().parents[4] / "reports/figures/spatial_distance"),
                        help="Output directory for figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading boundary…")
    rings = load_boundary_rings()

    print("Loading predictions and computing distances…")
    df = load_data()
    print(f"  {len(df)} holdout wells across {len(SEEDS)} seeds")

    plot_k1_uncapped(df, rings, out_dir)
    plot_k1_cap4km(df, rings, out_dir)
    plot_k1_cap4km(df, rings, out_dir, show_title=False)
    plot_k3_cap4km(df, rings, out_dir)
    plot_hydroraum(df, rings, out_dir)
    plot_hydroraum_recharge(df, rings, out_dir)
    plot_recharge_transit(df, rings, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
