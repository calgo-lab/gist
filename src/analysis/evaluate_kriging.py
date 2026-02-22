from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import json
import numpy as np
import pandas as pd
import yaml
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pyproj import Transformer
from libs.run_sig import resolve_tft_run_sig
from libs.spatial_split import resolve_split_path


def _load_boundary_rings(path, names):
    data = json.load(open(path))
    feats = [
        f for f in data.get("features", [])
        if f.get("properties", {}).get("shapeName") in names
    ]
    if not feats:
        return []
    tr = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
    rings = []
    for feat in feats:
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        polys = [coords] if gtype == "Polygon" else coords if gtype == "MultiPolygon" else []
        for poly in polys:
            for ring in poly:
                xs_r, ys_r = zip(*ring)
                xx_r, yy_r = tr.transform(xs_r, ys_r)
                rings.append((np.asarray(xx_r), np.asarray(yy_r)))
    return rings


def _plot_admin_boundaries(ax, rings, color="white", linewidth=1.2, alpha=0.9):
    for xr, yr in rings:
        ax.plot(xr, yr, color=color, linewidth=linewidth, alpha=alpha)


def main():
    global ROOT
    parser = argparse.ArgumentParser(
        description="Evaluate kriging outputs: plots + horizon variation analysis."
    )
    parser.add_argument("--run-sig", dest="run_sig", default=None)
    parser.add_argument("--dataset", dest="dataset", default=None)
    parser.add_argument("--model", dest="model", default=None)
    parser.add_argument(
        "--kriging-source", dest="kriging_source", choices=["pred", "true"], default=None
    )
    parser.add_argument("--split-tag", dest="spatial_split_tag", default=None)
    args = parser.parse_args()

    repo_root = ROOT.resolve()
    for p in [repo_root, *repo_root.parents]:
        if (p / "configs" / "data.yaml").exists():
            repo_root = p
            break
    else:
        raise FileNotFoundError("Repo root not found (configs/data.yaml missing).")
    ROOT = repo_root
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    cfg = yaml.safe_load((ROOT / "configs" / "kriging.yaml").read_text(encoding="utf-8")) or {}
    tft_cfg = yaml.safe_load((ROOT / "configs" / "tft.yaml").read_text(encoding="utf-8")) or {}
    data_cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8")) or {}
    for key in ["full_raw_path", "metadata_path", "full_merged_path", "sample_path"]:
        if key in data_cfg and data_cfg[key]:
            p = Path(data_cfg[key])
            data_cfg[key] = str(p if p.is_absolute() else (ROOT / p).resolve())

    dataset = args.dataset or cfg.get("dataset") or tft_cfg.get("dataset", "full_merged")
    model = args.model or cfg.get("model") or tft_cfg.get("model_name") or tft_cfg.get("model") or "TFT"
    if not isinstance(model, str):
        model = "TFT"
    run_sig = (args.run_sig or cfg.get("run_sig") or "").strip()
    if (not run_sig) or run_sig.lower() == "auto":
        run_sig = resolve_tft_run_sig(tft_cfg, model=str(model))
    if not run_sig:
        raise ValueError("run_sig must be set in configs/tft.yaml or configs/kriging.yaml")

    split_tag = str(args.spatial_split_tag or cfg.get("spatial_split_tag", "")).strip()
    kriging_source = str(args.kriging_source or cfg.get("kriging_source", "pred")).strip().lower()
    value_tag = "predobstrain" if kriging_source == "pred" else "trueobstrain"
    name_parts = [run_sig, split_tag, value_tag]
    run_tag = "__".join([p for p in name_parts if p])

    model_run_sig = run_sig if run_sig.startswith(f"{model}_") else f"{model}_{run_sig}"
    spatial_cfg = tft_cfg.get("spatial_split", {}) if isinstance(tft_cfg.get("spatial_split", {}), dict) else {}
    split_file_override = str(cfg.get("spatial_split_file", "")).strip()
    if split_file_override:
        spatial_cfg = dict(spatial_cfg)
        spatial_cfg["file"] = split_file_override
    split_path = resolve_split_path(ROOT / "splits", dataset, spatial_cfg)

    if dataset == "sample":
        data_path = Path(data_cfg["sample_path"])
    elif dataset == "full_raw":
        data_path = Path(data_cfg["full_raw_path"])
    else:
        data_path = Path(data_cfg["full_merged_path"])
    data_path = data_path if data_path.is_absolute() else (ROOT / data_path).resolve()

    gp_dir = ROOT / "outputs" / "gp" / run_tag
    gp_pred_path = gp_dir / "gp_pred.parquet"
    gp_metrics_path = gp_dir / "gp_metrics.parquet"
    gp_metrics_by_pair_path = gp_dir / "gp_metrics_by_pair.csv"
    gp_grid_path = gp_dir / "gp_grid.npz"

    for p, name in [
        (gp_pred_path, "gp_pred.parquet"),
        (gp_metrics_path, "gp_metrics.parquet"),
        (gp_grid_path, "gp_grid.npz"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name} at {p}. Run kriging.py first.")

    # --- Load GP outputs ---
    gp_pred = pq.read_table(gp_pred_path).to_pandas()
    gp_pred["datum"] = pd.to_datetime(gp_pred["datum"])
    gp_metrics_df = pq.read_table(gp_metrics_path).to_pandas()
    overall_metrics = gp_metrics_df.set_index("metric")["value"]
    if "NSE_pooled" not in overall_metrics.index and "NSE" in overall_metrics.index:
        overall_metrics.loc["NSE_pooled"] = overall_metrics.loc["NSE"]
    per_pair_metrics = (
        pd.read_csv(gp_metrics_by_pair_path)
        if gp_metrics_by_pair_path.exists()
        else pd.DataFrame()
    )

    # --- Load grid ---
    grid_data = np.load(gp_grid_path)
    pooled_grid = grid_data["pooled_grid"]
    xmin, xmax = float(grid_data["xmin"]), float(grid_data["xmax"])
    ymin, ymax = float(grid_data["ymin"]), float(grid_data["ymax"])
    grid_nx, grid_ny = int(grid_data["grid_nx"]), int(grid_data["grid_ny"])
    xs = np.linspace(xmin, xmax, grid_nx)
    ys = np.linspace(ymin, ymax, grid_ny)
    xx, yy = np.meshgrid(xs, ys)

    # --- Load spatial split + coordinates ---
    cols = ["id", "x_25833", "y_25833"]
    if str(data_path).lower().endswith(".csv"):
        raw_coords = pd.read_csv(data_path, usecols=cols, low_memory=False)
    else:
        raw_coords = pq.read_table(data_path, columns=cols).to_pandas()
    coords = raw_coords.drop_duplicates("id")

    split = pd.read_csv(split_path)
    train_ids = set(split.loc[split["spatial_split"] == "spatial_train", "id"])
    holdout_ids = set(split.loc[split["spatial_split"] == "spatial_holdout", "id"])
    train_coords = coords[coords["id"].isin(train_ids)]
    holdout_coords = coords[coords["id"].isin(holdout_ids)]

    # --- Compute pooled holdout errors from gp_pred ---
    gp_pred["diff"] = gp_pred["gws_forecast"] - gp_pred["gws_true"]
    gp_pred["abs_err"] = gp_pred["diff"].abs()
    pooled_holdout = (
        gp_pred.groupby(["id", "x_25833", "y_25833"], as_index=False)
        .agg(diff=("diff", "median"), abs_err=("abs_err", "median"))
    )

    # --- Load boundary ---
    boundary_path = ROOT / "data" / "boundaries" / "geoBoundaries-DEU-ADM1_simplified.geojson"
    if not boundary_path.exists():
        raise FileNotFoundError(f"Missing boundary file: {boundary_path}")
    boundary_rings = _load_boundary_rings(boundary_path, {"Brandenburg"})

    # --- Build filename suffix from the loaded data ---
    valid_pairs = gp_pred.groupby(["datum", "horizon"]).ngroups
    date_min = gp_pred["datum"].min().date()
    date_max = gp_pred["datum"].max().date()
    h_min = float(gp_pred["horizon"].min())
    h_max = float(gp_pred["horizon"].max())
    suffix = f"{run_tag}__pairs{valid_pairs}__d{date_min}_to_{date_max}__h{h_min:g}_to_{h_max:g}"

    figures_dir = ROOT / "reports" / "kriging" / "figures" / f"pairs{valid_pairs}"
    figures_dir.mkdir(parents=True, exist_ok=True)

    vmin = float(np.nanmin(pooled_grid))
    vmax = float(np.nanmax(pooled_grid))
    norm_pred = plt.Normalize(vmin=vmin, vmax=vmax)

    # --- Plot 1: Kriging map with train/holdout wells ---
    fig, ax = plt.subplots(figsize=(8, 7))
    cf = ax.pcolormesh(xx, yy, pooled_grid, shading="auto", cmap="turbo", norm=norm_pred)
    plt.colorbar(cf, ax=ax, label="GWS (pooled kriged)")
    _plot_admin_boundaries(ax, boundary_rings)
    ax.scatter(
        train_coords["x_25833"], train_coords["y_25833"],
        s=18, c="black", label="spatial_train", alpha=0.6,
    )
    ax.scatter(
        holdout_coords["x_25833"], holdout_coords["y_25833"],
        s=24, c="red", label="spatial_holdout", alpha=0.8,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Kriging map (pred) with train/holdout wells")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / f"pooled_pred_train_holdout__{suffix}.png", dpi=200)
    plt.close(fig)

    # --- Plot 1b: Spatial split styling on pooled kriged background ---
    fig1b, ax1b = plt.subplots(figsize=(8, 7))
    cf1b = ax1b.pcolormesh(xx, yy, pooled_grid, shading="auto", cmap="turbo", norm=norm_pred)
    plt.colorbar(cf1b, ax=ax1b, label="GWS (pooled kriged)")
    _plot_admin_boundaries(ax1b, boundary_rings)
    ax1b.scatter(
        train_coords["x_25833"], train_coords["y_25833"],
        s=20, c="tab:blue", label="spatial_train", alpha=0.75,
    )
    ax1b.scatter(
        holdout_coords["x_25833"], holdout_coords["y_25833"],
        s=24, c="tab:orange", label="spatial_holdout", alpha=0.85,
    )
    for spine in ax1b.spines.values():
        spine.set_visible(False)
    ax1b.set_xticks([])
    ax1b.set_yticks([])
    ax1b.set_title("Spatial split on pooled kriged GWS background")
    ax1b.legend(loc="lower left", frameon=False)
    fig1b.tight_layout()
    fig1b.savefig(figures_dir / f"pooled_pred_train_holdout_split__{suffix}.png", dpi=200)
    plt.close(fig1b)

    # --- Plot 2: Kriging map with holdout errors ---
    err_abs = pooled_holdout["abs_err"].values
    min_size, max_size = 20, 200
    if err_abs.max() == err_abs.min():
        sizes = np.full_like(err_abs, (min_size + max_size) / 2.0, dtype=float)
    else:
        sizes = (
            min_size
            + (err_abs - err_abs.min()) * (max_size - min_size) / (err_abs.max() - err_abs.min())
        )

    vmax_err = float(np.nanmax(np.abs(pooled_holdout["diff"].values))) if len(pooled_holdout) else 1.0
    if not np.isfinite(vmax_err) or vmax_err == 0:
        vmax_err = 1.0
    norm_err = TwoSlopeNorm(vmin=-vmax_err, vcenter=0.0, vmax=vmax_err)

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    cf2 = ax2.pcolormesh(xx, yy, pooled_grid, shading="auto", cmap="turbo", norm=norm_pred)
    plt.colorbar(cf2, ax=ax2, label="GWS (pooled kriged)")
    _plot_admin_boundaries(ax2, boundary_rings)
    sc = ax2.scatter(
        pooled_holdout["x_25833"], pooled_holdout["y_25833"],
        s=sizes, c=pooled_holdout["diff"],
        cmap="seismic", norm=norm_err, alpha=0.8,
        edgecolors="black", linewidths=0.5,
        label="spatial_holdout (pooled diff)",
    )
    plt.colorbar(sc, ax=ax2, label="Kriged - True (pooled)")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Kriging map with holdout error (size=|diff|, color=diff)")
    fig2.tight_layout()
    fig2.savefig(figures_dir / f"pooled_pred_holdout_error__{suffix}.png", dpi=200)
    plt.close(fig2)

    # --- Plot 3: Overall metrics table ---
    metric_order = [
        "RMSE",
        "nRMSE",
        "NSE_pooled",
        "NSE_id_median",
        "NSE_id_p25",
        "NSE_id_p50",
        "NSE_id_p75",
        "NSE_id_mean",
        "NSE_id_count",
        "MAE",
        "AbsErr_P95",
        "AbsErr_P99",
        "AbsErr_Max",
    ]
    metric_label = {
        "NSE_pooled": "NSE (pooled)",
        "NSE_id_median": "NSE id median",
        "NSE_id_p25": "NSE id p25",
        "NSE_id_p50": "NSE id p50",
        "NSE_id_p75": "NSE id p75",
        "NSE_id_mean": "NSE id mean",
        "NSE_id_count": "NSE id count",
        "AbsErr_P95": "AbsErr p95",
        "AbsErr_P99": "AbsErr p99",
        "AbsErr_Max": "AbsErr max",
    }
    metrics_display = overall_metrics.reindex(metric_order)
    table_rows = [[metric_label.get(m, m), f"{v:.3f}" if pd.notna(v) else ""] for m, v in metrics_display.items()]
    nrows = len(table_rows)
    fig_h = max(4.2, 0.34 * nrows)
    fig3, ax3 = plt.subplots(figsize=(7.2, fig_h))
    ax3.axis("off")
    tbl = ax3.table(
        cellText=table_rows, colLabels=["Metric", "Value"], cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.15)
    ax3.set_title("Kriging metrics (overall)", pad=18)
    fig3.tight_layout(rect=[0, 0, 1, 0.92])
    fig3.savefig(figures_dir / f"pooled_pred_metrics_table__{suffix}.png", dpi=200)
    plt.close(fig3)

    # --- Plot 4: Per-pair metrics table ---
    if not per_pair_metrics.empty:
        disp = per_pair_metrics.copy()
        disp["datum"] = pd.to_datetime(disp["datum"]).dt.date.astype(str)
        disp["horizon"] = disp["horizon"].astype(int, errors="ignore")
        disp["RMSE"] = disp["RMSE"].map(lambda v: f"{v:.3f}")
        disp["NSE"] = disp["NSE"].map(lambda v: f"{v:.3f}")
        table_rows4 = disp[["datum", "horizon", "RMSE", "NSE"]].values.tolist()
        nrows = len(table_rows4)
        fig_h = max(3.0, 0.28 * nrows)
        fig4, ax4 = plt.subplots(figsize=(8, fig_h))
        ax4.axis("off")
        tbl2 = ax4.table(
            cellText=table_rows4,
            colLabels=["Date", "Horizon", "RMSE", "NSE"],
            cellLoc="center",
            loc="center",
        )
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(9)
        tbl2.scale(1.0, 1.15)
        ax4.set_title("Kriging per-pair metrics (RMSE, NSE)", pad=18)
        fig4.tight_layout(rect=[0, 0, 1, 0.92])
        fig4.savefig(figures_dir / f"pooled_pred_metrics_by_pair__{suffix}.png", dpi=200)
        plt.close(fig4)

    print(f"Plots saved to {figures_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
