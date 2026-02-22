from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import yaml
from pyproj import Transformer
from libs.run_sig import resolve_tft_run_sig
from libs.spatial_split import resolve_split_path


ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _resolve_data_path(root: Path, dataset: str, data_cfg: dict) -> Path:
    if dataset == "sample":
        rel = data_cfg["sample_path"]
    elif dataset == "full_raw":
        rel = data_cfg["full_raw_path"]
    else:
        rel = data_cfg["full_merged_path"]
    p = Path(rel)
    return p if p.is_absolute() else (root / p).resolve()


def _resolve_run_tag(root: Path, args: argparse.Namespace) -> tuple[str, str]:
    cfg = _load_yaml(root / "configs" / "kriging.yaml")
    tft_cfg = _load_yaml(root / "configs" / "tft.yaml")

    dataset = args.dataset or cfg.get("dataset") or tft_cfg.get("dataset", "full_merged")
    run_sig = (args.run_sig or cfg.get("run_sig") or "").strip()
    if (not run_sig) or run_sig.lower() == "auto":
        run_sig = resolve_tft_run_sig(tft_cfg, model="TFT")
    if not run_sig:
        raise ValueError("run_sig must be set in configs/tft.yaml or passed via --run-sig.")

    split_tag = str(args.spatial_split_tag or cfg.get("spatial_split_tag", "")).strip()
    kriging_source = str(args.kriging_source or cfg.get("kriging_source", "pred")).strip().lower()
    if kriging_source not in {"pred", "true"}:
        raise ValueError("kriging_source must be 'pred' or 'true'.")
    value_tag = "predobstrain" if kriging_source == "pred" else "trueobstrain"
    run_tag = "__".join([p for p in [run_sig, split_tag, value_tag] if p])
    return str(dataset), run_tag


def _resolve_model_run_sig(root: Path, args: argparse.Namespace) -> tuple[str, str]:
    cfg = _load_yaml(root / "configs" / "kriging.yaml")
    tft_cfg = _load_yaml(root / "configs" / "tft.yaml")
    model = args.model or cfg.get("model") or tft_cfg.get("model_name") or tft_cfg.get("model") or "TFT"
    if not isinstance(model, str):
        model = "TFT"
    run_sig = (args.run_sig or cfg.get("run_sig") or "").strip()
    if (not run_sig) or run_sig.lower() == "auto":
        run_sig = resolve_tft_run_sig(tft_cfg, model=model if isinstance(model, str) else "TFT")
    if not run_sig:
        raise ValueError("run_sig must be set in configs/tft.yaml or passed via --run-sig.")
    model_run_sig = run_sig if run_sig.startswith(f"{model}_") else f"{model}_{run_sig}"
    return model, model_run_sig


def _load_boundary_rings(path: Path, names: set[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = [
        f for f in data.get("features", [])
        if f.get("properties", {}).get("shapeName") in names
    ]
    if not feats:
        return []
    tr = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
    rings: list[tuple[np.ndarray, np.ndarray]] = []
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


def _plot_admin_boundaries(ax, rings, color="0.35", linewidth=1.0, alpha=0.9):
    for xr, yr in rings:
        ax.plot(xr, yr, color=color, linewidth=linewidth, alpha=alpha, zorder=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze holdout kriging error by spatial cluster.")
    parser.add_argument("--run-sig", dest="run_sig", default=None)
    parser.add_argument("--dataset", dest="dataset", default=None)
    parser.add_argument("--model", dest="model", default=None)
    parser.add_argument("--kriging-source", dest="kriging_source", choices=["pred", "true"], default=None)
    parser.add_argument("--split-tag", dest="spatial_split_tag", default=None)
    parser.add_argument("--outlier-top-n", dest="outlier_top_n", type=int, default=12)
    args = parser.parse_args()

    data_cfg = _load_yaml(ROOT / "configs" / "data.yaml")
    dataset, run_tag = _resolve_run_tag(ROOT, args)
    gp_pred_path = ROOT / "outputs" / "gp" / run_tag / "gp_pred.parquet"
    if not gp_pred_path.exists():
        raise FileNotFoundError(f"Missing {gp_pred_path}. Run kriging first.")

    gp = pq.read_table(gp_pred_path).to_pandas()
    gp["datum"] = pd.to_datetime(gp["datum"])
    gp["err"] = gp["gws_forecast"] - gp["gws_true"]
    gp["abs_err"] = gp["err"].abs()

    cfg = _load_yaml(ROOT / "configs" / "kriging.yaml")
    tft_cfg = _load_yaml(ROOT / "configs" / "tft.yaml")
    spatial_cfg = tft_cfg.get("spatial_split", {}) if isinstance(tft_cfg.get("spatial_split", {}), dict) else {}
    split_file_override = str(cfg.get("spatial_split_file", "")).strip()
    if split_file_override:
        spatial_cfg = dict(spatial_cfg)
        spatial_cfg["file"] = split_file_override
    split_path = resolve_split_path(ROOT / "splits", dataset, spatial_cfg)
    split = pd.read_csv(split_path)
    split = split[["id", "cluster", "spatial_split"]].copy()
    train_ids = set(split.loc[split["spatial_split"] == "spatial_train", "id"])

    well = (
        gp.groupby(["id", "x_25833", "y_25833"], as_index=False)
        .agg(
            n_eval_points=("id", "size"),
            mae=("abs_err", "mean"),
            bias=("err", "mean"),
            median_abs_err=("abs_err", "median"),
        )
    )
    rmse = gp.groupby("id")["err"].apply(lambda e: float(np.sqrt(np.mean(np.square(e.to_numpy(dtype=float))))))
    well = well.merge(rmse.rename("rmse"), on="id", how="left")
    well = well.merge(split, on="id", how="left")

    data_path = _resolve_data_path(ROOT, dataset, data_cfg)
    base_cols = ["id", "datum", "gws", "x_25833", "y_25833"]
    if str(data_path).lower().endswith(".csv"):
        full = pd.read_csv(data_path, usecols=base_cols, low_memory=False)
    else:
        full = pq.read_table(data_path, columns=base_cols).to_pandas()
    n_obs = full.groupby("id", as_index=False).size().rename(columns={"size": "n_obs_full"})
    well = well.merge(n_obs, on="id", how="left")
    outlier_top_n = max(1, int(args.outlier_top_n))
    outlier_ids = set(well.sort_values("mae", ascending=False)["id"].head(outlier_top_n).tolist())
    well["is_outlier_topn"] = well["id"].isin(outlier_ids)

    cluster_df = (
        well.groupby("cluster", as_index=False)
        .agg(
            n_wells=("id", "nunique"),
            mae_median=("mae", "median"),
            rmse_median=("rmse", "median"),
            n_outliers=("is_outlier_topn", "sum"),
        )
        .sort_values("mae_median", ascending=False)
        .reset_index(drop=True)
    )
    cluster_df["n_outliers"] = cluster_df["n_outliers"].astype(int)
    total_wells_considered = int(well["id"].nunique())
    total_outliers = int(well["is_outlier_topn"].sum())
    outlier_rate = (total_outliers / total_wells_considered) if total_wells_considered > 0 else 0.0
    cluster_df["expected"] = cluster_df["n_wells"] * outlier_rate
    cluster_df = cluster_df[["cluster", "n_wells", "mae_median", "rmse_median", "n_outliers", "expected"]]

    out_dir = ROOT / "reports" / "kriging" / "outlier_analysis" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cluster_df.to_csv(out_dir / "error_by_cluster.csv", index=False)

    train_wells = (
        full[full["id"].isin(train_ids)][["id", "x_25833", "y_25833"]]
        .drop_duplicates("id")
        .merge(n_obs, on="id", how="left")
    )

    model, model_run_sig = _resolve_model_run_sig(ROOT, args)
    pred_path = ROOT / "outputs" / model / model_run_sig / "predictions" / "pred.parquet"
    train_mae = pd.DataFrame(columns=["id", "tft_mae_train"])
    if pred_path.exists():
        pred = pq.read_table(pred_path).to_pandas()
        pred["datum"] = pd.to_datetime(pred["datum"])
        pred["startzeitpunkt"] = pd.to_datetime(pred["startzeitpunkt"])
        pred["horizon"] = ((pred["datum"] - pred["startzeitpunkt"]) / pd.Timedelta(weeks=1)) + 1

        ids_keep = pd.unique(full[full["id"].isin(train_ids)]["id"])
        lookup_ids = pd.DataFrame(ids_keep, columns=["id"]).reset_index()

        pred = pred.merge(lookup_ids, on="index", how="left")
        pred = pred[pred["id"].isin(train_ids)].copy()
        pred = pred.rename(columns={"gws": "gws_forecast"})
        truth = full[["id", "datum", "gws"]].copy()
        pred = pred.merge(truth, on=["id", "datum"], how="inner")

        gp_pairs = gp[["datum", "horizon"]].drop_duplicates()
        pred = pred.merge(gp_pairs, on=["datum", "horizon"], how="inner")
        pred["abs_err"] = (pred["gws"] - pred["gws_forecast"]).abs()
        train_mae = pred.groupby("id", as_index=False)["abs_err"].mean().rename(columns={"abs_err": "tft_mae_train"})

    train_wells = train_wells.merge(train_mae, on="id", how="left")

    # Plot 1: n_obs_full map for SPATIAL TRAIN wells.
    boundary_path = ROOT / "data" / "boundaries" / "geoBoundaries-DEU-ADM1_simplified.geojson"
    boundary_rings = _load_boundary_rings(boundary_path, {"Brandenburg"}) if boundary_path.exists() else []
    fig2, ax2 = plt.subplots(figsize=(8, 6.2), dpi=180)
    _plot_admin_boundaries(ax2, boundary_rings)
    sc = ax2.scatter(
        train_wells["x_25833"], train_wells["y_25833"],
        c=train_wells["n_obs_full"],
        cmap="turbo",
        s=28,
        alpha=0.92,
        linewidths=0.2,
        edgecolors="black",
        zorder=2,
    )
    out = well[well["is_outlier_topn"]].copy()
    ax2.scatter(
        out["x_25833"], out["y_25833"],
        s=120,
        marker="*",
        c="#ff2d2d",
        edgecolors="white",
        linewidths=0.9,
        zorder=3,
        label=f"Top {outlier_top_n} MAE outliers",
    )
    for _, r in out.iterrows():
        ax2.annotate(
            str(r["id"]),
            (r["x_25833"], r["y_25833"]),
            fontsize=6.5,
            xytext=(3, 2),
            textcoords="offset points",
            zorder=4,
        )
    plt.colorbar(sc, ax=ax2, label="n_obs_full (spatial_train wells)")
    ax2.set_title("Spatial train wells: n_obs_full with top holdout outliers")
    ax2.set_xlabel("x_25833")
    ax2.set_ylabel("y_25833")
    ax2.legend(frameon=False, loc="lower left")
    fig2.tight_layout()
    fig2.savefig(out_dir / "n_obs_full_spatial_train_map.png", dpi=180)
    plt.close(fig2)

    # Plot 2: train-well error map (TFT MAE) with same outlier overlay.
    fig3, ax3 = plt.subplots(figsize=(8, 6.2), dpi=180)
    _plot_admin_boundaries(ax3, boundary_rings)
    tr = train_wells.dropna(subset=["tft_mae_train"]).copy()
    print(f"Train wells with TFT MAE for map: {len(tr)}")
    sc3 = ax3.scatter(
        tr["x_25833"], tr["y_25833"],
        c=tr["tft_mae_train"],
        cmap="turbo",
        s=28,
        alpha=0.92,
        linewidths=0.2,
        edgecolors="black",
        zorder=2,
    )
    ax3.scatter(
        out["x_25833"], out["y_25833"],
        s=120,
        marker="*",
        c="#ff2d2d",
        edgecolors="white",
        linewidths=0.9,
        zorder=3,
        label=f"Top {outlier_top_n} MAE outliers",
    )
    for _, r in out.iterrows():
        ax3.annotate(
            str(r["id"]),
            (r["x_25833"], r["y_25833"]),
            fontsize=6.5,
            xytext=(3, 2),
            textcoords="offset points",
            zorder=4,
        )
    plt.colorbar(sc3, ax=ax3, label="TFT MAE on spatial_train wells")
    ax3.set_title("Spatial train wells: TFT MAE with top holdout outliers")
    ax3.set_xlabel("x_25833")
    ax3.set_ylabel("y_25833")
    ax3.legend(frameon=False, loc="lower left")
    fig3.tight_layout()
    fig3.savefig(out_dir / "spatial_train_tft_mae_map.png", dpi=180)
    plt.close(fig3)

    print(f"Wrote: {out_dir / 'error_by_cluster.csv'}")
    print(f"Wrote: {out_dir / 'n_obs_full_spatial_train_map.png'}")
    print(f"Wrote: {out_dir / 'spatial_train_tft_mae_map.png'}")


if __name__ == "__main__":
    main()
