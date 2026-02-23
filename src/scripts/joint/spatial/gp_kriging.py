from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import yaml
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from libs.run_sig import resolve_tft_run_sig
from libs.spatial_split import resolve_split_path


def _resolve_gru_run_sig(tft_cfg: dict) -> str:
    run_sig_cfg = str(tft_cfg.get("run_sig", "")).strip()
    if run_sig_cfg and run_sig_cfg.lower() != "auto":
        return run_sig_cfg

    data_cfg = tft_cfg.get("data", {}) if isinstance(tft_cfg.get("data", {}), dict) else {}
    tr_cfg = tft_cfg.get("training", {}) if isinstance(tft_cfg.get("training", {}), dict) else {}
    sp_cfg = tft_cfg.get("spatial_split", {}) if isinstance(tft_cfg.get("spatial_split", {}), dict) else {}

    dataset = str(tft_cfg.get("dataset", "full_merged"))
    in_len = int(data_cfg.get("in_len", 52))
    out_len = int(data_cfg.get("out_len", 16))
    epochs = int(tr_cfg.get("epochs", 20))
    bs = int(tr_cfg.get("batch_size", 1024))
    seed = int(tr_cfg.get("seed", 40))
    spf = str(float(sp_cfg.get("train_fraction", 0.8))).replace(".", "p")
    sc = int(sp_cfg.get("cluster_count", 20))
    ss = int(sp_cfg.get("split_seed", 42))
    return f"in{in_len}_out{out_len}_ep{epochs}_bs{bs}_seed{seed}_{dataset}_spf{spf}_sc{sc}_ss{ss}"


def _split_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = []
        for v in value:
            items.extend(str(v).split(","))
    else:
        items = str(value).split(",")
    return [v.strip() for v in items if str(v).strip()]


def _load_pred(pred_path: Path) -> pd.DataFrame:
    pred = pq.read_table(pred_path).to_pandas()
    pred["datum"] = pd.to_datetime(pred["datum"])

    if "gws_forecast" in pred.columns:
        pred["gws_pred"] = pred["gws_forecast"]
    elif "gws" in pred.columns:
        pred["gws_pred"] = pred["gws"]
    else:
        raise ValueError("Prediction file must have either 'gws_forecast' or 'gws'.")

    if "horizon" not in pred.columns:
        if "startzeitpunkt" not in pred.columns:
            raise ValueError("Prediction file needs 'horizon' or 'startzeitpunkt'.")
        pred["startzeitpunkt"] = pd.to_datetime(pred["startzeitpunkt"])
        pred["horizon"] = ((pred["datum"] - pred["startzeitpunkt"]) / pd.Timedelta(weeks=1)) + 1

    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone GP/kriging baseline for TFT/GRU prediction outputs.")
    parser.add_argument("--config", default="configs/gp.yaml", help="Path to gp config yaml.")
    parser.add_argument("--model", default=None, help="Temporal model folder, e.g. TFT or GRU_FCOV.")
    parser.add_argument("--run-sig", default=None, help="Temporal run signature (without model prefix).")
    parser.add_argument("--dataset", default=None, help="Dataset key: sample/full_raw/full_merged.")
    parser.add_argument("--pred-path", default=None, help="Optional explicit pred.parquet path.")
    parser.add_argument("--split-tag", dest="spatial_split_tag", default=None)
    parser.add_argument("--date", dest="kriging_date", default=None)
    parser.add_argument("--dates", dest="kriging_dates", nargs="+", default=None)
    parser.add_argument("--horizon", type=float, default=None)
    parser.add_argument("--horizons", nargs="+", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg = cfg if isinstance(cfg, dict) else {}
    tft_cfg = yaml.safe_load((ROOT / "configs" / "tft.yaml").read_text(encoding="utf-8")) or {}
    data_cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8")) or {}

    for key in ["full_raw_path", "metadata_path", "full_merged_path", "sample_path"]:
        if key in data_cfg and data_cfg[key]:
            p = Path(data_cfg[key])
            data_cfg[key] = str(p if p.is_absolute() else (ROOT / p).resolve())

    model = str(args.model or cfg.get("model") or "GRU_FCOV").strip()
    dataset = str(args.dataset or cfg.get("dataset") or tft_cfg.get("dataset", "full_merged")).strip()
    run_sig = str(args.run_sig or cfg.get("run_sig") or "").strip()
    if not run_sig or run_sig.lower() == "auto":
        if model.upper() == "TFT":
            run_sig = resolve_tft_run_sig(tft_cfg, model="TFT")
        else:
            run_sig = _resolve_gru_run_sig(tft_cfg)
    if not run_sig:
        raise ValueError("Could not resolve run_sig. Set --run-sig or config run_sig.")

    spatial_cfg = tft_cfg.get("spatial_split", {}) if isinstance(tft_cfg.get("spatial_split", {}), dict) else {}
    split_file_override = str(cfg.get("spatial_split_file", "")).strip()
    if split_file_override:
        spatial_cfg = dict(spatial_cfg)
        spatial_cfg["file"] = split_file_override
    split_path = resolve_split_path(ROOT / "splits", dataset, spatial_cfg)

    if args.pred_path:
        pred_path = Path(args.pred_path)
    else:
        model_run_sig = run_sig if run_sig.startswith(f"{model}_") else f"{model}_{run_sig}"
        pred_path = ROOT / "outputs" / model / model_run_sig / "predictions" / "pred.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing pred.parquet: {pred_path}")

    if dataset == "sample":
        data_path = Path(data_cfg["sample_path"])
    elif dataset == "full_raw":
        data_path = Path(data_cfg["full_raw_path"])
    else:
        data_path = Path(data_cfg["full_merged_path"])
    data_path = data_path if data_path.is_absolute() else (ROOT / data_path).resolve()

    cols = ["datum", "id", "gws", "x_25833", "y_25833"]
    if str(data_path).lower().endswith(".csv"):
        gws = pd.read_csv(data_path, usecols=cols, low_memory=False)
    else:
        gws = pq.read_table(data_path, columns=cols).to_pandas()
    gws["datum"] = pd.to_datetime(gws["datum"])

    pred = _load_pred(pred_path)
    split = pd.read_csv(split_path)
    train_ids = set(split.loc[split["spatial_split"] == "spatial_train", "id"])
    holdout_ids = set(split.loc[split["spatial_split"] == "spatial_holdout", "id"])

    coords = gws[["id", "x_25833", "y_25833"]].drop_duplicates("id")
    if "id" not in pred.columns and "index" in pred.columns:
        gws_train = gws[gws["id"].isin(train_ids)]
        ids_keep = pd.unique(gws_train["id"])
        lookup_ids = pd.DataFrame(ids_keep, columns=["id"]).reset_index()
        pred = pred.merge(lookup_ids, on="index", how="left")
    pred = pred.merge(coords, on="id", how="left")

    xmin, xmax = coords["x_25833"].min(), coords["x_25833"].max()
    ymin, ymax = coords["y_25833"].min(), coords["y_25833"].max()
    grid_nx = int(cfg.get("grid_nx", 200))
    grid_ny = int(cfg.get("grid_ny", 200))
    xs = np.linspace(xmin, xmax, grid_nx)
    ys = np.linspace(ymin, ymax, grid_ny)
    xx, yy = np.meshgrid(xs, ys)
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])

    horizon_default = float(cfg.get("horizon", tft_cfg.get("data", {}).get("out_len", 16)))
    target_date_default = str(cfg.get("kriging_date", "")).strip()
    date_strategy = str(cfg.get("date_strategy", "")).strip().lower()

    cli_horizons = _split_list(args.horizons)
    cfg_horizons = _split_list(cfg.get("horizons"))
    if args.horizon is not None:
        horizons = [float(args.horizon)]
    elif cli_horizons:
        horizons = [float(h) for h in cli_horizons]
    elif cfg_horizons:
        horizons = [float(h) for h in cfg_horizons]
    else:
        horizons = [float(horizon_default)]

    cli_dates = _split_list(args.kriging_dates)
    cfg_dates = _split_list(cfg.get("kriging_dates"))
    target_date = (args.kriging_date or target_date_default).strip()
    if cli_dates:
        dates = [pd.to_datetime(d) for d in cli_dates]
    elif cfg_dates:
        dates = [pd.to_datetime(d) for d in cfg_dates]
    elif target_date:
        dates = [pd.to_datetime(target_date)]
    elif date_strategy in ("monthly", "biweekly"):
        avail = pred[np.isclose(pred["horizon"], horizons[0])]["datum"].drop_duplicates().sort_values()
        freq = "ME" if date_strategy == "monthly" else "2W"
        dates = pd.Series(avail.values, index=avail).resample(freq).first().dropna().tolist()
    else:
        dates = [pred["datum"].max()]

    kernel_cfg = cfg.get("kernel", {})
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=float(kernel_cfg.get("length_scale", 1.0)), nu=float(kernel_cfg.get("nu", 1.5)))
        + WhiteKernel(noise_level=float(kernel_cfg.get("noise", 1e-3)))
    )
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 40))
    restarts = int(kernel_cfg.get("restarts", 2))
    max_pairs = args.max_pairs if args.max_pairs is not None else int(cfg.get("max_pairs", 0) or 0)

    out_rows = []
    grid_rows = []
    valid_pairs = 0
    attempted_pairs = 0
    skipped = []
    for dt in dates:
        for h in horizons:
            attempted_pairs += 1
            gws_dt = gws[gws["datum"] == dt][["id", "gws", "x_25833", "y_25833"]]
            holdout_true = gws_dt[gws_dt["id"].isin(holdout_ids)].dropna()

            pred_sel = pred[(pred["datum"] == dt) & (np.isclose(pred["horizon"], h))]
            pred_sel = pred_sel[pred_sel["id"].isin(train_ids)].dropna(subset=["x_25833", "y_25833", "gws_pred"])
            X_train = pred_sel[["x_25833", "y_25833"]].to_numpy()
            y_train = pred_sel["gws_pred"].to_numpy()
            if X_train.size == 0 or holdout_true.empty:
                reason = "no training samples" if X_train.size == 0 else "no holdout samples"
                skipped.append((dt, h, reason))
                continue

            scaler = StandardScaler().fit(X_train)
            X_train_s = scaler.transform(X_train)
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.0,
                normalize_y=True,
                n_restarts_optimizer=restarts,
                random_state=seed,
            )
            gpr.fit(X_train_s, y_train)

            X_hold = scaler.transform(holdout_true[["x_25833", "y_25833"]].to_numpy())
            y_pred, y_std = gpr.predict(X_hold, return_std=True)

            X_grid_s = scaler.transform(X_grid)
            z = gpr.predict(X_grid_s).reshape(xx.shape)
            grid_rows.append(z)

            out = holdout_true.copy()
            out["gws_true"] = out["gws"]
            out.drop(columns=["gws"], inplace=True)
            out["gws_forecast"] = y_pred
            out["gws_forecast_std"] = y_std
            out["horizon"] = h
            out["datum"] = dt
            out_rows.append(out)
            valid_pairs += 1

            if max_pairs and valid_pairs >= max_pairs:
                break
        if max_pairs and valid_pairs >= max_pairs:
            break

    if not out_rows:
        raise ValueError("No GP outputs generated. Check date/horizon availability.")
    out_df = pd.concat(out_rows, ignore_index=True)

    def _nrmse(pred_vals, real):
        iqr = np.diff(np.quantile(real, q=[0.25, 0.75]))[0]
        return np.sqrt(np.square(pred_vals - real).mean(axis=0)) / iqr if iqr != 0 else np.nan

    def _rmse(pred_vals, real):
        return np.sqrt(np.square(pred_vals - real).mean(axis=0))

    def _nse(pred_vals, real):
        denom = np.sum((real - np.mean(real)) ** 2)
        if denom == 0:
            return np.nan
        return 1 - (np.sum((pred_vals - real) ** 2) / denom)

    pred_all = out_df["gws_forecast"].to_numpy()
    real_all = out_df["gws_true"].to_numpy()
    abs_err_all = np.abs(pred_all - real_all)

    per_id_nse_rows = []
    for well_id, g in out_df.groupby("id"):
        per_id_nse_rows.append({"id": well_id, "NSE_over_time": float(_nse(g["gws_forecast"].to_numpy(), g["gws_true"].to_numpy()))})
    per_id_nse_df = pd.DataFrame(per_id_nse_rows)
    per_id_vals = per_id_nse_df["NSE_over_time"].to_numpy(dtype=float) if not per_id_nse_df.empty else np.array([])
    per_id_valid = per_id_vals[np.isfinite(per_id_vals)] if per_id_vals.size else np.array([])

    overall_metrics = pd.Series(
        {
            "RMSE": float(_rmse(pred_all, real_all)),
            "nRMSE": float(_nrmse(pred_all, real_all)),
            "NSE_pooled": float(_nse(pred_all, real_all)),
            "NSE": float(_nse(pred_all, real_all)),
            "NSE_id_median": float(np.median(per_id_valid)) if per_id_valid.size else np.nan,
            "NSE_id_p25": float(np.percentile(per_id_valid, 25)) if per_id_valid.size else np.nan,
            "NSE_id_p50": float(np.percentile(per_id_valid, 50)) if per_id_valid.size else np.nan,
            "NSE_id_p75": float(np.percentile(per_id_valid, 75)) if per_id_valid.size else np.nan,
            "NSE_id_mean": float(np.mean(per_id_valid)) if per_id_valid.size else np.nan,
            "NSE_id_count": float(per_id_valid.size),
            "MAE": float(np.mean(abs_err_all)),
            "AbsErr_P95": float(np.percentile(abs_err_all, 95)) if abs_err_all.size else np.nan,
            "AbsErr_P99": float(np.percentile(abs_err_all, 99)) if abs_err_all.size else np.nan,
            "AbsErr_Max": float(np.max(abs_err_all)) if abs_err_all.size else np.nan,
        }
    )

    pair_rows = []
    for (dt, h), g in out_df.groupby(["datum", "horizon"]):
        real = g["gws_true"].to_numpy()
        pred_vals = g["gws_forecast"].to_numpy()
        pair_rows.append(
            {
                "datum": pd.to_datetime(dt),
                "horizon": float(h),
                "n": int(len(real)),
                "RMSE": float(_rmse(pred_vals, real)),
                "nRMSE": float(_nrmse(pred_vals, real)),
                "NSE": float(_nse(pred_vals, real)),
                "MAE": float(np.mean(np.abs(pred_vals - real))),
            }
        )
    per_pair_metrics = pd.DataFrame(pair_rows).sort_values(["datum", "horizon"])

    split_tag = str(args.spatial_split_tag or cfg.get("spatial_split_tag", "")).strip()
    run_tag = "__".join([p for p in [model, run_sig, split_tag, "predobstrain"] if p])
    gp_dir = ROOT / "outputs" / "gp" / run_tag
    gp_dir.mkdir(parents=True, exist_ok=True)

    pq.write_table(pa.Table.from_pandas(out_df), gp_dir / "gp_pred.parquet")
    metrics_df = overall_metrics.reset_index()
    metrics_df.columns = ["metric", "value"]
    pq.write_table(pa.Table.from_pandas(metrics_df), gp_dir / "gp_metrics.parquet")
    per_pair_metrics.to_csv(gp_dir / "gp_metrics_by_pair.csv", index=False)
    if not per_id_nse_df.empty:
        per_id_nse_df.to_csv(gp_dir / "gp_metrics_by_id.csv", index=False)

    pooled_grid = np.nanmedian(np.stack(grid_rows, axis=0), axis=0)
    np.savez(
        gp_dir / "gp_grid.npz",
        pooled_grid=pooled_grid,
        xmin=float(xmin),
        xmax=float(xmax),
        ymin=float(ymin),
        ymax=float(ymax),
        grid_nx=int(grid_nx),
        grid_ny=int(grid_ny),
    )

    print(f"Saved GP outputs to {gp_dir}")
    print(f"Valid pairs: {valid_pairs} / attempted: {attempted_pairs}")
    if skipped:
        preview = ", ".join([f"{d.date()}:{h}({r})" for d, h, r in skipped[:8]])
        print(f"Skipped {len(skipped)} pairs (preview: {preview})")


if __name__ == "__main__":
    main()

