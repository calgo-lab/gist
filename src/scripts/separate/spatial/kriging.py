from pathlib import Path
import sys
import argparse

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
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from libs.run_sig import resolve_tft_run_sig
from libs.spatial_split import resolve_split_path


def main():
    global ROOT
    parser = argparse.ArgumentParser(description="Run spatial kriging on one or more dates/horizons.")
    parser.add_argument("--date", dest="kriging_date", default=None, help="Single target date (YYYY-MM-DD).")
    parser.add_argument(
        "--dates",
        dest="kriging_dates",
        nargs="+",
        default=None,
        help="Multiple target dates (space- or comma-separated).",
    )
    parser.add_argument("--horizon", dest="horizon", type=float, default=None, help="Single horizon.")
    parser.add_argument(
        "--horizons",
        dest="horizons",
        nargs="+",
        type=float,
        default=None,
        help="Multiple horizons (space- or comma-separated).",
    )
    parser.add_argument("--run-sig", dest="run_sig", default=None)
    parser.add_argument("--dataset", dest="dataset", default=None)
    parser.add_argument("--model", dest="model", default=None)
    parser.add_argument("--kriging-source", dest="kriging_source", choices=["pred", "true"], default=None)
    parser.add_argument("--split-tag", dest="spatial_split_tag", default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--max-pairs", dest="max_pairs", type=int, default=None)
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

    dataset = (args.dataset or cfg.get("dataset") or tft_cfg.get("dataset", "full_merged"))
    model = (args.model or cfg.get("model") or tft_cfg.get("model_name") or tft_cfg.get("model") or "TFT")
    if not isinstance(model, str):
        model = "TFT"
    run_sig = (args.run_sig or cfg.get("run_sig") or "").strip()
    if (not run_sig) or run_sig.lower() == "auto":
        run_sig = resolve_tft_run_sig(tft_cfg, model=str(model))
    if not run_sig:
        raise ValueError("run_sig must be set in configs/tft.yaml or configs/kriging.yaml")

    horizon_default = float(cfg.get("horizon", tft_cfg.get("data", {}).get("out_len", 16)))
    target_date_default = str(cfg.get("kriging_date", "")).strip()
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 40))
    max_pairs = args.max_pairs if args.max_pairs is not None else int(cfg.get("max_pairs", 0) or 0)

    kernel_cfg = cfg.get("kernel", {})
    kernel_nu = float(kernel_cfg.get("nu", 1.5))
    kernel_len = float(kernel_cfg.get("length_scale", 1.0))
    kernel_noise = float(kernel_cfg.get("noise", 1e-3))
    restarts = int(kernel_cfg.get("restarts", 2))
    grid_nx = int(cfg.get("grid_nx", 200))
    grid_ny = int(cfg.get("grid_ny", 200))

    split_tag = str(args.spatial_split_tag or cfg.get("spatial_split_tag", "")).strip()
    kriging_source = str(args.kriging_source or cfg.get("kriging_source", "pred")).strip().lower()
    if kriging_source not in {"pred", "true"}:
        raise ValueError("kriging_source must be 'pred' or 'true' in configs/kriging.yaml")

    value_tag = "predobstrain" if kriging_source == "pred" else "trueobstrain"
    name_parts = [run_sig, split_tag, value_tag]
    run_tag = "__".join([p for p in name_parts if p])

    model_run_sig = run_sig
    if run_sig.startswith(f"{model}_"):
        model_run_sig = run_sig
    else:
        model_run_sig = f"{model}_{run_sig}"
    pred_path = ROOT / "outputs" / model / model_run_sig / "predictions" / "pred.parquet"
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

    cols = ["datum", "id", "gws", "x_25833", "y_25833"]
    if str(data_path).lower().endswith(".csv"):
        gws = pd.read_csv(data_path, usecols=cols, low_memory=False)
    else:
        gws = pq.read_table(data_path, columns=cols).to_pandas()
    gws["datum"] = pd.to_datetime(gws["datum"])

    pred = None
    if kriging_source == "pred":
        pred = pq.read_table(pred_path).to_pandas()
        pred = pred.rename(columns={"gws": "gws_pred"})
        pred["datum"] = pd.to_datetime(pred["datum"])
        pred["startzeitpunkt"] = pd.to_datetime(pred["startzeitpunkt"])
        pred["horizon"] = ((pred["datum"] - pred["startzeitpunkt"]) / pd.Timedelta(weeks=1)) + 1

    split = pd.read_csv(split_path)
    train_ids = set(split.loc[split["spatial_split"] == "spatial_train", "id"])
    holdout_ids = set(split.loc[split["spatial_split"] == "spatial_holdout", "id"])

    coords = gws[["id", "x_25833", "y_25833"]].drop_duplicates("id")

    xmin, xmax = coords["x_25833"].min(), coords["x_25833"].max()
    ymin, ymax = coords["y_25833"].min(), coords["y_25833"].max()
    xs = np.linspace(xmin, xmax, grid_nx)
    ys = np.linspace(ymin, ymax, grid_ny)
    xx, yy = np.meshgrid(xs, ys)
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])

    gws_train = gws[gws["id"].isin(train_ids)]
    ids_keep = pd.unique(gws_train["id"])
    lookup_ids = pd.DataFrame(ids_keep, columns=["id"]).reset_index()

    if pred is not None:
        pred = pred.merge(lookup_ids, on="index", how="left")
        pred = pred.merge(coords, on="id", how="left")

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
    date_strategy = str(cfg.get("date_strategy", "")).strip().lower()

    if cli_dates:
        dates = [pd.to_datetime(d) for d in cli_dates]
    elif cfg_dates:
        dates = [pd.to_datetime(d) for d in cfg_dates]
    elif target_date:
        dates = [pd.to_datetime(target_date)]
    elif date_strategy in ("monthly", "biweekly"):
        if pred is not None:
            avail = pred[np.isclose(pred["horizon"], horizons[0])]["datum"].drop_duplicates().sort_values()
        else:
            avail = gws["datum"].drop_duplicates().sort_values()
        freq = "ME" if date_strategy == "monthly" else "2W"
        dates = pd.Series(avail.values, index=avail).resample(freq).first().dropna().tolist()
    else:
        dates = [gws["datum"].max()]

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=kernel_len, nu=kernel_nu) + WhiteKernel(noise_level=kernel_noise)

    out_rows = []
    grid_rows = []
    skipped_pairs = []
    valid_pairs = 0
    attempted_pairs = 0
    for dt in dates:
        for h in horizons:
            attempted_pairs += 1
            gws_dt = gws[gws["datum"] == dt][["id", "gws", "x_25833", "y_25833"]]
            holdout_true = gws_dt[gws_dt["id"].isin(holdout_ids)].dropna()

            if kriging_source == "pred":
                pred_sel = pred[(pred["datum"] == dt) & (np.isclose(pred["horizon"], h))]
                pred_sel = pred_sel[pred_sel["id"].isin(train_ids)]
                pred_sel = pred_sel.dropna(subset=["x_25833", "y_25833", "gws_pred"])
                X_train = pred_sel[["x_25833", "y_25833"]].to_numpy()
                y_train = pred_sel["gws_pred"].to_numpy()
            else:
                train_true = gws_dt[gws_dt["id"].isin(train_ids)].dropna()
                X_train = train_true[["x_25833", "y_25833"]].to_numpy()
                y_train = train_true["gws"].to_numpy()

            if X_train.size == 0 or holdout_true.empty:
                reason = "no training samples" if X_train.size == 0 else "no holdout samples"
                skipped_pairs.append((dt, h, reason))
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
            print(f"  [{valid_pairs}/{len(dates) * len(horizons)}] {dt.date()} h={h:g} done", file=sys.stderr, flush=True)

            if max_pairs and valid_pairs >= max_pairs:
                break
        if max_pairs and valid_pairs >= max_pairs:
            break

    if not out_rows:
        raise ValueError("No kriging outputs generated. Check date/horizon availability and filters.")

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

    # Per-id NSE over time (across evaluated dates/horizons for each holdout well).
    # This complements pooled NSE, which can look optimistic when between-well variance is large.
    per_id_nse_rows = []
    for well_id, g in out_df.groupby("id"):
        real_id = g["gws_true"].to_numpy()
        pred_id = g["gws_forecast"].to_numpy()
        if len(real_id) == 0:
            continue
        per_id_nse_rows.append({"id": well_id, "NSE_over_time": float(_nse(pred_id, real_id))})
    per_id_nse_df = pd.DataFrame(per_id_nse_rows)
    per_id_nse_vals = per_id_nse_df["NSE_over_time"].to_numpy(dtype=float) if not per_id_nse_df.empty else np.array([])
    per_id_nse_valid = per_id_nse_vals[np.isfinite(per_id_nse_vals)] if per_id_nse_vals.size else np.array([])

    overall_metrics = pd.Series(
        {
            "RMSE": float(_rmse(pred_all, real_all)),
            "nRMSE": float(_nrmse(pred_all, real_all)),
            "NSE_pooled": float(_nse(pred_all, real_all)),
            "NSE": float(_nse(pred_all, real_all)),
            "NSE_id_median": float(np.median(per_id_nse_valid)) if per_id_nse_valid.size else np.nan,
            "NSE_id_p25": float(np.percentile(per_id_nse_valid, 25)) if per_id_nse_valid.size else np.nan,
            "NSE_id_p50": float(np.percentile(per_id_nse_valid, 50)) if per_id_nse_valid.size else np.nan,
            "NSE_id_p75": float(np.percentile(per_id_nse_valid, 75)) if per_id_nse_valid.size else np.nan,
            "NSE_id_mean": float(np.mean(per_id_nse_valid)) if per_id_nse_valid.size else np.nan,
            "NSE_id_count": float(per_id_nse_valid.size),
            "MAE": float(np.mean(abs_err_all)),
            "AbsErr_P95": float(np.percentile(abs_err_all, 95)) if abs_err_all.size else np.nan,
            "AbsErr_P99": float(np.percentile(abs_err_all, 99)) if abs_err_all.size else np.nan,
            "AbsErr_Max": float(np.max(abs_err_all)) if abs_err_all.size else np.nan,
        }
    )
    rows = []
    for (dt, h), g in out_df.groupby(["datum", "horizon"]):
        real = g["gws_true"].to_numpy()
        pred_vals = g["gws_forecast"].to_numpy()
        if len(real) == 0:
            continue
        rows.append(
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
    per_pair_metrics = pd.DataFrame(rows).sort_values(["datum", "horizon"])

    gp_dir = ROOT / "outputs" / "gp" / run_tag
    gp_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(out_df), gp_dir / "gp_pred.parquet")
    overall_df = overall_metrics.reset_index()
    overall_df.columns = ["metric", "value"]
    pq.write_table(pa.Table.from_pandas(overall_df), gp_dir / "gp_metrics.parquet")
    if not per_pair_metrics.empty:
        per_pair_metrics.to_csv(gp_dir / "gp_metrics_by_pair.csv", index=False)
    if not per_id_nse_df.empty:
        per_id_nse_df.to_csv(gp_dir / "gp_metrics_by_id.csv", index=False)

    pooled_grid = np.nanmedian(np.stack(grid_rows, axis=0), axis=0)
    np.savez(
        gp_dir / "gp_grid.npz",
        pooled_grid=pooled_grid,
        xmin=float(xmin), xmax=float(xmax),
        ymin=float(ymin), ymax=float(ymax),
        grid_nx=int(grid_nx), grid_ny=int(grid_ny),
    )

    print(
        f"Valid pairs: {valid_pairs} / attempted: {attempted_pairs}",
        file=sys.stderr,
    )
    if skipped_pairs:
        skipped_preview = ", ".join([f"{d.date()}:{h}({r})" for d, h, r in skipped_pairs[:10]])
        print(
            f"Skipped {len(skipped_pairs)} date/horizon pairs (preview: {skipped_preview})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
