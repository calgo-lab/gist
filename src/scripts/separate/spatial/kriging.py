from pathlib import Path
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
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

from libs.utils import get_metrics


def main():
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

    dataset = cfg.get("dataset") or tft_cfg.get("dataset", "full_merged")
    model = cfg.get("model") or tft_cfg.get("model", "TFT")
    run_sig = (cfg.get("run_sig") or tft_cfg.get("run_sig", "")).strip()
    if not run_sig:
        raise ValueError("run_sig must be set in configs/tft.yaml or configs/kriging.yaml")

    horizon = int(cfg.get("horizon", tft_cfg.get("data", {}).get("out_len", 16)))
    target_date = str(cfg.get("kriging_date", "")).strip()
    seed = int(cfg.get("seed", 40))

    kernel_cfg = cfg.get("kernel", {})
    kernel_nu = float(kernel_cfg.get("nu", 1.5))
    kernel_len = float(kernel_cfg.get("length_scale", 1.0))
    kernel_noise = float(kernel_cfg.get("noise", 1e-3))
    restarts = int(kernel_cfg.get("restarts", 2))

    split_tag = str(cfg.get("spatial_split_tag", "")).strip()
    kriging_source = str(cfg.get("kriging_source", "pred")).strip().lower()
    if kriging_source not in {"pred", "true"}:
        raise ValueError("kriging_source must be 'pred' or 'true' in configs/kriging.yaml")

    value_tag = "predobstrain" if kriging_source == "pred" else "trueobstrain"
    name_parts = [run_sig, split_tag, value_tag]
    run_tag = "__".join([p for p in name_parts if p])

    pred_path = ROOT / "outputs" / model / f"{model}_{run_sig}" / "predictions" / "pred.parquet"
    split_path = ROOT / "splits" / f"spatial_split_{dataset}.csv"

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

    gws_train = gws[gws["id"].isin(train_ids)]
    ids_keep = pd.unique(gws_train["id"])
    lookup_ids = pd.DataFrame(ids_keep, columns=["id"]).reset_index()

    if pred is not None:
        pred = pred.merge(lookup_ids, on="index", how="left")
        pred = pred.merge(coords, on="id", how="left")

    dates = [pd.to_datetime(target_date)] if target_date else [gws["datum"].max()]
    horizons = [float(horizon)]

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=kernel_len, nu=kernel_nu) + WhiteKernel(noise_level=kernel_noise)

    out_rows = []
    for dt in dates:
        for h in horizons:
            gws_dt = gws[gws["datum"] == dt][["id", "gws", "x_25833", "y_25833"]]
            holdout_true = gws_dt[gws_dt["id"].isin(holdout_ids)].dropna()

            if kriging_source == "pred":
                pred_sel = pred[(pred["datum"] == dt) & (pred["horizon"] == h)]
                pred_sel = pred_sel[pred_sel["id"].isin(train_ids)]
                pred_sel = pred_sel.dropna(subset=["x_25833", "y_25833", "gws_pred"])
                X_train = pred_sel[["x_25833", "y_25833"]].to_numpy()
                y_train = pred_sel["gws_pred"].to_numpy()
            else:
                train_true = gws_dt[gws_dt["id"].isin(train_ids)].dropna()
                X_train = train_true[["x_25833", "y_25833"]].to_numpy()
                y_train = train_true["gws"].to_numpy()

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

            out = holdout_true.copy()
            out["gws_true"] = out["gws"]
            out.drop(columns=["gws"], inplace=True)
            out["gws_forecast"] = y_pred
            out["gws_forecast_std"] = y_std
            out["horizon"] = h
            out["datum"] = dt
            out_rows.append(out)

    out_df = pd.concat(out_rows, ignore_index=True)

    group_sizes = out_df.groupby(["id", "horizon"]).size()
    if group_sizes.min() < 2:
        out_df_metrics = out_df.copy()
        out_df_metrics["id"] = "ALL"
    else:
        out_df_metrics = out_df

    metrics = get_metrics(
        prediction_df=out_df_metrics,
        real_col="gws_true",
        forecast_col="gws_forecast",
        id_col="id",
        metrics_subset=["nRMSE", "RMSE", "NSE", "KGE", "rMBE", "MAE"],
        lower_quantile=None,
        upper_quantile=None,
    )

    gp_dir = ROOT / "outputs" / "gp" / run_tag
    gp_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(out_df), gp_dir / "gp_pred.parquet")
    pq.write_table(pa.Table.from_pandas(metrics), gp_dir / "gp_metrics.parquet")


if __name__ == "__main__":
    main()
