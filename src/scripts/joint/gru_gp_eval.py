from pathlib import Path
import argparse
import math
import pickle
import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent

for _p in [
    str(SRC_ROOT),
    str(SCRIPT_DIR),
    str(SCRIPT_DIR / "temporal"),
    str(SCRIPT_DIR / "spatial"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from libs.spatial_split import resolve_split_path
from gru_model import GRUSeq2Seq
from gp_layer import make_gp_layer

STATIC_FEATURE_REGEX = (
    "eumohp_(.+)_(.+)_(.*[1])"
    "|shannongeom10kmsha"
    "|entgeom10kment"
    "|unigeom10kmuni"
    "|gwn"
    "|huek250.+_(kf).+"
    "|corine"
    "|twi"
)
COV_COLS = ["tas_5km", "hurs_5km", "pr_5km", "tag_sin", "tag_cos"]
TRAIN_CUTOFF = pd.Timestamp("20160101")
VAL_CUTOFF = pd.Timestamp("20200101")


def _load_yaml(path):
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_dataset(path):
    df = (
        pd.read_csv(path, low_memory=False)
        if str(path).lower().endswith(".csv")
        else pq.read_table(path).to_pandas()
    )
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"]).astype("datetime64[ns]")
    return df


def _build_windows(df, in_len, out_len, cov_cols, well_stats, well_static, target_col="gws"):
    x_past_rows, x_future_rows, y_rows, x_static_rows, meta = [], [], [], [], []
    static_size = len(next(iter(well_static.values()))) if well_static else 0

    for gid, g in df.groupby("id"):
        g = g.sort_values("datum")
        vals_y = g[target_col].to_numpy(dtype=np.float32)
        vals_cov = g[cov_cols].to_numpy(dtype=np.float32)
        times = g["datum"].to_numpy()

        mean_y, std_y = well_stats.get(gid, (float(vals_y.mean()), max(float(vals_y.std()), 1e-6)))
        vals_y_norm = (vals_y - mean_y) / std_y
        static_vec = well_static.get(gid, np.zeros(static_size, dtype=np.float32))

        for i in range(in_len, len(g) - out_len + 1):
            x_past_rows.append(np.concatenate(
                [vals_y_norm[i - in_len:i].reshape(in_len, 1), vals_cov[i - in_len:i]], axis=1
            ))
            x_future_rows.append(vals_cov[i:i + out_len])
            y_rows.append(vals_y_norm[i:i + out_len])
            x_static_rows.append(static_vec)
            meta.append((gid, times[i], times[i + out_len - 1], times[i:i + out_len]))

    x_past = np.stack(x_past_rows) if x_past_rows else np.zeros((0, in_len, 1 + len(cov_cols)), dtype=np.float32)
    x_future = np.stack(x_future_rows) if x_future_rows else np.zeros((0, out_len, len(cov_cols)), dtype=np.float32)
    y = np.stack(y_rows) if y_rows else np.zeros((0, out_len), dtype=np.float32)
    x_static = np.stack(x_static_rows) if x_static_rows else np.zeros((0, static_size), dtype=np.float32)
    return x_past, x_future, y, x_static, meta


def _nse(pred, real):
    denom = float(np.sum((real - np.mean(real)) ** 2))
    return float("nan") if denom == 0 else float(1 - np.sum((pred - real) ** 2) / denom)


def _rmse(pred, real):
    return float(np.sqrt(np.mean((pred - real) ** 2)))


def _nrmse(pred, real):
    iqr = float(np.diff(np.quantile(real, [0.25, 0.75]))[0])
    return float("nan") if iqr == 0 else _rmse(pred, real) / iqr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/joint/gru_gp.yaml")
    parser.add_argument("--run-sig", default=None, help="Joint run signature (auto-resolved from config if omitted)")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="Which spatial split to evaluate: 'val' (spatial_val) or 'test' (spatial_test, default)")
    parser.add_argument("--date-freq", default="ME", help="Resample freq for evaluation dates (default: ME = monthly)")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="GRU inference batch size per date")
    args = parser.parse_args()

    data_cfg = _load_yaml(ROOT / "configs" / "data.yaml")
    gru_cfg = _load_yaml(ROOT / args.config)

    dataset = gru_cfg.get("dataset", "full_merged")
    data_cfg_gru = gru_cfg.get("data", {})
    gp_cfg = gru_cfg.get("spatial_gp", {})

    in_len = int(data_cfg_gru.get("in_len", 52))
    out_len = int(data_cfg_gru.get("out_len", 16))
    lambda_spatial = float(gp_cfg.get("lambda_spatial", 0.5))

    run_sig = args.run_sig
    if not run_sig:
        run_sig_cfg = str(gru_cfg.get("run_sig", "")).strip()
        if run_sig_cfg and run_sig_cfg.lower() != "auto":
            run_sig = f"joint_{run_sig_cfg}"
        else:
            tr = gru_cfg.get("training", {})
            sc = gru_cfg.get("spatial_split", {})
            seed = int(tr.get("seed", 40))
            n_epochs = int(tr.get("epochs", 50))
            spatial_fraction = float(sc.get("train_fraction", 0.8))
            spatial_clusters = int(sc.get("cluster_count", 20))
            spatial_seed = int(sc.get("split_seed", 42))
            spf = str(spatial_fraction).replace(".", "p")
            lsp = str(lambda_spatial).replace(".", "p")
            run_sig = (
                f"in{in_len}_out{out_len}_ep{n_epochs}_seed{seed}_{dataset}"
                f"_spf{spf}_sc{spatial_clusters}_ss{spatial_seed}_lsp{lsp}"
            )

    run_dir = ROOT / "outputs" / "GRU_GP_JOINT" / f"GRU_GP_JOINT_{run_sig}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Joint run directory not found: {run_dir}")
    print(f"Loading from {run_dir}")

    gru_ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    gp_ckpt = torch.load(run_dir / "gp_models.pt", map_location="cpu")
    with (run_dir / "scalers.pkl").open("rb") as f:
        scalers = pickle.load(f)
    with (run_dir / "coord_scaler.pkl").open("rb") as f:
        coord_scaler = pickle.load(f)

    cov_scaler = scalers["cov_scaler"]
    well_stats = scalers["well_stats"]
    static_scaler = scalers.get("static_scaler")

    split_df = pd.read_csv(run_dir / "split_info.csv")
    train_ids = set(split_df.loc[split_df["spatial_split"] == "spatial_train", "id"])
    val_ids = set(split_df.loc[split_df["spatial_split"] == "spatial_val", "id"])
    test_ids = set(split_df.loc[split_df["spatial_split"] == "spatial_test", "id"])
    eval_ids = val_ids if args.split == "val" else test_ids
    print(f"Evaluating on spatial_{args.split}: {len(eval_ids)} wells")

    if dataset == "sample":
        data_path = Path(data_cfg["sample_path"])
    elif dataset == "full_raw":
        data_path = Path(data_cfg["full_raw_path"])
    else:
        data_path = Path(data_cfg["full_merged_path"])

    gws_full = _load_dataset(data_path)
    gws_train_wells = gws_full[gws_full["id"].isin(train_ids)].copy()

    static_cols = [c for c in gws_train_wells.columns if re.search(STATIC_FEATURE_REGEX, c)]
    well_static = {}
    for gid, g in gws_train_wells.groupby("id"):
        row = g[static_cols].iloc[0].fillna(0.0).to_numpy(dtype=np.float32)
        well_static[gid] = row
    if static_scaler is not None and well_static:
        for gid in well_static:
            well_static[gid] = static_scaler.transform(well_static[gid].reshape(1, -1)).squeeze(0)

    x_past_all, x_future_all, y_all, x_static_all, meta = _build_windows(
        gws_train_wells, in_len, out_len, COV_COLS, well_stats, well_static
    )

    end_times = np.array([m[2] for m in meta])
    test_mask = end_times > np.datetime64(VAL_CUTOFF)

    x_past_test = x_past_all[test_mask]
    x_future_test = x_future_all[test_mask]
    x_static_test = x_static_all[test_mask]
    test_meta = [m for i, m in enumerate(meta) if test_mask[i]]

    xp_cov = cov_scaler.transform(x_past_test[:, :, 1:].reshape(-1, len(COV_COLS))).reshape(x_past_test[:, :, 1:].shape)
    x_future_test = cov_scaler.transform(x_future_test.reshape(-1, len(COV_COLS))).reshape(x_future_test.shape)
    x_past_test = np.concatenate([x_past_test[:, :, :1], xp_cov], axis=2)

    coords_df = gws_full[["id", "x_25833", "y_25833"]].drop_duplicates("id").dropna()
    raw_coords = {
        row.id: np.array([row.x_25833, row.y_25833], dtype=np.float32)
        for row in coords_df.itertuples(index=False)
    }
    coords_scaled = {
        wid: coord_scaler.transform(raw_coords[wid].reshape(1, -1)).squeeze(0).astype(np.float32)
        for wid in raw_coords
    }

    eval_gws = gws_full[gws_full["id"].isin(eval_ids)][["datum", "id", "gws"]].dropna(subset=["gws"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GRUSeq2Seq(
        past_input_size=gru_ckpt["past_input_size"],
        future_input_size=gru_ckpt["future_input_size"],
        hidden_size=gru_ckpt["hidden_size"],
        num_layers=gru_ckpt["num_layers"],
        dropout=gru_ckpt["dropout"],
        out_len=gru_ckpt["out_len"],
        static_input_size=gru_ckpt.get("static_input_size", 0),
    ).to(device)
    model.load_state_dict(gru_ckpt["model"])
    model.eval()

    gp_config = gp_ckpt["gp_config"]
    gp_models = []
    for h in range(out_len):
        gp = make_gp_layer(
            backend=gp_config["backend"],
            n_spatial_dims=2,
            n_inducing=gp_config["n_inducing"],
            jitter=gp_config["jitter"],
            use_float64=gp_config["use_float64"],
            init_noise=gp_config["init_noise"],
        ).to(device)
        gp.load_state_dict(gp_ckpt["gp_state_dicts"][h])
        gp.eval()
        gp_models.append(gp)

    from collections import defaultdict

    date_to_idx = defaultdict(list)
    for i, (_, _, end_t, _) in enumerate(test_meta):
        key = str(np.datetime64(end_t, "D"))
        date_to_idx[key].append(i)

    all_dates_sorted = sorted(date_to_idx.keys())
    all_dates_ts = pd.Series(
        pd.to_datetime(all_dates_sorted), index=pd.to_datetime(all_dates_sorted)
    )
    eval_dates_ts = all_dates_ts.resample(args.date_freq).first().dropna()
    eval_date_keys = [str(np.datetime64(d, "D")) for d in eval_dates_ts]
    eval_date_keys = [k for k in eval_date_keys if k in date_to_idx]
    print(f"Evaluation dates: {len(eval_date_keys)}")

    eval_lookup = defaultdict(dict)
    for row in eval_gws.itertuples(index=False):
        key = str(np.datetime64(row.datum, "D"))
        eval_lookup[key][row.id] = float(row.gws)

    out_rows = []
    skipped = 0

    with torch.no_grad():
        for date_key in eval_date_keys:
            indices = date_to_idx[date_key]
            batch_well_ids = [test_meta[i][0] for i in indices]
            horizon_dates = test_meta[indices[0]][3]

            xp = torch.from_numpy(x_past_test[indices]).to(device)
            xf = torch.from_numpy(x_future_test[indices]).to(device)
            xs = torch.from_numpy(x_static_test[indices]).to(device)

            y_pred_norm = model(xp, xf, xs)

            X_train_sp = torch.tensor(
                np.array([coords_scaled[wid] for wid in batch_well_ids]), device=device
            )
            means_t = torch.tensor(
                [well_stats.get(wid, (0.0, 1.0))[0] for wid in batch_well_ids],
                dtype=torch.float32, device=device,
            )
            stds_t = torch.tensor(
                [well_stats.get(wid, (0.0, 1.0))[1] for wid in batch_well_ids],
                dtype=torch.float32, device=device,
            )

            for h_idx in range(out_len):
                hdate_key = str(np.datetime64(horizon_dates[h_idx], "D"))
                eval_data = eval_lookup.get(hdate_key, {})
                if not eval_data:
                    skipped += 1
                    continue

                eval_well_ids = list(eval_data.keys())
                gws_true = np.array([eval_data[wid] for wid in eval_well_ids])
                X_eval_sp = torch.tensor(
                    np.array([coords_scaled[wid] for wid in eval_well_ids]), device=device
                )

                y_pred_h = y_pred_norm[:, h_idx] * stds_t + means_t
                y_gp, y_std = gp_models[h_idx].forward_with_std(X_train_sp, y_pred_h, X_eval_sp)

                y_gp_np = y_gp.cpu().float().numpy()
                y_std_np = y_std.cpu().float().numpy()

                for j, wid in enumerate(eval_well_ids):
                    out_rows.append({
                        "id": wid,
                        "datum": pd.to_datetime(hdate_key),
                        "horizon": h_idx + 1,
                        "gws_true": gws_true[j],
                        "gws_forecast": float(y_gp_np[j]),
                        "gws_forecast_std": float(y_std_np[j]),
                        "x_25833": float(raw_coords[wid][0]) if wid in raw_coords else float("nan"),
                        "y_25833": float(raw_coords[wid][1]) if wid in raw_coords else float("nan"),
                    })

    if not out_rows:
        raise ValueError("No GP predictions generated. Check data availability.")

    out_df = pd.DataFrame(out_rows)
    print(f"Generated {len(out_df)} predictions, skipped {skipped} (date, horizon) pairs with no eval data.")

    pred_all = out_df["gws_forecast"].to_numpy()
    real_all = out_df["gws_true"].to_numpy()
    abs_err = np.abs(pred_all - real_all)

    per_id_nse = []
    for wid, g in out_df.groupby("id"):
        per_id_nse.append({"id": wid, "NSE_over_time": _nse(g["gws_forecast"].to_numpy(), g["gws_true"].to_numpy())})
    per_id_nse_df = pd.DataFrame(per_id_nse)
    per_id_valid = per_id_nse_df["NSE_over_time"].to_numpy(dtype=float)
    per_id_valid = per_id_valid[np.isfinite(per_id_valid)]

    overall = {
        "RMSE": _rmse(pred_all, real_all),
        "nRMSE": _nrmse(pred_all, real_all),
        "NSE_pooled": _nse(pred_all, real_all),
        "NSE_id_median": float(np.median(per_id_valid)) if per_id_valid.size else float("nan"),
        "NSE_id_p25": float(np.percentile(per_id_valid, 25)) if per_id_valid.size else float("nan"),
        "NSE_id_p50": float(np.percentile(per_id_valid, 50)) if per_id_valid.size else float("nan"),
        "NSE_id_p75": float(np.percentile(per_id_valid, 75)) if per_id_valid.size else float("nan"),
        "NSE_id_mean": float(np.mean(per_id_valid)) if per_id_valid.size else float("nan"),
        "NSE_id_count": float(per_id_valid.size),
        "MAE": float(np.mean(abs_err)),
        "AbsErr_P95": float(np.percentile(abs_err, 95)) if abs_err.size else float("nan"),
        "AbsErr_Max": float(np.max(abs_err)) if abs_err.size else float("nan"),
    }

    per_h_rows = []
    for h, g in out_df.groupby("horizon"):
        real = g["gws_true"].to_numpy()
        pv = g["gws_forecast"].to_numpy()
        per_h_rows.append({
            "horizon": int(h),
            "n": len(real),
            "RMSE": _rmse(pv, real),
            "nRMSE": _nrmse(pv, real),
            "NSE": _nse(pv, real),
            "MAE": float(np.mean(np.abs(pv - real))),
        })
    per_h_df = pd.DataFrame(per_h_rows).sort_values("horizon")

    eval_dir = run_dir / "eval" / args.split
    eval_dir.mkdir(parents=True, exist_ok=True)

    pq.write_table(pa.Table.from_pandas(out_df), eval_dir / "gp_pred.parquet")
    metrics_df = pd.Series(overall).reset_index()
    metrics_df.columns = ["metric", "value"]
    pq.write_table(pa.Table.from_pandas(metrics_df), eval_dir / "gp_metrics.parquet")
    per_h_df.to_csv(eval_dir / "gp_metrics_by_horizon.csv", index=False)
    if not per_id_nse_df.empty:
        per_id_nse_df.to_csv(eval_dir / "gp_metrics_by_id.csv", index=False)

    print(f"\nSaved evaluation to {eval_dir}")
    print("Overall metrics:")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) and not math.isnan(v) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
