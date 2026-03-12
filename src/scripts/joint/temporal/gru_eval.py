from pathlib import Path
import pickle
import re
import sys
import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from libs.spatial_split import load_or_create_split, resolve_split_path
from gru_model import GRUSeq2Seq

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
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data or {}


def _resolve_data_file(data_cfg, dataset):
    if dataset == "sample":
        return data_cfg["sample_path"]
    if dataset == "full_raw":
        return data_cfg["full_raw_path"]
    if dataset == "full_merged":
        return data_cfg["full_merged_path"]
    raise ValueError(f"Unknown DATASET={dataset}")


def _resolve_run_sig(tft_cfg, dataset, in_len, out_len, epochs, batch_size, seed, use_revin, use_scheduler, spatial_fraction, spatial_clusters, spatial_seed):
    run_sig_cfg = str(tft_cfg.get("run_sig", "")).strip()
    if run_sig_cfg and run_sig_cfg.lower() != "auto":
        return run_sig_cfg
    revin_tag = "r1" if bool(use_revin) else "r0"
    sched_tag = "s1" if bool(use_scheduler) else "s0"
    return (
        f"in{in_len}_out{out_len}_ep{epochs}_bs{batch_size}"
        f"_seed{seed}_{dataset}_{revin_tag}_{sched_tag}"
        f"_spf{str(spatial_fraction).replace('.', 'p')}_sc{spatial_clusters}_ss{spatial_seed}"
    )


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
    x_past_rows = []
    x_future_rows = []
    y_rows = []
    x_static_rows = []
    meta = []

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
            x_past_rows.append(np.concatenate([vals_y_norm[i - in_len : i].reshape(in_len, 1), vals_cov[i - in_len : i]], axis=1))
            x_future_rows.append(vals_cov[i : i + out_len])
            y_rows.append(vals_y_norm[i : i + out_len])
            x_static_rows.append(static_vec)
            meta.append((gid, times[i], times[i + out_len - 1], times[i : i + out_len]))

    x_past = np.stack(x_past_rows) if x_past_rows else np.zeros((0, in_len, 1 + len(cov_cols)), dtype=np.float32)
    x_future = np.stack(x_future_rows) if x_future_rows else np.zeros((0, out_len, len(cov_cols)), dtype=np.float32)
    y = np.stack(y_rows) if y_rows else np.zeros((0, out_len), dtype=np.float32)
    x_static = np.stack(x_static_rows) if x_static_rows else np.zeros((0, static_size), dtype=np.float32)
    return x_past, x_future, y, x_static, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gru.yaml")
    args = parser.parse_args()

    data_cfg = _load_yaml("configs/data.yaml")
    gru_cfg = _load_yaml(args.config)
    dataset = gru_cfg.get("dataset", "full_raw")
    data_file = _resolve_data_file(data_cfg, dataset)

    training_cfg = gru_cfg.get("training", {})
    data_cfg_tft = gru_cfg.get("data", {})
    spatial_cfg = gru_cfg.get("spatial_split", {})

    seed = int(training_cfg.get("seed", 40))
    in_len = int(data_cfg_tft.get("in_len", 52))
    out_len = int(data_cfg_tft.get("out_len", 16))
    epochs = int(training_cfg.get("epochs", 20))
    batch_size = int(training_cfg.get("batch_size", 1024))
    use_revin = bool(gru_cfg.get("model", {}).get("use_revin", False))
    use_scheduler = bool(gru_cfg.get("training", {}).get("lr_scheduler", {}).get("enabled", True))

    spatial_fraction = float(spatial_cfg.get("train_fraction", 0.5))
    spatial_clusters = int(spatial_cfg.get("cluster_count", 10))
    spatial_seed = int(spatial_cfg.get("split_seed", 42))
    spatial_excludes = tuple(
        s.strip().lower()
        for s in str(spatial_cfg.get("exclude_terms", "geometry,x_,y_,lon,lat,koord")).split(",")
        if s.strip()
    )

    gws_full = _load_dataset(data_file)

    split_path = resolve_split_path(ROOT / "splits", dataset, spatial_cfg)
    gws_bb, _ = load_or_create_split(
        gws_full,
        static_regex=STATIC_FEATURE_REGEX,
        train_fraction=spatial_fraction,
        n_clusters=spatial_clusters,
        rng_seed=spatial_seed,
        exclude_terms=spatial_excludes,
        save_path=split_path,
    )

    run_sig = _resolve_run_sig(
        tft_cfg=gru_cfg,
        dataset=dataset,
        in_len=in_len,
        out_len=out_len,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        use_revin=use_revin,
        use_scheduler=use_scheduler,
        spatial_fraction=spatial_fraction,
        spatial_clusters=spatial_clusters,
        spatial_seed=spatial_seed,
    )
    run_dir = ROOT / "outputs" / "GRU_FCOV" / f"GRU_FCOV_{run_sig}"
    model_path = run_dir / "model.pt"
    scaler_path = run_dir / "scalers.pkl"
    with scaler_path.open("rb") as f:
        scalers = pickle.load(f)
    cov_scaler = scalers["cov_scaler"]
    well_stats = scalers["well_stats"]
    static_scaler = scalers.get("static_scaler")

    # Build static features using same regex as train
    static_cols = [c for c in gws_bb.columns if re.search(STATIC_FEATURE_REGEX, c)]
    well_static = {}
    for gid, g in gws_bb.groupby("id"):
        row = g[static_cols].iloc[0].fillna(0.0).to_numpy(dtype=np.float32)
        well_static[gid] = row

    if static_scaler is not None and well_static:
        for gid in well_static:
            well_static[gid] = static_scaler.transform(well_static[gid].reshape(1, -1)).squeeze(0)

    x_past_all, x_future_all, y_all, x_static_all, meta = _build_windows(
        gws_bb, in_len, out_len, COV_COLS, well_stats, well_static
    )

    end_times = np.array([m[2] for m in meta])
    test_mask = end_times > np.datetime64(VAL_CUTOFF)

    x_past_val = x_past_all[test_mask]
    x_future_val = x_future_all[test_mask]
    y_val = y_all[test_mask]
    x_static_val = x_static_all[test_mask]
    meta_val = [m for i, m in enumerate(meta) if test_mask[i]]

    x_past_cov = cov_scaler.transform(x_past_val[:, :, 1:].reshape(-1, len(COV_COLS))).reshape(x_past_val[:, :, 1:].shape)
    x_future_val = cov_scaler.transform(x_future_val.reshape(-1, len(COV_COLS))).reshape(x_future_val.shape)
    # x_past[:, :, 0] is already per-well normalized; reconstruct with scaled covariates
    x_past_val = np.concatenate([x_past_val[:, :, :1], x_past_cov], axis=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = GRUSeq2Seq(
        past_input_size=checkpoint["past_input_size"],
        future_input_size=checkpoint["future_input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        out_len=checkpoint["out_len"],
        static_input_size=checkpoint.get("static_input_size", 0),
        use_revin=checkpoint.get("use_revin", False),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    eval_batch_size = int(training_cfg.get("batch_size", 4096))
    with torch.no_grad():
        chunks = []
        for start in range(0, len(x_past_val), eval_batch_size):
            xp = torch.from_numpy(x_past_val[start:start + eval_batch_size]).to(device)
            xf = torch.from_numpy(x_future_val[start:start + eval_batch_size]).to(device)
            xs = torch.from_numpy(x_static_val[start:start + eval_batch_size]).to(device)
            chunks.append(model(xp, xf, xs).cpu().numpy())
        pred = np.concatenate(chunks, axis=0)

    # Denormalize per-well
    pred_denorm = pred.copy()
    y_val_denorm = y_val.copy()
    for i, (gid, _start_time, _end_time, _times) in enumerate(meta_val):
        mean_y, std_y = well_stats.get(gid, (0.0, 1.0))
        pred_denorm[i] = pred[i] * std_y + mean_y
        y_val_denorm[i] = y_val[i] * std_y + mean_y

    rows = []
    for i, (gid, start_time, _, times) in enumerate(meta_val):
        for h in range(out_len):
            rows.append(
                {
                    "id": gid,
                    "startzeitpunkt": pd.to_datetime(start_time),
                    "datum": pd.to_datetime(times[h]),
                    "horizon": h + 1,
                    "gws_forecast": float(pred_denorm[i, h]),
                    "gws": float(y_val_denorm[i, h]),
                }
            )
    pred_df = pd.DataFrame(rows)

    metrics_rows = []
    for h, g in pred_df.groupby("horizon"):
        err = g["gws_forecast"] - g["gws"]
        metrics_rows.append({"horizon": int(h), "RMSE": float(np.sqrt(np.mean(err**2))), "MAE": float(np.mean(np.abs(err)))})
    metrics_df = pd.DataFrame(metrics_rows).sort_values("horizon")

    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pred_df), pred_dir / "pred.parquet")
    pq.write_table(pa.Table.from_pandas(metrics_df), run_dir / "metrics.parquet")


if __name__ == "__main__":
    main()
