from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from libs.spatial_split import load_or_create_split, resolve_split_path

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


class GRUSeq2Seq(nn.Module):
    def __init__(self, past_input_size, future_input_size, hidden_size, num_layers, dropout, out_len):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=past_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.GRU(
            input_size=future_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)
        self.out_len = out_len

    def forward(self, x_past, x_future):
        _, h = self.encoder(x_past)
        dec_out, _ = self.decoder(x_future, h)
        return self.head(dec_out).squeeze(-1)


def _load_yaml(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _resolve_data_file(data_cfg, dataset):
    if dataset == "sample":
        return data_cfg["sample_path"]
    if dataset == "full_raw":
        return data_cfg["full_raw_path"]
    if dataset == "full_merged":
        return data_cfg["full_merged_path"]
    raise ValueError(f"Unknown DATASET={dataset}")


def _resolve_run_sig(tft_cfg, dataset, in_len, out_len, epochs, batch_size, seed, spatial_fraction, spatial_clusters, spatial_seed):
    run_sig_cfg = str(tft_cfg.get("run_sig", "")).strip()
    if run_sig_cfg and run_sig_cfg.lower() != "auto":
        return run_sig_cfg
    return (
        f"in{in_len}_out{out_len}_ep{epochs}_bs{batch_size}"
        f"_seed{seed}_{dataset}"
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


def _build_windows(df, in_len, out_len, cov_cols, target_col="gws"):
    x_past_rows = []
    x_future_rows = []
    y_rows = []
    meta = []

    for gid, g in df.groupby("id"):
        g = g.sort_values("datum")
        vals_y = g[target_col].to_numpy(dtype=np.float32)
        vals_cov = g[cov_cols].to_numpy(dtype=np.float32)
        times = g["datum"].to_numpy()

        for i in range(in_len, len(g) - out_len + 1):
            x_past_rows.append(np.concatenate([vals_y[i - in_len : i].reshape(in_len, 1), vals_cov[i - in_len : i]], axis=1))
            x_future_rows.append(vals_cov[i : i + out_len])
            y_rows.append(vals_y[i : i + out_len])
            meta.append((gid, times[i], times[i + out_len - 1], times[i : i + out_len]))

    x_past = np.stack(x_past_rows) if x_past_rows else np.zeros((0, in_len, 1 + len(cov_cols)), dtype=np.float32)
    x_future = np.stack(x_future_rows) if x_future_rows else np.zeros((0, out_len, len(cov_cols)), dtype=np.float32)
    y = np.stack(y_rows) if y_rows else np.zeros((0, out_len), dtype=np.float32)
    return x_past, x_future, y, meta


def main():
    data_cfg = _load_yaml("configs/data.yaml")
    tft_cfg = _load_yaml("configs/tft.yaml")
    dataset = tft_cfg.get("dataset", "full_raw")
    data_file = _resolve_data_file(data_cfg, dataset)

    training_cfg = tft_cfg.get("training", {}) if isinstance(tft_cfg.get("training", {}), dict) else {}
    data_cfg_tft = tft_cfg.get("data", {}) if isinstance(tft_cfg.get("data", {}), dict) else {}
    spatial_cfg = tft_cfg.get("spatial_split", {}) if isinstance(tft_cfg.get("spatial_split", {}), dict) else {}

    seed = int(training_cfg.get("seed", 40))
    in_len = int(data_cfg_tft.get("in_len", 52))
    out_len = int(data_cfg_tft.get("out_len", 16))
    epochs = int(training_cfg.get("epochs", 20))
    batch_size = int(training_cfg.get("batch_size", 1024))

    spatial_fraction = float(spatial_cfg.get("train_fraction", 0.5))
    spatial_clusters = int(spatial_cfg.get("cluster_count", 10))
    spatial_seed = int(spatial_cfg.get("split_seed", 42))
    spatial_excludes = tuple(
        s.strip().lower()
        for s in str(spatial_cfg.get("exclude_terms", "geometry,x_,y_,lon,lat,koord")).split(",")
        if s.strip()
    )

    gws_full = _load_dataset(data_file)
    n_ids = int(gws_full["id"].nunique()) if "id" in gws_full.columns else 0
    if n_ids and spatial_clusters > n_ids:
        spatial_clusters = n_ids

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

    x_past_all, x_future_all, y_all, meta = _build_windows(gws_bb, in_len, out_len, COV_COLS)
    if not meta:
        raise ValueError("No windows created. Check data length and in/out lengths.")

    end_times = np.array([m[2] for m in meta])
    val_mask = (end_times > np.datetime64(TRAIN_CUTOFF)) & (end_times <= np.datetime64(VAL_CUTOFF))
    if val_mask.sum() == 0:
        raise ValueError("No validation windows found before VAL_CUTOFF.")

    x_past_val = x_past_all[val_mask]
    x_future_val = x_future_all[val_mask]
    y_val = y_all[val_mask]
    meta_val = [m for i, m in enumerate(meta) if val_mask[i]]

    run_sig = _resolve_run_sig(
        tft_cfg=tft_cfg,
        dataset=dataset,
        in_len=in_len,
        out_len=out_len,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        spatial_fraction=spatial_fraction,
        spatial_clusters=spatial_clusters,
        spatial_seed=spatial_seed,
    )
    run_dir = ROOT / "outputs" / "GRU_FCOV" / f"GRU_FCOV_{run_sig}"
    model_path = run_dir / "model.pt"
    scaler_path = run_dir / "scalers.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scalers not found: {scaler_path}")

    with scaler_path.open("rb") as f:
        scalers = pickle.load(f)
    cov_scaler = scalers["cov_scaler"]
    y_scaler = scalers["y_scaler"]

    x_past_cov = cov_scaler.transform(x_past_val[:, :, 1:].reshape(-1, len(COV_COLS))).reshape(x_past_val[:, :, 1:].shape)
    x_future_val = cov_scaler.transform(x_future_val.reshape(-1, len(COV_COLS))).reshape(x_future_val.shape)
    past_scaled = y_scaler.transform(x_past_val[:, :, 0].reshape(-1, 1)).reshape(x_past_val[:, :, 0].shape)
    x_past_val = np.concatenate([past_scaled.reshape(-1, in_len, 1), x_past_cov], axis=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = GRUSeq2Seq(
        past_input_size=checkpoint["past_input_size"],
        future_input_size=checkpoint["future_input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        out_len=checkpoint["out_len"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    eval_batch_size = int(training_cfg.get("batch_size", 4096))
    with torch.no_grad():
        chunks = []
        for start in range(0, len(x_past_val), eval_batch_size):
            xp = torch.from_numpy(x_past_val[start:start + eval_batch_size]).to(device)
            xf = torch.from_numpy(x_future_val[start:start + eval_batch_size]).to(device)
            chunks.append(model(xp, xf).cpu().numpy())
        pred = np.concatenate(chunks, axis=0)
    pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)

    rows = []
    for i, (gid, start_time, _, times) in enumerate(meta_val):
        for h in range(out_len):
            rows.append(
                {
                    "id": gid,
                    "startzeitpunkt": pd.to_datetime(start_time),
                    "datum": pd.to_datetime(times[h]),
                    "horizon": h + 1,
                    "gws_forecast": float(pred[i, h]),
                    "gws": float(y_val[i, h]),
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
