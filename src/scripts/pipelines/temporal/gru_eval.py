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

from libs.spatial_split import load_split, resolve_split_path
from libs.run_registry import lookup_run_id
from libs.experiment import resolve_experiment, snapshot_run, parse_set_overrides, apply_overrides
from gru_model import GRUSeq2Seq

STATIC_FEATURE_REGEX = (
    "eumohp_(.+)_(.+)_(.*[1])"
    "|shannongeom10kmsha"
    "|entgeom10kment"
    "|unigeom10kmuni"
    "|gwn"
    "|huek250.+_(kf).+"
    "|corine"
    "|^TWI_dgm50_r1000m$"
    "|^gok$"
    "|^parde_seasonality$"
    "|^GW_recharge_r1000m$"
    "|^siwa_verweilzeit_j$"
    "|^gw_gespannt_bin$"
    "|^hydroraum_Entlastungsgebiete$"
    "|^hydroraum_Transitgebiete$"
    "|^hydroraum_Speisungsgebiete$"
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
    parser.add_argument("--config", default="configs/gru/gru.yaml")
    parser.add_argument("--experiment", default=None,
                        help="registry entry from configs/experiments.yaml (beats --config)")
    parser.add_argument("--set", dest="set_overrides", nargs="*", default=None,
                        metavar="KEY=VALUE", help="extra dotted overrides, applied last")
    args = parser.parse_args()

    data_cfg = _load_yaml("configs/data.yaml")
    _cli_overrides = parse_set_overrides(args.set_overrides)
    if args.experiment:
        gru_cfg, _entry = resolve_experiment(args.experiment, extra_overrides=_cli_overrides)
    else:
        gru_cfg = _load_yaml(args.config)
        if _cli_overrides:
            apply_overrides(gru_cfg, _cli_overrides)
        _entry = {"base": args.config, "script": "src/scripts/pipelines/temporal/gru_eval.py", "overrides": {}}
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
    spatial_fraction = float(spatial_cfg.get("train_fraction", 0.5))
    spatial_clusters = int(spatial_cfg.get("cluster_count", 10))
    spatial_seed = int(spatial_cfg.get("split_seed", 42))
    spatial_split_type = str(spatial_cfg.get("split_type", "kmeans")).strip().lower()
    max_dist_k = int(spatial_cfg.get("max_dist_k", 3))
    max_dist_percentile = int(spatial_cfg.get("max_dist_percentile", 25))
    spatial_excludes = tuple(
        s.strip().lower()
        for s in str(spatial_cfg.get("exclude_terms", "geometry,x_,y_,lon,lat,koord")).split(",")
        if s.strip()
    )

    gws_full = _load_dataset(data_file)
    if "gw_gespannt" in gws_full.columns:
        gws_full["gw_gespannt_bin"] = (gws_full["gw_gespannt"] == "gespannt").astype("float32")

    if gru_cfg.get("add_hydroraum_onehot", False) and "hydroraum" in gws_full.columns:
        for cat in ["Entlastungsgebiete", "Transitgebiete", "Speisungsgebiete"]:
            gws_full[f"hydroraum_{cat}"] = (gws_full["hydroraum"] == cat).astype("float32")

    hydroraum_filter = gru_cfg.get("hydroraum_filter", None)
    if hydroraum_filter and "hydroraum" in gws_full.columns:
        gws_full = gws_full[gws_full["hydroraum"] == hydroraum_filter].copy()
        print(f"hydroraum_filter={hydroraum_filter!r}: {gws_full['id'].nunique()} wells retained")

    gwlk_exclude = gru_cfg.get("gwlk_exclude", None)
    if gwlk_exclude is not None and "gwlk" in gws_full.columns:
        exclude_vals = [gwlk_exclude] if not isinstance(gwlk_exclude, list) else gwlk_exclude
        mask = gws_full["gwlk"].astype(str).str.startswith(tuple(str(v) for v in exclude_vals))
        gws_full = gws_full[~mask].copy()
        print(f"gwlk_exclude={gwlk_exclude!r}: {gws_full['id'].nunique()} wells retained")

    split_path = resolve_split_path(ROOT / "splits", dataset, spatial_cfg)
    gws_bb, _ = load_split(
        gws_full,
        static_regex=STATIC_FEATURE_REGEX,
        train_fraction=spatial_fraction,
        n_clusters=spatial_clusters,
        rng_seed=spatial_seed,
        exclude_terms=spatial_excludes,
        save_path=split_path,
        split_type=spatial_split_type,
        max_dist_k=max_dist_k,
        max_dist_percentile=max_dist_percentile,
    )

    run_sig = _resolve_run_sig(
        tft_cfg=gru_cfg,
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
    run_id = lookup_run_id(ROOT / "outputs", "GRU_FCOV", run_sig)
    run_dir = ROOT / "outputs" / "GRU_FCOV" / f"GRU_FCOV_{run_id}"
    print(f"Evaluating run ID: {run_id}  (sig: {run_sig})")
    model_path = run_dir / "model.pt"
    scaler_path = run_dir / "scalers.pkl"
    with scaler_path.open("rb") as f:
        scalers = pickle.load(f)
    cov_scaler = scalers["cov_scaler"]
    well_stats = scalers["well_stats"]
    static_scaler = scalers.get("static_scaler")

    _ckpt_probe = torch.load(model_path, map_location="cpu", weights_only=False)
    if "static_cols" in _ckpt_probe:
        static_cols = _ckpt_probe["static_cols"]
        print(f"Loaded static_cols from checkpoint ({len(static_cols)} features)")
    else:
        static_cols = [c for c in gws_bb.columns if re.search(STATIC_FEATURE_REGEX, c)]
        exclude_static = gru_cfg.get("exclude_static_features", [])
        if exclude_static:
            static_cols = [c for c in static_cols if not any(re.search(p, c) for p in exclude_static)]
        extra_static = gru_cfg.get("extra_static_features", [])
        if extra_static:
            extra_present = [c for c in extra_static if c in gws_bb.columns and c not in static_cols]
            static_cols = static_cols + extra_present
    del _ckpt_probe
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
    TRAIN_CUTOFF_NP = np.datetime64(TRAIN_CUTOFF)
    VAL_CUTOFF_NP   = np.datetime64(VAL_CUTOFF)
    val_mask  = (end_times > TRAIN_CUTOFF_NP) & (end_times <= VAL_CUTOFF_NP)
    test_mask = end_times > VAL_CUTOFF_NP

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = GRUSeq2Seq(
        past_input_size=checkpoint["past_input_size"],
        future_input_size=checkpoint["future_input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        out_len=checkpoint["out_len"],
        static_input_size=checkpoint.get("static_input_size", 0),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    eval_batch_size = int(training_cfg.get("batch_size", 4096))

    def _run_period(mask):
        x_p = x_past_all[mask]
        x_f = x_future_all[mask]
        y   = y_all[mask]
        x_s = x_static_all[mask]
        meta_p = [m for i, m in enumerate(meta) if mask[i]]
        x_p_cov = cov_scaler.transform(x_p[:, :, 1:].reshape(-1, len(COV_COLS))).reshape(x_p[:, :, 1:].shape)
        x_f     = cov_scaler.transform(x_f.reshape(-1, len(COV_COLS))).reshape(x_f.shape)
        x_p     = np.concatenate([x_p[:, :, :1], x_p_cov], axis=2)
        with torch.no_grad():
            chunks = []
            for start in range(0, len(x_p), eval_batch_size):
                xp = torch.from_numpy(x_p[start:start + eval_batch_size]).to(device)
                xf = torch.from_numpy(x_f[start:start + eval_batch_size]).to(device)
                xs = torch.from_numpy(x_s[start:start + eval_batch_size]).to(device)
                chunks.append(model(xp, xf, xs).cpu().numpy())
        pred = np.concatenate(chunks, axis=0)
        pred_denorm = pred.copy()
        y_denorm    = y.copy()
        for i, (gid, _st, _et, _times) in enumerate(meta_p):
            mean_y, std_y = well_stats.get(gid, (0.0, 1.0))
            pred_denorm[i] = pred[i] * std_y + mean_y
            y_denorm[i]    = y[i]   * std_y + mean_y
        rows = []
        for i, (gid, start_time, _, times) in enumerate(meta_p):
            for h in range(out_len):
                rows.append({
                    "id": gid,
                    "startzeitpunkt": pd.to_datetime(start_time),
                    "datum": pd.to_datetime(times[h]),
                    "horizon": h + 1,
                    "gws_forecast": float(pred_denorm[i, h]),
                    "gws": float(y_denorm[i, h]),
                })
        return pd.DataFrame(rows)

    test_df = _run_period(test_mask)
    val_df  = _run_period(val_mask)

    def _metrics(df):
        rows = []
        for h, g in df.groupby("horizon"):
            err = g["gws_forecast"] - g["gws"]
            rows.append({"horizon": int(h), "RMSE": float(np.sqrt(np.mean(err**2))), "MAE": float(np.mean(np.abs(err)))})
        return pd.DataFrame(rows).sort_values("horizon")

    metrics_df     = _metrics(test_df)
    val_metrics_df = _metrics(val_df)

    per_well_rmse = test_df.groupby("id").apply(
        lambda g: float(np.sqrt(np.mean((g["gws_forecast"] - g["gws"]) ** 2))),
        include_groups=False,
    )
    rmse_pw = float(per_well_rmse.median())
    print(f"  RMSE_pw: {rmse_pw:.4f}")
    val_rmse_h16 = float(val_metrics_df[val_metrics_df["horizon"] == 16]["RMSE"].iloc[0]) if 16 in val_metrics_df["horizon"].values else float("nan")
    print(f"  val RMSE_h16: {val_rmse_h16:.4f}")

    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    snapshot_run(gru_cfg, _entry, pred_dir, experiment_name=args.experiment,
                 extra_overrides=_cli_overrides or None)
    pq.write_table(pa.Table.from_pandas(test_df),      pred_dir / "pred.parquet")
    pq.write_table(pa.Table.from_pandas(val_df),       pred_dir / "val_pred.parquet")
    pq.write_table(pa.Table.from_pandas(metrics_df),   run_dir  / "metrics.parquet")
    pq.write_table(pa.Table.from_pandas(val_metrics_df), run_dir / "val_metrics.parquet")


if __name__ == "__main__":
    main()
