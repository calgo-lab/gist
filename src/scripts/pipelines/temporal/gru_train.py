from pathlib import Path
import pickle
import re
import sys
import time
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import os
import yaml

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from libs.spatial_split import load_split, resolve_split_path
from libs.run_registry import assign_run_id
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


class Seq2SeqDataset(Dataset):
    def __init__(self, x_past, x_future, y_future, x_static):
        self.x_past = x_past
        self.x_future = x_future
        self.y_future = y_future
        self.x_static = x_static

    def __len__(self):
        return self.x_past.shape[0]

    def __getitem__(self, idx):
        return self.x_past[idx], self.x_future[idx], self.y_future[idx], self.x_static[idx]


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


def _resolve_run_sig(tft_cfg, dataset, in_len, out_len, n_epochs, batch_size, seed, spatial_fraction, spatial_clusters, spatial_seed):
    run_sig_cfg = str(tft_cfg.get("run_sig", "")).strip()
    if run_sig_cfg and run_sig_cfg.lower() != "auto":
        return run_sig_cfg
    return (
        f"in{in_len}_out{out_len}_ep{n_epochs}_bs{batch_size}"
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
            meta.append((gid, times[i + out_len - 1]))

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
        _entry = {"base": args.config, "script": "src/scripts/pipelines/temporal/gru_train.py", "overrides": {}}
    dataset = gru_cfg.get("dataset", "full_raw")
    data_file = _resolve_data_file(data_cfg, dataset)

    training_cfg = gru_cfg.get("training", {})
    data_cfg_tft = gru_cfg.get("data", {})
    model_cfg = gru_cfg.get("model", {})
    spatial_cfg = gru_cfg.get("spatial_split", {})

    seed = int(training_cfg.get("seed", 40))
    in_len = int(data_cfg_tft.get("in_len", 52))
    out_len = int(data_cfg_tft.get("out_len", 16))
    batch_size = int(training_cfg.get("batch_size", 1024))
    n_epochs = int(training_cfg.get("epochs", 20))
    lr = float(training_cfg.get("lr", 3e-4))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    es_cfg = training_cfg.get("early_stopping", {})
    es_patience = int(es_cfg.get("patience", 5))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_mode = str(es_cfg.get("mode", "min")).strip().lower()

    hidden_size = int(model_cfg.get("gru_hidden", 64))
    num_layers = int(model_cfg.get("gru_layers", 1))
    dropout = float(model_cfg.get("gru_dropout", 0.0))

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

    torch.manual_seed(seed)
    np.random.seed(seed)

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

    train_df = gws_bb[gws_bb["datum"] <= TRAIN_CUTOFF]
    well_stats = {}
    for gid, g in train_df.groupby("id"):
        vals = g["gws"].dropna().to_numpy(dtype=np.float32)
        if len(vals) > 0:
            well_stats[gid] = (float(vals.mean()), float(max(vals.std(), 1e-6)))

    static_cols = [c for c in gws_bb.columns if re.search(STATIC_FEATURE_REGEX, c)]
    exclude_static = gru_cfg.get("exclude_static_features", [])
    if exclude_static:
        static_cols = [c for c in static_cols if not any(re.search(p, c) for p in exclude_static)]
    extra_static = gru_cfg.get("extra_static_features", [])
    if extra_static:
        extra_present = [c for c in extra_static if c in gws_bb.columns and c not in static_cols]
        static_cols = static_cols + extra_present
    print(f"Static features ({len(static_cols)}): {static_cols}")
    well_static = {}
    for gid, g in gws_bb.groupby("id"):
        row = g[static_cols].iloc[0].fillna(0.0).to_numpy(dtype=np.float32)
        well_static[gid] = row

    if static_cols and well_static:
        static_scaler = StandardScaler()
        all_static = np.stack([well_static[gid] for gid in well_static])
        static_scaler.fit(all_static)
        for gid in well_static:
            well_static[gid] = static_scaler.transform(well_static[gid].reshape(1, -1)).squeeze(0)
    else:
        static_scaler = None

    x_past_all, x_future_all, y_all, x_static_all, meta = _build_windows(
        gws_bb, in_len, out_len, COV_COLS, well_stats, well_static
    )

    end_times = np.array([m[1] for m in meta])
    train_mask = end_times <= np.datetime64(TRAIN_CUTOFF)
    val_mask = (end_times > np.datetime64(TRAIN_CUTOFF)) & (end_times <= np.datetime64(VAL_CUTOFF))

    x_past_train = x_past_all[train_mask]
    x_future_train = x_future_all[train_mask]
    y_train = y_all[train_mask]
    x_static_train = x_static_all[train_mask]
    x_past_val = x_past_all[val_mask]
    x_future_val = x_future_all[val_mask]
    y_val = y_all[val_mask]
    x_static_val = x_static_all[val_mask]

    cov_scaler = StandardScaler()
    cov_train = np.concatenate(
        [
            x_past_train[:, :, 1:].reshape(-1, len(COV_COLS)),
            x_future_train.reshape(-1, len(COV_COLS)),
        ],
        axis=0,
    )
    cov_scaler.fit(cov_train)

    x_past_train_cov = cov_scaler.transform(x_past_train[:, :, 1:].reshape(-1, len(COV_COLS))).reshape(x_past_train[:, :, 1:].shape)
    x_past_val_cov = cov_scaler.transform(x_past_val[:, :, 1:].reshape(-1, len(COV_COLS))).reshape(x_past_val[:, :, 1:].shape)
    x_future_train = cov_scaler.transform(x_future_train.reshape(-1, len(COV_COLS))).reshape(x_future_train.shape)
    x_future_val = cov_scaler.transform(x_future_val.reshape(-1, len(COV_COLS))).reshape(x_future_val.shape)

    x_past_train = np.concatenate([x_past_train[:, :, :1], x_past_train_cov], axis=2)
    x_past_val = np.concatenate([x_past_val[:, :, :1], x_past_val_cov], axis=2)

    train_ds = Seq2SeqDataset(
        torch.from_numpy(x_past_train),
        torch.from_numpy(x_future_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_static_train),
    )
    val_ds = Seq2SeqDataset(
        torch.from_numpy(x_past_val),
        torch.from_numpy(x_future_val),
        torch.from_numpy(y_val),
        torch.from_numpy(x_static_val),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    _run_sig_early = _resolve_run_sig(
        tft_cfg=gru_cfg, dataset=dataset, in_len=in_len, out_len=out_len,
        n_epochs=n_epochs, batch_size=batch_size, seed=seed,
        spatial_fraction=spatial_fraction, spatial_clusters=spatial_clusters,
        spatial_seed=spatial_seed,
    )
    if _wandb is not None:
        _wandb_mode = os.getenv("WANDB_MODE", "online" if os.getenv("WANDB_API_KEY") else "disabled")
        _wandb.init(
            project=os.getenv("WANDB_PROJECT", "gwl-interpolation"),
            entity=os.getenv("WANDB_ENTITY") or None,
            name=_run_sig_early,
            mode=_wandb_mode,
            config={
                "pipeline": "decoupled_gru",
                "dataset": dataset, "in_len": in_len, "out_len": out_len,
                "seed": seed, "batch_size": batch_size, "n_epochs": n_epochs, "lr": lr,
                "hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout,
                "spatial_fraction": spatial_fraction, "spatial_seed": spatial_seed,
                "split_type": spatial_split_type,
            },
            tags=["gru", "decoupled"],
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = GRUSeq2Seq(
        past_input_size=x_past_train.shape[-1],
        future_input_size=x_future_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        out_len=out_len,
        static_input_size=len(static_cols),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = None
    best_epoch = 0
    no_improve = 0
    trained_epochs = 0
    best_state = {
        "model": model.state_dict(),
        "past_input_size": x_past_train.shape[-1],
        "future_input_size": x_future_train.shape[-1],
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "out_len": out_len,
        "static_input_size": len(static_cols),
        "static_cols": static_cols,
    }

    for epoch in range(1, n_epochs + 1):
        t_epoch = time.time()
        trained_epochs = epoch

        model.train()
        train_losses = []
        for xb_past, xb_future, yb, xb_static in train_loader:
            xb_past = xb_past.to(device)
            xb_future = xb_future.to(device)
            yb = yb.to(device)
            xb_static = xb_static.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb_past, xb_future, xb_static), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb_past, xb_future, yb, xb_static in val_loader:
                xb_past = xb_past.to(device)
                xb_future = xb_future.to(device)
                yb = yb.to(device)
                xb_static = xb_static.to(device)
                val_losses.append(loss_fn(model(xb_past, xb_future, xb_static), yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={time.time() - t_epoch:.1f}s")
        if _wandb is not None:
            _wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        improved = (
            best_val is None
            or (es_mode == "min" and val_loss < best_val - es_min_delta)
            or (es_mode == "max" and val_loss > best_val + es_min_delta)
        )
        if improved:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {
                "model": model.state_dict(),
                "past_input_size": x_past_train.shape[-1],
                "future_input_size": x_future_train.shape[-1],
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "out_len": out_len,
                "static_input_size": len(static_cols),
                "static_cols": static_cols,
            }
        else:
            no_improve += 1
            if no_improve >= es_patience:
                break

    run_sig = _resolve_run_sig(
        tft_cfg=gru_cfg,
        dataset=dataset,
        in_len=in_len,
        out_len=out_len,
        n_epochs=n_epochs,
        batch_size=batch_size,
        seed=seed,
        spatial_fraction=spatial_fraction,
        spatial_clusters=spatial_clusters,
        spatial_seed=spatial_seed,
    )
    run_id = assign_run_id(
        ROOT / "outputs", "GRU_FCOV", run_sig,
        meta={
            "dataset": dataset, "in_len": in_len, "out_len": out_len,
            "epochs": n_epochs, "batch_size": batch_size, "seed": seed,
            "split_file": str(spatial_cfg.get("file", "")),
            "hidden_size": hidden_size, "num_layers": num_layers,
            "dropout": dropout, "lr": lr,
            "hydroraum_filter": gru_cfg.get("hydroraum_filter", ""),
        },
    )
    run_dir = ROOT / "outputs" / "GRU_FCOV" / f"GRU_FCOV_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_run(gru_cfg, _entry, run_dir, experiment_name=args.experiment,
                 extra_overrides=_cli_overrides or None)
    print(f"Run ID: {run_id}  (sig: {run_sig})")

    torch.save(best_state, run_dir / "model.pt")
    with (run_dir / "scalers.pkl").open("wb") as f:
        pickle.dump({"cov_scaler": cov_scaler, "well_stats": well_stats, "static_scaler": static_scaler}, f)

    meta_out = {
        "run_id": run_id,
        "run_sig": run_sig,
        "dataset": dataset,
        "in_len": in_len,
        "out_len": out_len,
        "seed": seed,
        "batch_size": batch_size,
        "epochs": n_epochs,
        "trained_epochs": trained_epochs,
        "best_epoch": best_epoch,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "grad_clip": grad_clip,
        "n_static": len(static_cols),
        "early_stopping": {"patience": es_patience, "min_delta": es_min_delta, "mode": es_mode},
        "cov_cols": COV_COLS,
    }
    (run_dir / "meta.yaml").write_text(yaml.safe_dump(meta_out), encoding="utf-8")
    if _wandb is not None:
        _wandb.summary.update({"best_val_loss": best_val, "trained_epochs": trained_epochs, "best_epoch": best_epoch, "run_id": run_id})
        _wandb.finish()


if __name__ == "__main__":
    main()
