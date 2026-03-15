"""gru_gp_train.py — Joint GRU + GP training.

Loss: GRU_MSE + lambda_spatial * mean(GP_MSE_h  for h in 1..out_len)

Batches are grouped by forecast end-date so that all training-well predictions
at the same date can be fed into the GP for spatial interpolation.

Config is read from configs/gru.yaml:
  training.*    — same hyperparams as standalone GRU
  model.*       — same as standalone GRU
  spatial_gp.*  — GP backend, n_inducing, jitter, lambda_spatial, ...
  joint.*       — gru_lr, gp_lr, gru_pretrained_run_sig, spatial_val_fraction, ...

Outputs land in:  outputs/GRU_GP_JOINT/GRU_GP_JOINT_{run_sig}/
"""

from pathlib import Path
import pickle
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.preprocessing import StandardScaler
import yaml

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


# ---------------------------------------------------------------------------
# Helpers shared with gru_train.py / gru_eval.py
# ---------------------------------------------------------------------------

def _load_yaml(path):
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _resolve_data_file(data_cfg, dataset):
    if dataset == "sample":
        return data_cfg["sample_path"]
    if dataset == "full_raw":
        return data_cfg["full_raw_path"]
    if dataset == "full_merged":
        return data_cfg["full_merged_path"]
    raise ValueError(f"Unknown dataset={dataset!r}")


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
    """Sliding windows. meta entry = (gid, start_time, end_time, times_array)."""
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


# ---------------------------------------------------------------------------
# Joint-training helpers
# ---------------------------------------------------------------------------

def _make_three_way_split(split_df, val_fraction=0.5, rng_seed=0):
    """Split 'spatial_holdout' wells into 'spatial_val' and 'spatial_test'.

    Stratified by cluster so both subsets have representative coverage.
    Returns a copy of split_df with updated spatial_split column.
    """
    rng = np.random.default_rng(rng_seed)
    split_df = split_df.copy()
    holdout = split_df[split_df["spatial_split"] == "spatial_holdout"]

    val_ids = set()
    for _, group in holdout.groupby("cluster"):
        ids = group["id"].to_numpy()
        n_val = max(1, int(round(len(ids) * val_fraction)))
        chosen = rng.choice(ids, size=n_val, replace=False)
        val_ids.update(chosen.tolist())

    # First mark val, then remaining holdout → test
    split_df.loc[split_df["id"].isin(val_ids) & (split_df["spatial_split"] == "spatial_holdout"), "spatial_split"] = "spatial_val"
    split_df.loc[split_df["spatial_split"] == "spatial_holdout", "spatial_split"] = "spatial_test"
    return split_df


def _build_gws_lookup(gws_df, well_ids):
    """Build  date_str → {well_id → gws_value}  for a set of wells."""
    subset = gws_df[gws_df["id"].isin(well_ids)][["datum", "id", "gws"]].dropna(subset=["gws"])
    lookup = defaultdict(dict)
    for row in subset.itertuples(index=False):
        key = str(np.datetime64(row.datum, "D"))
        lookup[key][row.id] = float(row.gws)
    return lookup


def _build_date_index(meta_list):
    """Group window indices by forecast end-date string."""
    date_to_idx = defaultdict(list)
    for i, (_, _s, end_t, _ts) in enumerate(meta_list):
        key = str(np.datetime64(end_t, "D"))
        date_to_idx[key].append(i)
    return date_to_idx


def _pretrain_gp_kernels(
    gp_models, gru_model,
    x_past, x_future, x_static, meta,
    well_stats, coords_scaled,
    device, n_steps=100, lr=1e-2, max_pts=512,
):
    """Pretrain GP kernel via exact MLL on initial GRU predictions.

    Runs a short MLL optimisation for every horizon using a random subsample
    of all training-well predictions.  Gives the kernels a sensible starting
    point before joint training begins.
    """
    out_len = len(gp_models)
    gru_model.eval()

    # Collect GRU predictions for all training windows
    chunk = 512
    all_pred_norm = []
    with torch.no_grad():
        for s in range(0, len(x_past), chunk):
            xp = torch.from_numpy(x_past[s:s + chunk]).to(device)
            xf = torch.from_numpy(x_future[s:s + chunk]).to(device)
            xs = torch.from_numpy(x_static[s:s + chunk]).to(device)
            all_pred_norm.append(gru_model(xp, xf, xs).cpu().numpy())
    all_pred_norm = np.concatenate(all_pred_norm, axis=0)  # (N, out_len)

    # Denormalize per well
    all_pred_denorm = np.zeros_like(all_pred_norm)
    all_coords = np.zeros((len(meta), 2), dtype=np.float32)
    for i, (gid, _, _, _) in enumerate(meta):
        mean_y, std_y = well_stats.get(gid, (0.0, 1.0))
        all_pred_denorm[i] = all_pred_norm[i] * std_y + mean_y
        all_coords[i] = coords_scaled.get(gid, np.zeros(2, dtype=np.float32))

    for h_idx, gp in enumerate(gp_models):
        X_all = torch.tensor(all_coords, dtype=torch.float32, device=device)
        y_all = torch.tensor(all_pred_denorm[:, h_idx], dtype=torch.float32, device=device)

        if len(y_all) > max_pts:
            perm = torch.randperm(len(y_all))[:max_pts]
            X_s, y_s = X_all[perm], y_all[perm]
        else:
            X_s, y_s = X_all, y_all

        if hasattr(gp, "initialize_inducing"):
            gp.initialize_inducing(X_s)

        gp.train()
        params = list(gp.svgp.covar_module.parameters()) + list(gp.likelihood.parameters())
        opt = torch.optim.Adam(params, lr=lr)

        steps_ok = 0
        for step in range(n_steps):
            opt.zero_grad()
            try:
                loss = gp.marginal_log_likelihood(X_s, y_s)
            except RuntimeError as e:
                print(f"    [pretrain h={h_idx + 1}] MLL failed at step {step}: {e}")
                break
            if not torch.isfinite(loss):
                print(f"    [pretrain h={h_idx + 1}] non-finite MLL at step {step}, stopping")
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            steps_ok += 1

        gp.eval()
        ls = float(gp.length_scale().view(-1)[0].item())
        os_ = float(gp.output_scale().item())
        nz = float(gp.noise().item())
        print(f"  [pretrain h={h_idx + 1:2d}] steps_ok={steps_ok:3d}  ls={ls:.3f}  os={os_:.3f}  noise={nz:.4f}")

    gru_model.train()


def _resolve_joint_run_sig(gru_cfg, dataset, in_len, out_len, n_epochs, seed,
                            use_revin, spatial_fraction, spatial_clusters,
                            spatial_seed, lambda_spatial):
    run_sig_cfg = str(gru_cfg.get("run_sig", "")).strip()
    if run_sig_cfg and run_sig_cfg.lower() != "auto":
        return f"joint_{run_sig_cfg}"
    revin_tag = "r1" if use_revin else "r0"
    spf = str(spatial_fraction).replace(".", "p")
    lsp = str(lambda_spatial).replace(".", "p")
    return (
        f"in{in_len}_out{out_len}_ep{n_epochs}_seed{seed}_{dataset}_{revin_tag}"
        f"_spf{spf}_sc{spatial_clusters}_ss{spatial_seed}_lsp{lsp}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_cfg = _load_yaml(ROOT / "configs" / "data.yaml")
    gru_cfg = _load_yaml(ROOT / "configs" / "gru_gp.yaml")

    dataset = gru_cfg.get("dataset", "full_merged")
    data_file = _resolve_data_file(data_cfg, dataset)

    training_cfg = gru_cfg.get("training", {})
    data_cfg_gru = gru_cfg.get("data", {})
    model_cfg = gru_cfg.get("model", {})
    spatial_cfg = gru_cfg.get("spatial_split", {})
    gp_cfg = gru_cfg.get("spatial_gp", {})
    joint_cfg = gru_cfg.get("joint", {})

    # GRU hyperparams
    seed = int(training_cfg.get("seed", 40))
    in_len = int(data_cfg_gru.get("in_len", 52))
    out_len = int(data_cfg_gru.get("out_len", 16))
    n_epochs = int(training_cfg.get("epochs", 50))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    es_cfg = training_cfg.get("early_stopping", {})
    es_patience = int(es_cfg.get("patience", 5))
    es_min_delta = float(es_cfg.get("min_delta", 1e-4))

    hidden_size = int(model_cfg.get("gru_hidden", 64))
    num_layers = int(model_cfg.get("gru_layers", 1))
    dropout = float(model_cfg.get("gru_dropout", 0.0))
    use_revin = bool(model_cfg.get("use_revin", False))

    # Spatial split
    spatial_fraction = float(spatial_cfg.get("train_fraction", 0.8))
    spatial_clusters = int(spatial_cfg.get("cluster_count", 20))
    spatial_seed = int(spatial_cfg.get("split_seed", 42))
    spatial_excludes = tuple(
        s.strip().lower()
        for s in str(spatial_cfg.get("exclude_terms", "geometry,x_,y_,lon,lat,koord")).split(",")
        if s.strip()
    )

    # GP hyperparams
    backend = str(gp_cfg.get("backend", "gpytorch")).strip().lower()
    n_inducing = int(gp_cfg.get("n_inducing", 64))
    jitter = float(gp_cfg.get("jitter", 1e-4))
    use_float64 = bool(gp_cfg.get("use_float64", True))
    init_noise = float(gp_cfg.get("init_noise", 0.1))
    lambda_spatial = float(gp_cfg.get("lambda_spatial", 0.5))
    gp_pretrain_steps = int(gp_cfg.get("gp_pretrain_steps", 100))
    gp_pretrain_lr = float(gp_cfg.get("variational_lr", 1e-2))

    # Joint hyperparams
    gru_lr = float(joint_cfg.get("gru_lr", 3e-4))
    gp_lr = float(joint_cfg.get("gp_lr", 1e-3))
    gru_pretrained_run_sig = str(joint_cfg.get("gru_pretrained_run_sig", "")).strip()
    spatial_val_fraction = float(joint_cfg.get("spatial_val_fraction", 0.5))
    min_train_wells = int(joint_cfg.get("min_train_wells", 5))
    min_val_wells = int(joint_cfg.get("min_val_wells", 2))
    val_gp_stride = int(joint_cfg.get("val_gp_stride", 4))  # use every N-th val date for GP val

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading dataset...")
    gws_full = _load_dataset(data_file)

    # Load existing 2-way split and extend to 3-way
    split_path = resolve_split_path(ROOT / "splits", dataset, spatial_cfg)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Spatial split not found: {split_path}\n"
            "Run gru_train.py first to create the split file."
        )
    split_df = pd.read_csv(split_path)
    split_df = _make_three_way_split(split_df, val_fraction=spatial_val_fraction, rng_seed=spatial_seed)

    train_ids = set(split_df.loc[split_df["spatial_split"] == "spatial_train", "id"])
    val_ids = set(split_df.loc[split_df["spatial_split"] == "spatial_val", "id"])
    test_ids = set(split_df.loc[split_df["spatial_split"] == "spatial_test", "id"])
    print(f"Spatial split: train={len(train_ids)}, spatial_val={len(val_ids)}, spatial_test={len(test_ids)}")

    # GRU training data: spatial_train wells only (same as gru_train.py)
    gws_train_wells = gws_full[gws_full["id"].isin(train_ids)].copy()

    # Per-well GWS stats from temporal training period
    train_time_df = gws_train_wells[gws_train_wells["datum"] <= TRAIN_CUTOFF]
    well_stats = {}
    for gid, g in train_time_df.groupby("id"):
        vals = g["gws"].dropna().to_numpy(dtype=np.float32)
        if len(vals) > 0:
            well_stats[gid] = (float(vals.mean()), float(max(vals.std(), 1e-6)))

    # Static features (same regex as gru_train.py)
    static_cols = [c for c in gws_train_wells.columns if re.search(STATIC_FEATURE_REGEX, c)]
    well_static = {}
    for gid, g in gws_train_wells.groupby("id"):
        row = g[static_cols].iloc[0].fillna(0.0).to_numpy(dtype=np.float32)
        well_static[gid] = row

    static_scaler = None
    if static_cols and well_static:
        static_scaler = StandardScaler()
        all_static = np.stack([well_static[gid] for gid in well_static])
        static_scaler.fit(all_static)
        for gid in well_static:
            well_static[gid] = static_scaler.transform(well_static[gid].reshape(1, -1)).squeeze(0)

    # Build sliding windows
    print("Building windows...")
    x_past_all, x_future_all, y_all, x_static_all, meta = _build_windows(
        gws_train_wells, in_len, out_len, COV_COLS, well_stats, well_static
    )

    end_times = np.array([m[2] for m in meta])
    train_mask = end_times <= np.datetime64(TRAIN_CUTOFF)
    val_mask = (end_times > np.datetime64(TRAIN_CUTOFF)) & (end_times <= np.datetime64(VAL_CUTOFF))

    # Covariate scaling (fit on temporal training set)
    cov_scaler = StandardScaler()
    cov_scaler.fit(np.concatenate([
        x_past_all[train_mask][:, :, 1:].reshape(-1, len(COV_COLS)),
        x_future_all[train_mask].reshape(-1, len(COV_COLS)),
    ], axis=0))

    def _scale(xp, xf):
        xp_cov = cov_scaler.transform(xp[:, :, 1:].reshape(-1, len(COV_COLS))).reshape(xp[:, :, 1:].shape)
        xf_s = cov_scaler.transform(xf.reshape(-1, len(COV_COLS))).reshape(xf.shape)
        return np.concatenate([xp[:, :, :1], xp_cov], axis=2), xf_s

    x_past_train, x_future_train = _scale(x_past_all[train_mask], x_future_all[train_mask])
    x_past_val, x_future_val = _scale(x_past_all[val_mask], x_future_all[val_mask])
    y_train = y_all[train_mask]
    y_val = y_all[val_mask]
    x_static_train = x_static_all[train_mask]
    x_static_val = x_static_all[val_mask]
    train_meta = [m for i, m in enumerate(meta) if train_mask[i]]
    val_meta = [m for i, m in enumerate(meta) if val_mask[i]]

    # ------------------------------------------------------------------
    # Spatial coordinates
    # ------------------------------------------------------------------
    coords_df = gws_full[["id", "x_25833", "y_25833"]].drop_duplicates("id").dropna()
    raw_coords = {
        row.id: np.array([row.x_25833, row.y_25833], dtype=np.float32)
        for row in coords_df.itertuples(index=False)
    }

    # Fit coordinate scaler on spatial_train wells
    train_coord_arr = np.array([raw_coords[wid] for wid in train_ids if wid in raw_coords])
    coord_scaler = StandardScaler().fit(train_coord_arr)

    # Precompute scaled coordinates for every well (avoids repeated transform calls in the loop)
    coords_scaled = {
        wid: coord_scaler.transform(raw_coords[wid].reshape(1, -1)).squeeze(0).astype(np.float32)
        for wid in raw_coords
    }

    # ------------------------------------------------------------------
    # GWL lookup for spatial_val wells (target for GP loss)
    # ------------------------------------------------------------------
    print("Building spatial-val GWL lookup...")
    val_gws_lookup = _build_gws_lookup(gws_full, val_ids)

    # ------------------------------------------------------------------
    # Date-indexed batch structure
    # ------------------------------------------------------------------
    train_date_idx = _build_date_index(train_meta)
    val_date_idx = _build_date_index(val_meta)

    train_dates = sorted(k for k, v in train_date_idx.items() if len(v) >= min_train_wells)
    val_dates = sorted(val_date_idx.keys())
    val_dates_gp = val_dates[::val_gp_stride]

    print(f"Training date-batches: {len(train_dates)}  |  Val dates: {len(val_dates)}  |  Val GP dates: {len(val_dates_gp)}")

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GRUSeq2Seq(
        past_input_size=x_past_train.shape[-1],
        future_input_size=x_future_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        out_len=out_len,
        static_input_size=len(static_cols),
        use_revin=use_revin,
    ).to(device)

    # Optionally initialise from a pretrained GRU checkpoint
    if gru_pretrained_run_sig:
        ckpt_path = ROOT / "outputs" / "GRU_FCOV" / f"GRU_FCOV_{gru_pretrained_run_sig}" / "model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded pretrained GRU weights from {ckpt_path}")
        else:
            print(f"Warning: pretrained GRU not found at {ckpt_path}, training from scratch.")

    # 16 GP models (one per forecast horizon)
    gp_models = [
        make_gp_layer(
            backend=backend,
            n_spatial_dims=2,
            n_inducing=n_inducing,
            jitter=jitter,
            use_float64=use_float64,
            init_noise=init_noise,
        ).to(device)
        for _ in range(out_len)
    ]

    # ------------------------------------------------------------------
    # GP kernel pretraining
    # ------------------------------------------------------------------
    if gp_pretrain_steps > 0:
        print(f"\nPretraining {out_len} GP kernels ({gp_pretrain_steps} MLL steps, lr={gp_pretrain_lr})...")
        _pretrain_gp_kernels(
            gp_models, model,
            x_past_train, x_future_train, x_static_train, train_meta,
            well_stats, coords_scaled,
            device, n_steps=gp_pretrain_steps, lr=gp_pretrain_lr,
        )

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    gru_optimizer = torch.optim.Adam(model.parameters(), lr=gru_lr)
    gp_all_params = [p for gp in gp_models for p in gp.parameters()]
    gp_optimizer = torch.optim.Adam(gp_all_params, lr=gp_lr)
    loss_fn = torch.nn.MSELoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val = None
    best_epoch = 0
    no_improve = 0
    trained_epochs = 0
    best_gru_state = model.state_dict()
    best_gp_states = [gp.state_dict() for gp in gp_models]

    print(f"\nJoint training: {n_epochs} epochs  |  lambda_spatial={lambda_spatial}")

    for epoch in range(1, n_epochs + 1):
        t_epoch = time.time()
        trained_epochs = epoch

        model.train()
        for gp in gp_models:
            gp.train()

        rng.shuffle(train_dates)

        epoch_gru_losses = []
        epoch_gp_losses = []

        for date_key in train_dates:
            indices = train_date_idx[date_key]
            batch_well_ids = [train_meta[i][0] for i in indices]
            horizon_dates = train_meta[indices[0]][3]  # shape (out_len,)

            xp = torch.from_numpy(x_past_train[indices]).to(device)
            xf = torch.from_numpy(x_future_train[indices]).to(device)
            ys = torch.from_numpy(y_train[indices]).to(device)
            xs = torch.from_numpy(x_static_train[indices]).to(device)

            gru_optimizer.zero_grad()
            gp_optimizer.zero_grad()

            y_pred = model(xp, xf, xs)  # (n_wells, out_len) normalized
            gru_mse = loss_fn(y_pred, ys)

            # Spatial coords for training wells in this date-batch
            X_train_sp = torch.tensor(
                np.array([coords_scaled[wid] for wid in batch_well_ids]),
                device=device,
            )
            # Per-well denormalization constants (no grad needed)
            means_t = torch.tensor(
                [well_stats.get(wid, (0.0, 1.0))[0] for wid in batch_well_ids],
                dtype=torch.float32, device=device,
            )
            stds_t = torch.tensor(
                [well_stats.get(wid, (0.0, 1.0))[1] for wid in batch_well_ids],
                dtype=torch.float32, device=device,
            )

            # GP loss — one forward per horizon
            gp_mses = []
            for h_idx in range(out_len):
                hdate_key = str(np.datetime64(horizon_dates[h_idx], "D"))
                val_data = val_gws_lookup.get(hdate_key, {})
                if len(val_data) < min_val_wells:
                    continue

                val_well_ids = list(val_data.keys())
                val_true = torch.tensor(
                    [val_data[wid] for wid in val_well_ids],
                    dtype=torch.float32, device=device,
                )
                X_val_sp = torch.tensor(
                    np.array([coords_scaled[wid] for wid in val_well_ids]),
                    device=device,
                )

                # Denormalize GRU predictions; gradient flows through y_pred
                y_pred_h = y_pred[:, h_idx] * stds_t + means_t  # (n_wells,)

                y_gp = gp_models[h_idx](X_train_sp, y_pred_h, X_val_sp)
                gp_mses.append(loss_fn(y_gp.float(), val_true))

            if gp_mses:
                gp_loss = torch.stack(gp_mses).mean()
                loss = gru_mse + lambda_spatial * gp_loss
                epoch_gp_losses.append(gp_loss.item())
            else:
                loss = gru_mse

            epoch_gru_losses.append(gru_mse.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            gru_optimizer.step()
            gp_optimizer.step()

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        model.eval()
        for gp in gp_models:
            gp.eval()

        val_gru_losses = []
        val_gp_losses = []

        with torch.no_grad():
            # GRU val: all val dates
            for date_key in val_dates:
                indices = val_date_idx[date_key]
                xp = torch.from_numpy(x_past_val[indices]).to(device)
                xf = torch.from_numpy(x_future_val[indices]).to(device)
                ys = torch.from_numpy(y_val[indices]).to(device)
                xs = torch.from_numpy(x_static_val[indices]).to(device)
                val_gru_losses.append(loss_fn(model(xp, xf, xs), ys).item())

            # GP val: subsampled val dates
            for date_key in val_dates_gp:
                if date_key not in val_date_idx:
                    continue
                indices = val_date_idx[date_key]
                batch_well_ids = [val_meta[i][0] for i in indices]
                horizon_dates = val_meta[indices[0]][3]

                xp = torch.from_numpy(x_past_val[indices]).to(device)
                xf = torch.from_numpy(x_future_val[indices]).to(device)
                xs = torch.from_numpy(x_static_val[indices]).to(device)
                y_pred = model(xp, xf, xs)

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

                h_gp_losses = []
                for h_idx in range(out_len):
                    hdate_key = str(np.datetime64(horizon_dates[h_idx], "D"))
                    val_data = val_gws_lookup.get(hdate_key, {})
                    if len(val_data) < min_val_wells:
                        continue
                    val_well_ids = list(val_data.keys())
                    val_true = torch.tensor(
                        [val_data[wid] for wid in val_well_ids],
                        dtype=torch.float32, device=device,
                    )
                    X_val_sp = torch.tensor(
                        np.array([coords_scaled[wid] for wid in val_well_ids]), device=device
                    )
                    y_pred_h = y_pred[:, h_idx] * stds_t + means_t
                    y_gp = gp_models[h_idx](X_train_sp, y_pred_h, X_val_sp)
                    h_gp_losses.append(loss_fn(y_gp.float(), val_true).item())

                if h_gp_losses:
                    val_gp_losses.append(float(np.mean(h_gp_losses)))

        train_gru = float(np.mean(epoch_gru_losses)) if epoch_gru_losses else float("nan")
        train_gp = float(np.mean(epoch_gp_losses)) if epoch_gp_losses else float("nan")
        val_gru = float(np.mean(val_gru_losses)) if val_gru_losses else float("nan")
        val_gp = float(np.mean(val_gp_losses)) if val_gp_losses else float("nan")
        val_combined = val_gru + lambda_spatial * val_gp if val_gp_losses else val_gru

        print(
            f"epoch={epoch:3d}  "
            f"train_gru={train_gru:.5f}  train_gp={train_gp:.5f}  "
            f"val_gru={val_gru:.5f}  val_gp={val_gp:.5f}  "
            f"val_combined={val_combined:.5f}  "
            f"time={time.time() - t_epoch:.1f}s"
        )

        improved = best_val is None or val_combined < best_val - es_min_delta
        if improved:
            best_val = val_combined
            best_epoch = epoch
            no_improve = 0
            best_gru_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_gp_states = [{k: v.cpu().clone() for k, v in gp.state_dict().items()} for gp in gp_models]
        else:
            no_improve += 1
            if no_improve >= es_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {es_patience} epochs)")
                break

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    run_sig = _resolve_joint_run_sig(
        gru_cfg=gru_cfg,
        dataset=dataset,
        in_len=in_len,
        out_len=out_len,
        n_epochs=n_epochs,
        seed=seed,
        use_revin=use_revin,
        spatial_fraction=spatial_fraction,
        spatial_clusters=spatial_clusters,
        spatial_seed=spatial_seed,
        lambda_spatial=lambda_spatial,
    )
    run_dir = ROOT / "outputs" / "GRU_GP_JOINT" / f"GRU_GP_JOINT_{run_sig}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # GRU checkpoint (same format as gru_train.py for eval script compatibility)
    torch.save(
        {
            "model": best_gru_state,
            "past_input_size": x_past_train.shape[-1],
            "future_input_size": x_future_train.shape[-1],
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "out_len": out_len,
            "static_input_size": len(static_cols),
            "use_revin": use_revin,
        },
        run_dir / "model.pt",
    )

    # GP checkpoints
    torch.save(
        {
            "gp_state_dicts": {h: best_gp_states[h] for h in range(out_len)},
            "gp_config": {
                "backend": backend,
                "n_inducing": n_inducing,
                "jitter": jitter,
                "use_float64": use_float64,
                "init_noise": init_noise,
                "out_len": out_len,
            },
        },
        run_dir / "gp_models.pt",
    )

    # Scalers (same format as gru_train.py)
    with (run_dir / "scalers.pkl").open("wb") as f:
        pickle.dump({"cov_scaler": cov_scaler, "well_stats": well_stats, "static_scaler": static_scaler}, f)

    with (run_dir / "coord_scaler.pkl").open("wb") as f:
        pickle.dump(coord_scaler, f)

    # 3-way split assignment (for eval script)
    split_df.to_csv(run_dir / "split_info.csv", index=False)

    meta_out = {
        "dataset": dataset,
        "in_len": in_len,
        "out_len": out_len,
        "seed": seed,
        "n_epochs": n_epochs,
        "trained_epochs": trained_epochs,
        "best_epoch": best_epoch,
        "best_val_combined": float(best_val) if best_val is not None else None,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "gru_lr": gru_lr,
        "gp_lr": gp_lr,
        "grad_clip": grad_clip,
        "lambda_spatial": lambda_spatial,
        "backend": backend,
        "n_inducing": n_inducing,
        "n_static": len(static_cols),
        "spatial_train_wells": len(train_ids),
        "spatial_val_wells": len(val_ids),
        "spatial_test_wells": len(test_ids),
        "early_stopping": {"patience": es_patience, "min_delta": es_min_delta},
        "cov_cols": COV_COLS,
        "gru_pretrained_run_sig": gru_pretrained_run_sig or None,
    }
    (run_dir / "meta.yaml").write_text(yaml.safe_dump(meta_out), encoding="utf-8")

    print(f"\nSaved joint model to {run_dir}")
    print(f"Best epoch: {best_epoch}  |  Best val combined: {best_val:.5f}")


if __name__ == "__main__":
    main()
