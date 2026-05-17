from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
CUSTOM_GP_DIR = SCRIPT_DIR / "custom_gp"
for _p in [str(SRC_ROOT), str(SCRIPT_DIR), str(CUSTOM_GP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import yaml

from sklearn.preprocessing import StandardScaler
from libs.spatial_split import resolve_split_path
from gp_layer_custom import GPLayer

# ---------------------------------------------------------------------------
# Comprehensive static feature set (used with --all-features)
# ---------------------------------------------------------------------------

ALL_NUMERIC_FEATURES = [
    "gok", "ausbausohle",
    "siwa_verweilzeit_j",
    "TWI_dgm50_r1000m", "Slope_dgm50_r1000m",
    "GW_recharge_r1000m", "Percolation_r1000m",
    "DSD_order1_r1000m", "LP_order1_r1000m", "SD_order1_r1000m",
    "DSD_order5_r1000m", "LP_order5_r1000m", "SD_order5_r1000m",
    "parde_seasonality", "interannual_variation", "baseflow_index",
    "distance_wsg_z1", "mean_range_1991-2020", "acf_1991-2020",
    "pr_52W_xcorr", "pr_52W_lag",
]
ALL_ONEHOT_FEATURES = ["hydroraum", "gw_gespannt", "gfa_klasse"]

TRAIN_CUTOFF = pd.Timestamp("20200101")  # dates before this used for per-date training


# ---------------------------------------------------------------------------
# Deep-kernel GP model
# ---------------------------------------------------------------------------

class DeepKernelGPLayer(nn.Module):
    def __init__(self, n_input_dims: int, latent_dims: int = 8,
                 hidden_sizes: list[int] | None = None, **gp_kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 32]
        layers: list[nn.Module] = []
        in_d = n_input_dims
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_d, h), nn.Tanh()])
            in_d = h
        layers.append(nn.Linear(in_d, latent_dims))
        self.feature_net = nn.Sequential(*layers)
        self.gp = GPLayer(n_spatial_dims=latent_dims, **gp_kwargs)
        self.latent_dims = latent_dims
        self.n_input_dims = n_input_dims

    def _encode(self, X: torch.Tensor) -> torch.Tensor:
        return self.feature_net(X.float())

    def forward(self, X_tr, y_tr, X_te):
        return self.gp.forward(self._encode(X_tr), y_tr, self._encode(X_te))

    def forward_with_std(self, X_tr, y_tr, X_te):
        return self.gp.forward_with_std(self._encode(X_tr), y_tr, self._encode(X_te))

    def marginal_log_likelihood(self, X, y):
        return self.gp.marginal_log_likelihood(self._encode(X), y)


# ---------------------------------------------------------------------------
# Training — mean mode (v1: deduplicate and average y per well)
# ---------------------------------------------------------------------------

def _training_step(model, X_batch, y_batch, opt, cur_lr):
    last_good = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt.zero_grad()
    try:
        loss = model.marginal_log_likelihood(X_batch, y_batch)
    except RuntimeError as e:
        return None, cur_lr, last_good, str(e)
    if not torch.isfinite(loss):
        return None, cur_lr, last_good, "non-finite"
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    if not torch.isfinite(grad_norm) or not all(
        torch.isfinite(p).all() for p in model.parameters()
    ):
        model.load_state_dict(last_good)
        return None, cur_lr * 0.5, last_good, "bad-grad"
    opt.step()
    return loss.item(), cur_lr, None, None


def train_dkl(model, X_all, y_all, n_steps=500, lr=3e-3, max_pts=2000):
    """Train on time-averaged GWL per well (deduplicated)."""
    model.train()
    X_np, y_np = X_all.numpy(), y_all.numpy()
    keys = np.round(X_np, 6)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        n_u = len(counts)
        X_d = np.zeros((n_u, X_np.shape[1]), dtype=np.float32)
        y_d = np.zeros(n_u, dtype=np.float32)
        np.add.at(X_d, inv, X_np)
        np.add.at(y_d, inv, y_np)
        X_all = torch.from_numpy(X_d / counts[:, None])
        y_all = torch.from_numpy(y_d / counts)

    if X_all.size(0) > max_pts:
        idx = torch.randperm(X_all.size(0))[:max_pts]
        X_s, y_s = X_all[idx], y_all[idx]
    else:
        X_s, y_s = X_all, y_all

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr * 0.01)
    best_loss, best_state = float("inf"), {k: v.detach().clone() for k, v in model.state_dict().items()}
    cur_lr, steps_ok = lr, 0

    for step in range(n_steps):
        lv, cur_lr, _, reason = _training_step(model, X_s, y_s, opt, cur_lr)
        if lv is None:
            if reason in ("non-finite", None):
                break
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            if cur_lr < 1e-7:
                break
            continue
        scheduler.step()
        steps_ok += 1
        if lv < best_loss:
            best_loss = lv
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (step + 1) % 100 == 0:
            ls = float(model.gp.length_scale().detach().float().view(-1)[0])
            nz = float(model.gp.noise().detach().float())
            print(f"    [dkl] step {step+1}/{n_steps}  mll={lv:.2f}  ls={ls:.3f}  noise={nz:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    return {"steps_ok": steps_ok, "best_mll": best_loss}


# ---------------------------------------------------------------------------
# Training — per-date mode (v2: sample a random training-period date each step)
# ---------------------------------------------------------------------------

def train_dkl_perdate(model, X_wells, date_obs, n_steps=500, lr=3e-3, max_pts=2000):
    """
    Per-date training: at each step, sample a random training-period date and
    use actual observed GWL values for that date (not time-averaged means).
    X_wells: [n_wells, n_features] tensor (features are the same for all dates).
    date_obs: list of (well_indices, y_norm_tensor) tuples, one per training date.
    """
    model.train()
    rng = np.random.default_rng(42)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr * 0.01)
    best_loss, best_state = float("inf"), {k: v.detach().clone() for k, v in model.state_dict().items()}
    cur_lr, steps_ok = lr, 0

    for step in range(n_steps):
        i = int(rng.integers(len(date_obs)))
        well_idx, y_t = date_obs[i]
        X_batch = X_wells[well_idx].float()
        y_batch = y_t.float()
        if X_batch.size(0) > max_pts:
            sel = torch.randperm(X_batch.size(0))[:max_pts]
            X_batch, y_batch = X_batch[sel], y_batch[sel]

        lv, cur_lr, _, reason = _training_step(model, X_batch, y_batch, opt, cur_lr)
        if lv is None:
            if reason in ("non-finite", None):
                break
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            if cur_lr < 1e-7:
                break
            continue
        scheduler.step()
        steps_ok += 1
        if lv < best_loss:
            best_loss = lv
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (step + 1) % 100 == 0:
            ls = float(model.gp.length_scale().detach().float().view(-1)[0])
            nz = float(model.gp.noise().detach().float())
            print(f"    [dkl] step {step+1}/{n_steps}  mll={lv:.2f}  ls={ls:.3f}  noise={nz:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    return {"steps_ok": steps_ok, "best_mll": best_loss}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _nse(pred, real):
    denom = float(np.sum((real - np.mean(real)) ** 2))
    return float("nan") if denom == 0 else float(1 - np.sum((pred - real) ** 2) / denom)

def _rmse(pred, real):
    return float(np.sqrt(np.mean((pred - real) ** 2)))

def _nrmse(pred, real):
    iqr = float(np.diff(np.quantile(real, [0.25, 0.75]))[0])
    return float("nan") if iqr == 0 else _rmse(pred, real) / iqr


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_yaml(path):
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deep-kernel GP oracle evaluation")
    parser.add_argument("--config", default="configs/gp/gp.yaml")
    parser.add_argument("--latent-dims", type=int, default=8)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--train-lr", type=float, default=3e-3)
    parser.add_argument("--max-train-pts", type=int, default=2000)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--date-freq", default="D")
    parser.add_argument("--jitter", type=float, default=1e-5)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--all-features", action="store_true",
                        help="Use comprehensive static feature set instead of config features")
    parser.add_argument("--per-date", action="store_true",
                        help="Train per-date (stochastic) instead of on time-averaged means")
    parser.add_argument("--kernel", default="matern32", choices=["rbf", "matern32"],
                        help="GP kernel (matern32 is now safe with epsilon fix)")
    args = parser.parse_args()

    # --- load configs -------------------------------------------------------
    gp_cfg  = _load_yaml(ROOT / args.config)
    data_cfg = _load_yaml(ROOT / "configs" / "data.yaml")
    gru_cfg  = _load_yaml(ROOT / "configs" / "gru" / "gru.yaml")

    for key in ["full_raw_path", "full_merged_path", "sample_path", "metadata_path"]:
        if key in data_cfg and data_cfg[key]:
            p = Path(data_cfg[key])
            data_cfg[key] = str(p if p.is_absolute() else (ROOT / p).resolve())

    dataset    = str(gp_cfg.get("dataset", "full_merged"))
    spatial_cfg = gp_cfg.get("spatial_split", gru_cfg.get("spatial_split", {}))
    split_path  = ROOT / args.split_file if args.split_file else resolve_split_path(
        ROOT / "splits", dataset, spatial_cfg)
    split       = pd.read_csv(split_path)
    train_ids   = set(split.loc[split["spatial_split"] == "spatial_train", "id"])
    holdout_ids = set(split.loc[split["spatial_split"].isin(
        ["spatial_holdout", "spatial_test"]), "id"])
    print(f"Split: {len(train_ids)} train wells, {len(holdout_ids)} holdout wells")

    # --- load data ----------------------------------------------------------
    data_path = Path(data_cfg[{
        "sample": "sample_path", "full_raw": "full_raw_path"
    }.get(dataset, "full_merged_path")])

    base_cols = ["datum", "id", "gws", "x_25833", "y_25833"]
    gws = (
        pd.read_csv(data_path, usecols=base_cols, low_memory=False)
        if str(data_path).lower().endswith(".csv")
        else pq.read_table(data_path, columns=base_cols).to_pandas()
    )
    gws["datum"] = pd.to_datetime(gws["datum"])

    # --- build static feature matrix ----------------------------------------
    if args.all_features:
        gp_features        = list(ALL_NUMERIC_FEATURES)
        gp_features_onehot = list(ALL_ONEHOT_FEATURES)
    else:
        gp_features        = list(gp_cfg.get("gp_features", []))
        gp_features_onehot = list(gp_cfg.get("gp_features_onehot", []))

    coords = gws[["id", "x_25833", "y_25833"]].drop_duplicates("id").copy()
    onehot_cols: list[str] = []

    if gp_features or gp_features_onehot:
        meta_path = Path(data_cfg.get("metadata_path", ""))
        all_meta_cols = pd.read_csv(meta_path, sep=";", nrows=0).columns.tolist()
        gp_features        = [f for f in gp_features        if f in all_meta_cols]
        gp_features_onehot = [f for f in gp_features_onehot if f in all_meta_cols]
        meta = pd.read_csv(meta_path, sep=";")[["id"] + gp_features + gp_features_onehot]
        coords = coords.merge(meta, on="id", how="left")
        for f in gp_features:
            coords[f] = pd.to_numeric(coords[f], errors="coerce")
            coords[f] = coords[f].fillna(coords[f].median())
        for f in gp_features_onehot:
            dummies = pd.get_dummies(coords[f], prefix=f, drop_first=True).astype(float)
            onehot_cols.extend(dummies.columns.tolist())
            coords = pd.concat([coords.drop(columns=[f]), dummies], axis=1)

    feature_cols  = ["x_25833", "y_25833"] + gp_features + onehot_cols
    n_input_dims  = len(feature_cols)
    print(f"Features ({n_input_dims}): {feature_cols}")

    gws = gws.merge(coords[["id"] + gp_features + onehot_cols], on="id", how="left")

    # --- fit scaler + y normalisation on training wells ---------------------
    train_obs = gws[gws["id"].isin(train_ids)].dropna(subset=["gws"] + feature_cols).copy()
    feature_scaler = StandardScaler().fit(train_obs[feature_cols].to_numpy())
    y_raw  = train_obs["gws"].to_numpy()
    y_mean = float(np.mean(y_raw))
    y_std  = float(np.std(y_raw)) or 1.0

    # --- build model --------------------------------------------------------
    model = DeepKernelGPLayer(
        n_input_dims=n_input_dims,
        latent_dims=args.latent_dims,
        hidden_sizes=args.hidden,
        kernel_type=args.kernel,
        isotropic=True,
        jitter=args.jitter,
    )
    print(f"DNN: {n_input_dims} → {args.hidden} → {args.latent_dims}  kernel={args.kernel}")

    # --- train --------------------------------------------------------------
    if args.per_date:
        # Build well-order aligned X matrix
        well_order = sorted(train_ids)
        well_to_idx = {wid: i for i, wid in enumerate(well_order)}
        well_feats = coords[coords["id"].isin(train_ids)].set_index("id").reindex(well_order)[feature_cols]
        well_feats = well_feats.fillna(well_feats.median())
        X_wells = torch.tensor(
            feature_scaler.transform(well_feats.to_numpy()), dtype=torch.float32
        )

        # Pivot training-period GWL by date
        train_period = train_obs[train_obs["datum"] < TRAIN_CUTOFF][["id", "datum", "gws"]]
        pivot = train_period.pivot_table(index="datum", columns="id", values="gws")
        pivot = pivot.reindex(columns=well_order)

        date_obs = []
        for _, row in pivot.iterrows():
            valid = ~row.isna()
            if valid.sum() < 50:
                continue
            idx = torch.tensor(np.where(valid.values)[0], dtype=torch.long)
            y_t = torch.tensor(
                (row[valid].values.astype(np.float32) - y_mean) / y_std, dtype=torch.float32
            )
            date_obs.append((idx, y_t))
        print(f"\nPer-date training: {len(date_obs)} training dates, "
              f"{len(well_order)} wells, lr={args.train_lr}")
        print(f"Training DKL ({args.train_steps} steps) ...")
        train_stats = train_dkl_perdate(
            model, X_wells, date_obs,
            n_steps=args.train_steps, lr=args.train_lr, max_pts=args.max_train_pts,
        )
    else:
        X_all = torch.tensor(
            feature_scaler.transform(train_obs[feature_cols].to_numpy()), dtype=torch.float32
        )
        y_all = torch.tensor((y_raw - y_mean) / y_std, dtype=torch.float32)
        print(f"\nGlobal training set: {X_all.size(0)} obs from {train_obs['id'].nunique()} wells")
        print(f"Training DKL ({args.train_steps} steps, lr={args.train_lr}) ...")
        train_stats = train_dkl(
            model, X_all, y_all,
            n_steps=args.train_steps, lr=args.train_lr, max_pts=args.max_train_pts,
        )

    ls  = float(model.gp.length_scale().detach().float().view(-1)[0])
    os_ = float(model.gp.output_scale().detach().float())
    nz  = float(model.gp.noise().detach().float())
    print(f"Training done: steps_ok={train_stats['steps_ok']}  best_mll={train_stats['best_mll']:.2f}")
    print(f"  GP hyperparams: ls={ls:.4f}  os={os_:.4f}  noise={nz:.4f}")

    # --- per-date oracle inference ------------------------------------------
    all_dates = gws["datum"].drop_duplicates().sort_values()
    dates = (
        pd.Series(all_dates.values, index=all_dates)
        .resample(args.date_freq).first().dropna().tolist()
    )
    print(f"\nEvaluating on {len(dates)} dates ...")

    out_rows, skipped_n = [], 0
    for dt in dates:
        day = gws[gws["datum"] == dt]
        tr = day[day["id"].isin(train_ids)].dropna(subset=["gws"] + feature_cols)
        ho = day[day["id"].isin(holdout_ids)].dropna(subset=["gws"] + feature_cols)
        if tr.empty or ho.empty:
            skipped_n += 1
            continue

        X_tr = torch.tensor(feature_scaler.transform(tr[feature_cols].to_numpy()), dtype=torch.float32)
        y_tr = torch.tensor((tr["gws"].to_numpy() - y_mean) / y_std, dtype=torch.float32)
        X_te = torch.tensor(feature_scaler.transform(ho[feature_cols].to_numpy()), dtype=torch.float32)

        with torch.no_grad():
            y_pred_n, y_std_n = model.forward_with_std(X_tr, y_tr, X_te)

        y_pred = y_pred_n.cpu().numpy() * y_std + y_mean
        y_unc  = y_std_n.cpu().numpy()  * y_std

        out = ho[["id", "x_25833", "y_25833"]].copy().reset_index(drop=True)
        out["gws_true"]         = ho["gws"].to_numpy()
        out["gws_forecast"]     = y_pred
        out["gws_forecast_std"] = y_unc
        out["datum"]            = dt
        out["horizon"]          = 1
        out_rows.append(out)

    if not out_rows:
        raise ValueError("No oracle DKL outputs generated.")

    out_df = pd.concat(out_rows, ignore_index=True)
    print(f"Skipped {skipped_n} dates (no train or holdout obs).")

    # --- metrics ------------------------------------------------------------
    pred_all = out_df["gws_forecast"].to_numpy()
    real_all = out_df["gws_true"].to_numpy()
    abs_err  = np.abs(pred_all - real_all)

    per_id_rows = []
    for wid, g in out_df.groupby("id"):
        pw, tw = g["gws_forecast"].to_numpy(), g["gws_true"].to_numpy()
        per_id_rows.append({"id": wid, "NSE_over_time": _nse(pw, tw),
                             "RMSE_over_time": _rmse(pw, tw), "nRMSE_over_time": _nrmse(pw, tw)})
    per_id_df = pd.DataFrame(per_id_rows)
    valid_nse  = per_id_df["NSE_over_time"].to_numpy(float)
    valid_rmse = per_id_df["RMSE_over_time"].to_numpy(float)
    valid_nse  = valid_nse[np.isfinite(valid_nse)]
    valid_rmse = valid_rmse[np.isfinite(valid_rmse)]

    overall = {
        "RMSE":          _rmse(pred_all, real_all),
        "nRMSE":         _nrmse(pred_all, real_all),
        "NSE_pooled":    _nse(pred_all, real_all),
        "NSE_id_median": float(np.median(valid_nse))       if valid_nse.size  else float("nan"),
        "NSE_id_mean":   float(np.mean(valid_nse))         if valid_nse.size  else float("nan"),
        "NSE_id_p25":    float(np.percentile(valid_nse, 25)) if valid_nse.size else float("nan"),
        "NSE_id_p75":    float(np.percentile(valid_nse, 75)) if valid_nse.size else float("nan"),
        "NSE_id_count":  float(valid_nse.size),
        "RMSE_pw":       float(np.median(valid_rmse))      if valid_rmse.size else float("nan"),
        "MAE":           float(np.mean(abs_err)),
        "AbsErr_P95":    float(np.percentile(abs_err, 95)) if abs_err.size    else float("nan"),
        "n_dates":       float(len(out_rows)),
        "n_train_wells": float(len(train_ids)),
        "n_holdout_wells": float(len(holdout_ids)),
        "dkl_latent_dims":  float(args.latent_dims),
        "dkl_train_steps":  float(args.train_steps),
        "gp_ls": ls, "gp_os": os_, "gp_noise": nz,
        "dkl_best_mll": train_stats["best_mll"],
        "y_mean": y_mean, "y_std": y_std,
    }

    # --- save ---------------------------------------------------------------
    run_tag = args.run_tag or (
        f"dkl_oracle_lat{args.latent_dims}_h{'x'.join(str(h) for h in args.hidden)}"
        f"_steps{args.train_steps}"
    )
    out_dir = ROOT / "outputs" / "gp_dkl" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    pq.write_table(pa.Table.from_pandas(out_df), out_dir / "gp_dkl_pred.parquet")
    metrics_df = pd.Series(overall).reset_index()
    metrics_df.columns = ["metric", "value"]
    pq.write_table(pa.Table.from_pandas(metrics_df), out_dir / "gp_dkl_metrics.parquet")
    metrics_df.to_csv(out_dir / "gp_dkl_metrics.csv", index=False)
    per_id_df.to_csv(out_dir / "gp_dkl_metrics_by_id.csv", index=False)
    pd.Series({
        "run_tag": run_tag, "config": args.config,
        "dkl_hidden": str(args.hidden), "feature_cols": str(feature_cols),
        "split_path": str(split_path), "kernel": args.kernel,
        "per_date": str(args.per_date), "all_features": str(args.all_features),
    }).to_csv(out_dir / "run_config.csv", header=False)

    print(f"\nSaved to {out_dir}")
    print("Overall metrics:")
    for k, v in overall.items():
        if isinstance(v, float) and not math.isnan(v):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
