from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from libs.spatial_split import resolve_split_path
from libs.run_registry import lookup_run_id, read_registry
from gp_layer import make_gp_layer


def _load_yaml(path):
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _nse(pred, real):
    denom = float(np.sum((real - np.mean(real)) ** 2))
    if denom == 0:
        return float("nan")
    return float(1 - np.sum((pred - real) ** 2) / denom)


def _rmse(pred, real):
    return float(np.sqrt(np.mean((pred - real) ** 2)))


def _nrmse(pred, real):
    iqr = float(np.diff(np.quantile(real, [0.25, 0.75]))[0])
    if iqr == 0:
        return float("nan")
    return _rmse(pred, real) / iqr


def pretrain_mll_kernel(gp, X_train, y_train, n_steps=200, lr=1e-2, max_train_pts=2000, device=torch.device("cpu")):
    gp.train()
    if X_train.size(0) > max_train_pts:
        idx = torch.randperm(X_train.size(0))[:max_train_pts]
        X_s, y_s = X_train[idx], y_train[idx]
    else:
        X_s, y_s = X_train, y_train

    X_s = X_s.detach().to(device)
    y_s = y_s.detach().to(device)

    start_ls = float(gp.length_scale().detach().float().view(-1)[0])
    start_os = float(gp.output_scale().detach().float().view(-1)[0])
    start_nz = float(gp.noise().detach().float().view(-1)[0])

    cur_lr = float(lr)
    params = list(gp.svgp.covar_module.parameters()) + list(gp.likelihood.parameters())
    opt = torch.optim.Adam(params, lr=cur_lr)
    steps_ok = 0

    for step in range(n_steps):
        last_good = {k: v.detach().clone() for k, v in gp.state_dict().items()}
        opt.zero_grad()
        try:
            loss = gp.marginal_log_likelihood(X_s, y_s)
        except RuntimeError as e:
            print(f"    mll failed at step {step + 1}/{n_steps}: {e}")
            break
        if not torch.isfinite(loss):
            print(f"    non-finite mll at step {step + 1}/{n_steps}; stop pretrain")
            break
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        if not torch.isfinite(grad_norm):
            gp.load_state_dict(last_good)
            cur_lr *= 0.5
            if cur_lr < 1e-6:
                break
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            continue
        opt.step()
        if not all(torch.isfinite(p).all() for p in params):
            gp.load_state_dict(last_good)
            cur_lr *= 0.5
            if cur_lr < 1e-6:
                break
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            continue
        steps_ok += 1

    gp.eval()
    end_ls = float(gp.length_scale().detach().float().view(-1)[0])
    end_os = float(gp.output_scale().detach().float().view(-1)[0])
    end_nz = float(gp.noise().detach().float().view(-1)[0])
    changed = (
        abs(end_ls - start_ls) > 1e-6
        or abs(end_os - start_os) > 1e-6
        or abs(end_nz - start_nz) > 1e-6
    )
    return {
        "steps_ok": steps_ok, "lr_final": cur_lr, "changed": changed,
        "ls_start": start_ls, "os_start": start_os, "nz_start": start_nz,
        "ls_end": end_ls, "os_end": end_os, "nz_end": end_nz,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gp/gp.yaml")
    parser.add_argument("--gru-run-sig", default=None)
    parser.add_argument("--model-prefix", default=None)
    parser.add_argument("--gp-run-tag", default=None)
    parser.add_argument("--pred-path", default=None)
    parser.add_argument("--pretrain-steps", type=int, default=None)
    parser.add_argument("--pretrain-lr", type=float, default=None)
    parser.add_argument("--max-pretrain-pts", type=int, default=None)
    parser.add_argument("--date-freq", default=None)
    parser.add_argument("--jitter", type=float, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--n-inducing", type=int, default=None)
    parser.add_argument("--variational-lr", type=float, default=None)
    parser.add_argument("--use-float64", default=None)
    parser.add_argument("--mean-type", default=None)

    config_args, _ = parser.parse_known_args()
    gp_cfg = _load_yaml(ROOT / config_args.config)
    parser.set_defaults(
        model_prefix     = str(gp_cfg.get("model",            "GRU_FCOV")),
        gru_run_sig      = gp_cfg.get("gru_run_sig",          None) or None,
        gp_run_tag       = str(gp_cfg.get("gp_run_tag",       "gp_pytorch")),
        pretrain_steps   = int(gp_cfg.get("pretrain_steps",   200)),
        pretrain_lr      = float(gp_cfg.get("pretrain_lr",    1e-2)),
        max_pretrain_pts = int(gp_cfg.get("max_pretrain_pts", 2000)),
        date_freq        = str(gp_cfg.get("date_freq",        "ME")),
        jitter           = float(gp_cfg.get("jitter",         1e-5)),
        backend          = str(gp_cfg.get("backend",          "gpytorch")),
        n_inducing       = int(gp_cfg.get("n_inducing",       64)),
        variational_lr   = float(gp_cfg.get("variational_lr", 1e-2)),
        use_float64      = str(gp_cfg.get("use_float64",      True)).lower(),
        mean_type        = str(gp_cfg.get("mean_type",        "zero")),
    )
    args = parser.parse_args()

    gru_cfg = _load_yaml(ROOT / "configs" / "gru.yaml")
    tft_cfg = _load_yaml(ROOT / "configs" / "tft.yaml")
    data_cfg = _load_yaml(ROOT / "configs" / "data.yaml")

    for key in ["full_raw_path", "full_merged_path", "sample_path", "metadata_path"]:
        if key in data_cfg and data_cfg[key]:
            p = Path(data_cfg[key])
            data_cfg[key] = str(p if p.is_absolute() else (ROOT / p).resolve())

    cfg_src = gru_cfg if gru_cfg else tft_cfg
    dataset = str(cfg_src.get("dataset", "full_merged"))
    spatial_cfg = gp_cfg.get("spatial_split", cfg_src.get("spatial_split", {}))

    gru_run_sig = args.gru_run_sig
    if not gru_run_sig:
        run_sig_cfg = str(cfg_src.get("run_sig", "")).strip()
        if run_sig_cfg and run_sig_cfg.lower() != "auto":
            gru_run_sig = run_sig_cfg
        else:
            tr = cfg_src.get("training", {})
            dc = cfg_src.get("data", {})
            mc = cfg_src.get("model", {})
            sc = spatial_cfg
            in_len  = int(dc.get("in_len", 52))
            out_len = int(dc.get("out_len", 16))
            epochs  = int(tr.get("epochs", 50))
            bs      = int(tr.get("batch_size", 4096))
            seed    = int(tr.get("seed", 40))
            spf     = str(float(sc.get("train_fraction", 0.8))).replace(".", "p")
            sc_cnt  = int(sc.get("cluster_count", 20))
            ss      = int(sc.get("split_seed", 42))
            gru_run_sig = f"in{in_len}_out{out_len}_ep{epochs}_bs{bs}_seed{seed}_{dataset}_spf{spf}_sc{sc_cnt}_ss{ss}"

    model_prefix = str(args.model_prefix).strip()
    backend      = str(args.backend).strip().lower()
    n_inducing   = int(args.n_inducing)
    variational_lr = float(args.variational_lr)
    use_float64  = _as_bool(args.use_float64)
    mean_type    = str(args.mean_type).strip().lower()
    if args.pred_path:
        pred_path = Path(args.pred_path)
    else:
        # Try registry lookup first; fall back to old-style long-sig directory
        try:
            gru_run_id = lookup_run_id(ROOT / "outputs", model_prefix, gru_run_sig)
            pred_path = ROOT / "outputs" / model_prefix / f"{model_prefix}_{gru_run_id}" / "predictions" / "pred.parquet"
        except (FileNotFoundError, ValueError):
            pred_path = ROOT / "outputs" / model_prefix / f"{model_prefix}_{gru_run_sig}" / "predictions" / "pred.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"pred.parquet not found: {pred_path}")

    split_path = resolve_split_path(ROOT / "splits", dataset, spatial_cfg)
    split = pd.read_csv(split_path)
    train_ids   = set(split.loc[split["spatial_split"] == "spatial_train", "id"])
    holdout_ids = set(split.loc[split["spatial_split"] == "spatial_holdout", "id"])

    if dataset == "sample":
        data_path = Path(data_cfg["sample_path"])
    elif dataset == "full_raw":
        data_path = Path(data_cfg["full_raw_path"])
    else:
        data_path = Path(data_cfg["full_merged_path"])

    gp_features = list(gp_cfg.get("gp_features", []))
    gp_features_onehot = list(gp_cfg.get("gp_features_onehot", []))

    cols = ["datum", "id", "gws", "x_25833", "y_25833"]
    gws = (
        pd.read_csv(data_path, usecols=cols, low_memory=False)
        if str(data_path).lower().endswith(".csv")
        else pq.read_table(data_path, columns=cols).to_pandas()
    )
    gws["datum"] = pd.to_datetime(gws["datum"])

    pred = pq.read_table(pred_path).to_pandas()
    pred["datum"] = pd.to_datetime(pred["datum"])
    if "gws_forecast" in pred.columns:
        pred["gws_pred"] = pred["gws_forecast"]
    else:
        pred["gws_pred"] = pred["gws"]

    coords = gws[["id", "x_25833", "y_25833"]].drop_duplicates("id")
    onehot_cols = []
    if gp_features or gp_features_onehot:
        meta_path = Path(data_cfg.get("metadata_path", ""))
        meta = pd.read_csv(meta_path, sep=";")[["id"] + gp_features + gp_features_onehot]
        coords = coords.merge(meta, on="id", how="left")
        for f in gp_features:
            coords[f] = coords[f].fillna(coords[f].median())
        for f in gp_features_onehot:
            dummies = pd.get_dummies(coords[f], prefix=f, drop_first=True).astype(float)
            onehot_cols.extend(dummies.columns.tolist())
            coords = pd.concat([coords.drop(columns=[f]), dummies], axis=1)
    feature_cols = ["x_25833", "y_25833"] + gp_features + onehot_cols
    pred = pred.merge(coords, on="id", how="left")

    all_dates = pred["datum"].drop_duplicates().sort_values()
    dates = pd.Series(all_dates.values, index=all_dates).resample(args.date_freq).first().dropna().tolist()
    horizons = sorted(pred["horizon"].unique().tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model prefix: {model_prefix}")
    print(f"Run sig: {gru_run_sig}")
    print(f"Dates ({len(dates)}): {[str(d.date()) for d in dates[:4]]} ...")
    print(f"Horizons: {horizons}")

    coords_arr = coords.set_index("id")
    train_coords_df = coords_arr.loc[coords_arr.index.isin(train_ids)]
    feature_scaler = StandardScaler().fit(train_coords_df[feature_cols].to_numpy())
    n_dims = len(feature_cols)

    out_rows = []
    valid_pairs = 0
    skipped = []

    for h in horizons:
        pred_h = pred[pred["horizon"] == h]
        train_pred_h = pred_h[pred_h["id"].isin(train_ids)].dropna(subset=["x_25833", "y_25833", "gws_pred"])

        gp = make_gp_layer(
            backend=backend, n_spatial_dims=n_dims,
            jitter=args.jitter, n_inducing=n_inducing, use_float64=use_float64,
            mean_type=mean_type,
        ).to(device)

        if train_pred_h.empty:
            print(f"  horizon {h}: no train predictions — skipping kernel pretrain")
        else:
            X_all = torch.tensor(
                feature_scaler.transform(train_pred_h[feature_cols].to_numpy()),
                dtype=torch.float32, device=device
            )
            y_all = torch.tensor(train_pred_h["gws_pred"].to_numpy(), dtype=torch.float32, device=device)
            print(f"  horizon {h}: pre-training GP kernel ({args.pretrain_steps} steps, backend={backend}) on {X_all.size(0)} pts ...")

            pretrain_stats = pretrain_mll_kernel(
                gp, X_all, y_all, n_steps=args.pretrain_steps,
                lr=variational_lr, max_train_pts=args.max_pretrain_pts, device=device,
            )
            print(
                f"    pretrain_effect:"
                f" steps_ok={pretrain_stats['steps_ok']}"
                f" changed={pretrain_stats['changed']}"
                f" lr_final={pretrain_stats['lr_final']:.2e}"
            )

            ls  = float(gp.length_scale().detach().float().view(-1)[0].item())
            os_ = float(gp.output_scale().detach().float().view(-1)[0].item())
            nz  = float(gp.noise().detach().float().view(-1)[0].item())
            if not np.isfinite(ls) or not np.isfinite(os_) or not np.isfinite(nz):
                print("    non-finite GP hyperparams after pretrain; re-init GP without pretrain for this horizon")
                gp = make_gp_layer(
                    backend=backend, n_spatial_dims=n_dims,
                    jitter=args.jitter, n_inducing=n_inducing, use_float64=use_float64,
                ).to(device)
                ls  = float(gp.length_scale().detach().float().view(-1)[0].item())
                os_ = float(gp.output_scale().detach().float().view(-1)[0].item())
                nz  = float(gp.noise().detach().float().view(-1)[0].item())
            print(f"    ls={ls:.4f}  os={os_:.4f}  noise={nz:.4f}")

        for dt in dates:
            gws_dt = gws[gws["datum"] == dt][["id", "gws", "x_25833", "y_25833"]]
            holdout_true = gws_dt[gws_dt["id"].isin(holdout_ids)].dropna(subset=["gws", "x_25833", "y_25833"])
            if gp_features or onehot_cols:
                holdout_true = holdout_true.merge(coords[["id"] + gp_features + onehot_cols], on="id", how="left")
            if holdout_true.empty:
                skipped.append((dt, h, "no holdout obs"))
                continue

            pred_sel = pred[(pred["datum"] == dt) & (pred["horizon"] == h)]
            pred_sel = pred_sel[pred_sel["id"].isin(train_ids)].dropna(subset=["x_25833", "y_25833", "gws_pred"])
            if pred_sel.empty:
                skipped.append((dt, h, "no train predictions"))
                continue

            X_train_np = feature_scaler.transform(pred_sel[feature_cols].to_numpy())
            y_train_np = pred_sel["gws_pred"].to_numpy()
            X_test_np  = feature_scaler.transform(holdout_true[feature_cols].to_numpy())

            X_train_t = torch.tensor(X_train_np, dtype=torch.float32, device=device)
            y_train_t = torch.tensor(y_train_np, dtype=torch.float32, device=device)
            X_test_t  = torch.tensor(X_test_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                y_pred_t, y_std_t = gp.forward_with_std(X_train_t, y_train_t, X_test_t)

            out = holdout_true.copy().reset_index(drop=True)
            out["gws_true"]         = out["gws"]
            out.drop(columns=["gws"], inplace=True)
            out["gws_forecast"]     = y_pred_t.cpu().numpy()
            out["gws_forecast_std"] = y_std_t.cpu().numpy()
            out["horizon"]          = h
            out["datum"]            = dt
            out_rows.append(out)
            valid_pairs += 1

    if not out_rows:
        raise ValueError("No GP outputs generated. Check date/horizon availability in GRU predictions.")

    out_df = pd.concat(out_rows, ignore_index=True)

    pred_all = out_df["gws_forecast"].to_numpy()
    real_all = out_df["gws_true"].to_numpy()
    abs_err  = np.abs(pred_all - real_all)

    per_id_nse_rows = []
    for well_id, g in out_df.groupby("id"):
        per_id_nse_rows.append({
            "id": well_id,
            "NSE_over_time": _nse(g["gws_forecast"].to_numpy(), g["gws_true"].to_numpy()),
        })
    per_id_nse_df = pd.DataFrame(per_id_nse_rows)
    per_id_valid = per_id_nse_df["NSE_over_time"].to_numpy(dtype=float) if not per_id_nse_df.empty else np.array([])
    per_id_valid = per_id_valid[np.isfinite(per_id_valid)]

    overall_metrics = {
        "RMSE":          _rmse(pred_all, real_all),
        "nRMSE":         _nrmse(pred_all, real_all),
        "NSE_pooled":    _nse(pred_all, real_all),
        "NSE":           _nse(pred_all, real_all),
        "NSE_id_median": float(np.median(per_id_valid)) if per_id_valid.size else float("nan"),
        "NSE_id_p25":    float(np.percentile(per_id_valid, 25)) if per_id_valid.size else float("nan"),
        "NSE_id_p50":    float(np.percentile(per_id_valid, 50)) if per_id_valid.size else float("nan"),
        "NSE_id_p75":    float(np.percentile(per_id_valid, 75)) if per_id_valid.size else float("nan"),
        "NSE_id_mean":   float(np.mean(per_id_valid)) if per_id_valid.size else float("nan"),
        "NSE_id_count":  float(per_id_valid.size),
        "MAE":           float(np.mean(abs_err)),
        "AbsErr_P95":    float(np.percentile(abs_err, 95)) if abs_err.size else float("nan"),
        "AbsErr_P99":    float(np.percentile(abs_err, 99)) if abs_err.size else float("nan"),
        "AbsErr_Max":    float(np.max(abs_err)) if abs_err.size else float("nan"),
    }

    pair_rows = []
    for (dt, hv), g in out_df.groupby(["datum", "horizon"]):
        real = g["gws_true"].to_numpy()
        pv   = g["gws_forecast"].to_numpy()
        pair_rows.append({
            "datum":   pd.to_datetime(dt),
            "horizon": float(hv),
            "n":       int(len(real)),
            "RMSE":    _rmse(pv, real),
            "nRMSE":   _nrmse(pv, real),
            "NSE":     _nse(pv, real),
            "MAE":     float(np.mean(np.abs(pv - real))),
        })
    per_pair_metrics = pd.DataFrame(pair_rows).sort_values(["datum", "horizon"])

    run_tag = f"{model_prefix}_{gru_run_sig}__{args.gp_run_tag}__predobstrain"
    gp_dir  = ROOT / "outputs" / "gp" / run_tag
    gp_dir.mkdir(parents=True, exist_ok=True)

    pq.write_table(pa.Table.from_pandas(out_df), gp_dir / "gp_pred.parquet")
    metrics_df = pd.Series(overall_metrics).reset_index()
    metrics_df.columns = ["metric", "value"]
    pq.write_table(pa.Table.from_pandas(metrics_df), gp_dir / "gp_metrics.parquet")
    per_pair_metrics.to_csv(gp_dir / "gp_metrics_by_pair.csv", index=False)
    if not per_id_nse_df.empty:
        per_id_nse_df.to_csv(gp_dir / "gp_metrics_by_id.csv", index=False)

    print(f"\nSaved GP outputs to {gp_dir}")
    print(f"Valid pairs: {valid_pairs}  |  Skipped: {len(skipped)}")
    print("Overall metrics:")
    for k, v in overall_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) and not math.isnan(v) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
