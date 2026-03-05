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
from gp_layer import GPLayer, make_gp_layer


def _load_yaml(path):
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _as_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


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


def pretrain_gp_kernel(
    gp,
    X_train,
    y_train,
    n_steps=200,
    lr=1e-2,
    max_train_pts=2000,
    device=torch.device("cpu"),
):
    def _params_finite(module):
        for p in module.parameters():
            if not torch.isfinite(p).all():
                return False
        return True

    gp.train()
    if X_train.size(0) > max_train_pts:
        idx = torch.randperm(X_train.size(0), device=X_train.device)[:max_train_pts]
        X_s, y_s = X_train[idx], y_train[idx]
    else:
        X_s, y_s = X_train, y_train

    # Run MLL pretraining on CPU float64 for numerical stability.
    X_s = X_s.detach().to(device=torch.device("cpu"), dtype=torch.float64)
    y_s = y_s.detach().to(device=torch.device("cpu"), dtype=torch.float64)
    gp_pre = GPLayer(
        n_spatial_dims=gp.n_spatial_dims,
        kernel_type=gp.kernel_type,
        isotropic=gp.isotropic,
        jitter=gp.jitter,
        mll_diag_eps=getattr(gp, "mll_diag_eps", 1e-4),
    )
    gp_pre.load_state_dict(gp.state_dict())
    gp_pre = gp_pre.to(device=torch.device("cpu"), dtype=torch.float64)

    start_ls = float(gp.length_scale().detach().float().view(-1)[0].item())
    start_os = float(gp.output_scale().detach().float().view(-1)[0].item())
    start_nz = float(gp.noise().detach().float().view(-1)[0].item())
    cur_lr = float(lr)
    opt = torch.optim.Adam(gp_pre.parameters(), lr=cur_lr)
    steps_ok = 0

    for step in range(n_steps):
        last_good = {k: v.detach().clone() for k, v in gp_pre.state_dict().items()}
        opt.zero_grad()
        try:
            loss = gp_pre.marginal_log_likelihood(X_s, y_s)
        except RuntimeError as e:
            print(f"    mll failed at step {step + 1}/{n_steps}: {e}")
            print("    continue without more pretrain steps for this horizon")
            break
        if not torch.isfinite(loss):
            print(f"    non-finite mll at step {step + 1}/{n_steps}; stop pretrain for this horizon")
            break
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(gp_pre.parameters(), max_norm=5.0)
        if not torch.isfinite(grad_norm):
            gp_pre.load_state_dict(last_good)
            cur_lr *= 0.5
            if cur_lr < 1e-6:
                print("    grad norm non-finite and lr too small; stop pretrain for this horizon")
                break
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            print(f"    non-finite grad norm; rollback step and reduce lr to {cur_lr:.2e}")
            continue
        opt.step()
        if not _params_finite(gp_pre):
            gp_pre.load_state_dict(last_good)
            cur_lr *= 0.5
            if cur_lr < 1e-6:
                print(f"    non-finite params after step {step + 1}/{n_steps}; restore and stop pretrain")
                break
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            print(f"    non-finite params after step {step + 1}/{n_steps}; rollback and reduce lr to {cur_lr:.2e}")
            continue
        steps_ok += 1

    # Copy trained parameters back to runtime GP module.
    if _params_finite(gp_pre):
        gp.load_state_dict(gp_pre.state_dict())

    gp.eval()
    end_ls = float(gp.length_scale().detach().float().view(-1)[0].item())
    end_os = float(gp.output_scale().detach().float().view(-1)[0].item())
    end_nz = float(gp.noise().detach().float().view(-1)[0].item())
    changed = (
        abs(end_ls - start_ls) > 1e-6
        or abs(end_os - start_os) > 1e-6
        or abs(end_nz - start_nz) > 1e-6
    )
    return {
        "steps_ok": steps_ok,
        "lr_final": cur_lr,
        "changed": changed,
        "ls_start": start_ls,
        "os_start": start_os,
        "nz_start": start_nz,
        "ls_end": end_ls,
        "os_end": end_os,
        "nz_end": end_nz,
    }


def pretrain_svgp_kernel(
    gp,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    n_steps: int = 200,
    lr: float = 1e-2,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Optimise SVGPLayer kernel hyperparameters via variational ELBO.

    Works with gpytorch backend (SVGPLayer).  Mini-batches of size
    ``batch_size`` are drawn at each step so this scales to large datasets.

    Returns a summary dict with start/end hyperparameter values.
    """
    gp.train()
    if hasattr(gp, "likelihood"):
        gp.likelihood.train()

    X_s = X_train.detach().to(device)
    y_s = y_train.detach().to(device)
    n_data = X_s.size(0)

    # Initialise inducing points from a random subset of the training data.
    if hasattr(gp, "initialize_inducing"):
        gp.initialize_inducing(X_s)

    opt = torch.optim.Adam(gp.parameters(), lr=lr)

    start_ls = float(gp.length_scale().detach().float().view(-1)[0].item()) if hasattr(gp, "length_scale") else float("nan")
    start_os = float(gp.output_scale().detach().float().view(-1)[0].item()) if hasattr(gp, "output_scale") else float("nan")
    start_nz = float(gp.noise().detach().float().view(-1)[0].item()) if hasattr(gp, "noise") else float("nan")

    steps_ok = 0
    for step in range(n_steps):
        idx = torch.randperm(n_data, device=device)[:min(batch_size, n_data)]
        xb = X_s[idx]
        yb = y_s[idx]

        opt.zero_grad()
        try:
            loss = gp.elbo_loss(xb, yb, n_data)
        except RuntimeError as e:
            print(f"    elbo_loss failed at step {step + 1}/{n_steps}: {e}")
            break
        if not torch.isfinite(loss):
            print(f"    non-finite ELBO at step {step + 1}/{n_steps}; stopping pretrain")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gp.parameters(), max_norm=5.0)
        opt.step()
        steps_ok += 1

    gp.eval()
    if hasattr(gp, "likelihood"):
        gp.likelihood.eval()

    end_ls = float(gp.length_scale().detach().float().view(-1)[0].item()) if hasattr(gp, "length_scale") else float("nan")
    end_os = float(gp.output_scale().detach().float().view(-1)[0].item()) if hasattr(gp, "output_scale") else float("nan")
    end_nz = float(gp.noise().detach().float().view(-1)[0].item()) if hasattr(gp, "noise") else float("nan")
    changed = (
        abs(end_ls - start_ls) > 1e-6
        or abs(end_os - start_os) > 1e-6
        or abs(end_nz - start_nz) > 1e-6
    )
    return {
        "steps_ok": steps_ok,
        "changed": changed,
        "ls_start": start_ls, "os_start": start_os, "nz_start": start_nz,
        "ls_end": end_ls,   "os_end": end_os,   "nz_end": end_nz,
    }


def main():
    parser = argparse.ArgumentParser(description="Standalone differentiable GP on GRU predictions.")
    parser.add_argument("--gru-run-sig", default=None, help="GRU run signature.")
    parser.add_argument("--model-prefix", default="GRU_FCOV", help="Temporal model folder/prefix, e.g. GRU_FCOV or TFT.")
    parser.add_argument("--gp-run-tag", default="gp_pytorch", help="Tag appended to output dir name.")
    parser.add_argument("--pred-path", default=None, help="Optional explicit pred.parquet path.")
    parser.add_argument("--pretrain-steps", type=int, default=200, help="Optimisation steps per horizon.")
    parser.add_argument("--pretrain-lr", type=float, default=1e-2)
    parser.add_argument("--max-pretrain-pts", type=int, default=2000, help="Max train pts for MLL / ELBO batch.")
    parser.add_argument("--date-freq", default="ME", help="Pandas resample freq for date selection (ME=monthly).")
    parser.add_argument("--jitter", type=float, default=1e-5)
    parser.add_argument("--kernel-type", default="matern32", help="matern32 or rbf (custom backend only)")
    parser.add_argument("--isotropic", default="true", help="true/false (custom backend only)")
    # New gpytorch backend args
    parser.add_argument("--backend", default="custom", help="GP backend: 'custom' or 'gpytorch'")
    parser.add_argument("--n-inducing", type=int, default=64, help="Inducing point count (gpytorch backend)")
    parser.add_argument("--variational-lr", type=float, default=1e-2, help="SVGP ELBO learning rate (gpytorch)")
    parser.add_argument("--use-float64", default="true", help="Use float64 in GP kernel (gpytorch backend)")
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
    spatial_cfg = cfg_src.get("spatial_split", {})

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
            revin_tag = "r1" if bool(mc.get("use_revin", False)) else "r0"
            spf     = str(float(sc.get("train_fraction", 0.8))).replace(".", "p")
            sc_cnt  = int(sc.get("cluster_count", 20))
            ss      = int(sc.get("split_seed", 42))
            gru_run_sig = f"in{in_len}_out{out_len}_ep{epochs}_bs{bs}_seed{seed}_{dataset}_{revin_tag}_spf{spf}_sc{sc_cnt}_ss{ss}"

    model_prefix = str(args.model_prefix).strip()
    kernel_type  = str(args.kernel_type).strip().lower()
    isotropic    = _as_bool(args.isotropic)
    backend      = str(args.backend).strip().lower()
    n_inducing   = int(args.n_inducing)
    variational_lr = float(args.variational_lr)
    use_float64  = _as_bool(args.use_float64)
    pred_path = (
        Path(args.pred_path)
        if args.pred_path
        else ROOT / "outputs" / model_prefix / f"{model_prefix}_{gru_run_sig}" / "predictions" / "pred.parquet"
    )
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
    pred = pred.merge(coords, on="id", how="left")

    all_dates = pred["datum"].drop_duplicates().sort_values()
    dates = pd.Series(all_dates.values, index=all_dates).resample(args.date_freq).first().dropna().tolist()
    horizons = sorted(pred["horizon"].unique().tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model prefix: {model_prefix}")
    print(f"Run sig: {gru_run_sig}")
    print(f"Kernel: {kernel_type} | isotropic={isotropic}")
    print(f"Dates ({len(dates)}): {[str(d.date()) for d in dates[:4]]} ...")
    print(f"Horizons: {horizons}")

    coords_arr = coords.set_index("id")
    train_coords_df = coords_arr.loc[coords_arr.index.isin(train_ids)]
    coord_scaler = StandardScaler().fit(train_coords_df[["x_25833", "y_25833"]].to_numpy())

    out_rows = []
    valid_pairs = 0
    skipped = []

    for h in horizons:
        pred_h = pred[pred["horizon"] == h]
        train_pred_h = pred_h[pred_h["id"].isin(train_ids)].dropna(subset=["x_25833", "y_25833", "gws_pred"])

        # Build the GP model for this horizon (fresh each horizon)
        gp = make_gp_layer(
            backend=backend,
            n_spatial_dims=2,
            kernel_type=kernel_type,
            isotropic=isotropic,
            jitter=args.jitter,
            n_inducing=n_inducing,
            use_float64=use_float64,
        ).to(device)

        if train_pred_h.empty:
            print(f"  horizon {h}: no train predictions — skipping kernel pretrain")
        else:
            X_all = torch.tensor(
                coord_scaler.transform(train_pred_h[["x_25833", "y_25833"]].to_numpy()),
                dtype=torch.float32, device=device
            )
            y_all = torch.tensor(train_pred_h["gws_pred"].to_numpy(), dtype=torch.float32, device=device)
            print(f"  horizon {h}: pre-training GP kernel ({args.pretrain_steps} steps, backend={backend}) on {X_all.size(0)} pts ...")

            if backend == "gpytorch":
                pretrain_stats = pretrain_svgp_kernel(
                    gp, X_all, y_all,
                    n_steps=args.pretrain_steps,
                    lr=variational_lr,
                    batch_size=args.max_pretrain_pts,
                    device=device,
                )
                print(
                    f"    pretrain_effect:"
                    f" steps_ok={pretrain_stats['steps_ok']}"
                    f" changed={pretrain_stats['changed']}"
                )
            else:
                pretrain_stats = pretrain_gp_kernel(
                    gp, X_all, y_all, n_steps=args.pretrain_steps, lr=args.pretrain_lr,
                    max_train_pts=args.max_pretrain_pts, device=device
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
                    backend=backend,
                    n_spatial_dims=2,
                    kernel_type=kernel_type,
                    isotropic=isotropic,
                    jitter=args.jitter,
                    n_inducing=n_inducing,
                    use_float64=use_float64,
                ).to(device)
                ls  = float(gp.length_scale().detach().float().view(-1)[0].item())
                os_ = float(gp.output_scale().detach().float().view(-1)[0].item())
                nz  = float(gp.noise().detach().float().view(-1)[0].item())
            print(f"    ls={ls:.4f}  os={os_:.4f}  noise={nz:.4f}")

        for dt in dates:
            gws_dt = gws[gws["datum"] == dt][["id", "gws", "x_25833", "y_25833"]]
            holdout_true = gws_dt[gws_dt["id"].isin(holdout_ids)].dropna()
            if holdout_true.empty:
                skipped.append((dt, h, "no holdout obs"))
                continue

            pred_sel = pred[(pred["datum"] == dt) & (pred["horizon"] == h)]
            pred_sel = pred_sel[pred_sel["id"].isin(train_ids)].dropna(subset=["x_25833", "y_25833", "gws_pred"])
            if pred_sel.empty:
                skipped.append((dt, h, "no train predictions"))
                continue

            X_train_np = coord_scaler.transform(pred_sel[["x_25833", "y_25833"]].to_numpy())
            y_train_np = pred_sel["gws_pred"].to_numpy()
            X_test_np  = coord_scaler.transform(holdout_true[["x_25833", "y_25833"]].to_numpy())

            X_train_t = torch.tensor(X_train_np, dtype=torch.float32, device=device)
            y_train_t = torch.tensor(y_train_np, dtype=torch.float32, device=device)
            X_test_t  = torch.tensor(X_test_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                y_pred_t, y_std_t = gp.forward_with_std(X_train_t, y_train_t, X_test_t)

            y_pred_np = y_pred_t.cpu().numpy()
            y_std_np  = y_std_t.cpu().numpy()

            out = holdout_true.copy().reset_index(drop=True)
            out["gws_true"]         = out["gws"]
            out.drop(columns=["gws"], inplace=True)
            out["gws_forecast"]     = y_pred_np
            out["gws_forecast_std"] = y_std_np
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
        "RMSE":         _rmse(pred_all, real_all),
        "nRMSE":        _nrmse(pred_all, real_all),
        "NSE_pooled":   _nse(pred_all, real_all),
        "NSE":          _nse(pred_all, real_all),
        "NSE_id_median":float(np.median(per_id_valid)) if per_id_valid.size else float("nan"),
        "NSE_id_p25":   float(np.percentile(per_id_valid, 25)) if per_id_valid.size else float("nan"),
        "NSE_id_p50":   float(np.percentile(per_id_valid, 50)) if per_id_valid.size else float("nan"),
        "NSE_id_p75":   float(np.percentile(per_id_valid, 75)) if per_id_valid.size else float("nan"),
        "NSE_id_mean":  float(np.mean(per_id_valid)) if per_id_valid.size else float("nan"),
        "NSE_id_count": float(per_id_valid.size),
        "MAE":          float(np.mean(abs_err)),
        "AbsErr_P95":   float(np.percentile(abs_err, 95)) if abs_err.size else float("nan"),
        "AbsErr_P99":   float(np.percentile(abs_err, 99)) if abs_err.size else float("nan"),
        "AbsErr_Max":   float(np.max(abs_err)) if abs_err.size else float("nan"),
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
    metrics_ser = pd.Series(overall_metrics)
    metrics_df  = metrics_ser.reset_index()
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
