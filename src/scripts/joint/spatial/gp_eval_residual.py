from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in [str(SRC_ROOT), str(SCRIPT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from libs.spatial_split import resolve_split_path
from libs.run_registry import lookup_run_id
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
    return float("nan") if denom == 0 else float(1 - np.sum((pred - real) ** 2) / denom)


def _rmse(pred, real):
    return float(np.sqrt(np.mean((pred - real) ** 2)))


def _nrmse(pred, real):
    iqr = float(np.diff(np.quantile(real, [0.25, 0.75]))[0])
    return float("nan") if iqr == 0 else _rmse(pred, real) / iqr


def pretrain_mll_kernel(gp, X_train, y_train, n_steps=200, lr=1e-2, max_train_pts=2000, device=torch.device("cpu")):
    gp.train()
    if X_train.size(0) > max_train_pts:
        idx = torch.randperm(X_train.size(0))[:max_train_pts]
        X_s, y_s = X_train[idx], y_train[idx]
    else:
        X_s, y_s = X_train, y_train

    X_s = X_s.detach().to(device)
    y_s = y_s.detach().to(device)

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
    return steps_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gru/gru.yaml")
    parser.add_argument("--gru-run-sig", default=None)
    parser.add_argument("--model-prefix", default="GRU_FCOV")
    parser.add_argument("--gp-run-tag", default="gp_residual")
    parser.add_argument("--pred-path", default=None)
    parser.add_argument("--pretrain-steps", type=int, default=200)
    parser.add_argument("--pretrain-lr", type=float, default=1e-2)
    parser.add_argument("--max-pretrain-pts", type=int, default=2000)
    parser.add_argument("--date-freq", default="ME")
    parser.add_argument("--jitter", type=float, default=1e-4)
    parser.add_argument("--backend", default="gpytorch")
    parser.add_argument("--n-inducing", type=int, default=256)
    parser.add_argument("--use-float64", default="true")
    args = parser.parse_args()

    gru_cfg = _load_yaml(ROOT / args.config)
    data_cfg = _load_yaml(ROOT / "configs" / "data.yaml")

    for key in ["full_raw_path", "full_merged_path", "sample_path", "metadata_path"]:
        if key in data_cfg and data_cfg[key]:
            p = Path(data_cfg[key])
            data_cfg[key] = str(p if p.is_absolute() else (ROOT / p).resolve())

    dataset = str(gru_cfg.get("dataset", "full_merged"))
    spatial_cfg = gru_cfg.get("spatial_split", {})
    use_float64 = _as_bool(args.use_float64)

    gru_run_sig = args.gru_run_sig or str(gru_cfg.get("run_sig", "")).strip()
    model_prefix = args.model_prefix

    if args.pred_path:
        pred_path = Path(args.pred_path)
    else:
        try:
            run_id = lookup_run_id(ROOT / "outputs", model_prefix, gru_run_sig)
            pred_path = ROOT / "outputs" / model_prefix / f"{model_prefix}_{run_id}" / "predictions" / "pred.parquet"
        except (FileNotFoundError, ValueError):
            pred_path = ROOT / "outputs" / model_prefix / f"{model_prefix}_{gru_run_sig}" / "predictions" / "pred.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"pred.parquet not found: {pred_path}")

    split_path = resolve_split_path(ROOT / "splits", dataset, spatial_cfg)
    split = pd.read_csv(split_path)
    train_ids   = set(split.loc[split["spatial_split"] == "spatial_train", "id"])
    holdout_ids = set(split.loc[split["spatial_split"].isin(["spatial_holdout", "spatial_test"]), "id"])

    if dataset == "full_merged":
        data_path = Path(data_cfg["full_merged_path"])
    elif dataset == "full_raw":
        data_path = Path(data_cfg["full_raw_path"])
    else:
        data_path = Path(data_cfg["sample_path"])

    gp_features = list(gru_cfg.get("gp_features", []))
    gp_features_onehot = list(gru_cfg.get("gp_features_onehot", []))

    # Load true GWL from original dataset
    cols = ["datum", "id", "gws", "x_25833", "y_25833"]
    gws = (
        pd.read_csv(data_path, usecols=cols, low_memory=False)
        if str(data_path).lower().endswith(".csv")
        else pq.read_table(data_path, columns=cols).to_pandas()
    )
    gws["datum"] = pd.to_datetime(gws["datum"])

    # Load GRU predictions (has gws_forecast = GRU pred, gws = true value)
    pred = pq.read_table(pred_path).to_pandas()
    pred["datum"] = pd.to_datetime(pred["datum"])
    # gws_forecast = GRU prediction, gws = actual true value
    if "gws_forecast" not in pred.columns:
        raise ValueError("pred.parquet must have 'gws_forecast' column (GRU predictions)")
    if "gws" not in pred.columns:
        raise ValueError("pred.parquet must have 'gws' column (true values)")

    # Build coordinate + feature lookup
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
    print(f"Residual GP | run_sig: {gru_run_sig} | tag: {args.gp_run_tag}")
    print(f"Dates ({len(dates)}): {[str(d.date()) for d in dates[:4]]} ...")
    print(f"Horizons: {horizons}")

    train_coords_df = coords.set_index("id")
    train_coords_df = train_coords_df.loc[train_coords_df.index.isin(train_ids)]
    feature_scaler = StandardScaler().fit(train_coords_df[feature_cols].to_numpy())
    n_dims = len(feature_cols)

    out_rows = []
    valid_pairs = 0
    skipped = []

    for h in horizons:
        pred_h = pred[pred["horizon"] == h]

        # Residuals at train wells: gws (true) - gws_forecast (GRU pred)
        train_pred_h = pred_h[pred_h["id"].isin(train_ids)].dropna(
            subset=["x_25833", "y_25833", "gws_forecast", "gws"]
        ).copy()
        train_pred_h["residual"] = train_pred_h["gws"] - train_pred_h["gws_forecast"]

        gp = make_gp_layer(
            backend=args.backend, n_spatial_dims=n_dims,
            jitter=args.jitter, n_inducing=args.n_inducing, use_float64=use_float64,
        ).to(device)

        if not train_pred_h.empty:
            X_all = torch.tensor(
                feature_scaler.transform(train_pred_h[feature_cols].to_numpy()),
                dtype=torch.float32, device=device,
            )
            y_all = torch.tensor(train_pred_h["residual"].to_numpy(), dtype=torch.float32, device=device)
            print(f"  horizon {h}: pre-training GP on residuals ({args.pretrain_steps} steps, {X_all.size(0)} pts)")
            pretrain_mll_kernel(
                gp, X_all, y_all, n_steps=args.pretrain_steps,
                lr=args.pretrain_lr, max_train_pts=args.max_pretrain_pts, device=device,
            )
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

            pred_dt_h = pred[(pred["datum"] == dt) & (pred["horizon"] == h)]

            # Train wells: need both GRU forecast and true value for residual
            train_sel = pred_dt_h[pred_dt_h["id"].isin(train_ids)].dropna(
                subset=["x_25833", "y_25833", "gws_forecast", "gws"]
            )
            if train_sel.empty:
                skipped.append((dt, h, "no train predictions"))
                continue

            # Test wells: need GRU forecast to add correction
            test_sel = pred_dt_h[pred_dt_h["id"].isin(holdout_ids)].dropna(
                subset=["x_25833", "y_25833", "gws_forecast"]
            )
            # Align with holdout_true (inner join on id)
            test_aligned = holdout_true.merge(test_sel[["id", "gws_forecast"]], on="id", how="inner")
            if test_aligned.empty:
                skipped.append((dt, h, "no test gru predictions"))
                continue

            residuals = (train_sel["gws"] - train_sel["gws_forecast"]).to_numpy()
            X_train_np = feature_scaler.transform(train_sel[feature_cols].to_numpy())
            X_test_np  = feature_scaler.transform(test_aligned[feature_cols].to_numpy())

            X_train_t = torch.tensor(X_train_np, dtype=torch.float32, device=device)
            y_train_t = torch.tensor(residuals, dtype=torch.float32, device=device)
            X_test_t  = torch.tensor(X_test_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                y_residual_t, y_std_t = gp.forward_with_std(X_train_t, y_train_t, X_test_t)

            out = test_aligned.copy().reset_index(drop=True)
            out["gws_true"]         = out["gws"]
            out.drop(columns=["gws"], inplace=True)
            # Final prediction = GRU forecast + GP residual correction
            out["gws_forecast"]     = out["gws_forecast"] + y_residual_t.cpu().numpy()
            out["gws_forecast_std"] = y_std_t.cpu().numpy()
            out["horizon"]          = h
            out["datum"]            = dt
            out_rows.append(out)
            valid_pairs += 1

    if not out_rows:
        raise ValueError("No GP outputs generated.")

    out_df = pd.concat(out_rows, ignore_index=True)

    pred_all = out_df["gws_forecast"].to_numpy()
    real_all = out_df["gws_true"].to_numpy()

    per_id_nrmse = out_df.groupby("id").apply(
        lambda g: _nrmse(g["gws_forecast"].to_numpy(), g["gws_true"].to_numpy()),
        include_groups=False,
    )
    nrmse_pw = float(per_id_nrmse.median())

    print(f"\nValid pairs: {valid_pairs}  |  Skipped: {len(skipped)}")
    print(f"Overall metrics:")
    print(f"  RMSE: {_rmse(pred_all, real_all):.4f}")
    print(f"  nRMSE_pw: {nrmse_pw:.4f}")
    print(f"  NSE: {_nse(pred_all, real_all):.4f}")

    # Save outputs
    out_dir = ROOT / "outputs" / "gp" / f"{model_prefix}_{gru_run_sig}__{args.gp_run_tag}__residual"
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(out_df), out_dir / "gp_pred.parquet")

    metrics = {
        "RMSE": _rmse(pred_all, real_all),
        "nRMSE_pw": nrmse_pw,
        "NSE": _nse(pred_all, real_all),
        "valid_pairs": valid_pairs,
        "n_skipped": len(skipped),
        "n_test_wells": int(out_df["id"].nunique()),
    }
    import json
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
