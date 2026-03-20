"""
cross_model_table.py

Builds a cross-model comparison table from the best available run of each
model type. Auto-selects best runs from outputs/ — no hardcoded run signatures.

Models compared (all selected by lowest nRMSE or weighted RMSE h1–16):
  - TFT          — lowest weighted RMSE h1–16
  - GP on TFT    — lowest nRMSE
  - GRU          — lowest weighted RMSE h1–16
  - GP on GRU    — lowest nRMSE
  - Joint GRU+GP — lowest nRMSE

Metrics reported:
  RMSE, MAE, NSE_pooled, NSE_id_median  (from each model's eval output)

Output:
  reports/cross_model/cross_model_table.csv
  (printed to stdout as well)

Usage:
    python src/analysis/build/cross_model_table.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
GRU_SUMMARY = ROOT / "reports" / "gru" / "metrics" / "gru_metrics_summary.csv"
TFT_SUMMARY = ROOT / "reports" / "tft" / "metrics" / "tft_metrics_summary.csv"
GP_ROOT = ROOT / "outputs" / "gp"
JOINT_ROOT = ROOT / "outputs" / "GRU_GP_JOINT"
OUT_DIR = ROOT / "reports" / "cross_model"
OUT_CSV = OUT_DIR / "cross_model_table.csv"

WEIGHTS_H1_16 = np.linspace(1.0, 2.0, 16)
METRIC_COLS = ["RMSE", "MAE", "NSE_pooled", "NSE_id_median"]


def _load_kv_metrics(path):
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "metric" in df.columns and "value" in df.columns:
        return dict(zip(df["metric"], df["value"].astype(float)))
    return df.iloc[0].to_dict()


def _extract(metrics, keys):
    if metrics is None:
        return {k: float("nan") for k in keys}
    return {k: float(metrics.get(k, float("nan"))) for k in keys}


def _weighted_mean(series, weights):
    vals = series.to_numpy(dtype=float)
    if len(vals) != len(weights):
        return float(np.nanmean(vals))
    return float(np.average(vals, weights=weights))


def best_gru():
    if not GRU_SUMMARY.exists():
        return ("—", {k: float("nan") for k in METRIC_COLS})

    df = pd.read_csv(GRU_SUMMARY)
    col = "RMSE_weighted_mean"
    if col not in df.columns or df.empty:
        return ("—", {k: float("nan") for k in METRIC_COLS})

    row = df.loc[df[col].idxmin()]
    run_sig = str(row["run_sig"])

    pred_path = ROOT / "outputs" / "GRU_FCOV" / f"GRU_FCOV_{run_sig}" / "predictions" / "pred.parquet"
    metrics = _gru_metrics_from_pred(pred_path)
    if metrics is None:
        metrics = {
            "RMSE": float(row.get("RMSE_weighted_mean", float("nan"))),
            "MAE": float(row.get("MAE_weighted_mean", float("nan"))),
            "NSE_pooled": float(row.get("NSE_weighted_mean", float("nan"))),
            "NSE_id_median": float("nan"),
        }
    return (run_sig, metrics)


def _gru_metrics_from_pred(pred_path):
    if not pred_path.exists():
        return None
    pred = pd.read_parquet(pred_path)
    y, t = pred["gws_forecast"].to_numpy(), pred["gws"].to_numpy()
    err = y - t
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    denom = float(np.sum((t - t.mean()) ** 2))
    nse_pooled = float(1 - np.sum(err ** 2) / denom) if denom else float("nan")
    per_id = []
    for _, g in pred.groupby("id"):
        r, p = g["gws"].to_numpy(), g["gws_forecast"].to_numpy()
        d = float(np.sum((r - r.mean()) ** 2))
        if d > 0:
            per_id.append(float(1 - np.sum((p - r) ** 2) / d))
    nse_id_median = float(np.median(per_id)) if per_id else float("nan")
    return {"RMSE": rmse, "MAE": mae, "NSE_pooled": nse_pooled, "NSE_id_median": nse_id_median}


def best_gp_on(base_tag, label):
    if not GP_ROOT.exists():
        return ("—", {k: float("nan") for k in METRIC_COLS})

    candidates = [d for d in GP_ROOT.iterdir() if d.is_dir() and base_tag in d.name]
    if not candidates:
        candidates = [d for d in GP_ROOT.iterdir()
                      if d.is_dir() and base_tag in d.name.replace("GRU_FCOV_", "")]
    if not candidates:
        return ("—", {k: float("nan") for k in METRIC_COLS})

    best_dir, best_nrmse, best_metrics = None, float("inf"), None
    for d in candidates:
        m = _load_kv_metrics(d / "gp_metrics.parquet")
        if m is None:
            continue
        nrmse = float(m.get("nRMSE", float("inf")))
        if nrmse < best_nrmse:
            best_nrmse, best_dir, best_metrics = nrmse, d, m

    if best_dir is None:
        return ("—", {k: float("nan") for k in METRIC_COLS})

    return (best_dir.name, _extract(best_metrics, METRIC_COLS))


def best_tft():
    if not TFT_SUMMARY.exists():
        return ("—", {k: float("nan") for k in METRIC_COLS})

    tft = pd.read_csv(TFT_SUMMARY)
    if tft.empty or "RMSE" not in tft.columns:
        return ("—", {k: float("nan") for k in METRIC_COLS})

    ranked = (
        tft[tft["horizon"].between(1, 16)]
        .groupby("run")
        .apply(lambda g: _weighted_mean(g.sort_values("horizon")["RMSE"], WEIGHTS_H1_16), include_groups=False)
    )
    best_run = ranked.idxmin()
    sub = tft[tft["run"] == best_run]
    h_all = sub[sub["horizon"].between(1, 16)].sort_values("horizon")

    rmse = float(np.average(h_all["RMSE"].to_numpy(dtype=float), weights=WEIGHTS_H1_16)) if len(h_all) == 16 else float(h_all["RMSE"].mean())
    mae = float(np.average(h_all["MAE"].to_numpy(dtype=float), weights=WEIGHTS_H1_16)) if len(h_all) == 16 else float(h_all["MAE"].mean())
    nse = _weighted_mean(h_all["NSE"], WEIGHTS_H1_16)

    return (best_run, {"RMSE": rmse, "MAE": mae, "NSE_pooled": nse, "NSE_id_median": float("nan")})


def best_joint():
    if not JOINT_ROOT.exists():
        return ("—", {k: float("nan") for k in METRIC_COLS})

    best_dir, best_nrmse, best_metrics = None, float("inf"), None
    for d in JOINT_ROOT.iterdir():
        if not d.is_dir():
            continue
        mp = d / "eval" / "test" / "gp_metrics.parquet"
        m = _load_kv_metrics(mp)
        if m is None:
            continue
        nrmse = float(m.get("nRMSE", float("inf")))
        if nrmse < best_nrmse:
            best_nrmse, best_dir, best_metrics = nrmse, d, m

    if best_dir is None:
        return ("—", {k: float("nan") for k in METRIC_COLS})

    return (best_dir.name, _extract(best_metrics, METRIC_COLS))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gru_sig, gru_m = best_gru()
    tft_run, tft_m = best_tft()
    gp_tft_tag, gp_tft_m = best_gp_on("TFT", "TFT")
    gp_gru_tag, gp_gru_m = best_gp_on(gru_sig, "GRU")
    joint_tag, joint_m = best_joint()

    rows = [
        {"model": "TFT (best run)", "run": tft_run, **tft_m},
        {"model": "GP on TFT", "run": gp_tft_tag, **gp_tft_m},
        {"model": "GRU (best run)", "run": gru_sig, **gru_m},
        {"model": "GP on GRU", "run": gp_gru_tag, **gp_gru_m},
        {"model": "Joint GRU+GP (best run)", "run": joint_tag, **joint_m},
    ]

    table = pd.DataFrame(rows).set_index("model")
    run_col = table.pop("run")
    table.insert(0, "run", run_col)

    print("\n=== Cross-model comparison (best available runs) ===")
    print(table[METRIC_COLS].to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    table.to_csv(OUT_CSV)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
