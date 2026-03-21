"""
Post-hoc GOK clipping: clip gws_forecast to gok for unconfined (ungespannt) wells only.
Recomputes metrics after clipping and saves clipped predictions + metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import math
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to gp_pred.parquet")
    parser.add_argument("--meta", default=None, help="Path to meta_LFU_info.csv")
    parser.add_argument("--clip-all", action="store_true", help="Clip all wells regardless of aquifer type")
    args = parser.parse_args()

    data_cfg_path = ROOT / "configs" / "data.yaml"
    if data_cfg_path.exists():
        import yaml
        data_cfg = yaml.safe_load(data_cfg_path.read_text()) or {}
    else:
        data_cfg = {}

    meta_path = Path(args.meta) if args.meta else Path(data_cfg.get("metadata_path", ""))
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata not found: {meta_path}")

    meta = pd.read_csv(meta_path, sep=";")[["id", "gok", "gw_gespannt"]]
    meta["ungespannt"] = meta["gw_gespannt"] == "ungespannt"

    pred_path = Path(args.pred)
    df = pq.read_table(pred_path).to_pandas()
    df = df.merge(meta, on="id", how="left")

    before = df["gws_forecast"].copy()
    mask = df["gok"].notna() if args.clip_all else (df["ungespannt"] & df["gok"].notna())
    df.loc[mask, "gws_forecast"] = df.loc[mask, ["gws_forecast", "gok"]].min(axis=1)

    n_clipped = int((df["gws_forecast"] < before).sum())
    print(f"Clipped {n_clipped} predictions ({100*n_clipped/len(df):.2f}%) for unconfined wells")

    pred_all = df["gws_forecast"].to_numpy()
    real_all = df["gws_true"].to_numpy()
    abs_err = np.abs(pred_all - real_all)

    per_id_rows = []
    for well_id, g in df.groupby("id"):
        iqr = float(np.diff(np.quantile(g["gws_true"], [0.25, 0.75]))[0])
        nrmse_well = _rmse(g["gws_forecast"].to_numpy(), g["gws_true"].to_numpy()) / iqr if iqr > 0 else float("nan")
        per_id_rows.append({
            "id": well_id,
            "NSE_over_time": _nse(g["gws_forecast"].to_numpy(), g["gws_true"].to_numpy()),
            "nRMSE": nrmse_well,
        })
    per_id_df = pd.DataFrame(per_id_rows)
    per_id_valid = per_id_df["NSE_over_time"].to_numpy(dtype=float)
    per_id_valid = per_id_valid[np.isfinite(per_id_valid)]
    nrmse_per_well = per_id_df["nRMSE"].to_numpy(dtype=float)
    nrmse_per_well = nrmse_per_well[np.isfinite(nrmse_per_well)]

    metrics = {
        "RMSE":          _rmse(pred_all, real_all),
        "nRMSE":         _nrmse(pred_all, real_all),
        "NSE_pooled":    _nse(pred_all, real_all),
        "NSE_id_median": float(np.median(per_id_valid)) if per_id_valid.size else float("nan"),
        "NSE_id_mean":   float(np.mean(per_id_valid)) if per_id_valid.size else float("nan"),
        "MAE":           float(np.mean(abs_err)),
        "nRMSE_id_median": float(np.median(nrmse_per_well)) if nrmse_per_well.size else float("nan"),
        "nRMSE_id_mean":   float(np.mean(nrmse_per_well)) if nrmse_per_well.size else float("nan"),
        "n_clipped":       float(n_clipped),
    }

    out_dir = pred_path.parent / "gok_clipped"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = df.drop(columns=["gok", "gw_gespannt", "ungespannt"])
    pq.write_table(pa.Table.from_pandas(df_out), out_dir / "gp_pred_clipped.parquet")

    metrics_df = pd.Series(metrics).reset_index()
    metrics_df.columns = ["metric", "value"]
    pq.write_table(pa.Table.from_pandas(metrics_df), out_dir / "gp_metrics_clipped.parquet")

    print(f"Saved clipped predictions to {out_dir}")
    print("Metrics after GOK clipping:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) and not math.isnan(v) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
