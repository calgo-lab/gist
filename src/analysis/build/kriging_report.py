"""
kriging_report.py

Builds reports/kriging/metrics/kriging_metrics_summary.csv from outputs/gp/.

Scans all kriging eval outputs (non-GRU_FCOV directories in outputs/gp/),
pivots key/value parquet metrics to wide format.

Run after kriging eval completes:
    python src/analysis/build/kriging_report.py

Outputs
-------
reports/kriging/metrics/kriging_metrics_summary.csv
    One row per kriging run. Columns: run_number, RMSE_pooled, nRMSE,
    NSE_pooled, NSE_id_median, MAE_pooled, run_tag.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
GP_ROOT = ROOT / "outputs" / "gp"
METRICS_DIR = ROOT / "reports" / "kriging" / "metrics"
OUT_CSV = METRICS_DIR / "kriging_metrics_summary.csv"

KEEP_METRICS = ["RMSE", "nRMSE", "NSE_pooled", "NSE_id_median", "MAE"]
METRIC_RENAME = {"RMSE": "RMSE_pooled", "MAE": "MAE_pooled"}


def _load_metrics(run_dir):
    p = run_dir / "gp_metrics.parquet"
    if not p.exists():
        return None
    df = pq.read_table(p).to_pandas().set_index("metric")["value"]
    row = {}
    for m in KEEP_METRICS:
        if m in df.index:
            key = METRIC_RENAME.get(m, m)
            row[key] = df[m]
    return row


def build_metrics_table():
    rows = []
    for run_dir in sorted(GP_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("GRU_FCOV"):
            continue
        metrics = _load_metrics(run_dir)
        if metrics is None:
            continue
        metrics["run_tag"] = run_dir.name
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).reset_index(drop=True)
    df.insert(0, "run_number", range(1, len(df) + 1))

    metric_cols = [c for c in ["RMSE_pooled", "nRMSE", "NSE_pooled", "NSE_id_median", "MAE_pooled"] if c in df.columns]
    trailing = ["run_tag"]
    other = [c for c in df.columns if c not in ["run_number"] + metric_cols + trailing]
    df = df[["run_number"] + metric_cols + other + trailing]
    return df


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = build_metrics_table()
    if df.empty:
        print("No kriging runs found in outputs/gp/")
        return
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows → {OUT_CSV}")
    print(df[["run_number", "RMSE_pooled", "NSE_pooled", "run_tag"]].to_string(index=False))


if __name__ == "__main__":
    main()
