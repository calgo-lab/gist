"""
gp_report.py

Builds reports/gp/metrics/gp_metrics_summary.csv from outputs/gp/.

Scans all GP eval outputs, pivots key/value parquet metrics to wide format,
and joins HPO metadata from reports/gp/hpo/*_results.csv.

Run after GP eval completes:
    python src/analysis/build/gp_report.py

Outputs
-------
reports/gp/metrics/gp_metrics_summary.csv
    One row per GP run. Columns: run_number, hpo_name, trial, objective,
    elapsed_s, model_prefix, gru_run_sig, gp_run_tag, split,
    RMSE, nRMSE, NSE_pooled, NSE_id_median, MAE,
    kernel, pretrain_steps, jitter,
    max_pretrain_pts, variational_lr, isotropic, use_float64, dir_name.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
GP_ROOT = ROOT / "outputs" / "gp"
HPO_DIR = ROOT / "reports" / "gp" / "hpo"
METRICS_DIR = ROOT / "reports" / "gp" / "metrics"
OUT_CSV = METRICS_DIR / "gp_metrics_summary.csv"

KEEP_METRICS = ["RMSE", "nRMSE", "NSE_pooled", "NSE_id_median", "MAE"]
METRIC_RENAME = {"RMSE": "RMSE_pooled", "MAE": "MAE_pooled"}

HP_RENAME: dict[str, str] = {
    "hp.n_inducing": "n_inducing",
    "hp.pretrain_steps": "pretrain_steps",
    "hp.jitter": "jitter",
    "hp.max_pretrain_pts": "max_pretrain_pts",
    "hp.variational_lr": "variational_lr",
}


def _parse_dir_name(name):
    parts = name.split("__")
    base = parts[0]
    gp_run_tag = parts[1] if len(parts) >= 2 else ""
    split = parts[2] if len(parts) >= 3 else ""

    if base.startswith("GRU_FCOV_"):
        model_prefix = "GRU_FCOV"
        gru_run_sig = base[len("GRU_FCOV_"):]
    elif base.startswith("TFT_"):
        model_prefix = "TFT"
        gru_run_sig = base[len("TFT_"):]
    else:
        model_prefix = ""
        gru_run_sig = base

    return {
        "model_prefix": model_prefix,
        "gru_run_sig": gru_run_sig,
        "gp_run_tag": gp_run_tag,
        "split": split,
    }


def load_gp_outputs():
    rows = []
    for run_dir in sorted(GP_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        mp = run_dir / "gp_metrics.parquet"
        if not mp.exists():
            continue
        df = pd.read_parquet(mp)
        if "metric" not in df.columns or "value" not in df.columns:
            continue

        metrics = dict(zip(df["metric"], df["value"].astype(float)))
        row = _parse_dir_name(run_dir.name)
        row["dir_name"] = run_dir.name
        for m in KEEP_METRICS:
            row[m] = metrics.get(m, np.nan)
            if m == "NSE_pooled" and np.isnan(row[m]):
                row[m] = metrics.get("NSE", np.nan)
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_hpo_meta():
    parts = []
    for p in sorted(HPO_DIR.glob("*_results.csv")):
        df = pd.read_csv(p)
        if "gp_run_tag" not in df.columns:
            continue
        df = df.copy()
        df["hpo_name"] = p.stem.replace("_results", "")
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    return combined.drop_duplicates(subset=["gp_run_tag"], keep="first")


def build_metrics_table(runs_df, hpo_df):
    out = runs_df.copy()

    if not hpo_df.empty:
        hpo_extra = ["hpo_name", "trial", "objective", "elapsed_s"]
        hp_cols = [c for c in hpo_df.columns if c.startswith("hp.")]
        hpo_keep = ["gp_run_tag"] + [c for c in hpo_extra + hp_cols if c in hpo_df.columns]
        out = out.merge(hpo_df[hpo_keep].drop_duplicates("gp_run_tag"), on="gp_run_tag", how="left")

    out = out.rename(columns={k: v for k, v in HP_RENAME.items() if k in out.columns})
    for drop_col in ("hp.isotropic", "hp.use_float64", "hp.kernel_type", "isotropic", "use_float64", "kernel"):
        if drop_col in out.columns:
            out = out.drop(columns=drop_col)

    out = out.rename(columns={k: v for k, v in METRIC_RENAME.items() if k in out.columns})

    if "objective" not in out.columns:
        out["objective"] = np.nan
    if "RMSE_pooled" in out.columns:
        out["objective"] = out["objective"].combine_first(out["RMSE_pooled"])

    out = out.sort_values("NSE_pooled", ascending=False).reset_index(drop=True)
    out.insert(0, "run_number", range(1, len(out) + 1))

    leading = ["run_number", "objective", "elapsed_s", "model_prefix"]
    metric_cols = ["RMSE_pooled", "nRMSE", "NSE_pooled", "NSE_id_median", "MAE_pooled"]
    hp_keys = ["pretrain_steps", "jitter", "max_pretrain_pts", "variational_lr"]
    extra_hp = sorted(c for c in out.columns if c.startswith("hp."))
    trailing = ["hpo_name", "trial", "gru_run_sig", "gp_run_tag", "split", "dir_name"]
    ordered = (
        [c for c in leading if c in out.columns]
        + [c for c in metric_cols if c in out.columns]
        + [c for c in hp_keys if c in out.columns]
        + extra_hp
        + [c for c in trailing if c in out.columns]
    )
    remaining = [c for c in out.columns if c not in ordered]
    return out[ordered + remaining]


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    runs_df = load_gp_outputs()
    if runs_df.empty:
        print("No GP eval outputs found in", GP_ROOT)
        return

    hpo_df = load_hpo_meta()
    table = build_metrics_table(runs_df, hpo_df)
    table.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}  ({len(table)} runs)")


if __name__ == "__main__":
    main()
