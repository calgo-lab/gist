import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

KUNZ_FILE = ROOT / "reports/metrics/kunz_metrics_aggregated.parquet"
FULL_TABLE_CSV  = ROOT / "reports/metrics/tft_metrics_summary.csv"
FULL_TABLE_HTML  = ROOT / "reports/metrics/tft_metrics_summary.html"
SPARSE_TABLE_CSV  = ROOT / "reports/metrics/tft_metrics_summary_sparse.csv"
SPARSE_TABLE_HTML = ROOT / "reports/metrics/tft_metrics_summary_sparse.html"


MODEL = "TFT"
IN_LEN = 52
OUT_LEN = 16
STATICS = True
DATASET = "full_raw"
BATCH_SIZE = 4096
SEEDS = [40, 94, 673, 1899, 2100, 2149, 2230, 6013, 9595, 9898]

MY_RUNS = [
    ("robert_ep50", 50),
    ("robert_ep1", 1)
]

METRICS_KEEP = ["NSE", "RMSE", "MAE", "rMBE"]
HORIZONS = list(range(1, 17))
# -------------------------------

def run_sig(in_len, out_len, epochs, statics, seed, dataset, bs):
    return f"in{in_len}_out{out_len}_ep{epochs}_bs{bs}_stat{int(statics)}_seed{seed}_{dataset}"

def load_kunz():
    df = pd.read_parquet(KUNZ_FILE)
    df = df[df["architecture"] == "TFT_dyn_stat"]
    df = df[df["metric"].isin(METRICS_KEEP)]
    df = df.groupby(["metric", "horizon"])["value"].median().reset_index()
    df = df[df["horizon"].isin(HORIZONS)]
    kunz_wide = df.pivot(index="horizon", columns="metric", values="value").reset_index()
    kunz_wide["run"] = "kunz_original"
    kunz_wide["skill_vs_persistence"] = np.nan
    cols = ["run", "horizon"] + METRICS_KEEP + ["skill_vs_persistence"]
    return kunz_wide[cols]

def load_my_run(run_name, epochs):
    rows_metrics = []
    rows_skill = []

    for seed in SEEDS:
        sig = run_sig(IN_LEN, OUT_LEN, epochs, STATICS, seed, DATASET, BATCH_SIZE)
        model_name = f"{MODEL}_{sig}"
        run_dir = ROOT / "outputs" / MODEL / model_name
        metrics_path = run_dir / "metrics.parquet"
        skill_path   = run_dir / "skill_by_horizon.parquet"

        if not metrics_path.exists():
            continue

        m = pd.read_parquet(metrics_path)
        m = m[m["metric"].isin(METRICS_KEEP)]
        m = m[m["horizon"].isin(HORIZONS)]

        m = (m.groupby(["metric", "horizon"])["value"]
               .median().reset_index())
        m["seed"] = seed
        rows_metrics.append(m)

        if skill_path.exists():
            s = pd.read_parquet(skill_path)
            s["horizon"] = s["horizon"].astype(int)
            s = s[s["horizon"].isin(HORIZONS)]
            s["seed"] = seed
            rows_skill.append(s)

    if not rows_metrics:
        return pd.DataFrame()

    m_all = pd.concat(rows_metrics, ignore_index=True)
    m_wide = m_all.pivot(index=["seed", "horizon"], columns="metric", values="value").reset_index()

    if rows_skill:
        s_all = pd.concat(rows_skill, ignore_index=True)
        s_all = s_all[["seed", "horizon", "skill_vs_persistence"]]
        df = m_wide.merge(s_all, on=["seed", "horizon"], how="left")
    else:
        df = m_wide
        df["skill_vs_persistence"] = np.nan

    df["run"] = run_name
    return df 

def aggregate_median(df_run):
    if df_run.empty:
        return pd.DataFrame()

    # median over seeds per horizon
    median = (df_run
            .groupby(["run", "horizon"])[METRICS_KEEP + ["skill_vs_persistence"]]
            .median()
            .reset_index())
    median = median[["run", "horizon"] + METRICS_KEEP + ["skill_vs_persistence"]]

    return pd.concat([median], ignore_index=True)

def main():
    parts = []

    parts.append(load_kunz())

    for run_name, epochs in MY_RUNS:
        df_run = load_my_run(run_name, epochs)
        med = aggregate_median(df_run)
        parts.append(med)

    run_order = {
        "kunz_original": 0,
        "robert_ep50": 1,
        "robert_ep1": 2,
    }

    # full table
    final = pd.concat(parts, ignore_index=True)
    final["run_order"] = final["run"].map(run_order)
    final = final.sort_values(["run_order", "horizon"]).reset_index(drop=True)
    final = final.drop(columns="run_order")

    numeric_cols = ["NSE", "RMSE", "MAE", "rMBE", "skill_vs_persistence"]
    final[numeric_cols] = final[numeric_cols].round(3)

    final.to_csv(FULL_TABLE_CSV, index=False)
    final.to_html(FULL_TABLE_HTML, index=False)

    # sparse table
    SPARSE_HORIZONS = [1, 8, 16]
    sparse = final[final["horizon"].isin(SPARSE_HORIZONS) & (final["run"] != "robert_ep1")].copy()
    sparse = sparse[["horizon", "run", "NSE", "RMSE"]]
    sparse = sparse.sort_values(["horizon"])

    sparse.to_csv(SPARSE_TABLE_CSV, index=False)
    sparse.to_html(SPARSE_TABLE_HTML, index=False)

if __name__ == "__main__":
    main()
