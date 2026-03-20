"""
Compare Joint GRU+GP predictions vs global TFT on spatial holdout wells.

Outputs (all in reports/cross_model/):
  - joint_vs_tft_holdout_summary.csv      : avg nRMSE over all 16 horizons (one row per model)
  - joint_vs_tft_nrmse_by_horizon.png     : line plot nRMSE per horizon
  - joint_vs_tft_nse_by_horizon.png       : line plot NSE per horizon
  - joint_vs_tft_top5_tables.csv          : top-5 wells for joint / TFT / closest performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "cross_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_CSV  = ROOT / "splits" / "spatial_split_full_merged_spf0p8_sc20_ss42_90_10.csv"
JOINT_PRED = ROOT / "outputs" / "GRU_GP_JOINT" / \
    "GRU_GP_JOINT_in52_out16_ep50_seed40_full_merged_r0_spf0p8_sc20_ss42_lsp0p5" / \
    "eval" / "test" / "gp_pred.parquet"
TFT_METRICS = ROOT / "outputs" / "TFT" / \
    "TFT_in52_out16_ep50_bs4096_stat1_seed40_full_raw" / "metrics.parquet"

HORIZONS = list(range(1, 17))

# ── helpers ────────────────────────────────────────────────────────────────────
def nrmse_iqr(pred: np.ndarray, true: np.ndarray) -> float:
    iqr = float(np.diff(np.quantile(true, [0.25, 0.75]))[0])
    if iqr == 0:
        return np.nan
    return float(np.sqrt(np.mean((pred - true) ** 2))) / iqr


def nse(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


# ── load data ──────────────────────────────────────────────────────────────────
split = pd.read_csv(SPLIT_CSV)
holdout_ids = set(split.loc[split["spatial_split"] == "spatial_holdout", "id"])

# Joint predictions: long format (id, datum, horizon, gws_true, gws_forecast)
joint = pd.read_parquet(JOINT_PRED)
joint = joint[joint["id"].isin(holdout_ids)].copy()

# TFT metrics: (metric, value, id, horizon)
tft_m = pd.read_parquet(TFT_METRICS)
tft_m = tft_m[tft_m["id"].isin(holdout_ids)].copy()
tft_m["horizon"] = tft_m["horizon"].astype(int)

# Wells present in both
common_ids = sorted(set(joint["id"].unique()) & set(tft_m["id"].unique()))
print(f"Holdout wells: {len(holdout_ids)}")
print(f"Wells in Joint: {joint['id'].nunique()}")
print(f"Wells in TFT:   {tft_m['id'].nunique()}")
print(f"Common wells:   {len(common_ids)}")

joint = joint[joint["id"].isin(common_ids)]
tft_m = tft_m[tft_m["id"].isin(common_ids)]

# ── compute per-well per-horizon metrics for joint model ───────────────────────
rows = []
for well_id, wdf in joint.groupby("id"):
    for h, hdf in wdf.groupby("horizon"):
        pred = hdf["gws_forecast"].values
        true = hdf["gws_true"].values
        rows.append({
            "id": well_id,
            "horizon": int(h),
            "nRMSE": nrmse_iqr(pred, true),
            "NSE": nse(pred, true),
        })

joint_hw = pd.DataFrame(rows)  # per-well per-horizon

# ── extract TFT per-well per-horizon nRMSE and NSE ────────────────────────────
tft_nrmse = tft_m[tft_m["metric"] == "nRMSE"][["id", "horizon", "value"]].rename(columns={"value": "nRMSE"})
tft_nse   = tft_m[tft_m["metric"] == "NSE"][["id", "horizon", "value"]].rename(columns={"value": "NSE"})
tft_hw    = tft_nrmse.merge(tft_nse, on=["id", "horizon"])

# ── per-horizon aggregation (mean + median across wells) ──────────────────────
joint_by_h = joint_hw.groupby("horizon")[["nRMSE", "NSE"]].agg(
    nRMSE_mean=("nRMSE", "mean"), nRMSE_median=("nRMSE", "median"),
    NSE_mean=("NSE", "mean"),     NSE_median=("NSE", "median"),
).reset_index()
tft_by_h = tft_hw.groupby("horizon")[["nRMSE", "NSE"]].agg(
    nRMSE_mean=("nRMSE", "mean"), nRMSE_median=("nRMSE", "median"),
    NSE_mean=("NSE", "mean"),     NSE_median=("NSE", "median"),
).reset_index()

n_positive_nse_joint = (joint_hw.groupby("id")["NSE"].mean() > 0).sum()

# ── summary table: avg nRMSE over all horizons ────────────────────────────────
summary = pd.DataFrame({
    "model": ["Joint GRU+GP", "TFT (global)"],
    "avg_nRMSE_h1_h16": [
        joint_by_h["nRMSE_mean"].mean(),
        tft_by_h["nRMSE_mean"].mean(),
    ],
    "median_nRMSE_h1_h16": [
        joint_by_h["nRMSE_median"].mean(),
        tft_by_h["nRMSE_median"].mean(),
    ],
    "avg_NSE_h1_h16": [
        joint_by_h["NSE_mean"].mean(),
        tft_by_h["NSE_mean"].mean(),
    ],
    "median_NSE_h1_h16": [
        joint_by_h["NSE_median"].mean(),
        tft_by_h["NSE_median"].mean(),
    ],
    "n_wells": [len(common_ids), len(common_ids)],
    "n_wells_positive_NSE": [n_positive_nse_joint, len(common_ids)],
})
summary_path = OUT_DIR / "joint_vs_tft_holdout_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"\nSummary saved → {summary_path}")
print(summary.to_string(index=False))

# ── line plot: nRMSE by horizon ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(joint_by_h["horizon"], joint_by_h["nRMSE_median"], marker="o", label="Joint GRU+GP (holdout)")
ax.plot(tft_by_h["horizon"],   tft_by_h["nRMSE_median"],   marker="s", label="TFT global (holdout wells)")
ax.set_xlabel("Forecast horizon (weeks)")
ax.set_ylabel("nRMSE (IQR-normalised, median across wells)")
ax.set_title(f"nRMSE by horizon — spatial holdout wells (n={len(common_ids)})")
ax.set_xticks(HORIZONS)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
nrmse_plot = OUT_DIR / "joint_vs_tft_nrmse_by_horizon.png"
fig.savefig(nrmse_plot, dpi=150)
plt.close(fig)
print(f"nRMSE plot saved → {nrmse_plot}")

# ── line plot: NSE by horizon ──────────────────────────────────────────────────
# Use median NSE — mean is dominated by large negative outliers in the joint model
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(joint_by_h["horizon"], joint_by_h["NSE_median"], marker="o", label="Joint GRU+GP (holdout)")
ax.plot(tft_by_h["horizon"],   tft_by_h["NSE_median"],   marker="s", label="TFT global (holdout wells)")
ax.set_xlabel("Forecast horizon (weeks)")
ax.set_ylabel("NSE (median across wells)")
ax.set_title(
    f"NSE by horizon — spatial holdout wells (n={len(common_ids)})\n"
    f"Note: Joint model has {n_positive_nse_joint}/{len(common_ids)} wells with positive median NSE"
)
ax.set_xticks(HORIZONS)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
nse_plot = OUT_DIR / "joint_vs_tft_nse_by_horizon.png"
fig.savefig(nse_plot, dpi=150)
plt.close(fig)
print(f"NSE plot saved → {nse_plot}")

# ── per-well summary (avg over horizons) ──────────────────────────────────────
joint_well = joint_hw.groupby("id")[["nRMSE", "NSE"]].mean().rename(
    columns={"nRMSE": "nRMSE_joint", "NSE": "NSE_joint"}
)
tft_well = tft_hw.groupby("id")[["nRMSE", "NSE"]].mean().rename(
    columns={"nRMSE": "nRMSE_tft", "NSE": "NSE_tft"}
)
well_df = joint_well.join(tft_well, how="inner")
well_df["nRMSE_diff"] = (well_df["nRMSE_joint"] - well_df["nRMSE_tft"]).abs()

cols_base = ["nRMSE_joint", "NSE_joint", "nRMSE_tft", "NSE_tft"]

# top-5 joint (lowest nRMSE_joint)
top5_joint = well_df.nsmallest(5, "nRMSE_joint")[cols_base].reset_index()
top5_joint.insert(0, "rank", range(1, 6))
top5_joint.insert(0, "category", "Top 5 Joint GRU+GP")

# top-5 tft (lowest nRMSE_tft)
top5_tft = well_df.nsmallest(5, "nRMSE_tft")[cols_base].reset_index()
top5_tft.insert(0, "rank", range(1, 6))
top5_tft.insert(0, "category", "Top 5 TFT global")

# top-5 closest (smallest |nRMSE_joint - nRMSE_tft|)
top5_close = well_df.nsmallest(5, "nRMSE_diff")[cols_base + ["nRMSE_diff"]].reset_index()
top5_close.insert(0, "rank", range(1, 6))
top5_close.insert(0, "category", "Top 5 Closest")

tables = pd.concat([top5_joint, top5_tft, top5_close], ignore_index=True)
tables_path = OUT_DIR / "joint_vs_tft_top5_tables.csv"
tables.to_csv(tables_path, index=False)
print(f"Top-5 tables saved → {tables_path}")

# pretty print
for cat, grp in tables.groupby("category", sort=False):
    print(f"\n{'─'*60}")
    print(f"  {cat}")
    print(grp.drop(columns="category").to_string(index=False))
