"""
Plot joint GRU+GP vs separate (GRU → GP) per-horizon metrics.

Both models are evaluated on the spatial holdout wells using GP spatial
interpolation; the difference is whether GRU and GP were trained jointly
or as two independent stages.

Outputs: reports/cross_model/joint_vs_separate_by_horizon.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
JOINT_BY_H = (
    ROOT / "outputs"
    / "GRU_GP_JOINT"
    / "GRU_GP_JOINT_in52_out16_ep50_seed40_full_merged_r0_spf0p8_sc20_ss42_lsp0p5"
    / "eval" / "test" / "gp_metrics_by_horizon.csv"
)

# Best separate GP-on-GRU run (hpo_v1_t004, NSE_id_median = 0.4149)
SEPARATE_PRED = (
    ROOT / "outputs" / "gp"
    / "GRU_FCOV_gru_l2_seed40_ep50_full_merged_spf0p8_sc20_ss42"
      "__gp_gpytorch_hpo_v1_t004__predobstrain"
    / "gp_pred.parquet"
)

OUT_DIR = ROOT / "reports" / "cross_model"
OUT_PNG = OUT_DIR / "joint_vs_separate_by_horizon.png"


# ---------------------------------------------------------------------------
# Metric helpers (same as gru_gp_eval.py)
# ---------------------------------------------------------------------------
def _nse(pred, real):
    denom = float(np.sum((real - np.mean(real)) ** 2))
    if denom == 0:
        return float("nan")
    return float(1 - np.sum((pred - real) ** 2) / denom)


def _rmse(pred, real):
    return float(np.sqrt(np.mean((pred - real) ** 2)))


def _mae(pred, real):
    return float(np.mean(np.abs(pred - real)))


def per_horizon_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pooled NSE, RMSE, MAE per horizon from a predictions DataFrame."""
    rows = []
    for h, g in df.groupby("horizon"):
        real = g["gws_true"].to_numpy(dtype=float)
        pred = g["gws_forecast"].to_numpy(dtype=float)
        mask = np.isfinite(real) & np.isfinite(pred)
        real, pred = real[mask], pred[mask]
        rows.append({
            "horizon": int(h),
            "n": int(mask.sum()),
            "NSE": _nse(pred, real),
            "RMSE": _rmse(pred, real),
            "MAE": _mae(pred, real),
        })
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
joint = pd.read_csv(JOINT_BY_H)

sep_pred = pq.read_table(SEPARATE_PRED).to_pandas()
sep_pred.rename(columns={"gws": "gws_forecast"}, errors="ignore", inplace=True)
if "gws_forecast" not in sep_pred.columns and "gws_pred" in sep_pred.columns:
    sep_pred["gws_forecast"] = sep_pred["gws_pred"]
separate = per_horizon_metrics(sep_pred)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

horizons = joint["horizon"].to_numpy()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

metrics = [
    ("NSE",  "NSE (pooled)", False),
    ("RMSE", "RMSE [m]",     False),
    ("MAE",  "MAE [m]",      False),
]

colors = {"joint": "#2166ac", "separate": "#d6604d"}

for ax, (col, ylabel, invert) in zip(axes, metrics):
    ax.plot(horizons, joint[col].to_numpy(),
            color=colors["joint"], marker="o", markersize=4,
            linewidth=1.8, label="Joint GRU+GP")
    ax.plot(separate["horizon"].to_numpy(), separate[col].to_numpy(),
            color=colors["separate"], marker="s", markersize=4,
            linewidth=1.8, label="Separate (GRU → GP)")
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(horizons)
    ax.set_xticklabels(horizons, fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    if invert:
        ax.invert_yaxis()

axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Joint GRU+GP vs Separate Pipeline — Spatial Holdout by Horizon",
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")
plt.show()
