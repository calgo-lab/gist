"""
joint_report.py

Builds reports/gru_gp_joint/metrics/joint_metrics_summary.csv and figures
from outputs/GRU_GP_JOINT/.

Each joint run is expected to have:
    eval/test/gp_metrics.parquet       — pooled metrics (key/value)
    eval/test/gp_metrics_by_horizon.csv — per-horizon RMSE/MAE/NSE
    eval/test/gp_pred.parquet          — raw predictions with coordinates

Run after joint eval completes:
    python src/analysis/build/joint_report.py

Outputs
-------
reports/gru_gp_joint/metrics/joint_metrics_summary.csv
    One row per joint run.

reports/gru_gp_joint/figures/rmse_by_horizon.png
    RMSE by forecast horizon: best joint run vs best standalone GRU vs best TFT.

reports/gru_gp_joint/figures/mae_per_well_map.png
    Spatial scatter: per-well MAE at well coordinates (x_25833, y_25833).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
JOINT_ROOT = ROOT / "outputs" / "GRU_GP_JOINT"
METRICS_DIR = ROOT / "reports" / "gru_gp_joint" / "metrics"
FIG_DIR = ROOT / "reports" / "gru_gp_joint" / "figures"
OUT_CSV = METRICS_DIR / "joint_metrics_summary.csv"

GRU_SUMMARY = ROOT / "reports" / "gru" / "metrics" / "gru_metrics_summary.csv"
GP_SUMMARY = ROOT / "reports" / "gp" / "metrics" / "gp_metrics_summary.csv"
TFT_SUMMARY = ROOT / "reports" / "tft" / "metrics" / "tft_metrics_summary.csv"
GP_ROOT = ROOT / "outputs" / "gp"

KEEP_METRICS = ["RMSE_pooled", "nRMSE", "NSE_pooled", "NSE_id_median", "MAE_pooled"]

BOUNDARY_FILE = ROOT / "data" / "boundaries" / "geoBoundaries-DEU-ADM1_simplified.geojson"


def load_joint_runs():
    rows = []
    horizon_dfs = {}
    pred_dfs = {}

    for run_dir in sorted(JOINT_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        test_dir = run_dir / "eval" / "test"
        metrics_path = test_dir / "gp_metrics.parquet"
        if not metrics_path.exists():
            continue

        run_sig = run_dir.name.removeprefix("GRU_GP_JOINT_")

        df = pd.read_parquet(metrics_path)
        if "metric" not in df.columns or "value" not in df.columns:
            continue
        metrics = dict(zip(df["metric"], df["value"].astype(float)))
        row = {"run_sig": run_sig}
        row["RMSE_pooled"] = metrics.get("RMSE", np.nan)
        row["nRMSE"] = metrics.get("nRMSE", np.nan)
        row["NSE_pooled"] = metrics.get("NSE_pooled", metrics.get("NSE", np.nan))
        row["NSE_id_median"] = metrics.get("NSE_id_median", np.nan)
        row["MAE_pooled"] = metrics.get("MAE", np.nan)
        rows.append(row)

        hz_path = test_dir / "gp_metrics_by_horizon.csv"
        if hz_path.exists():
            horizon_dfs[run_sig] = pd.read_csv(hz_path)

        pred_path = test_dir / "gp_pred.parquet"
        if pred_path.exists():
            pred_dfs[run_sig] = pd.read_parquet(pred_path)

    summary_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return summary_df, horizon_dfs, pred_dfs


def build_metrics_table(summary_df):
    if summary_df.empty:
        return summary_df
    out = summary_df.sort_values("NSE_pooled", ascending=False).reset_index(drop=True)
    out.insert(0, "run_number", range(1, len(out) + 1))
    ordered = ["run_number"] + [c for c in KEEP_METRICS if c in out.columns] + ["run_sig"]
    remaining = [c for c in out.columns if c not in ordered]
    return out[ordered + remaining]


def _nrmse_from_pred(pred, obs_col="gws_true"):
    rows = []
    obs_std_global = float(pred[obs_col].std())
    for h, g in pred.groupby("horizon"):
        err = g["gws_forecast"].to_numpy() - g[obs_col].to_numpy()
        rmse = float(np.sqrt((err ** 2).mean()))
        rows.append({"horizon": int(h), "nRMSE": rmse / obs_std_global if obs_std_global > 0 else np.nan})
    return pd.DataFrame(rows).sort_values("horizon")


def _load_gp_best_horizon():
    if not GP_SUMMARY.exists():
        return None
    gp = pd.read_csv(GP_SUMMARY)
    if gp.empty or "NSE_pooled" not in gp.columns or "dir_name" not in gp.columns:
        return None
    gp_gru = gp[gp["model_prefix"] == "GRU_FCOV"]
    if gp_gru.empty:
        gp_gru = gp
    best_dir = gp_gru.loc[gp_gru["NSE_pooled"].idxmax(), "dir_name"]
    pred_path = GP_ROOT / best_dir / "gp_pred.parquet"
    if not pred_path.exists():
        return None
    pred = pd.read_parquet(pred_path)
    return _nrmse_from_pred(pred, obs_col="gws_true")


def _load_tft_best_horizon():
    if not TFT_SUMMARY.exists():
        return None
    tft = pd.read_csv(TFT_SUMMARY)
    if tft.empty or "RMSE" not in tft.columns:
        return None
    best_run = tft.groupby("run")["NSE"].mean().idxmax()
    sub = tft[tft["run"] == best_run].sort_values("horizon")
    if "nRMSE" in sub.columns:
        return sub[["horizon", "nRMSE"]]
    return None


def plot_nrmse_by_horizon(horizon_dfs, table):
    if not horizon_dfs or table.empty:
        return

    best_sig = table.loc[table["RMSE_pooled"].idxmin(), "run_sig"]
    hz = horizon_dfs.get(best_sig)
    if hz is None or "nRMSE" not in hz.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(hz["horizon"], hz["nRMSE"], marker="o", markersize=4, label="Joint GRU+GP (best)")

    gp_hz = _load_gp_best_horizon()
    if gp_hz is not None:
        ax.plot(gp_hz["horizon"], gp_hz["nRMSE"], marker="s", markersize=4,
                linestyle="--", label="GP on GRU (best)")

    tft_hz = _load_tft_best_horizon()
    if tft_hz is not None:
        ax.plot(tft_hz["horizon"], tft_hz["nRMSE"], marker="^", markersize=4,
                linestyle=":", color="black", label="TFT (best)")

    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("nRMSE")
    ax.set_title("nRMSE by forecast horizon: Joint GRU+GP vs GP on GRU")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "nrmse_by_horizon.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out}")


def plot_mae_per_well_map(pred_dfs, table):
    import geopandas as gpd

    if not pred_dfs or table.empty:
        return

    best_sig = table.loc[table["RMSE_pooled"].idxmin(), "run_sig"]
    pred = pred_dfs.get(best_sig)
    if pred is None:
        return
    if not {"id", "gws_true", "gws_forecast", "x_25833", "y_25833"}.issubset(pred.columns):
        return

    per_well = (
        pred.groupby("id")
        .apply(lambda g: pd.Series({
            "MAE": float(np.abs(g["gws_forecast"].to_numpy() - g["gws_true"].to_numpy()).mean()),
            "x": float(g["x_25833"].iloc[0]),
            "y": float(g["y_25833"].iloc[0]),
        }), include_groups=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 9))

    if BOUNDARY_FILE.exists():
        gdf = gpd.read_file(BOUNDARY_FILE)
        name_col = next((c for c in gdf.columns if "name" in c.lower()), None)
        if name_col:
            gdf = gdf[gdf[name_col].str.contains("Brandenburg", case=False, na=False)]
        gdf = gdf.to_crs("EPSG:25833")
        gdf.plot(ax=ax, facecolor="none", edgecolor="#555555", linewidth=1.2)

    sc = ax.scatter(
        per_well["x"], per_well["y"],
        c=per_well["MAE"], cmap="RdBu_r",
        s=45, alpha=0.9, edgecolors="white", linewidths=0.3,
        vmin=per_well["MAE"].quantile(0.02), vmax=per_well["MAE"].quantile(0.98),
    )
    plt.colorbar(sc, ax=ax, label="MAE (m)", shrink=0.7)
    ax.set_xlabel("Easting (EPSG:25833, m)")
    ax.set_ylabel("Northing (EPSG:25833, m)")
    ax.set_title("Per-well MAE — Joint GRU+GP (best run)")
    ax.set_aspect("equal")
    fig.tight_layout()
    out = FIG_DIR / "mae_per_well_map.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out}")


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary_df, horizon_dfs, pred_dfs = load_joint_runs()
    if summary_df.empty:
        print("No joint run outputs found in", JOINT_ROOT)
        return

    table = build_metrics_table(summary_df)
    table.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}  ({len(table)} runs)")

    plot_nrmse_by_horizon(horizon_dfs, table)
    plot_mae_per_well_map(pred_dfs, table)


if __name__ == "__main__":
    main()
