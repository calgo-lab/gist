from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GRU_ROOT = ROOT / "outputs" / "GRU_FCOV"
HPO_DIR = ROOT / "reports" / "gru" / "hpo"
METRICS_DIR = ROOT / "reports" / "gru" / "metrics"
FIG_DIR = ROOT / "reports" / "gru" / "figures"
TFT_SUMMARY = ROOT / "reports" / "tft" / "metrics" / "tft_metrics_summary.csv"

BASELINE_RUN_SIG = "gru_l2_seed40_ep50_full_merged_spf0p8_sc20_ss42"
OUT_TABLE = METRICS_DIR / "gru_top10_overall_plotstyle_metrics.csv"
OUT_PLOT = FIG_DIR / "gru_top6_plus_baseline_plus_tft.png"


def nse(pred: np.ndarray, real: np.ndarray) -> float:
    denom = np.sum((real - np.mean(real)) ** 2)
    if denom <= 0:
        return np.nan
    return float(1 - np.sum((pred - real) ** 2) / denom)


def rmse(pred: np.ndarray, real: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - real) ** 2)))


def mae(pred: np.ndarray, real: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - real)))


def load_hpo_meta() -> pd.DataFrame:
    files = sorted(HPO_DIR.glob("*_results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return pd.DataFrame()

    parts = []
    for p in files:
        df = pd.read_csv(p)
        if "run_sig" not in df.columns:
            continue
        hpo_name = p.name.replace("_results.csv", "")
        df = df.copy()
        df["hpo_name"] = hpo_name
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    all_hpo = pd.concat(parts, ignore_index=True)
    # Keep newest file's row on duplicate run_sig.
    all_hpo = all_hpo.drop_duplicates(subset=["run_sig"], keep="first")
    return all_hpo


def per_well_horizon_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (well_id, h), g in pred_df.groupby(["id", "horizon"]):
        p = g["gws_forecast"].to_numpy()
        r = g["gws"].to_numpy()
        rows.append(
            {
                "id": well_id,
                "horizon": int(h),
                "NSE": nse(p, r),
                "RMSE": rmse(p, r),
                "MAE": mae(p, r),
            }
        )
    return pd.DataFrame(rows)


def collect_all_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    horizon_rows: list[pd.DataFrame] = []

    for run_dir in sorted(GRU_ROOT.glob("GRU_FCOV_*")):
        pred_path = run_dir / "predictions" / "pred.parquet"
        if not pred_path.exists():
            continue

        pred_df = pd.read_parquet(pred_path)
        if pred_df.empty:
            continue
        if not {"id", "horizon", "gws_forecast", "gws"}.issubset(pred_df.columns):
            continue

        run_sig = run_dir.name.replace("GRU_FCOV_", "", 1)
        wh = per_well_horizon_metrics(pred_df)
        if wh.empty:
            continue

        hz = wh.groupby("horizon", as_index=False)[["NSE", "RMSE", "MAE"]].median()
        hz["run_sig"] = run_sig
        horizon_rows.append(hz)

        h16 = hz[hz["horizon"] == 16]
        summary_rows.append(
            {
                "run_sig": run_sig,
                "NSE_mean_h1_16": float(hz["NSE"].mean()),
                "NSE_h16_plotstyle": float(h16["NSE"].iloc[0]) if not h16.empty else np.nan,
                "RMSE_h16_plotstyle": float(h16["RMSE"].iloc[0]) if not h16.empty else np.nan,
                "MAE_h16_plotstyle": float(h16["MAE"].iloc[0]) if not h16.empty else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    horizon_df = pd.concat(horizon_rows, ignore_index=True) if horizon_rows else pd.DataFrame()
    return summary_df, horizon_df


def build_table(summary_df: pd.DataFrame, hpo_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    out = summary_df.merge(hpo_df, on="run_sig", how="left") if not hpo_df.empty else summary_df.copy()
    out = out.sort_values("NSE_mean_h1_16", ascending=False)

    top10 = out.head(10).copy()
    if BASELINE_RUN_SIG in out["run_sig"].values and BASELINE_RUN_SIG not in top10["run_sig"].values:
        baseline_row = out[out["run_sig"] == BASELINE_RUN_SIG].head(1)
        top10 = pd.concat([top10, baseline_row], ignore_index=True)

    first_cols = [
        "trial",
        "run_sig",
        "objective",
        "NSE_mean_h1_16",
        "NSE_h16_plotstyle",
        "RMSE_h16_plotstyle",
        "MAE_h16_plotstyle",
    ]
    ordered = [c for c in first_cols if c in top10.columns] + [c for c in top10.columns if c not in first_cols]
    top10 = top10[ordered]
    return top10


def plot_top_runs(horizon_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    if horizon_df.empty or summary_df.empty:
        print("No GRU runs available for plot.")
        return

    ranked = summary_df.sort_values("NSE_mean_h1_16", ascending=False)
    top6 = ranked.head(6)["run_sig"].tolist()
    chosen = set(top6)
    if BASELINE_RUN_SIG in summary_df["run_sig"].values:
        chosen.add(BASELINE_RUN_SIG)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for run_sig in sorted(chosen):
        sub = horizon_df[horizon_df["run_sig"] == run_sig].sort_values("horizon")
        if sub.empty:
            continue
        tm = re.search(r"_t(\d{3})_", run_sig)
        if run_sig == BASELINE_RUN_SIG:
            label = "gru 2-layer baseline"
        elif tm:
            label = f"gru hpo t{tm.group(1)}"
        else:
            label = run_sig
        ax.plot(sub["horizon"], sub["NSE"], marker="o", markersize=3.5, label=label)

    if TFT_SUMMARY.exists():
        tft = pd.read_csv(TFT_SUMMARY)
        tft_ref = tft[tft["run"] == "robert_ep50_full_merged_spatial_split"].copy()
        if not tft_ref.empty:
            tft_ref = tft_ref.sort_values("horizon")
            ax.plot(
                tft_ref["horizon"],
                tft_ref["NSE"],
                linestyle="--",
                linewidth=2,
                color="black",
                label="tft_ep50_full_merged_spatial_split",
            )

    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("NSE (median across wells)")
    ax.set_title("Top 6 GRU Runs + Baseline + TFT Reference")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=200)
    plt.close(fig)
    print(f"Wrote: {OUT_PLOT}")


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary_df, horizon_df = collect_all_runs()
    if summary_df.empty:
        print("No GRU prediction runs found.")
        return

    hpo_df = load_hpo_meta()
    table = build_table(summary_df, hpo_df)
    table.to_csv(OUT_TABLE, index=False)
    print(f"Wrote: {OUT_TABLE}")

    plot_top_runs(horizon_df, summary_df)


if __name__ == "__main__":
    main()
