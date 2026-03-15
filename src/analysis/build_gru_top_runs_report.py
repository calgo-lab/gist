from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
GRU_ROOT = ROOT / "outputs" / "GRU_FCOV"
HPO_DIR = ROOT / "reports" / "gru" / "hpo"
METRICS_DIR = ROOT / "reports" / "gru" / "metrics"
FIG_DIR = ROOT / "reports" / "gru" / "figures"
TFT_SUMMARY = ROOT / "reports" / "tft" / "metrics" / "tft_metrics_summary.csv"

BASELINE_RUN_SIG = "gru_l2_seed40_ep50_full_merged_spf0p8_sc20_ss42"
OUT_TABLE = METRICS_DIR / "gru_top10_overall_plotstyle_metrics.csv"
OUT_ALL_TABLE = METRICS_DIR / "gru_all_runs_plotstyle_metrics.csv"
OUT_PLOT = FIG_DIR / "gru_top6_plus_baseline_plus_tft.png"
SELECTED_RUN_SIGS = [
    "gru_small_grid_v2_t012_seed40_full_merged",
    "gru_small_grid_v3_t011_seed40_full_merged",
]
OUT_SELECTED_TABLE = METRICS_DIR / "gru_selected2_plotstyle_metrics.csv"
OUT_SELECTED_PLOT = FIG_DIR / "gru_selected2_plus_baseline_plus_tft.png"
OUT_SELECTED_RMSE_PLOT = FIG_DIR / "gru_selected2_rmse_plus_baseline_plus_tft.png"
HORIZON_WEIGHTS = np.linspace(1.0, 2.0, 16)


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

        trained_epochs = None
        meta_path = run_dir / "meta.yaml"
        if meta_path.exists():
            with meta_path.open() as f:
                meta = yaml.safe_load(f)
            trained_epochs = meta.get("trained_epochs")

        wh = per_well_horizon_metrics(pred_df)
        if wh.empty:
            continue

        hz = wh.groupby("horizon", as_index=False)[["NSE", "RMSE", "MAE"]].median()
        hz["run_sig"] = run_sig
        horizon_rows.append(hz)

        h16 = hz[hz["horizon"] == 16]
        hz_1_16 = hz[hz["horizon"].between(1, 16)].sort_values("horizon")
        nse_weighted = np.nan
        rmse_weighted = np.nan
        if len(hz_1_16) == 16:
            nse_weighted = float(np.average(hz_1_16["NSE"].to_numpy(dtype=float), weights=HORIZON_WEIGHTS))
            rmse_weighted = float(np.average(hz_1_16["RMSE"].to_numpy(dtype=float), weights=HORIZON_WEIGHTS))
        summary_rows.append(
            {
                "run_sig": run_sig,
                "trained_epochs": trained_epochs,
                "NSE_mean_h1_16": float(hz["NSE"].mean()),
                "NSE_weighted_mean_h1_16": nse_weighted,
                "RMSE_mean_h1_16": float(hz["RMSE"].mean()),
                "RMSE_weighted_mean_h1_16": rmse_weighted,
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
    out = out.sort_values("NSE_weighted_mean_h1_16", ascending=False)

    top10 = out.head(10).copy()
    if BASELINE_RUN_SIG in out["run_sig"].values and BASELINE_RUN_SIG not in top10["run_sig"].values:
        baseline_row = out[out["run_sig"] == BASELINE_RUN_SIG].head(1)
        top10 = pd.concat([top10, baseline_row], ignore_index=True)

    first_cols = [
        "trial",
        "run_sig",
        "objective",
        "train_s",
        "trained_epochs",
        "NSE_weighted_mean_h1_16",
        "NSE_mean_h1_16",
        "RMSE_mean_h1_16",
        "RMSE_weighted_mean_h1_16",
        "NSE_h16_plotstyle",
        "RMSE_h16_plotstyle",
        "MAE_h16_plotstyle",
    ]
    ordered = [c for c in first_cols if c in top10.columns] + [c for c in top10.columns if c not in first_cols]
    top10 = top10[ordered]
    return top10


def build_all_runs_table(summary_df: pd.DataFrame, hpo_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    out = summary_df.merge(hpo_df, on="run_sig", how="left") if not hpo_df.empty else summary_df.copy()
    out = out.sort_values("NSE_weighted_mean_h1_16", ascending=False)

    first_cols = [
        "trial",
        "run_sig",
        "objective",
        "train_s",
        "trained_epochs",
        "NSE_weighted_mean_h1_16",
        "NSE_mean_h1_16",
        "RMSE_mean_h1_16",
        "RMSE_weighted_mean_h1_16",
        "NSE_h16_plotstyle",
        "RMSE_h16_plotstyle",
        "MAE_h16_plotstyle",
    ]
    ordered = [c for c in first_cols if c in out.columns] + [c for c in out.columns if c not in first_cols]
    return out[ordered]


def build_selected_table(summary_df: pd.DataFrame, hpo_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    out = summary_df.merge(hpo_df, on="run_sig", how="left") if not hpo_df.empty else summary_df.copy()
    chosen = SELECTED_RUN_SIGS.copy()
    if BASELINE_RUN_SIG in out["run_sig"].values:
        chosen.append(BASELINE_RUN_SIG)

    selected = out[out["run_sig"].isin(chosen)].copy()
    if selected.empty:
        return selected

    selected["plot_order"] = selected["run_sig"].apply(lambda sig: chosen.index(sig) if sig in chosen else 999)
    selected = selected.sort_values(["plot_order", "NSE_weighted_mean_h1_16"], ascending=[True, False]).drop(columns=["plot_order"])

    first_cols = [
        "trial",
        "run_sig",
        "objective",
        "NSE_weighted_mean_h1_16",
        "NSE_mean_h1_16",
        "RMSE_mean_h1_16",
        "RMSE_weighted_mean_h1_16",
        "NSE_h16_plotstyle",
        "RMSE_h16_plotstyle",
        "MAE_h16_plotstyle",
    ]
    ordered = [c for c in first_cols if c in selected.columns] + [c for c in selected.columns if c not in first_cols]
    return selected[ordered]


def _build_hpo_map(hpo_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    hpo_map = {}
    if not hpo_df.empty and "run_sig" in hpo_df.columns:
        for _, r in hpo_df.drop_duplicates(subset=["run_sig"], keep="first").iterrows():
            payload = {k: r.get(k) for k in hpo_df.columns}
            payload["hpo_name"] = str(r.get("hpo_name", ""))
            payload["trial"] = r.get("trial")
            hpo_map[str(r["run_sig"])] = payload
    return hpo_map


def _label_for_run(run_sig: str, hpo_map: dict[str, dict[str, object]]) -> str:
    if run_sig == BASELINE_RUN_SIG:
        return "hidden=128, layers=2, dropout=0.2, lr=0.0003"
    if run_sig in hpo_map:
        row = hpo_map[run_sig]
        layers = row.get("hp.model.gru_layers")
        hidden = row.get("hp.model.gru_hidden")
        lr = row.get("hp.training.lr")
        dropout = row.get("hp.model.gru_dropout")
        bits = []
        if pd.notna(hidden):
            bits.append(f"hidden={int(hidden)}")
        if pd.notna(layers):
            bits.append(f"layers={int(layers)}")
        if pd.notna(dropout):
            bits.append(f"dropout={float(dropout):g}")
        if pd.notna(lr):
            bits.append(f"lr={float(lr):g}")
        if bits:
            return ", ".join(bits)
    return run_sig


def _run_colors(chosen: list[str]) -> dict[str, str]:
    palette = list(plt.get_cmap("tab10").colors)
    color_map: dict[str, str] = {}
    next_color = 0
    for run_sig in chosen:
        if run_sig == BASELINE_RUN_SIG:
            color_map[run_sig] = "#666666"
            continue
        color_map[run_sig] = palette[next_color % len(palette)]
        next_color += 1
    return color_map


def _plot_runs(
    horizon_df: pd.DataFrame,
    chosen: list[str],
    hpo_df: pd.DataFrame,
    out_plot: Path,
    title: str,
    metric: str,
    ylabel: str,
    color_map: dict[str, str] | None = None,
) -> None:
    if horizon_df.empty:
        print("No GRU runs available for plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    hpo_map = _build_hpo_map(hpo_df)
    color_map = color_map or _run_colors(chosen)

    for run_sig in chosen:
        sub = horizon_df[horizon_df["run_sig"] == run_sig].sort_values("horizon")
        if sub.empty:
            continue
        label = _label_for_run(run_sig, hpo_map)
        ax.plot(
            sub["horizon"],
            sub[metric],
            marker="o",
            markersize=3.5,
            label=label,
            color=color_map.get(run_sig),
        )

    if TFT_SUMMARY.exists():
        tft = pd.read_csv(TFT_SUMMARY)
        tft_ref = tft[tft["run"] == "robert_ep50_full_merged_spatial_split"].copy()
        if not tft_ref.empty and metric in tft_ref.columns:
            tft_ref = tft_ref.sort_values("horizon")
            ax.plot(
                tft_ref["horizon"],
                tft_ref[metric],
                linestyle="--",
                linewidth=2,
                color="black",
                label="tft reference",
            )

    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out_plot}")


def plot_top_runs(horizon_df: pd.DataFrame, summary_df: pd.DataFrame, hpo_df: pd.DataFrame) -> None:
    if horizon_df.empty or summary_df.empty:
        print("No GRU runs available for plot.")
        return

    ranked = summary_df.sort_values("NSE_mean_h1_16", ascending=False)
    top6 = ranked.head(6)["run_sig"].tolist()
    chosen = set(top6)
    if BASELINE_RUN_SIG in summary_df["run_sig"].values:
        chosen.add(BASELINE_RUN_SIG)
    _plot_runs(
        horizon_df=horizon_df,
        chosen=sorted(chosen),
        hpo_df=hpo_df,
        out_plot=OUT_PLOT,
        title="Top 6 GRU Runs + Baseline + TFT Reference",
        metric="NSE",
        ylabel="NSE (median across wells)",
    )


def plot_selected_runs(horizon_df: pd.DataFrame, summary_df: pd.DataFrame, hpo_df: pd.DataFrame) -> None:
    if horizon_df.empty or summary_df.empty:
        print("No GRU runs available for selected-runs plot.")
        return

    chosen = [sig for sig in SELECTED_RUN_SIGS if sig in summary_df["run_sig"].values]
    if BASELINE_RUN_SIG in summary_df["run_sig"].values:
        chosen.append(BASELINE_RUN_SIG)
    if not chosen:
        print("No selected GRU runs available for plot.")
        return
    color_map = _run_colors(chosen)

    _plot_runs(
        horizon_df=horizon_df,
        chosen=chosen,
        hpo_df=hpo_df,
        out_plot=OUT_SELECTED_PLOT,
        title="Selected GRU Runs + Baseline + TFT Reference",
        metric="NSE",
        ylabel="NSE (median across wells)",
        color_map=color_map,
    )
    _plot_runs(
        horizon_df=horizon_df,
        chosen=chosen,
        hpo_df=hpo_df,
        out_plot=OUT_SELECTED_RMSE_PLOT,
        title="Selected GRU Runs + Baseline + TFT Reference (RMSE)",
        metric="RMSE",
        ylabel="RMSE (median across wells)",
        color_map=color_map,
    )


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
    all_table = build_all_runs_table(summary_df, hpo_df)
    all_table.to_csv(OUT_ALL_TABLE, index=False)
    print(f"Wrote: {OUT_ALL_TABLE}")
    selected_table = build_selected_table(summary_df, hpo_df)
    selected_table.to_csv(OUT_SELECTED_TABLE, index=False)
    print(f"Wrote: {OUT_SELECTED_TABLE}")

    plot_top_runs(horizon_df, summary_df, hpo_df)
    plot_selected_runs(horizon_df, summary_df, hpo_df)


if __name__ == "__main__":
    main()
