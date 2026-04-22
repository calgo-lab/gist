from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
OUT_MET = ROOT / "reports" / "presentation_assets" / "metrics"
OUT_FIG = ROOT / "reports" / "presentation_assets" / "figures"

RUNS = {
    "Global GRU": ROOT / "outputs/GRU_FCOV/GRU_FCOV_in52_out16_ep50_bs4096_seed40_full_merged_all_train/predictions/pred.parquet",
    "GP on true obs": ROOT / "outputs/gp/GRU_FCOV_oracle_rmd90_test52__oracle_hpo_t020_ni128_ps200_j1e-03__predobstrain/gp_pred.parquet",
    "GP on GRU preds": ROOT / "outputs/gp/GRU_FCOV_in52_out16_ep50_bs8192_seed40_full_merged_test52_hpo_sep_t118__hpo_sep_t118__predobstrain/gp_pred.parquet",
    "Joint training GRU + GP": ROOT / "outputs/GRU_GP_JOINT/GRU_GP_JOINT_0260/eval/test/gp_pred.parquet",
}
MODEL_ORDER = [
    "Global GRU",
    "GP on true obs",
    "GP on GRU preds",
    "Joint training GRU + GP",
]
COLORS = {
    "Global GRU": "#1f77b4",
    "GP on true obs": "#2ca02c",
    "GP on GRU preds": "#ff7f0e",
    "Joint training GRU + GP": "#d62728",
}


def _load_split() -> set:
    split = pd.read_csv(ROOT / "splits/rmd90_test52.csv")
    col = "spatial_split" if "spatial_split" in split.columns else "split"
    return set(split.loc[split[col].isin(["spatial_test", "spatial_holdout"]), "id"])


def _normalize_pred(path: Path, test_ids: set) -> pd.DataFrame:
    df = pq.read_table(path).to_pandas()
    cols = set(df.columns)
    pred_col = next((c for c in ["gws_forecast", "pred", "y_mean"] if c in cols), None)
    true_col = next((c for c in ["gws_true", "gws", "target", "y"] if c in cols), None)
    if pred_col is None or true_col is None:
        raise ValueError(f"Could not infer pred/true columns for {path}: {list(cols)}")
    out = df.rename(columns={pred_col: "pred", true_col: "true"})[["id", "datum", "horizon", "pred", "true"]].copy()
    out["datum"] = pd.to_datetime(out["datum"])
    out = out[out["id"].isin(test_ids)].copy()
    out["datum_s"] = out["datum"].astype(str)
    return out


def _per_well_horizon_rmse(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (wid, h), g in df.groupby(["id", "horizon"]):
        pred = g["pred"].to_numpy(dtype=float)
        true = g["true"].to_numpy(dtype=float)
        mask = np.isfinite(pred) & np.isfinite(true)
        if mask.sum() < 2:
            continue
        rmse = float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))
        rows.append({"id": wid, "horizon": int(h), "rmse": rmse})
    return pd.DataFrame(rows)


def _summary_stats(df: pd.DataFrame) -> dict:
    per_well = []
    for wid, g in df.groupby("id"):
        pred = g["pred"].to_numpy(dtype=float)
        true = g["true"].to_numpy(dtype=float)
        mask = np.isfinite(pred) & np.isfinite(true)
        if mask.sum() < 2:
            continue
        rmse = float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))
        iqr = float(np.quantile(true[mask], 0.75) - np.quantile(true[mask], 0.25))
        per_well.append({"rmse": rmse, "ratio": rmse / iqr if iqr > 0 else float("nan")})
    s = pd.DataFrame(per_well)
    return {
        "median_nrmse_per_well": float(s["ratio"].median()),
        "median_rmse_per_well_m": float(s["rmse"].median()),
        "common_keys": len(df),
    }


def _make_plot(
    plot_df: pd.DataFrame,
    labels: list[str],
    title: str,
    out_path: Path,
    legend_loc: str = "center left",
    legend_anchor: tuple[float, float] = (1.02, 0.5),
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    for label in labels:
        sub = plot_df.loc[plot_df["model"] == label].sort_values("horizon")
        ax.plot(sub["horizon"], sub["median_rmse_per_well"], marker="o",
                linewidth=2.2, markersize=4.8, color=COLORS[label], label=label)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Median RMSE per well (m)")
    ax.set_title(title)
    ax.set_xticks(sorted(plot_df["horizon"].unique()))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc=legend_loc, bbox_to_anchor=legend_anchor)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_MET.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    test_ids = _load_split()
    frames = {name: _normalize_pred(path, test_ids) for name, path in RUNS.items()}

    # Find common (id, datum_s, horizon) keys across all 4 models
    common = None
    for df in frames.values():
        keys = set(zip(df["id"], df["datum_s"], df["horizon"]))
        common = keys if common is None else common & keys
    common_df = pd.DataFrame(sorted(common), columns=["id", "datum_s", "horizon"])

    summary_rows = []
    horizon_rows = []
    for model, df in frames.items():
        sub = df.merge(common_df, on=["id", "datum_s", "horizon"], how="inner")
        stats = _summary_stats(sub)
        stats["model"] = model
        summary_rows.append(stats)

        pw_h = _per_well_horizon_rmse(sub)
        by_h = pw_h.groupby("horizon", as_index=False)["rmse"].median().rename(columns={"rmse": "median_rmse_per_well"})
        by_h["model"] = model
        horizon_rows.append(by_h)

    summary = pd.DataFrame(summary_rows)
    horizon = pd.concat(horizon_rows, ignore_index=True)
    summary.to_csv(OUT_MET / "slide_table_consistent_metrics.csv", index=False)
    horizon.to_csv(OUT_MET / "slide_table_rmse_per_horizon_consistent.csv", index=False)

    horizon["model"] = pd.Categorical(horizon["model"], categories=MODEL_ORDER, ordered=True)
    horizon = horizon.sort_values(["model", "horizon"]).reset_index(drop=True)

    _make_plot(
        plot_df=horizon,
        labels=MODEL_ORDER,
        title="Per-Well RMSE by Horizon on the Common Evaluation Window",
        out_path=OUT_FIG / "slide_table_rmse_per_horizon.png",
    )
    _make_plot(
        plot_df=horizon,
        labels=[m for m in MODEL_ORDER if m != "Global GRU"],
        title="Per-Well RMSE by Horizon Without the Global GRU Baseline",
        out_path=OUT_FIG / "slide_table_rmse_per_horizon_no_global_gru.png",
        legend_loc="upper left",
        legend_anchor=(1.02, 1.0),
    )

    print("Done.")
    for _, row in summary.iterrows():
        print(f"  {row['model']}: nRMSE_pw={row['median_nrmse_per_well']:.4f}, RMSE_pw={row['median_rmse_per_well_m']:.4f} m")


if __name__ == "__main__":
    main()
