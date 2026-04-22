from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_MET = ROOT / "reports" / "presentation_assets" / "metrics"
OUT_FIG = ROOT / "reports" / "presentation_assets" / "figures"
OUT_TXT = ROOT / "reports" / "presentation_assets" / "notes"


SOURCE_MAP = {
    "Global GRU": "cluster: outputs/GRU_FCOV/GRU_FCOV_in52_out16_ep50_bs4096_seed40_full_merged_all_train/predictions/pred.parquet",
    "GP on true obs": "cluster: outputs/gp/GRU_FCOV_oracle_rmd90_test52__oracle_hpo_t020_ni128_ps200_j1e-03__predobstrain/gp_pred.parquet",
    "GP on GRU preds": "cluster: outputs/gp/GRU_FCOV_in52_out16_ep50_bs8192_seed40_full_merged_test52_hpo_sep_t118__hpo_sep_t118__predobstrain/gp_pred.parquet",
    "Joint training GRU + GP": "cluster: outputs/GRU_GP_JOINT/GRU_GP_JOINT_0260/eval/test/gp_pred.parquet",
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


def _ensure_dirs() -> None:
    for path in [OUT_MET, OUT_FIG, OUT_TXT]:
        path.mkdir(parents=True, exist_ok=True)


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
        ax.plot(
            sub["horizon"],
            sub["median_rmse_per_well"],
            marker="o",
            linewidth=2.2,
            markersize=4.8,
            color=COLORS[label],
            label=label,
        )

    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Median RMSE per well (m)")
    ax.set_title(title)
    ax.set_xticks(sorted(plot_df["horizon"].unique()))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc=legend_loc, bbox_to_anchor=legend_anchor)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_table_plot() -> tuple[Path, Path, Path]:
    _ensure_dirs()

    plot_df = pd.read_csv(OUT_MET / "slide_table_rmse_per_horizon_consistent.csv")
    plot_df["source"] = plot_df["model"].map(SOURCE_MAP)
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=MODEL_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["model", "horizon"]).reset_index(drop=True)

    summary_df = pd.read_csv(OUT_MET / "slide_table_consistent_metrics.csv")
    summary_df["model"] = pd.Categorical(summary_df["model"], categories=MODEL_ORDER, ordered=True)
    summary_df = summary_df.sort_values("model").reset_index(drop=True)

    csv_path = OUT_MET / "slide_table_rmse_per_horizon.csv"
    plot_df.to_csv(csv_path, index=False)

    fig_path = OUT_FIG / "slide_table_rmse_per_horizon.png"
    _make_plot(
        plot_df=plot_df,
        labels=MODEL_ORDER,
        title="Per-Well RMSE by Horizon on the Common Evaluation Window",
        out_path=fig_path,
        legend_loc="center left",
        legend_anchor=(1.02, 0.5),
    )

    fig_no_gru_path = OUT_FIG / "slide_table_rmse_per_horizon_no_global_gru.png"
    _make_plot(
        plot_df=plot_df,
        labels=[label for label in MODEL_ORDER if label != "Global GRU"],
        title="Per-Well RMSE by Horizon Without the Global GRU Baseline",
        out_path=fig_no_gru_path,
        legend_loc="upper left",
        legend_anchor=(1.02, 1.0),
    )

    rounded = {
        row["model"]: (
            f"{row['median_nrmse_per_well']:.4f}",
            f"{row['median_rmse_per_well_m']:.4f}",
        )
        for _, row in summary_df.iterrows()
    }
    horizons = ", ".join(str(int(h)) for h in sorted(plot_df["horizon"].unique()))
    common_keys = int(summary_df["common_keys"].iloc[0])
    note = "\n".join([
        "Figures: median per-well RMSE by forecast horizon for the slide-table pipelines.",
        "This version is fully consistent across all four rows and plots.",
        "Common evaluation slice:",
        "- 52 holdout wells",
        f"- {common_keys:,} common (well, date, horizon) keys",
        f"- Horizons present in all four saved artifacts: {horizons}",
        "Sources:",
        "- Global GRU uses the all-1040-well GRU run `GRU_FCOV_in52_out16_ep50_bs4096_seed40_full_merged_all_train`.",
        "- GP on true obs uses the oracle GP HPO best (`oracle_hpo_t020`).",
        "- GP on GRU preds uses the decoupled HPO best (`hpo_sep_t118`).",
        "- Joint training GRU + GP uses `GRU_GP_JOINT_0260`.",
        "Consistent table values on that common slice:",
        f"- Global GRU: nRMSE_pw={rounded['Global GRU'][0]}, RMSE_pw={rounded['Global GRU'][1]} m",
        f"- GP on true obs: nRMSE_pw={rounded['GP on true obs'][0]}, RMSE_pw={rounded['GP on true obs'][1]} m",
        f"- GP on GRU preds: nRMSE_pw={rounded['GP on GRU preds'][0]}, RMSE_pw={rounded['GP on GRU preds'][1]} m",
        f"- Joint training GRU + GP: nRMSE_pw={rounded['Joint training GRU + GP'][0]}, RMSE_pw={rounded['Joint training GRU + GP'][1]} m",
        "- `slide_table_rmse_per_horizon.png` shows all four lines with the legend moved outside the axes.",
        "- `slide_table_rmse_per_horizon_no_global_gru.png` removes the global GRU baseline.",
    ])
    note_path = OUT_TXT / "slide_table_rmse_per_horizon_note.md"
    note_path.write_text(note + "\n", encoding="utf-8")

    return csv_path, fig_path, note_path


if __name__ == "__main__":
    csv_path, fig_path, note_path = build_table_plot()
    print(csv_path)
    print(fig_path)
    print(note_path)
