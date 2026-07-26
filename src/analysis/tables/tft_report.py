from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]

KUNZ_FILE = ROOT / "data" / "metrics_aggregated.parquet"
FULL_TABLE_CSV = ROOT / "reports/metrics/tft/tft_metrics_summary.csv"
SPARSE_TABLE_CSV = ROOT / "reports/metrics/tft/tft_metrics_summary_sparse.csv"
FIGURES_DIR = ROOT / "reports/figures/temporal_gru_vs_tft"

MODEL = "TFT"
IN_LEN = 52
OUT_LEN = 16
STATICS = True
DATASETS = ["full_raw", "full_merged"]
BATCH_SIZE = 4096
SEEDS = [40, 94, 673, 1899, 2100, 2149, 2230, 6013, 9595, 9898]

RUN_SPECS = [
    ("robert_ep50", 50),
    ("robert_ep1", 1),
]

METRICS_KEEP = ["NSE", "RMSE", "MAE", "rMBE"]
HORIZONS = list(range(1, 17))


def horizon_variation_analysis(pred_path):
    pred = pq.read_table(pred_path).to_pandas()
    pred["datum"] = pd.to_datetime(pred["datum"])
    pred["startzeitpunkt"] = pd.to_datetime(pred["startzeitpunkt"])
    pred["horizon"] = ((pred["datum"] - pred["startzeitpunkt"]) / pd.Timedelta(weeks=1)) + 1

    all_horizons = pred.groupby("datum")["horizon"].nunique()
    dates_with_all16 = pd.DatetimeIndex(all_horizons[all_horizons == 16].index)

    monthly = {}
    for d in dates_with_all16:
        key = (d.year, d.month)
        if key not in monthly:
            monthly[key] = d
    sample_dates = sorted(monthly.values())

    rows = []
    for dt in sample_dates:
        d_rows = pred[pred["datum"] == dt]
        well_ranges = d_rows.groupby("index")["gws"].agg(lambda x: x.max() - x.min())
        median_horizon_range = well_ranges.median()
        n_wells = int(well_ranges.count())
        h1_vals = d_rows[np.isclose(d_rows["horizon"], 1.0)]["gws"]
        spatial_range = float(h1_vals.max() - h1_vals.min())
        ratio = median_horizon_range / spatial_range if spatial_range > 0 else np.nan
        rows.append({
            "date": str(dt.date()),
            "n_wells": n_wells,
            "median_horizon_range_m": round(float(median_horizon_range), 4),
            "spatial_range_m": round(spatial_range, 4),
            "ratio_pct": round(float(ratio * 100), 3),
        })

    df = pd.DataFrame(rows)
    overall_median_horizon_range = round(float(df["median_horizon_range_m"].median()), 4)
    overall_median_spatial_range = round(float(df["spatial_range_m"].median()), 4)
    overall_median_ratio = round(float(df["ratio_pct"].median()), 3)

    print()
    print("=" * 74)
    print("HORIZON VARIATION ANALYSIS  (reproducible from pred.parquet)")
    print("Question: how much do TFT predictions vary across horizons h=1..16")
    print("(per-well max-min range) vs spatial spread (max-min across wells)?")
    print("=" * 74)
    print(
        f"{'Date':<12}  {'n_wells':>7}  "
        f"{'Median per-well range h1..h16 (m)':>33}  "
        f"{'Spatial range at h=1 (m)':>24}  {'Ratio (%)':>9}"
    )
    print("-" * 74)
    for _, r in df.iterrows():
        print(
            f"{r['date']:<12}  {int(r['n_wells']):>7}  "
            f"{r['median_horizon_range_m']:>23.4f}  "
            f"{r['spatial_range_m']:>17.4f}  "
            f"{r['ratio_pct']:>9.3f}"
        )
    print("-" * 74)
    print(
        f"{'OVERALL':<12}  {'':>7}  "
        f"{overall_median_horizon_range:>23.4f}  "
        f"{overall_median_spatial_range:>17.4f}  "
        f"{overall_median_ratio:>9.3f}"
    )
    print("=" * 74)
    print()
    print(f"Conclusion: the horizon effect ({overall_median_horizon_range} m) is {overall_median_ratio}% of")
    print(f"the spatial spread ({overall_median_spatial_range} m). Evaluating at a fixed")
    print(f"horizon (h=16) across many dates captures the meaningful variation.")
    print()

    return df, overall_median_horizon_range, overall_median_spatial_range, overall_median_ratio


def run_sig(in_len, out_len, epochs, statics, seed, dataset, bs):
    return f"in{in_len}_out{out_len}_ep{epochs}_bs{bs}_stat{int(statics)}_seed{seed}_{dataset}"


def label_run(run_name, dataset):
    label = f"{run_name}_{dataset}"
    if dataset == "full_merged":
        label = f"{label}_spatial_split"
    return label


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


def load_my_run(run_name, epochs, dataset):
    rows_metrics = []
    rows_skill = []

    for seed in SEEDS:
        sig = run_sig(IN_LEN, OUT_LEN, epochs, STATICS, seed, dataset, BATCH_SIZE)
        model_name = f"{MODEL}_{sig}"
        run_dir = ROOT / "outputs" / MODEL / model_name
        metrics_path = run_dir / "metrics.parquet"
        skill_path = run_dir / "skill_by_horizon.parquet"

        if not metrics_path.exists():
            continue

        m = pd.read_parquet(metrics_path)
        m = m[m["metric"].isin(METRICS_KEEP)]
        m = m[m["horizon"].isin(HORIZONS)]

        m = m.groupby(["metric", "horizon"])["value"].median().reset_index()
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
    df = m_all.pivot(index=["seed", "horizon"], columns="metric", values="value").reset_index()

    if rows_skill:
        s_all = pd.concat(rows_skill, ignore_index=True)
        s_all = s_all[["seed", "horizon", "skill_vs_persistence"]]
        df = df.merge(s_all, on=["seed", "horizon"], how="left")
    else:
        df["skill_vs_persistence"] = np.nan

    df["run"] = label_run(run_name, dataset)
    return df


def aggregate_median(df_run):
    if df_run.empty:
        return pd.DataFrame()

    median = (
        df_run.groupby(["run", "horizon"])[METRICS_KEEP + ["skill_vs_persistence"]]
        .median()
        .reset_index()
    )
    median = median[["run", "horizon"] + METRICS_KEEP + ["skill_vs_persistence"]]
    return median


def main():
    parts = [load_kunz()]

    for run_name, epochs in RUN_SPECS:
        for dataset in DATASETS:
            if run_name == "robert_ep1" and dataset != "full_raw":
                continue
            df_run = load_my_run(run_name, epochs, dataset)
            med = aggregate_median(df_run)
            parts.append(med)

    run_labels = [
        "kunz_original",
        "robert_ep50_full_raw",
        "robert_ep50_full_merged_spatial_split",
        "robert_ep1_full_raw",
    ]
    run_order = {name: idx for idx, name in enumerate(run_labels)}

    final = pd.concat(parts, ignore_index=True)
    final["run_order"] = final["run"].map(run_order).fillna(9999)
    final = final.sort_values(["run_order", "horizon"]).reset_index(drop=True)
    final = final.drop(columns="run_order")

    numeric_cols = ["NSE", "RMSE", "MAE", "rMBE", "skill_vs_persistence"]
    final[numeric_cols] = final[numeric_cols].round(3)

    final.to_csv(FULL_TABLE_CSV, index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for run_label in final["run"].unique():
        sub = final[final["run"] == run_label].sort_values("horizon")
        ax.plot(sub["horizon"], sub["NSE"], marker="o", markersize=4, label=run_label)
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("NSE (median across wells & seeds)")
    ax.set_title("TFT NSE by forecast horizon")
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tft_nse_by_horizon.png", dpi=200)
    plt.close(fig)

    SPARSE_HORIZONS = [1, 8, 16]
    sparse = final[final["horizon"].isin(SPARSE_HORIZONS)].copy()
    sparse = sparse[["horizon", "run", "NSE", "RMSE"]]
    sparse["run_order"] = sparse["run"].map(run_order).fillna(9999)
    sparse = sparse.sort_values(["horizon", "run_order"]).drop(columns="run_order")
    sparse.to_csv(SPARSE_TABLE_CSV, index=False)


if __name__ == "__main__":
    main()
