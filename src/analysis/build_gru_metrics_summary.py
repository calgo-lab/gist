from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[2]

GRU_OUTPUT_ROOT = ROOT / "outputs" / "GRU_FCOV"
REPORTS_DIR = ROOT / "reports" / "gru" / "metrics"
FIGURES_DIR = ROOT / "reports" / "gru" / "figures"
TFT_SUMMARY_CSV = ROOT / "reports" / "tft" / "metrics" / "tft_metrics_summary.csv"

FULL_TABLE_CSV = REPORTS_DIR / "gru_metrics_summary.csv"
SPARSE_TABLE_CSV = REPORTS_DIR / "gru_metrics_summary_sparse.csv"


RUN_RE = re.compile(
    r"^GRU_FCOV_"
    r"in(?P<in_len>\d+)_out(?P<out_len>\d+)_ep(?P<epochs>\d+)_bs(?P<batch_size>\d+)"
    r"_seed(?P<seed>\d+)_(?P<dataset>.+)$"
)


def _is_spatial_split(run_sig: str) -> bool:
    return "_spf" in run_sig and "_sc" in run_sig and "_ss" in run_sig


def _split_tag(run_sig: str) -> str:
    m = re.search(r"(_spf[^_]+_sc\d+_ss\d+)", run_sig)
    return m.group(1) if m else ""


def _run_label(
    in_len: int,
    out_len: int,
    epochs: int,
    dataset: str,
    spatial_split: bool,
    split_tag: str,
) -> str:
    label = f"gru_in{in_len}_out{out_len}_ep{epochs}_{dataset}"
    if spatial_split:
        if dataset == "full_merged":
            label = f"{label}_spatial_split"
        if split_tag:
            label = f"{label}{split_tag}"
    return label


def _nse(pred: np.ndarray, real: np.ndarray) -> float:
    denom = np.sum((real - np.mean(real)) ** 2)
    if denom <= 0:
        return np.nan
    return float(1 - (np.sum((pred - real) ** 2) / denom))


def _rmse(pred: np.ndarray, real: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - real) ** 2)))


def _mae(pred: np.ndarray, real: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - real)))


def _rmbe(pred: np.ndarray, real: np.ndarray) -> float:
    std_real = np.std(real)
    if std_real <= 0:
        return np.nan
    return float(np.mean(pred - real) / std_real)


def _metrics_by_id_horizon(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (well_id, horizon), group in pred_df.groupby(["id", "horizon"]):
        pred = group["gws_forecast"].to_numpy()
        real = group["gws"].to_numpy()
        rows.append(
            {
                "id": well_id,
                "horizon": int(horizon),
                "NSE": _nse(pred, real),
                "RMSE": _rmse(pred, real),
                "MAE": _mae(pred, real),
                "rMBE": _rmbe(pred, real),
            }
        )
    return pd.DataFrame(rows)


def _collect_runs() -> pd.DataFrame:
    rows = []
    if not GRU_OUTPUT_ROOT.exists():
        return pd.DataFrame()

    for run_dir in sorted(GRU_OUTPUT_ROOT.glob("GRU_FCOV_*")):
        m = RUN_RE.match(run_dir.name)
        if m is None:
            continue

        pred_path = run_dir / "predictions" / "pred.parquet"
        if not pred_path.exists():
            continue

        run_sig = run_dir.name.replace("GRU_FCOV_", "", 1)
        split_tag = _split_tag(run_sig)
        in_len = int(m.group("in_len"))
        out_len = int(m.group("out_len"))
        epochs = int(m.group("epochs"))
        seed = int(m.group("seed"))
        dataset = m.group("dataset")
        spatial_split = _is_spatial_split(run_sig)

        meta_path = run_dir / "meta.yaml"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
            if isinstance(meta, dict):
                in_len = int(meta.get("in_len", in_len))
                out_len = int(meta.get("out_len", out_len))
                epochs = int(meta.get("epochs", epochs))
                seed = int(meta.get("seed", seed))
                dataset = str(meta.get("dataset", dataset))

        pred_df = pd.read_parquet(pred_path)
        if pred_df.empty:
            continue

        metrics_well_h = _metrics_by_id_horizon(pred_df)
        metrics_well_h["seed"] = seed
        metrics_well_h["run"] = _run_label(
            in_len=in_len,
            out_len=out_len,
            epochs=epochs,
            dataset=dataset,
            spatial_split=spatial_split,
            split_tag=split_tag,
        )
        metrics_well_h["in_len"] = in_len
        metrics_well_h["out_len"] = out_len
        metrics_well_h["epochs"] = epochs
        metrics_well_h["dataset"] = dataset
        rows.append(metrics_well_h)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # Match TFT logic:
    # 1) median across wells for each seed + horizon
    # 2) median across seeds for each run + horizon
    by_seed = (
        df.groupby(["run", "seed", "horizon"], as_index=False)[["NSE", "RMSE", "MAE", "rMBE"]]
        .median()
    )
    agg = (
        by_seed.groupby(["run", "horizon"], as_index=False)[["NSE", "RMSE", "MAE", "rMBE"]]
        .median()
    )
    agg["skill_vs_persistence"] = np.nan
    return agg


def _run_sort_key(run_name: str) -> tuple:
    m = re.match(r"^gru_in(\d+)_out(\d+)_ep(\d+)_(.+)$", run_name)
    if m is None:
        return (9999, 9999, 9999, run_name)
    in_len = int(m.group(1))
    out_len = int(m.group(2))
    epochs = int(m.group(3))
    dataset = m.group(4)
    dataset_order = {"full_raw": 0, "full_merged_spatial_split": 1, "full_merged": 2, "sample": 3}
    d_ord = dataset_order.get(dataset, 9)
    return (d_ord, in_len, out_len, epochs, run_name)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw = _collect_runs()
    if raw.empty:
        print("No GRU runs with predictions found. Nothing to summarize.")
        return

    final = _aggregate(raw)
    run_order = {r: i for i, r in enumerate(sorted(final["run"].unique(), key=_run_sort_key))}
    final["run_order"] = final["run"].map(run_order)
    final = final.sort_values(["run_order", "horizon"]).drop(columns="run_order").reset_index(drop=True)

    numeric_cols = ["NSE", "RMSE", "MAE", "rMBE", "skill_vs_persistence"]
    final[numeric_cols] = final[numeric_cols].round(3)
    final.to_csv(FULL_TABLE_CSV, index=False)

    # Plot NSE by horizon for each run
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = final[~final["run"].str.contains("_sample", na=False)].copy()
    for run_label in plot_df["run"].unique():
        sub = plot_df[plot_df["run"] == run_label].sort_values("horizon")
        ax.plot(sub["horizon"], sub["NSE"], marker="o", markersize=4, label=run_label)

    # Overlay a TFT reference line (if available) for direct comparison in one figure.
    if TFT_SUMMARY_CSV.exists():
        tft = pd.read_csv(TFT_SUMMARY_CSV)
        tft_ep1 = tft[tft["run"] == "robert_ep1_full_raw"].copy()
        if not tft_ep1.empty:
            tft_ep1 = tft_ep1.sort_values("horizon")
            ax.plot(
                tft_ep1["horizon"],
                tft_ep1["NSE"],
                linestyle="--",
                linewidth=2,
                color="black",
                label="tft_robert_ep1_full_raw",
            )

    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("NSE (median across wells & seeds)")
    ax.set_title("GRU NSE by forecast horizon")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gru_nse_by_horizon.png", dpi=200)
    plt.close(fig)

    sparse_horizons = [1, 8, 16]
    sparse = final[final["horizon"].isin(sparse_horizons)].copy()
    sparse = sparse[["horizon", "run", "NSE", "RMSE"]]
    sparse["run_order"] = sparse["run"].map(run_order)
    sparse = sparse.sort_values(["horizon", "run_order"]).drop(columns="run_order")
    sparse.to_csv(SPARSE_TABLE_CSV, index=False)

    print(f"Wrote: {FULL_TABLE_CSV}")
    print(f"Wrote: {SPARSE_TABLE_CSV}")
    print(f"Wrote: {FIGURES_DIR / 'gru_nse_by_horizon.png'}")


if __name__ == "__main__":
    main()
