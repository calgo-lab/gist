from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
GRU_ROOT = ROOT / "outputs" / "GRU_FCOV"
METRICS_DIR = ROOT / "reports" / "gru" / "metrics"
FIG_DIR = ROOT / "reports" / "gru" / "figures"
TFT_SUMMARY = ROOT / "reports" / "tft" / "metrics" / "tft_metrics_summary.csv"

FULL_CSV = METRICS_DIR / "gru_metrics_summary.csv"
SPARSE_CSV = METRICS_DIR / "gru_metrics_summary_sparse.csv"

RUN_RE = re.compile(
    r"^GRU_FCOV_"
    r"in(?P<in_len>\d+)_out(?P<out_len>\d+)_ep(?P<epochs>\d+)_bs(?P<batch_size>\d+)"
    r"_seed(?P<seed>\d+)_(?P<dataset>[a-z]+(?:_[a-z]+)*)"
    r"(?:_(?P<revin>r[01]))?"
    r"(?:_(?P<scheduler>s[01]))?"
    r"(?:_spf(?P<spf>[^_]+)_sc(?P<sc>\d+)_ss(?P<ss>\d+))?$"
)


def nse(pred: np.ndarray, real: np.ndarray) -> float:
    denom = np.sum((real - np.mean(real)) ** 2)
    if denom <= 0:
        return np.nan
    return float(1 - np.sum((pred - real) ** 2) / denom)


def rmse(pred: np.ndarray, real: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - real) ** 2)))


def mae(pred: np.ndarray, real: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - real)))


def rmbe(pred: np.ndarray, real: np.ndarray) -> float:
    s = np.std(real)
    if s <= 0:
        return np.nan
    return float(np.mean(pred - real) / s)


def split_tag(run_sig: str) -> str:
    m = re.search(r"(_spf[^_]+_sc\d+_ss\d+)", run_sig)
    return m.group(1) if m else ""


def run_label(in_len: int, out_len: int, epochs: int, dataset: str, layers: int | None, run_sig: str, use_revin: bool | None, use_scheduler: bool | None) -> str:
    prefix = f"gru_l{layers}" if layers is not None else "gru"
    label = f"{prefix}_in{in_len}_out{out_len}_ep{epochs}_{dataset}"
    if use_revin is not None:
        label += "_r1" if use_revin else "_r0"
    if use_scheduler is not None:
        label += "_s1" if use_scheduler else "_s0"
    if "_spf" in run_sig and "_sc" in run_sig and "_ss" in run_sig:
        if dataset == "full_merged":
            label += "_spatial_split"
        tag = split_tag(run_sig)
        if tag:
            label += tag
    return label


def load_run_meta(run_dir: Path) -> dict | None:
    name = run_dir.name
    m = RUN_RE.match(name)
    meta = {}
    if m:
        meta = {
            "in_len": int(m.group("in_len")),
            "out_len": int(m.group("out_len")),
            "epochs": int(m.group("epochs")),
            "seed": int(m.group("seed")),
            "dataset": m.group("dataset"),
            "num_layers": None,
            "use_revin": (m.group("revin") == "r1") if m.group("revin") else None,
            "use_scheduler": (m.group("scheduler") == "s1") if m.group("scheduler") else None,
        }
    meta_path = run_dir / "meta.yaml"
    if meta_path.exists():
        y = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        if isinstance(y, dict):
            if not meta:
                req = ["in_len", "out_len", "epochs", "seed", "dataset"]
                if not all(k in y for k in req):
                    return None
            meta.update({
                "in_len": int(y.get("in_len", meta.get("in_len"))),
                "out_len": int(y.get("out_len", meta.get("out_len"))),
                "epochs": int(y.get("epochs", meta.get("epochs"))),
                "seed": int(y.get("seed", meta.get("seed"))),
                "dataset": str(y.get("dataset", meta.get("dataset"))),
                "num_layers": y.get("num_layers", meta.get("num_layers")),
                "use_revin": y.get("use_revin", meta.get("use_revin")),
                "use_scheduler": y.get("use_scheduler", meta.get("use_scheduler")),
            })
    return meta or None


def metrics_by_well_horizon(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (well_id, h), g in pred_df.groupby(["id", "horizon"]):
        p = g["gws_forecast"].to_numpy()
        r = g["gws"].to_numpy()
        rows.append({
            "id": well_id,
            "horizon": int(h),
            "NSE": nse(p, r),
            "RMSE": rmse(p, r),
            "MAE": mae(p, r),
            "rMBE": rmbe(p, r),
        })
    return pd.DataFrame(rows)


def collect() -> pd.DataFrame:
    if not GRU_ROOT.exists():
        return pd.DataFrame()

    parts = []
    for run_dir in sorted(GRU_ROOT.glob("GRU_FCOV_*")):
        pred_path = run_dir / "predictions" / "pred.parquet"
        if not pred_path.exists():
            continue

        meta = load_run_meta(run_dir)
        if not meta:
            continue

        pred_df = pd.read_parquet(pred_path)
        if pred_df.empty:
            continue

        run_sig = run_dir.name.replace("GRU_FCOV_", "", 1)
        label = run_label(
            in_len=meta["in_len"],
            out_len=meta["out_len"],
            epochs=meta["epochs"],
            dataset=meta["dataset"],
            layers=meta.get("num_layers"),
            run_sig=run_sig,
            use_revin=meta.get("use_revin"),
            use_scheduler=meta.get("use_scheduler"),
        )

        m = metrics_by_well_horizon(pred_df)
        m["seed"] = meta["seed"]
        m["run"] = label
        parts.append(m)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    by_seed = (
        raw.groupby(["run", "seed", "horizon"], as_index=False)[["NSE", "RMSE", "MAE", "rMBE"]]
        .median()
    )
    out = (
        by_seed.groupby(["run", "horizon"], as_index=False)[["NSE", "RMSE", "MAE", "rMBE"]]
        .median()
    )
    out["skill_vs_persistence"] = np.nan
    return out


def sort_key(run_name: str) -> tuple:
    m = re.match(r"^gru(?:_l(\d+))?_in(\d+)_out(\d+)_ep(\d+)_(.+)$", run_name)
    if m is None:
        return (9999, 9999, 9999, 9999, 9999, run_name)
    layers = int(m.group(1)) if m.group(1) else 0
    in_len = int(m.group(2))
    out_len = int(m.group(3))
    epochs = int(m.group(4))
    dataset = m.group(5)
    order = {"full_raw": 0, "full_merged_spatial_split": 1, "full_merged": 2, "sample": 3}
    return (order.get(dataset, 9), in_len, out_len, epochs, layers, run_name)


def legend_label(run_name: str) -> str:
    m = re.match(r"^gru(?:_l(\d+))?_in\d+_out\d+_ep\d+_.+$", run_name)
    if m and m.group(1):
        base = f"gru {int(m.group(1))}-layer"
    else:
        base = "gru"

    tags = []
    if "_r1" in run_name:
        tags.append("revin")
    elif "_r0" in run_name:
        tags.append("no-revin")

    if "_s0" in run_name:
        tags.append("no-sched")

    if tags:
        return f"{base} " + " ".join(tags)
    return base


def plot_nse(final: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = final[
        final["run"].str.contains("_ep50_", na=False)
        & final["run"].str.contains("spf0p8", na=False)
    ].copy()
    for run_label in plot_df["run"].unique():
        sub = plot_df[plot_df["run"] == run_label].sort_values("horizon")
        ax.plot(sub["horizon"], sub["NSE"], marker="o", markersize=4, label=legend_label(run_label))

    # Overlay a TFT reference line (if available) for direct comparison in one figure.
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
    ax.set_title("GRU NSE by forecast horizon")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gru_nse_by_horizon.png", dpi=200)
    plt.close(fig)

    sparse_horizons = [1, 8, 16]
    sparse = final[final["horizon"].isin(sparse_horizons)].copy()
    sparse = sparse[["horizon", "run", "NSE", "RMSE"]]
    sparse["run_order"] = sparse["run"].map(sort_key)
    sparse = sparse.sort_values(["horizon", "run_order"]).drop(columns="run_order")
    sparse.to_csv(SPARSE_CSV, index=False)

    print(f"Wrote: {FULL_CSV}")
    print(f"Wrote: {SPARSE_CSV}")
    print(f"Wrote: {FIG_DIR / 'gru_nse_by_horizon.png'}")


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = collect()
    if raw.empty:
        print("No GRU prediction files found. Nothing to summarize.")
        return

    final = aggregate(raw)
    final["run_order"] = final["run"].map(sort_key)
    final = final.sort_values(["horizon", "run_order"]).drop(columns="run_order")
    final.to_csv(FULL_CSV, index=False)

    plot_nse(final)


if __name__ == "__main__":
    main()
    
