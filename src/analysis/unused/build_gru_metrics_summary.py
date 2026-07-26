from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
GRU_ROOT = ROOT / "outputs" / "GRU_FCOV"
METRICS_DIR = ROOT / "reports" / "metrics" / "gru"
FIG_DIR = ROOT / "reports" / "figures" / "gru"
TFT_SUMMARY = ROOT / "reports" / "metrics" / "tft" / "tft_metrics_summary.csv"

FULL_CSV = METRICS_DIR / "gru_metrics_summary_long.csv"
SPARSE_CSV = METRICS_DIR / "gru_metrics_summary_sparse.csv"
HPO_SPARSE_CSV = METRICS_DIR / "gru_metrics_summary_hpo_sparse.csv"
HPO_BASELINE_TABLE_CSV = METRICS_DIR / "gru_hpo_vs_baseline_summary.csv"
BASELINE_RUN_SIG = "gru_l2_seed40_ep50_full_merged_spf0p8_sc20_ss42"

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
        m["run_sig"] = run_sig
        m["seed"] = meta["seed"]
        m["run"] = label
        parts.append(m)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    out = (
        raw.groupby(["run_sig", "run", "seed", "horizon"], as_index=False)[["NSE", "RMSE", "MAE", "rMBE"]]
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


def legend_label(run_name: str, run_sig: str | None = None) -> str:
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
        base = f"{base} " + " ".join(tags)

    if run_sig:
        tm = re.search(r"_t(\d{3})_", run_sig)
        if tm:
            base = f"{base} t{tm.group(1)}"
    return base


def plot_nse(final: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    hpo_df = final[final["run_sig"].str.contains(r"_t\d{3}_", regex=True, na=False)].copy()
    baseline_df = final[final["run_sig"] == BASELINE_RUN_SIG].copy()

    if not hpo_df.empty:
        hpo_rank = (
            hpo_df.groupby(["run_sig", "run"], as_index=False)["NSE"]
            .mean()
            .rename(columns={"NSE": "NSE_mean_h1_16"})
            .sort_values("NSE_mean_h1_16", ascending=False)
        )
        top5 = hpo_rank.head(5)["run_sig"].tolist()
    else:
        top5 = []

    chosen = set(top5)
    if not baseline_df.empty:
        chosen.add(BASELINE_RUN_SIG)

    plot_df = final[final["run_sig"].isin(chosen)].copy()
    for run_sig in sorted(plot_df["run_sig"].unique()):
        sub = plot_df[plot_df["run_sig"] == run_sig].sort_values("horizon")
        if run_sig == BASELINE_RUN_SIG:
            lbl = "gru 2-layer baseline"
        else:
            tm = re.search(r"_t(\d{3})_", str(run_sig))
            lbl = f"gru hpo t{tm.group(1)}" if tm else legend_label(str(sub["run"].iloc[0]), run_sig)
        ax.plot(sub["horizon"], sub["NSE"], marker="o", markersize=4, label=lbl)

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
    ax.set_title("GRU NSE by Forecast Horizon (Baseline + Top5 HPO)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gru_nse_by_horizon.png", dpi=200)
    plt.close(fig)

    sparse_horizons = [1, 8, 16]
    sparse = final[final["horizon"].isin(sparse_horizons)].copy()
    sparse = sparse[["horizon", "run", "run_sig", "NSE", "RMSE"]]
    sparse["run_order"] = sparse["run"].map(sort_key)
    sparse = sparse.sort_values(["horizon", "run_order"]).drop(columns="run_order")
    sparse.to_csv(SPARSE_CSV, index=False)

    print(f"Wrote: {FULL_CSV}")
    print(f"Wrote: {SPARSE_CSV}")
    print(f"Wrote: {FIG_DIR / 'gru_nse_by_horizon.png'}")


def plot_nse_hpo(final: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    hpo_df = final[final["run_sig"].str.contains(r"_t\d{3}_", regex=True, na=False)].copy()
    if hpo_df.empty:
        print("No HPO runs found for HPO-only figure.")
        return

    for run_sig in sorted(hpo_df["run_sig"].unique()):
        sub = hpo_df[hpo_df["run_sig"] == run_sig].sort_values("horizon")
        tm = re.search(r"_t(\d{3})_", str(run_sig))
        lbl = f"trial t{tm.group(1)}" if tm else str(run_sig)
        ax.plot(sub["horizon"], sub["NSE"], marker="o", markersize=3, label=lbl)

    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("NSE (median across wells)")
    ax.set_title("GRU HPO Trials NSE by Forecast Horizon")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_fig = FIG_DIR / "gru_hpo_nse_by_horizon.png"
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    sparse_horizons = [1, 8, 16]
    sparse = hpo_df[hpo_df["horizon"].isin(sparse_horizons)][["horizon", "run", "run_sig", "NSE", "RMSE"]].copy()
    sparse.to_csv(HPO_SPARSE_CSV, index=False)
    print(f"Wrote: {HPO_SPARSE_CSV}")
    print(f"Wrote: {out_fig}")


def write_hpo_baseline_table(final: pd.DataFrame) -> None:
    hpo_df = final[final["run_sig"].str.contains(r"_t\d{3}_", regex=True, na=False)].copy()
    baseline_df = final[final["run_sig"] == BASELINE_RUN_SIG].copy()
    comp = pd.concat([hpo_df, baseline_df], ignore_index=True)
    if comp.empty:
        print("No HPO/baseline rows found for comparison table.")
        return

    by_run = (
        comp.groupby(["run_sig", "run"], as_index=False)
        .agg(NSE_mean_h1_16=("NSE", "mean"), RMSE_mean_h1_16=("RMSE", "mean"))
    )
    h16 = comp[comp["horizon"] == 16][["run_sig", "NSE", "RMSE"]].rename(
        columns={"NSE": "NSE_h16", "RMSE": "RMSE_h16"}
    )
    out = by_run.merge(h16, on="run_sig", how="left")
    out["is_baseline"] = out["run_sig"] == BASELINE_RUN_SIG
    out["is_hpo"] = out["run_sig"].str.contains(r"_t\d{3}_", regex=True, na=False)
    out = out.sort_values(["is_baseline", "NSE_mean_h1_16"], ascending=[False, False])
    out.to_csv(HPO_BASELINE_TABLE_CSV, index=False)
    print(f"Wrote: {HPO_BASELINE_TABLE_CSV}")


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
    plot_nse_hpo(final)
    write_hpo_baseline_table(final)


if __name__ == "__main__":
    main()
    
