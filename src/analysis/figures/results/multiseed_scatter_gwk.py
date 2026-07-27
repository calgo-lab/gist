import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT      = Path(__file__).resolve().parents[4]
DATA_PATH = ROOT.parent / "data" / "merged.parquet"
GP_DIR    = ROOT / "outputs" / "gp"
GRU_DIR   = ROOT / "outputs" / "GRU_FCOV"
OUT_DIR   = ROOT / "reports" / "figures" / "_dead_ends"
VAL_CUTOFF = pd.Timestamp("2020-01-01")

RE_GP = re.compile(
    r"in\d+_out\d+_ep\d+_bs\d+_seed(?P<ms>\d+)_coloc"
    r"_(?P<split>rand|km|md\d+)"
    r"_f(?P<frac>\d+)"
    r"_ss(?P<ss>\d+)"
)
RE_GRU_GLOBAL = re.compile(r"gru_1040all_ep50_bs\d+_seed(?P<seed>\d+)$")

SPLITS = [
    ("md30",  95, "MaxDist-30 f=0.95"),
    ("rand",  95, "Random f=0.95"),
    ("km",    95, "KMeans f=0.95"),
]

GWK_CLASSES   = [1, 2, 3, 4, 5]
GWK_LABELS    = ["very low", "low", "normal", "high", "very high"]
GWK_QUANTILES = [0.10, 0.30, 0.70, 0.90]



def load_gp_preds(split: str, frac: int) -> pd.DataFrame:
    rows = []
    for gp_dir in GP_DIR.iterdir():
        if not (gp_dir / "gp_pred.parquet").exists():
            continue
        run_name = gp_dir.name
        if not run_name.startswith("GRU_FCOV_"):
            continue
        run_sig = run_name[len("GRU_FCOV_"):].split("__")[0]
        m = RE_GP.search(run_sig)
        if not m or m["split"] != split or int(m["frac"]) != frac:
            continue
        print(f"  loading: {run_name}")
        try:
            df = pq.read_table(
                gp_dir / "gp_pred.parquet",
                columns=["id", "datum", "horizon", "gws_true", "gws_forecast"],
            ).to_pandas()
            df["datum"] = pd.to_datetime(df["datum"])
            rows.append(df)
        except Exception as e:
            print(f"  SKIP {run_name}: {e}")
    if not rows:
        raise RuntimeError(f"No GP preds found for {split}_f{frac}")
    return pd.concat(rows, ignore_index=True)


def load_gru_global_preds() -> pd.DataFrame:
    rows = []
    seen_seeds = set()
    for run_dir in sorted(GRU_DIR.iterdir()):
        meta = run_dir / "meta.yaml"
        pred = run_dir / "predictions" / "pred.parquet"
        if not meta.exists() or not pred.exists():
            continue
        txt = meta.read_text()
        m = None
        for line in txt.splitlines():
            if line.strip().startswith("run_sig:"):
                sig = line.split(":", 1)[1].strip().strip("'\"")
                m = RE_GRU_GLOBAL.match(sig)
                break
        if not m:
            continue
        seed = int(m["seed"])
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)
        try:
            df = pq.read_table(
                pred,
                columns=["id", "datum", "horizon", "gws", "gws_forecast"],
            ).to_pandas()
            df["datum"] = pd.to_datetime(df["datum"])
            df = df.rename(columns={"gws": "gws_true"})
            rows.append(df)
        except Exception as e:
            print(f"  SKIP {run_dir.name}: {e}")
    if not rows:
        raise RuntimeError("No global GRU preds found")
    print(f"  Loaded {len(seen_seeds)} GRU global seeds: {sorted(seen_seeds)}")
    return pd.concat(rows, ignore_index=True)



def per_well_metrics(preds: pd.DataFrame, forecast_col: str = "gws_forecast",
                     true_col: str = "gws_true",
                     suffix: str = "") -> pd.DataFrame:
    result = []
    for well_id, g in preds.groupby("id"):
        t = g[true_col].to_numpy(float)
        p = g[forecast_col].to_numpy(float)
        mask = np.isfinite(t) & np.isfinite(p)
        t, p = t[mask], p[mask]
        if len(t) == 0:
            continue
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        denom = float(np.sum((t - t.mean()) ** 2))
        nse  = float(1.0 - np.sum((p - t) ** 2) / denom) if denom > 0 else np.nan
        result.append({"id": well_id, f"rmse{suffix}": rmse, f"nse{suffix}": nse})
    return pd.DataFrame(result)


def gwk_thresholds(full_gws: pd.DataFrame) -> dict:
    thresholds = {}
    for well_id, g in full_gws.groupby("id"):
        vals = g["gws"].dropna().to_numpy(float)
        if len(vals) < 20:
            continue
        thresholds[well_id] = np.quantile(vals, GWK_QUANTILES)
    return thresholds


def classify_gwk(values: np.ndarray, thresh: np.ndarray) -> np.ndarray:
    classes = np.ones(len(values), dtype=int)
    classes[values >= thresh[0]] = 2
    classes[values >= thresh[1]] = 3
    classes[values >= thresh[2]] = 4
    classes[values >= thresh[3]] = 5
    return classes


def gwk_accuracy(gp_preds: pd.DataFrame, gru_preds: pd.DataFrame,
                 thresholds: dict, well_ids: set) -> dict:
    KEY = ["id", "datum", "horizon"]

    gp_avg = (
        gp_preds[gp_preds["id"].isin(well_ids)]
        .groupby(KEY)[["gws_true", "gws_forecast"]]
        .mean()
        .reset_index()
        .rename(columns={"gws_forecast": "gws_forecast_gp"})
    )
    gru_avg = (
        gru_preds[gru_preds["id"].isin(well_ids)]
        .groupby(KEY)[["gws_forecast"]]
        .mean()
        .reset_index()
        .rename(columns={"gws_forecast": "gws_forecast_gru"})
    )

    both = gp_avg.merge(gru_avg, on=KEY)
    n_obs = len(both)
    print(f"  GWK eval: {both['id'].nunique()} wells, {n_obs:,} aligned (id,datum,horizon) tuples")

    def _acc(forecast_col):
        accs = {}
        for c in GWK_CLASSES:
            true_c, pred_c, total = 0, 0, 0
            for well_id, g in both.groupby("id"):
                if well_id not in thresholds:
                    continue
                thresh = thresholds[well_id]
                t = g["gws_true"].to_numpy(float)
                p = g[forecast_col].to_numpy(float)
                mask = np.isfinite(t) & np.isfinite(p)
                t, p = t[mask], p[mask]
                if len(t) == 0:
                    continue
                t_cls = classify_gwk(t, thresh)
                p_cls = classify_gwk(p, thresh)
                mask_c = t_cls == c
                total  += int(mask_c.sum())
                pred_c += int((p_cls[mask_c] == c).sum())
            accs[c] = {
                "total":   total,
                "correct": pred_c,
                "acc":     pred_c / total if total > 0 else np.nan,
            }
        return accs

    return {
        "gp":  _acc("gws_forecast_gp"),
        "gru": _acc("gws_forecast_gru"),
    }



def scatter_plot(merged: pd.DataFrame, metric_gp: str, metric_gru: str,
                 xlabel: str, window: tuple, title: str, out_path: Path,
                 median_gwl: dict):
    fig, ax = plt.subplots(figsize=(7, 7))
    lo, hi = window

    gp_vals  = merged[metric_gp].to_numpy(float)
    gru_vals = merged[metric_gru].to_numpy(float)
    colors   = [median_gwl.get(wid, np.nan) for wid in merged["id"]]

    gp_plot  = np.clip(gp_vals,  lo, hi)
    gru_plot = np.clip(gru_vals, lo, hi)

    sc = ax.scatter(gp_plot, gru_plot, c=colors, cmap="RdYlGn_r",
                    vmin=0, vmax=80, s=60, alpha=0.85, zorder=3)
    plt.colorbar(sc, ax=ax, label="Median GWL [m a.s.l.]")

    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Decoupled GRU+GP — per-well {xlabel}", fontsize=11)
    ax.set_ylabel(f"GRU temporal only — per-well {xlabel}", fontsize=11)

    n = len(merged)
    gp_better = int(np.sum(gp_vals < gru_vals) if "rmse" in metric_gp.lower()
                    else np.sum(gp_vals > gru_vals))
    clipped = int(np.sum((gp_vals < lo) | (gp_vals > hi) |
                         (gru_vals < lo) | (gru_vals > hi)))
    info = f"n = {n}   GP better: {gp_better} / {n}\n{clipped} points clipped"
    ax.text(0.03, 0.97, info, transform=ax.transAxes, va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path.name}")


def gwk_count_plot(accs: dict, title: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    colors = {"gp": "#4C72B0", "gru": "#DD8452"}
    labels_map = {"gp": "Decoupled GP pipeline", "gru": "GRU temporal only"}

    for ax, (model, color) in zip(axes, colors.items()):
        totals   = [accs[model][c]["total"]   for c in GWK_CLASSES]
        corrects = [accs[model][c]["correct"] for c in GWK_CLASSES]
        x = np.arange(len(GWK_CLASSES))

        ax.bar(x, totals,   color=color, alpha=0.25, label="Incorrectly classified")
        ax.bar(x, corrects, color=color, alpha=1.0,  label="Correctly classified")
        for xi, (tot, cor) in enumerate(zip(totals, corrects)):
            ax.text(xi, tot + 5, str(tot), ha="center", va="bottom", fontsize=9, color="gray")
            if cor > 0:
                ax.text(xi, cor / 2, str(cor), ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(GWK_LABELS, fontsize=9)
        ax.set_xlabel("GWK class", fontsize=10)
        ax.set_ylabel("Number of observations", fontsize=10)
        ax.set_title(f"{labels_map[model]}\nAbsolute GWK classification counts", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path.name}")


def gwk_accuracy_plot(accs: dict, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(GWK_CLASSES))
    width = 0.35
    colors = {"gp": "#4C72B0", "gru": "#DD8452"}
    labels_map = {"gp": "Decoupled GP pipeline", "gru": "GRU temporal only"}

    for i, (model, color) in enumerate(colors.items()):
        acc_vals = [accs[model][c]["acc"] * 100 for c in GWK_CLASSES]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, acc_vals, width, color=color,
                      label=labels_map[model], alpha=0.85)
        for bar, val in zip(bars, acc_vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color=color)

    ax.axhline(20, color="gray", ls="--", lw=1, label="Random baseline (20%)")
    ax.set_xticks(x)
    ax.set_xticklabels(GWK_LABELS, fontsize=10)
    ax.set_xlabel("GWK class", fontsize=11)
    ax.set_ylabel("Classification accuracy (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path.name}")



def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GRU global predictions (10 seeds, all 1040 wells) ...")
    gru_global = load_gru_global_preds()
    gru_global = gru_global[gru_global["datum"] > VAL_CUTOFF]
    print(f"  {gru_global['id'].nunique()} wells, {len(gru_global):,} rows")

    print("Loading full-history GWL for GWK thresholds ...")
    full_gws = pq.read_table(DATA_PATH, columns=["id", "gws"]).to_pandas().dropna(subset=["gws"])
    thresholds = gwk_thresholds(full_gws)
    median_gwl = full_gws.groupby("id")["gws"].median().to_dict()
    print(f"  {len(thresholds)} wells with GWK thresholds")

    for split, frac, label in SPLITS:
        tag = f"{split}_f{frac}"
        print(f"\n=== {label} ({tag}) ===")

        print("  Loading GP predictions ...")
        gp_preds = load_gp_preds(split, frac)
        gp_preds = gp_preds[gp_preds["datum"] > VAL_CUTOFF]
        holdout_wells = set(gp_preds["id"].unique())
        print(f"  {len(holdout_wells)} unique holdout wells, {len(gp_preds):,} rows")

        gru_holdout = gru_global[gru_global["id"].isin(holdout_wells)]
        print(f"  GRU global: {gru_holdout['id'].nunique()} holdout wells covered")

        print("  Computing per-well metrics ...")
        gp_m  = per_well_metrics(gp_preds, suffix="_gp")
        gru_m = per_well_metrics(gru_holdout, suffix="_gru")
        scatter_df = gp_m.merge(gru_m, on="id").dropna()
        print(f"  {len(scatter_df)} wells with both GP and GRU metrics")

        scatter_plot(
            scatter_df, "rmse_gp", "rmse_gru", "mean RMSE [m]",
            window=(0.0, 0.6),
            title=f"RMSE — clipped to [0, 0.6] m\n({len(scatter_df)} holdout wells, all horizons)  {label}",
            out_path=OUT_DIR / f"scatter_rmse_{tag}.png",
            median_gwl=median_gwl,
        )

        scatter_plot(
            scatter_df, "nse_gp", "nse_gru", "mean NSE",
            window=(-0.5, 1.0),
            title=f"NSE — clipped to [-0.5, 1.0]\n({len(scatter_df)} holdout wells, all horizons)  {label}",
            out_path=OUT_DIR / f"scatter_nse_{tag}.png",
            median_gwl=median_gwl,
        )

        print("  Computing GWK classification ...")
        gru_holdout_full = gru_global[gru_global["id"].isin(holdout_wells)].copy()
        accs = gwk_accuracy(gp_preds, gru_holdout_full, thresholds, holdout_wells)

        gwk_count_plot(
            accs,
            title=f"GWK classification: absolute counts — {label}",
            out_path=OUT_DIR / f"gwk_counts_{tag}.png",
        )
        gwk_accuracy_plot(
            accs,
            title=f"GWK per-class accuracy: Decoupled GP vs GRU temporal only\n{label}",
            out_path=OUT_DIR / f"gwk_accuracy_{tag}.png",
        )

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
