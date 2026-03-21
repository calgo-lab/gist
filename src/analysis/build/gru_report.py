"""
gru_report.py

Builds reports/gru/metrics/gru_metrics_summary.csv and all GRU figures
from outputs/GRU_FCOV/.

Run after training completes:
    python src/analysis/build/gru_report.py

Outputs
-------
reports/gru/metrics/gru_metrics_summary.csv
    One row per run. Columns: run_number, hpo_name, trial, objective, train_s,
    trained_epochs, NSE/RMSE/MAE × {weighted_mean, mean, weighted_mean_h13,
    mean_h13, h16}, layers, hidden, dropout, lr, batch_size, revin,
    lr_scheduler, run_sig.

reports/gru/figures/gru_top5_nse_vs_tft.png
reports/gru/figures/gru_top5_rmse_vs_tft.png
    Top 5 runs by weighted mean NSE/RMSE (h1–16) vs best TFT run.

reports/gru/figures/gru_best_nse_vs_tft.png
reports/gru/figures/gru_best_rmse_vs_tft.png
    Best run by weighted h1–16 + best run by weighted h13–16 + TFT.

reports/gru/figures/config_seed_comparison.png
    Performance vs runtime for hardcoded config groups across seeds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from libs.run_registry import read_registry

GRU_ROOT = ROOT / "outputs" / "GRU_FCOV"
HPO_DIR = ROOT / "reports" / "gru" / "hpo"
METRICS_DIR = ROOT / "reports" / "gru" / "metrics"
FIG_DIR = ROOT / "reports" / "gru" / "figures"
TFT_SUMMARY = ROOT / "reports" / "tft" / "metrics" / "tft_metrics_summary.csv"
OUT_CSV = METRICS_DIR / "gru_metrics_summary.csv"

WEIGHTS_H1_16 = np.linspace(1.0, 2.0, 16)
WEIGHTS_H13_16 = np.linspace(1.0, 2.0, 4)

META_TO_HP: dict[str, str] = {
    "num_layers": "layers",
    "hidden_size": "hidden",
    "dropout": "dropout",
    "lr": "lr",
    "batch_size": "batch_size",
    "use_revin": "revin",
    "use_scheduler": "lr_scheduler",
}

METRIC_RENAME: dict[str, str] = {}
for _m in ("NSE", "RMSE", "MAE"):
    METRIC_RENAME[f"{_m}_weighted_mean_h1_16"] = f"{_m}_weighted_mean"
    METRIC_RENAME[f"{_m}_mean_h1_16"] = f"{_m}_mean"
    METRIC_RENAME[f"{_m}_weighted_mean_h13_16"] = f"{_m}_weighted_mean_h13"
    METRIC_RENAME[f"{_m}_mean_h13_16"] = f"{_m}_mean_h13"

HP_RENAME: dict[str, str] = {
    "hp.model.gru_layers": "layers",
    "hp.model.gru_hidden": "hidden",
    "hp.model.gru_dropout": "dropout",
    "hp.training.lr": "lr",
    "hp.training.batch_size": "batch_size",
    "hp.model.use_revin": "revin",
    "hp.training.lr_scheduler.enabled": "lr_scheduler",
}

SEED_COMPARISON_CONFIGS: dict[str, list[str]] = {
    "h192, l4, lr=4e-4": [
        "gru_small_grid_v2_t012_seed40_full_merged",
        "gru_l4_h192_bs4096_seed41_ep50",
        "gru_l4_h192_bs4096_seed42_ep50",
    ],
    "h256, l4, lr=2e-4": [
        "gru_small_grid_v2_t005_seed40_full_merged",
        "gru_l4_h256_bs4096_seed41_ep50",
        "gru_l4_h256_bs4096_seed42_ep50",
    ],
}


def _nse(p, r):
    d = np.sum((r - r.mean()) ** 2)
    return float(1 - np.sum((p - r) ** 2) / d) if d > 0 else np.nan


def _rmse(p, r):
    return float(np.sqrt(np.mean((p - r) ** 2)))


def _mae(p, r):
    return float(np.mean(np.abs(p - r)))


def _wmean(vals, weights):
    mask = ~np.isnan(vals)
    if mask.sum() < len(weights):
        return np.nan
    return float(np.average(vals[mask], weights=weights[mask]))


def load_hpo_meta():
    parts = []
    for p in sorted(HPO_DIR.glob("*_results.csv")):
        df = pd.read_csv(p)
        if "run_sig" not in df.columns:
            continue
        df = df.copy()
        df["hpo_name"] = p.stem.replace("_results", "")
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    return combined.drop_duplicates(subset=["run_sig"], keep="first")


def _hp_from_meta(meta):
    return {short: meta[meta_key] for meta_key, short in META_TO_HP.items() if meta_key in meta}


def _horizon_medians(pred_df):
    rows = []
    for (_, h), g in pred_df.groupby(["id", "horizon"]):
        p = g["gws_forecast"].to_numpy()
        r = g["gws"].to_numpy()
        rows.append({"horizon": int(h), "NSE": _nse(p, r), "RMSE": _rmse(p, r), "MAE": _mae(p, r)})
    return pd.DataFrame(rows).groupby("horizon", as_index=False)[["NSE", "RMSE", "MAE"]].median()


def _summary_row(hz, run_sig, trained_epochs):
    hz = hz.sort_values("horizon")
    h1_16 = hz[hz["horizon"].between(1, 16)]
    h13_16 = hz[hz["horizon"].between(13, 16)]
    h16 = hz[hz["horizon"] == 16]

    row = {"run_sig": run_sig, "trained_epochs": trained_epochs}
    for col in ("NSE", "RMSE", "MAE"):
        v1_16 = h1_16[col].to_numpy(dtype=float)
        v13_16 = h13_16[col].to_numpy(dtype=float)
        row[f"{col}_weighted_mean_h1_16"] = _wmean(v1_16, WEIGHTS_H1_16) if len(v1_16) == 16 else np.nan
        row[f"{col}_mean_h1_16"] = float(np.nanmean(v1_16)) if len(v1_16) > 0 else np.nan
        row[f"{col}_weighted_mean_h13_16"] = _wmean(v13_16, WEIGHTS_H13_16) if len(v13_16) == 4 else np.nan
        row[f"{col}_mean_h13_16"] = float(np.nanmean(v13_16)) if len(v13_16) > 0 else np.nan
        row[f"{col}_h16"] = float(h16[col].iloc[0]) if not h16.empty else np.nan
    return row


def collect_runs():
    summary_rows = []
    horizon_parts = []
    meta_hp_rows = []

    registry = read_registry(ROOT / "outputs")
    reg_by_id = {}
    if not registry.empty and "run_id" in registry.columns and "model_type" in registry.columns:
        for _, row in registry[registry["model_type"] == "GRU_FCOV"].iterrows():
            reg_by_id[f"{int(row['run_id']):04d}"] = row.to_dict()

    for run_dir in sorted(GRU_ROOT.glob("GRU_FCOV_*")):
        pred_path = run_dir / "predictions" / "pred.parquet"
        if not pred_path.exists():
            continue

        pred_df = pd.read_parquet(pred_path)
        if pred_df.empty or not {"id", "horizon", "gws_forecast", "gws"}.issubset(pred_df.columns):
            continue

        dir_suffix = run_dir.name.removeprefix("GRU_FCOV_")
        if dir_suffix.isdigit() and len(dir_suffix) == 4 and dir_suffix in reg_by_id:
            run_sig = reg_by_id[dir_suffix].get("run_sig", dir_suffix)
            run_id_label = dir_suffix
        else:
            run_sig = dir_suffix
            run_id_label = None

        meta = {}
        trained_epochs = None
        meta_path = run_dir / "meta.yaml"
        if meta_path.exists():
            loaded = yaml.safe_load(meta_path.read_text()) or {}
            if isinstance(loaded, dict):
                meta = loaded
                trained_epochs = meta.get("trained_epochs")

        hz = _horizon_medians(pred_df)
        if hz.empty:
            continue

        hz["run_sig"] = run_sig
        horizon_parts.append(hz)
        row_summary = _summary_row(hz, run_sig, trained_epochs)
        if run_id_label is not None:
            row_summary["run_id"] = run_id_label
        summary_rows.append(row_summary)

        hp = _hp_from_meta(meta)
        if run_id_label is not None and run_id_label in reg_by_id:
            reg = reg_by_id[run_id_label]
            for src, dst in [("hidden_size", "hidden"), ("num_layers", "layers"),
                              ("dropout", "dropout"), ("lr", "lr")]:
                if src in reg and not hp.get(dst):
                    hp[dst] = reg[src]
        if hp:
            meta_hp_rows.append({"run_sig": run_sig, **hp})

    summary_df = pd.DataFrame(summary_rows)
    horizon_df = pd.concat(horizon_parts, ignore_index=True) if horizon_parts else pd.DataFrame()
    meta_hp_df = pd.DataFrame(meta_hp_rows) if meta_hp_rows else pd.DataFrame(columns=["run_sig"])
    return summary_df, horizon_df, meta_hp_df


def build_metrics_table(summary_df, hpo_df, meta_hp_df):
    if summary_df.empty:
        return summary_df

    hpo_extra = ["trial", "objective", "train_s", "hpo_name"]
    hpo_hp_cols = [c for c in hpo_df.columns if c.startswith("hp.")] if not hpo_df.empty else []
    hpo_keep = ["run_sig"] + [c for c in hpo_extra + hpo_hp_cols if c in (hpo_df.columns if not hpo_df.empty else [])]

    out = summary_df.copy()

    if not hpo_df.empty:
        out = out.merge(hpo_df[hpo_keep].drop_duplicates("run_sig"), on="run_sig", how="left")

    out = out.rename(columns={k: v for k, v in HP_RENAME.items() if k in out.columns})
    out = out.rename(columns={k: v for k, v in METRIC_RENAME.items() if k in out.columns})

    if not meta_hp_df.empty:
        meta_hp_cols = [c for c in meta_hp_df.columns if c != "run_sig"]
        out = out.merge(meta_hp_df.drop_duplicates("run_sig"), on="run_sig", how="left", suffixes=("", "_meta"))
        for col in meta_hp_cols:
            if col in out.columns and f"{col}_meta" in out.columns:
                out[col] = out[col].combine_first(out.pop(f"{col}_meta"))
            elif f"{col}_meta" in out.columns:
                out = out.rename(columns={f"{col}_meta": col})

    if "objective" not in out.columns:
        out["objective"] = np.nan
    if "RMSE_weighted_mean" in out.columns:
        out["objective"] = out["objective"].combine_first(out["RMSE_weighted_mean"])

    for col in ("revin", "lr_scheduler"):
        if col in out.columns:
            out[col] = out[col].fillna(False)

    out = out.sort_values("NSE_weighted_mean", ascending=False).reset_index(drop=True)
    out.insert(0, "run_number", range(1, len(out) + 1))

    leading = ["run_number", "objective", "train_s", "trained_epochs"]
    metric_cols = []
    for m in ("NSE", "RMSE", "MAE"):
        metric_cols += [
            f"{m}_weighted_mean", f"{m}_mean",
            f"{m}_weighted_mean_h13", f"{m}_mean_h13",
            f"{m}_h16",
        ]
    hp_keys = ["layers", "hidden", "dropout", "lr", "batch_size", "revin", "lr_scheduler"]
    extra_hp = sorted(c for c in out.columns if c.startswith("hp."))
    trailing = ["hpo_name", "trial", "run_sig"]
    ordered = (
        [c for c in leading if c in out.columns]
        + [c for c in metric_cols if c in out.columns]
        + [c for c in hp_keys if c in out.columns]
        + extra_hp
        + [c for c in trailing if c in out.columns]
    )
    remaining = [c for c in out.columns if c not in ordered]
    return out[ordered + remaining]


def load_tft_best():
    if not TFT_SUMMARY.exists():
        return None
    tft = pd.read_csv(TFT_SUMMARY)
    if tft.empty or "NSE" not in tft.columns or "run" not in tft.columns:
        return None
    best_run = tft.groupby("run")["NSE"].mean().idxmax()
    return tft[tft["run"] == best_run].sort_values("horizon")


def _overlay_tft(ax, tft_df, metric):
    if tft_df is None or metric not in tft_df.columns:
        return
    ax.plot(
        tft_df["horizon"], tft_df[metric],
        linestyle="--", linewidth=2, color="black", label="TFT (best run)",
    )


def _save(ax, fig, ylabel, title, out):
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out}")


def _run_label(row):
    parts = []
    if pd.notna(row.get("layers")):
        parts.append(f"L{int(row['layers'])}")
    if pd.notna(row.get("hidden")):
        parts.append(f"H{int(row['hidden'])}")
    if pd.notna(row.get("lr")):
        parts.append(f"lr={float(row['lr']):g}")
    if pd.notna(row.get("dropout")):
        parts.append(f"do={float(row['dropout']):g}")
    if pd.notna(row.get("batch_size")):
        parts.append(f"bs={int(row['batch_size'])}")
    return " ".join(parts) if parts else row.get("run_sig", "?")[:40]


def plot_top5(horizon_df, table, tft_df, metric):
    rank_col = f"{metric}_weighted_mean"
    if rank_col not in table.columns or horizon_df.empty:
        return
    ascending = metric != "NSE"
    top5 = table.nsmallest(5, rank_col) if ascending else table.nlargest(5, rank_col)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for _, row in top5.iterrows():
        run_sig = row["run_sig"]
        sub = horizon_df[horizon_df["run_sig"] == run_sig].sort_values("horizon")
        ax.plot(sub["horizon"], sub[metric], marker="o", markersize=3.5, label=_run_label(row))
    _overlay_tft(ax, tft_df, metric)
    _save(ax, fig,
          ylabel=f"{metric} (median across wells)",
          title=f"Top 5 GRU runs vs TFT — {metric}",
          out=FIG_DIR / f"gru_top5_{metric.lower()}_vs_tft.png")


def plot_best_highlight(horizon_df, table, tft_df, metric):
    col_h1_16 = f"{metric}_weighted_mean"
    col_h13_16 = f"{metric}_weighted_mean_h13"
    if col_h1_16 not in table.columns or col_h13_16 not in table.columns or horizon_df.empty:
        return

    ascending = metric != "NSE"
    idx_h1_16 = table[col_h1_16].idxmin() if ascending else table[col_h1_16].idxmax()
    idx_h13_16 = table[col_h13_16].idxmin() if ascending else table[col_h13_16].idxmax()

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, suffix in [(idx_h1_16, "best h1–16"), (idx_h13_16, "best h13–16")]:
        row = table.loc[idx]
        sub = horizon_df[horizon_df["run_sig"] == row["run_sig"]].sort_values("horizon")
        ax.plot(sub["horizon"], sub[metric], marker="o", markersize=4,
                label=f"{_run_label(row)} ({suffix})")
    _overlay_tft(ax, tft_df, metric)
    _save(ax, fig,
          ylabel=f"{metric} (median across wells)",
          title=f"Best GRU runs vs TFT — {metric}",
          out=FIG_DIR / f"gru_best_{metric.lower()}_vs_tft.png")


def plot_config_seed_comparison(table):
    if table.empty or "NSE_mean" not in table.columns:
        return
    has_runtime = "train_s" in table.columns

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, run_sigs in SEED_COMPARISON_CONFIGS.items():
        rows = table[table["run_sig"].isin(run_sigs)]
        if rows.empty:
            continue
        nse_vals = rows["NSE_mean"].dropna()
        if nse_vals.empty:
            continue
        x = rows["train_s"].dropna().mean() if has_runtime and not rows["train_s"].dropna().empty else 0
        ax.errorbar(
            x, nse_vals.mean(),
            yerr=[[nse_vals.mean() - nse_vals.min()], [nse_vals.max() - nse_vals.mean()]],
            fmt="o", capsize=6, markersize=7, label=label,
        )

    ax.set_xlabel("Mean training time (s)")
    ax.set_ylabel("NSE mean h1–16 (median across wells)")
    ax.set_title("Config comparison: performance vs runtime\n(error bars = seed range)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "config_seed_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out}")


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary_df, horizon_df, meta_hp_df = collect_runs()
    if summary_df.empty:
        print("No GRU prediction files found in", GRU_ROOT)
        return

    hpo_df = load_hpo_meta()
    table = build_metrics_table(summary_df, hpo_df, meta_hp_df)
    table.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}  ({len(table)} runs)")

    tft_df = load_tft_best()

    plot_top5(horizon_df, table, tft_df, "NSE")
    plot_top5(horizon_df, table, tft_df, "RMSE")
    plot_best_highlight(horizon_df, table, tft_df, "NSE")
    plot_best_highlight(horizon_df, table, tft_df, "RMSE")
    plot_config_seed_comparison(table)


if __name__ == "__main__":
    main()
