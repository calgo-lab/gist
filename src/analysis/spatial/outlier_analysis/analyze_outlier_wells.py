from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from libs.run_sig import resolve_tft_run_sig

PROFILE_EXPORT_COLUMNS = [
    "id",
    "is_outlier",
    "mae",
    "rmse",
    "bias",
    "n_obs_full",
    "gws_iqr_full",
]
SUMMARY_METRICS = ["mae", "rmse", "bias", "n_obs_full", "gws_iqr_full"]
SUMMARY_EXPORT_COLUMNS = [
    "metric",
    "outlier_median",
    "rest_median",
    "median_diff_outlier_minus_rest",
]


def _load_yaml(path):
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _resolve_data_path(root, dataset, data_cfg):
    if dataset == "sample":
        rel = data_cfg["sample_path"]
    elif dataset == "full_raw":
        rel = data_cfg["full_raw_path"]
    else:
        rel = data_cfg["full_merged_path"]
    p = Path(rel)
    return p if p.is_absolute() else (root / p).resolve()


def _resolve_run_tag(root, args):
    cfg = _load_yaml(root / "configs" / "baselines" / "kriging.yaml")
    tft_cfg = _load_yaml(root / "configs" / "baselines" / "tft.yaml")

    dataset = args.dataset or cfg.get("dataset") or tft_cfg.get("dataset", "full_merged")
    model = args.model or cfg.get("model") or tft_cfg.get("model_name") or tft_cfg.get("model") or "TFT"
    run_sig = (args.run_sig or cfg.get("run_sig") or "").strip()
    if (not run_sig) or run_sig.lower() == "auto":
        run_sig = resolve_tft_run_sig(tft_cfg, model=str(model))
    if not run_sig:
        raise ValueError("run_sig must be set in configs/baselines/tft.yaml or passed via --run-sig.")

    split_tag = str(args.spatial_split_tag or cfg.get("spatial_split_tag", "")).strip()
    kriging_source = str(args.kriging_source or cfg.get("kriging_source", "pred")).strip().lower()
    if kriging_source not in {"pred", "true"}:
        raise ValueError("kriging_source must be 'pred' or 'true'.")

    value_tag = "predobstrain" if kriging_source == "pred" else "trueobstrain"
    run_tag = "__".join([p for p in [run_sig, split_tag, value_tag] if p])
    return dataset, str(model), run_sig, kriging_source, run_tag


def _iqr(s):
    if s.empty:
        return float("nan")
    q = s.quantile([0.25, 0.75])
    return float(q.iloc[1] - q.iloc[0])


def _rmse_from_series(err):
    if err.empty:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(err.to_numpy(dtype=float)))))


def _build_profile(gp_pred, full):
    gp_pred = gp_pred.copy()
    gp_pred["err"] = gp_pred["gws_forecast"] - gp_pred["gws_true"]
    gp_pred["abs_err"] = gp_pred["err"].abs()
    gp_pred["sq_err"] = gp_pred["err"] ** 2

    profile = (
        gp_pred.groupby("id", as_index=False)
        .agg(
            n_points=("id", "size"),
            mae=("abs_err", "mean"),
            bias=("err", "mean"),
            median_abs_err=("abs_err", "median"),
            p95_abs_err=("abs_err", lambda x: float(np.percentile(x, 95))),
            max_abs_err=("abs_err", "max"),
            obs_mean_eval=("gws_true", "mean"),
            obs_std_eval=("gws_true", "std"),
            obs_min_eval=("gws_true", "min"),
            obs_max_eval=("gws_true", "max"),
            pred_std_mean=("gws_forecast_std", "mean"),
        )
    )
    profile["rmse"] = gp_pred.groupby("id")["err"].apply(_rmse_from_series).reindex(profile["id"]).to_numpy()
    profile["obs_range_eval"] = profile["obs_max_eval"] - profile["obs_min_eval"]
    profile["obs_iqr_eval"] = gp_pred.groupby("id")["gws_true"].apply(_iqr).reindex(profile["id"]).to_numpy()

    full_stats = (
        full.groupby("id", as_index=False)
        .agg(
            n_obs_full=("id", "size"),
            first_date_full=("datum", "min"),
            last_date_full=("datum", "max"),
            gws_mean_full=("gws", "mean"),
            gws_std_full=("gws", "std"),
            gws_min_full=("gws", "min"),
            gws_max_full=("gws", "max"),
        )
    )
    full_stats["span_days_full"] = (full_stats["last_date_full"] - full_stats["first_date_full"]).dt.days
    full_stats["gws_range_full"] = full_stats["gws_max_full"] - full_stats["gws_min_full"]
    full_stats["gws_iqr_full"] = full.groupby("id")["gws"].apply(_iqr).reindex(full_stats["id"]).to_numpy()

    profile = profile.merge(full_stats, on="id", how="left")
    return profile


def _select_outliers(profile, outlier_metric, outlier_top_n, outlier_ids):
    profile = profile.copy()
    profile["is_outlier"] = False

    if outlier_ids:
        known = set(profile["id"])
        chosen = [w for w in outlier_ids if w in known]
        if not chosen:
            raise ValueError("None of the provided --outlier-ids are present in gp_pred.parquet.")
        profile.loc[profile["id"].isin(chosen), "is_outlier"] = True
        return profile

    if outlier_metric not in profile.columns:
        raise ValueError(f"Unknown outlier metric: {outlier_metric}")

    pick_n = max(1, int(outlier_top_n))
    ranked = profile.sort_values(outlier_metric, ascending=False)
    chosen_ids = ranked["id"].head(pick_n).tolist()
    profile.loc[profile["id"].isin(chosen_ids), "is_outlier"] = True
    return profile


def _build_summary(profile, metrics):
    out = profile[profile["is_outlier"]].copy()
    rest = profile[~profile["is_outlier"]].copy()
    rows = []
    for m in metrics:
        o = pd.to_numeric(out[m], errors="coerce").dropna()
        r = pd.to_numeric(rest[m], errors="coerce").dropna()
        o_mean = float(o.mean()) if len(o) else np.nan
        r_mean = float(r.mean()) if len(r) else np.nan
        o_med = float(o.median()) if len(o) else np.nan
        r_med = float(r.median()) if len(r) else np.nan
        rows.append(
            {
                "metric": m,
                "n_outlier_non_na": int(len(o)),
                "n_rest_non_na": int(len(r)),
                "outlier_mean": o_mean,
                "rest_mean": r_mean,
                "mean_diff_outlier_minus_rest": o_mean - r_mean if np.isfinite(o_mean) and np.isfinite(r_mean) else np.nan,
                "mean_ratio_outlier_over_rest": (o_mean / r_mean) if np.isfinite(o_mean) and np.isfinite(r_mean) and r_mean != 0 else np.nan,
                "outlier_median": o_med,
                "rest_median": r_med,
                "median_diff_outlier_minus_rest": o_med - r_med if np.isfinite(o_med) and np.isfinite(r_med) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _boxplot(profile, metrics, out_path, title):
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.6), dpi=170)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        out_vals = pd.to_numeric(profile.loc[profile["is_outlier"], metric], errors="coerce").dropna().to_numpy()
        rest_vals = pd.to_numeric(profile.loc[~profile["is_outlier"], metric], errors="coerce").dropna().to_numpy()
        ax.boxplot([rest_vals, out_vals], tick_labels=["rest", "outliers"], showfliers=True)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _scatter(profile, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=170)
    rest = profile[~profile["is_outlier"]]
    out = profile[profile["is_outlier"]]
    ax.scatter(rest["gws_range_full"], rest["mae"], s=20, alpha=0.6, c="#6c757d", label="rest")
    ax.scatter(out["gws_range_full"], out["mae"], s=55, alpha=0.95, c="#d62728", label="outliers")
    for _, r in out.iterrows():
        ax.annotate(str(r["id"]), (r["gws_range_full"], r["mae"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Full-history GWS range (m)")
    ax.set_ylabel("Kriging MAE (m)")
    ax.set_title("Outlier wells: MAE vs full-history range")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze outlier wells for a kriging run.")
    parser.add_argument("--run-sig", dest="run_sig", default=None)
    parser.add_argument("--dataset", dest="dataset", default=None)
    parser.add_argument("--model", dest="model", default=None)
    parser.add_argument("--kriging-source", dest="kriging_source", choices=["pred", "true"], default=None)
    parser.add_argument("--split-tag", dest="spatial_split_tag", default=None)
    parser.add_argument("--outlier-top-n", dest="outlier_top_n", type=int, default=12)
    parser.add_argument(
        "--outlier-metric",
        dest="outlier_metric",
        default="median_abs_err",
        choices=["median_abs_err", "mae", "rmse", "p95_abs_err", "max_abs_err"],
    )
    parser.add_argument(
        "--outlier-ids",
        dest="outlier_ids",
        default="",
        help="Comma-separated list of well IDs to force as outliers (overrides top-n selection).",
    )
    args = parser.parse_args()

    data_cfg = _load_yaml(ROOT / "configs" / "data.yaml")
    dataset, _model, _run_sig, _kriging_source, run_tag = _resolve_run_tag(ROOT, args)

    gp_dir = ROOT / "outputs" / "gp" / run_tag
    gp_pred_path = gp_dir / "gp_pred.parquet"
    if not gp_pred_path.exists():
        raise FileNotFoundError(f"Missing {gp_pred_path}. Run kriging first.")

    gp_pred = pq.read_table(gp_pred_path).to_pandas()
    gp_pred["datum"] = pd.to_datetime(gp_pred["datum"])

    data_path = _resolve_data_path(ROOT, dataset, data_cfg)
    if str(data_path).lower().endswith(".csv"):
        full = pd.read_csv(data_path, usecols=["id", "datum", "gws"], low_memory=False)
    else:
        full = pq.read_table(data_path, columns=["id", "datum", "gws"]).to_pandas()
    full["datum"] = pd.to_datetime(full["datum"])
    full = full[full["id"].isin(gp_pred["id"].unique())].copy()

    profile = _build_profile(gp_pred=gp_pred, full=full)
    outlier_ids = [x.strip() for x in str(args.outlier_ids).split(",") if x.strip()]
    profile = _select_outliers(
        profile=profile,
        outlier_metric=args.outlier_metric,
        outlier_top_n=args.outlier_top_n,
        outlier_ids=outlier_ids,
    )

    n_out = int(profile["is_outlier"].sum())
    n_rest = int((~profile["is_outlier"]).sum())
    if n_out == 0 or n_rest == 0:
        raise ValueError(f"Need both outlier and rest groups. Got outliers={n_out}, rest={n_rest}.")

    summary = _build_summary(profile, metrics=SUMMARY_METRICS)

    out_dir = ROOT / "reports" / "outlier_analysis" / "kriging"
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = profile.sort_values(["is_outlier", args.outlier_metric], ascending=[False, False]).reset_index(drop=True)
    profile_out = profile[PROFILE_EXPORT_COLUMNS].copy()
    summary_out = summary[SUMMARY_EXPORT_COLUMNS].copy()
    summary_out.to_csv(out_dir / "outlier_vs_rest_summary.csv", index=False)
    _scatter(profile, out_path=out_dir / "outlier_vs_rest_scatter.png")

    chosen = profile.loc[profile["is_outlier"], "id"].tolist()
    print(f"Run tag: {run_tag}")
    print(f"Outliers ({len(chosen)}): {', '.join(chosen)}")
    print(f"Wrote: {out_dir / 'outlier_vs_rest_summary.csv'}")
    print(f"Wrote plots in: {out_dir}")


if __name__ == "__main__":
    main()
