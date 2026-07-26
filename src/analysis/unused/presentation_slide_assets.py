import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[3]

DATA_CFG = ROOT / "configs" / "data.yaml"
SPLITS_DIR = ROOT / "splits"
OUT_DIR = ROOT / "reports"
OUT_FIG = ROOT / "reports" / "figures" / "slides"
OUT_MET = ROOT / "reports" / "tables" / "slides"
OUT_TXT = ROOT / "reports" / "tables" / "slides"

BOUNDARY_FILE = ROOT / "data" / "boundaries" / "geoBoundaries-DEU-ADM1_simplified.geojson"

HYDRORAUM_LABELS_EN = {
    "Speisungsgebiete": "Recharge zones",
    "Transitgebiete": "Transit zones",
    "Entlastungsgebiete": "Discharge zones",
    "NaN": "Missing",
}

CONFINEMENT_LABELS_EN = {
    "gespannt": "Confined",
    "ungespannt": "Unconfined",
    "NaN": "Missing",
}

METADATA_FEATURE_LABELS_EN = {
    "hydroraum": "Hydrogeological zone",
    "gw_gespannt": "Aquifer condition",
    "gwlk": "Groundwater level class",
    "siwa_verweilzeit_j": "Residence time (SIWA)",
    "fok": "Screen top depth (fok)",
    "fuk": "Screen bottom depth (fuk)",
    "gok": "Ground elevation (gok)",
}

GP_FEATURE_IMPORTANCE_ROWS = [
    {
        "feature_group": "Coordinates only",
        "best_tested_setup": "x/y only",
        "best_nrmse_pw": 1.7831,
        "delta_vs_coords": 0.0,
        "verdict": "Baseline",
        "takeaway": "Reference point",
    },
    {
        "feature_group": "Best metadata combination",
        "best_tested_setup": "SIWA + hydrogeological zone + aquifer condition",
        "best_nrmse_pw": 1.1760,
        "delta_vs_coords": -0.6071,
        "verdict": "Best overall",
        "takeaway": "Use this",
    },
    {
        "feature_group": "Groundwater residence time",
        "best_tested_setup": "SIWA only",
        "best_nrmse_pw": 1.7313,
        "delta_vs_coords": -0.0518,
        "verdict": "Helpful alone",
        "takeaway": "Main driver in combinations",
    },
    {
        "feature_group": "Hydrogeological zone",
        "best_tested_setup": "Zone only",
        "best_nrmse_pw": 1.5776,
        "delta_vs_coords": -0.2055,
        "verdict": "Important",
        "takeaway": "Strong spatial context",
    },
    {
        "feature_group": "Aquifer condition",
        "best_tested_setup": "Confined vs unconfined only",
        "best_nrmse_pw": 1.6548,
        "delta_vs_coords": -0.1283,
        "verdict": "Useful",
        "takeaway": "Adds modest gain",
    },
    {
        "feature_group": "Autocorrelation class",
        "best_tested_setup": "Best case: SIWA + acf_class",
        "best_nrmse_pw": 1.8954,
        "delta_vs_coords": 0.1123,
        "verdict": "Hurts",
        "takeaway": "Do not use",
    },
    {
        "feature_group": "Groundwater level class",
        "best_tested_setup": "gwlk only",
        "best_nrmse_pw": 2.4942,
        "delta_vs_coords": 0.7111,
        "verdict": "Hurts",
        "takeaway": "Do not use",
    },
    {
        "feature_group": "Precipitation-response features",
        "best_tested_setup": "Best case: pr1lag only",
        "best_nrmse_pw": 2.0930,
        "delta_vs_coords": 0.3099,
        "verdict": "Hurts to catastrophic",
        "takeaway": "Not useful as spatial covariates",
    },
    {
        "feature_group": "Well screen depths",
        "best_tested_setup": "Best case: SIWA + zone + confinement + fok",
        "best_nrmse_pw": 3.6766,
        "delta_vs_coords": 1.8935,
        "verdict": "Catastrophic",
        "takeaway": "Strongly avoid",
    },
]


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _resolve_data_paths() -> dict[str, Path]:
    cfg = _load_yaml(DATA_CFG)
    out: dict[str, Path] = {}
    for k, v in cfg.items():
        p = Path(v)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        out[k] = p
    return out


def _load_coords(merged_path: Path) -> pd.DataFrame:
    coords = pq.read_table(merged_path, columns=["id", "x_25833", "y_25833"]).to_pandas()
    coords = coords.drop_duplicates("id").dropna(subset=["x_25833", "y_25833"]).copy()
    return coords


def _load_boundary_rings(path: Path, names: set[str]) -> list[tuple[list[float], list[float]]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = [
        f for f in data.get("features", [])
        if f.get("properties", {}).get("shapeName") in names
    ]
    if not feats:
        return []
    tr = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
    rings: list[tuple[list[float], list[float]]] = []
    for feat in feats:
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        polys = [coords] if gtype == "Polygon" else coords if gtype == "MultiPolygon" else []
        for poly in polys:
            for ring in poly:
                xs_r, ys_r = zip(*ring)
                xx_r, yy_r = tr.transform(xs_r, ys_r)
                rings.append((list(xx_r), list(yy_r)))
    return rings


def _plot_boundary(ax, rings: Iterable[tuple[list[float], list[float]]]) -> None:
    for xr, yr in rings:
        ax.plot(xr, yr, color="0.35", linewidth=1.0, alpha=0.9, zorder=1)


def _weighted(values: np.ndarray) -> float:
    weights = np.linspace(1.0, 2.0, 16)
    return float(np.average(values, weights=weights))


def _ensure_dirs() -> None:
    for p in [OUT_DIR, OUT_FIG, OUT_MET, OUT_TXT]:
        p.mkdir(parents=True, exist_ok=True)


def _per_well_horizon_metrics_from_pred(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (wid, h), g in pred.groupby(["id", "horizon"]):
        y = g["gws"].to_numpy(dtype=float)
        p = g["gws_forecast"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        mae = float(np.mean(np.abs(p - y)))
        denom = float(np.sum((y - y.mean()) ** 2))
        nse = float(1.0 - np.sum((p - y) ** 2) / denom) if denom > 0 else np.nan
        iqr = float(np.diff(np.quantile(y, [0.25, 0.75]))[0])
        nrmse = rmse / iqr if iqr > 0 else np.nan
        rows.append({
            "id": wid,
            "horizon": int(h),
            "RMSE": rmse,
            "MAE": mae,
            "NSE": nse,
            "nRMSE": nrmse,
        })
    return pd.DataFrame(rows)


def _horizon_median(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("horizon", as_index=False)[["NSE", "RMSE", "nRMSE", "MAE"]]
        .median()
        .sort_values("horizon")
    )


def _metadata_group(col: str) -> str:
    if col in {"id", "id_original", "name", "typ", "parameter"}:
        return "Identifiers"
    if col in {"x_25833", "y_25833", "geometry"}:
        return "Coordinates and Geometry"
    if col in {"gok", "ausbausohle", "fok", "fuk", "gwlk"}:
        return "Well Construction and Water-Level Class"
    if col in {
        "logger_beginn_lfu", "logger_beginn", "messintervall", "sensibel",
        "messnetz", "datenquelle", "gws_logger", "outlier_count",
        "imputed_proportion", "interpolated_proportion",
        "imputed_proportion_ml_prognose", "interpolated_proportion_ml_prognose",
        "gaps_count", "min_gap_w", "max_gap_w", "dropped_lfu",
    }:
        return "Monitoring, Source, and Quality"
    if col in {
        "gw_gespannt", "siwa_verweilzeit_j", "oezg_moor_habitat", "distance_wsg_z1",
        "uEZG_Zuordnung", "hydroraum", "HGN_anthro", "DFU", "MST_pro_uEZG",
        "MeanDist5clostestMST", "hydroraum_1991-2020",
    }:
        return "Hydrogeologic Context"
    if col in {
        "TWI_dgm50_r1000m", "Slope_dgm50_r1000m", "GW_recharge_r1000m",
        "Percolation_r1000m", "DSD_order1_r1000m", "LP_order1_r1000m",
        "SD_order1_r1000m", "DSD_order5_r1000m", "LP_order5_r1000m",
        "SD_order5_r1000m",
    }:
        return "Terrain and Recharge Features"
    if col in {
        "acf_2018-2023_preproc", "gws_beginn_ml_prognose", "ml_prognose",
        "dss_standsklassen", "gws_beginn", "konsistenz", "konsistenz_1991-2020",
        "median_1991-2020", "median_ugok_1991-2020", "mean_range_1991-2020",
        "mean_range_hyj_1991-2020", "acf_1991-2020", "range_ratio_1991-2020",
        "skew", "min_month", "max_month", "longest_recession", "parde_seasonality",
        "interannual_variation", "low_pulse_count", "high_pulse_count",
        "low_pulse_duration", "high_pulse_duration", "baseflow_index",
        "trend_slope_GS", "trend_bewertung_GS", "acf_class",
    }:
        return "Time-Series Signatures"
    if col in {"KIT_rank_old", "PCA_R2", "KIT_rank", "KIT_rank_RMSE", "LSTM_16w_NSE", "TFT_16w_NSE", "LSTM_16w_RMSE", "TFT_16w_RMSE"}:
        return "Model-Derived Scores"
    if col in {"Cluster_1980", "Silhouette_score_1980", "Cluster_1990", "Silhouette_score_1990"}:
        return "Clustering Features"
    if col in {"pr_1W_lag", "pr_1W_xcorr", "pr_52W_lag", "pr_52W_xcorr"}:
        return "Precipitation-Derived Features"
    return "Other"


def _mapped_counts(series: pd.Series, label_map: dict[str, str], name: str) -> pd.DataFrame:
    counts = series.fillna("NaN").astype(str).value_counts().rename_axis(name).reset_index(name="count")
    counts["label_en"] = counts[name].map(lambda v: label_map.get(v, v))
    return counts


def _plot_bar_counts(ax, labels: list[str], counts: list[int], title: str, ylabel: str, color: str,
                     rotation: int = 0, annotate: bool = True) -> None:
    bars = ax.bar(labels, counts, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=rotation)
    if annotate:
        ymax = max(counts) if counts else 0
        pad = max(1.0, 0.02 * ymax) if ymax > 0 else 1.0
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + pad,
                f"{int(cnt)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )


def _save_text_panel(lines: list[str], out_name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.01, 0.98, "\n".join([title, ""] + lines), va="top", fontsize=11, family="monospace")
    fig.tight_layout()
    fig.savefig(OUT_FIG / out_name, dpi=220)
    plt.close(fig)


def build_gp_feature_importance_assets() -> None:
    df = pd.DataFrame(GP_FEATURE_IMPORTANCE_ROWS)
    df["delta_vs_coords_str"] = df["delta_vs_coords"].map(lambda v: f"{v:+.3f}")
    df["best_nrmse_pw_str"] = df["best_nrmse_pw"].map(lambda v: f"{v:.3f}")
    df.to_csv(OUT_MET / "gp_feature_importance_table.csv", index=False)

    note_lines = [
        "# Slide 17 GP Feature-Importance Table",
        "",
        "Recommended slide wording:",
        "- Values come from the oracle GP ablation on `rmd90_test52`.",
        "- This isolates the spatial GP feature effect without temporal-model error.",
        "- The GP kernel and metadata feature space match the decoupled pipeline, so the ranking should transfer qualitatively.",
        "- Use this as a compact replacement for the longer feature-ablation text.",
        "",
        "Metric:",
        "- `nRMSE_pw = median_well(RMSE_well / IQR_well)`; lower is better.",
        "",
        "Main message:",
        "- `SIWA + hydrogeological zone + aquifer condition` is the clear winner.",
        "- `acf_class`, `gwlk`, precipitation-response features, and `fok/fuk` hurt badly.",
    ]
    (OUT_TXT / "slide17_feature_importance_table.md").write_text("\n".join(note_lines), encoding="utf-8")

    display_cols = [
        "Feature group",
        "Best tested setup",
        "Best nRMSE_pw",
        "Delta vs coords",
        "Takeaway",
    ]
    display_df = pd.DataFrame(
        {
            display_cols[0]: df["feature_group"],
            display_cols[1]: df["best_tested_setup"],
            display_cols[2]: df["best_nrmse_pw_str"],
            display_cols[3]: df["delta_vs_coords_str"],
            display_cols[4]: df["takeaway"],
        }
    )

    fig_h = 0.62 * len(display_df) + 1.8
    fig, ax = plt.subplots(figsize=(12.8, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.22, 0.34, 0.12, 0.12, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)

    header_color = "#2F4B7C"
    for c in range(len(display_df.columns)):
        cell = table[(0, c)]
        cell.set_facecolor(header_color)
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
        cell.set_edgecolor("#DDDDDD")

    verdict_colors = {
        "Baseline": "#F2F2F2",
        "Best overall": "#D8F0D2",
        "Helpful alone": "#E8F5E4",
        "Important": "#DFF0DA",
        "Useful": "#E8F5E4",
        "Hurts": "#FCE8E6",
        "Hurts to catastrophic": "#F9D7D3",
        "Catastrophic": "#F4B6B0",
    }
    for r in range(1, len(display_df) + 1):
        verdict = df.iloc[r - 1]["verdict"]
        row_color = verdict_colors.get(verdict, "#FFFFFF")
        for c in range(len(display_df.columns)):
            cell = table[(r, c)]
            cell.set_facecolor(row_color)
            cell.set_edgecolor("#DDDDDD")
            if c in {2, 3}:
                cell.get_text().set_ha("right")

    fig.suptitle(
        "GP Spatial Feature Importance (Oracle Ablation, rmd90_test52)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.01,
        0.02,
        "Metric: nRMSE_pw = median over wells of (RMSE_well / IQR_well), lower is better. "
        "Use as the clean spatial ranking for the decoupled GP as well.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT_FIG / "gp_feature_importance_table.png", dpi=220)
    plt.close(fig)


def build_metadata_assets(meta_path: Path, model_ids: set[str]) -> None:
    meta = pd.read_csv(meta_path, sep=";")

    meta_w = meta.drop_duplicates("id").copy()
    meta_w = meta_w[meta_w["id"].astype(str).isin(model_ids)].copy()
    n_wells = int(meta_w["id"].nunique())

    feature_cols = [
        "hydroraum",
        "gw_gespannt",
        "gwlk",
        "siwa_verweilzeit_j",
        "fok",
        "fuk",
        "gok",
    ]
    feature_cols = [c for c in feature_cols if c in meta_w.columns]

    miss_rows = []
    for c in feature_cols:
        miss_rows.append({
            "feature": c,
            "missing_n": int(meta_w[c].isna().sum()),
            "missing_pct": float(meta_w[c].isna().mean() * 100.0),
            "n_unique": int(meta_w[c].nunique(dropna=True)),
        })
    miss_df = pd.DataFrame(miss_rows).sort_values("missing_pct", ascending=False)
    miss_df.to_csv(OUT_MET / "metadata_missingness_key_features.csv", index=False)

    hydroraum_counts = (
        _mapped_counts(meta_w["hydroraum"], HYDRORAUM_LABELS_EN, "hydroraum")
        if "hydroraum" in meta_w.columns else pd.DataFrame(columns=["hydroraum", "count", "label_en"])
    )
    gesp_counts = (
        _mapped_counts(meta_w["gw_gespannt"], CONFINEMENT_LABELS_EN, "gw_gespannt")
        if "gw_gespannt" in meta_w.columns else pd.DataFrame(columns=["gw_gespannt", "count", "label_en"])
    )
    gwlk_counts = (
        meta_w["gwlk"].fillna("NaN").astype(str).value_counts().rename_axis("gwlk").reset_index(name="count")
        if "gwlk" in meta_w.columns else pd.DataFrame(columns=["gwlk", "count"])
    )
    if "siwa_verweilzeit_j" in meta_w.columns:
        siwa_nonmissing = (
            meta_w.loc[meta_w["siwa_verweilzeit_j"].notna(), "siwa_verweilzeit_j"]
            .astype(float)
            .value_counts()
            .sort_index()
            .rename_axis("siwa_verweilzeit_j")
            .reset_index(name="count")
        )
        siwa_missing_n = int(meta_w["siwa_verweilzeit_j"].isna().sum())
        siwa_counts = siwa_nonmissing.copy()
        siwa_counts["label_en"] = siwa_counts["siwa_verweilzeit_j"].map(lambda v: f"{int(v)} years")
        if siwa_missing_n > 0:
            siwa_counts = pd.concat(
                [
                    siwa_counts,
                    pd.DataFrame(
                        {
                            "siwa_verweilzeit_j": [np.nan],
                            "count": [siwa_missing_n],
                            "label_en": ["Missing"],
                        }
                    ),
                ],
                ignore_index=True,
            )
    else:
        siwa_counts = pd.DataFrame(columns=["siwa_verweilzeit_j", "count", "label_en"])

    hydroraum_counts.to_csv(OUT_MET / "metadata_hydroraum_counts.csv", index=False)
    gesp_counts.to_csv(OUT_MET / "metadata_gw_gespannt_counts.csv", index=False)
    gwlk_counts.to_csv(OUT_MET / "metadata_gwlk_counts.csv", index=False)
    hydroraum_counts.to_csv(OUT_MET / "metadata_hydroraum_counts_en.csv", index=False)
    gesp_counts.to_csv(OUT_MET / "metadata_confined_status_counts_en.csv", index=False)
    siwa_counts.to_csv(OUT_MET / "metadata_siwa_counts_en.csv", index=False)

    inventory_df = pd.DataFrame({
        "column": list(meta_w.columns),
        "group": [_metadata_group(c) for c in meta_w.columns],
    })
    inventory_df.to_csv(OUT_MET / "metadata_all_columns.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    top_txt = [
        f"- Wells with metadata: {n_wells}",
        f"- Metadata columns: {meta_w.shape[1]}",
    ]
    if "hydroraum" in meta_w.columns:
        top_txt.append(f"- Hydrogeological zone available: {(~meta_w['hydroraum'].isna()).sum()} wells")
    if "gw_gespannt" in meta_w.columns:
        top_txt.append(f"- Aquifer condition available: {(~meta_w['gw_gespannt'].isna()).sum()} wells")
    if "siwa_verweilzeit_j" in meta_w.columns:
        top_txt.append(f"- Residence time (SIWA) available: {(~meta_w['siwa_verweilzeit_j'].isna()).sum()} wells")
    if "fok" in meta_w.columns and "fuk" in meta_w.columns:
        top_txt.append(f"- Screen top/bottom available: {(~meta_w['fok'].isna() & ~meta_w['fuk'].isna()).sum()} wells")
    axes[0, 0].axis("off")
    axes[0, 0].text(0.01, 0.98, "\n".join(["Metadata Overview (1 row per well)", ""] + top_txt), va="top", fontsize=11, family="monospace")
    _save_text_panel(top_txt, "metadata_summary_text.png", "Metadata Overview (1 row per well)")

    if not hydroraum_counts.empty:
        labels = hydroraum_counts["label_en"].tolist()
        counts = hydroraum_counts["count"].astype(int).tolist()
        _plot_bar_counts(
            axes[0, 1],
            labels=labels,
            counts=counts,
            title="Hydrogeological Zone Distribution",
            ylabel="Wells",
            color="#4C78A8",
            rotation=15,
        )
        fig_single, ax_single = plt.subplots(figsize=(7, 4.5))
        _plot_bar_counts(
            ax_single,
            labels=labels,
            counts=counts,
            title="Hydrogeological Zone Distribution",
            ylabel="Wells",
            color="#4C78A8",
            rotation=15,
        )
        fig_single.tight_layout()
        fig_single.savefig(OUT_FIG / "metadata_hydroraum_distribution_en.png", dpi=220)
        plt.close(fig_single)
    else:
        axes[0, 1].axis("off")

    if not gesp_counts.empty:
        labels = gesp_counts["label_en"].tolist()
        counts = gesp_counts["count"].astype(int).tolist()
        _plot_bar_counts(
            axes[1, 0],
            labels=labels,
            counts=counts,
            title="Aquifer Condition Distribution",
            ylabel="Wells",
            color="#59A14F",
        )
        fig_single, ax_single = plt.subplots(figsize=(7, 4.5))
        _plot_bar_counts(
            ax_single,
            labels=labels,
            counts=counts,
            title="Aquifer Condition Distribution",
            ylabel="Wells",
            color="#59A14F",
        )
        fig_single.tight_layout()
        fig_single.savefig(OUT_FIG / "metadata_confined_status_distribution_en.png", dpi=220)
        plt.close(fig_single)
    else:
        axes[1, 0].axis("off")

    if not miss_df.empty:
        miss_plot = miss_df.copy()
        miss_plot["feature_en"] = miss_plot["feature"].map(lambda c: METADATA_FEATURE_LABELS_EN.get(c, c))
        axes[1, 1].barh(miss_plot["feature_en"], miss_plot["missing_pct"], color="#E15759")
        axes[1, 1].set_title("Missingness in Key Metadata Features")
        axes[1, 1].set_xlabel("Missing (%)")
        axes[1, 1].grid(True, axis="x", alpha=0.25)
        fig_single, ax_single = plt.subplots(figsize=(8, 4.8))
        ax_single.barh(miss_plot["feature_en"], miss_plot["missing_pct"], color="#E15759")
        ax_single.set_title("Missingness in Key Metadata Features")
        ax_single.set_xlabel("Missing (%)")
        ax_single.grid(True, axis="x", alpha=0.25)
        fig_single.tight_layout()
        fig_single.savefig(OUT_FIG / "metadata_missingness_key_features_en.png", dpi=220)
        plt.close(fig_single)
    else:
        axes[1, 1].axis("off")

    if not siwa_counts.empty:
        fig_single, ax_single = plt.subplots(figsize=(8, 4.8))
        colors = ["#4C78A8" if label != "Missing" else "#B8B8B8" for label in siwa_counts["label_en"]]
        bars = ax_single.bar(siwa_counts["label_en"], siwa_counts["count"], color=colors)
        ax_single.set_title("SIWA Residence Time Distribution")
        ax_single.set_xlabel("Residence time category")
        ax_single.set_ylabel("Wells")
        ax_single.grid(True, axis="y", alpha=0.25)
        ymax = int(siwa_counts["count"].max()) if len(siwa_counts) else 0
        pad = max(1.0, 0.02 * ymax) if ymax > 0 else 1.0
        for bar, cnt in zip(bars, siwa_counts["count"].astype(int).tolist()):
            ax_single.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + pad,
                f"{cnt}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig_single.tight_layout()
        fig_single.savefig(OUT_FIG / "metadata_siwa_distribution_en.png", dpi=220)
        plt.close(fig_single)

    fig.tight_layout()
    fig.savefig(OUT_FIG / "metadata_overview.png", dpi=220)
    fig.savefig(OUT_FIG / "metadata_overview_en.png", dpi=220)
    plt.close(fig)

    note_lines = [
        "# Slide 9 Metadata Notes",
        "",
        "This file is the short summary version.",
        "For the complete grouped inventory, use `slide9_metadata_full_inventory.md` and `metadata_all_columns.csv`.",
    ]
    (OUT_TXT / "slide9_metadata_notes.md").write_text("\n".join(note_lines), encoding="utf-8")

    group_order = [
        "Identifiers",
        "Coordinates and Geometry",
        "Well Construction and Water-Level Class",
        "Monitoring, Source, and Quality",
        "Hydrogeologic Context",
        "Terrain and Recharge Features",
        "Time-Series Signatures",
        "Model-Derived Scores",
        "Clustering Features",
        "Precipitation-Derived Features",
        "Other",
    ]
    full_lines = [
        "# Slide 9 Metadata Full Inventory",
        "",
        f"Model-well metadata rows considered: {n_wells}",
        f"Total metadata columns available on model wells: {meta_w.shape[1]}",
        "",
        "Use this if you want the full metadata list on the slide.",
        "A practical slide layout would be 2-3 columns grouped by category.",
        "",
    ]
    for group in group_order:
        cols = inventory_df.loc[inventory_df["group"] == group, "column"].tolist()
        if not cols:
            continue
        full_lines.append(f"## {group}")
        full_lines.append(", ".join(cols))
        full_lines.append("")
    (OUT_TXT / "slide9_metadata_full_inventory.md").write_text("\n".join(full_lines), encoding="utf-8")


def build_gru_tft_horizon_plots() -> None:
    comparison_label = "full_merged spatial split"
    comparison_slug = "full_merged_spatial_split"

    gru_summary = pd.read_csv(ROOT / "reports" / "metrics" / "gru" / "gru_metrics_summary.csv")
    gru_best = gru_summary.loc[gru_summary["RMSE_weighted_mean"].idxmin()]
    gru_sig = str(gru_best["run_sig"])
    gru_pred = pq.read_table(
        ROOT / "outputs" / "GRU_FCOV" / f"GRU_FCOV_{gru_sig}" / "predictions" / "pred.parquet"
    ).to_pandas()
    gru_h = _horizon_median(_per_well_horizon_metrics_from_pred(gru_pred))
    gru_h["model"] = "GRU"
    gru_h["run"] = gru_sig

    tft_root = ROOT / "outputs" / "TFT"
    tft_rows = []
    tft_hz_map: dict[str, pd.DataFrame] = {}
    for run_dir in sorted(tft_root.glob("TFT_in52_out16_ep50_bs4096_stat1_seed*_full_merged")):
        mp = run_dir / "metrics.parquet"
        if not mp.exists():
            continue
        m = pq.read_table(mp).to_pandas()
        m = m[m["metric"].isin(["RMSE", "MAE", "NSE", "nRMSE"])]
        piv = m.pivot_table(index=["id", "horizon"], columns="metric", values="value").reset_index()
        hz = (
            piv.groupby("horizon", as_index=False)[["NSE", "RMSE", "nRMSE", "MAE"]]
            .median()
            .sort_values("horizon")
        )
        hz16 = hz[hz["horizon"].between(1, 16)]
        if len(hz16) != 16:
            continue
        w_rmse = _weighted(hz16["RMSE"].to_numpy(dtype=float))
        tft_rows.append({"run": run_dir.name, "RMSE_weighted_h1_16": w_rmse})
        tft_hz_map[run_dir.name] = hz

    if not tft_rows:
        raise RuntimeError("No TFT run with usable metrics.parquet found for horizon comparison.")

    tft_rank = pd.DataFrame(tft_rows).sort_values("RMSE_weighted_h1_16")
    tft_best_run = str(tft_rank.iloc[0]["run"])
    tft_h = tft_hz_map[tft_best_run].copy()
    tft_h["model"] = "TFT"
    tft_h["run"] = tft_best_run

    both = pd.concat([gru_h, tft_h], ignore_index=True)
    both.to_csv(OUT_MET / "gru_vs_tft_best_horizon_metrics.csv", index=False)
    both.to_csv(OUT_MET / f"gru_vs_tft_best_horizon_metrics_{comparison_slug}.csv", index=False)
    tft_rank.to_csv(OUT_MET / "tft_run_ranking_weighted_rmse_h1_16.csv", index=False)
    tft_rank.to_csv(OUT_MET / f"tft_run_ranking_weighted_rmse_h1_16_{comparison_slug}.csv", index=False)

    comp = (
        gru_h[["horizon", "NSE", "RMSE", "nRMSE", "MAE"]]
        .merge(
            tft_h[["horizon", "NSE", "RMSE", "nRMSE", "MAE"]],
            on="horizon",
            suffixes=("_gru", "_tft"),
        )
        .sort_values("horizon")
    )
    win_counts = {
        "RMSE": int((comp["RMSE_gru"] < comp["RMSE_tft"]).sum()),
        "MAE": int((comp["MAE_gru"] < comp["MAE_tft"]).sum()),
        "nRMSE": int((comp["nRMSE_gru"] < comp["nRMSE_tft"]).sum()),
        "NSE": int((comp["NSE_gru"] > comp["NSE_tft"]).sum()),
    }

    metrics = ["NSE", "RMSE", "nRMSE", "MAE"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        series = [("GRU", "#1f77b4", "Best GRU overall"), ("TFT", "#ff7f0e", "Best TFT")]
        for model, col, label in series:
            sub = both[both["model"] == model].sort_values("horizon")
            ax.plot(sub["horizon"], sub[metric], marker="o", linewidth=2.0, markersize=4.0, color=col, label=label)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(metric)
        ax.set_title(f"Best GRU vs Best TFT by Horizon ({comparison_label}): {metric}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT_FIG / f"gru_vs_tft_best_{metric.lower()}_by_horizon.png", dpi=220)
        fig.savefig(OUT_FIG / f"gru_vs_tft_best_{comparison_slug}_{metric.lower()}_by_horizon.png", dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        series = [("GRU", "#1f77b4", "Best GRU overall"), ("TFT", "#ff7f0e", "Best TFT")]
        for model, col, label in series:
            sub = both[both["model"] == model].sort_values("horizon")
            ax.plot(sub["horizon"], sub[metric], marker="o", linewidth=1.8, markersize=3.5, color=col, label=label)
        ax.set_title(metric)
        ax.set_xlabel("Horizon")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(f"Best GRU vs Best TFT (H1..H16, {comparison_label})")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "gru_vs_tft_best_all_metrics_by_horizon.png", dpi=220)
    fig.savefig(OUT_FIG / f"gru_vs_tft_best_all_metrics_by_horizon_{comparison_slug}.png", dpi=220)
    plt.close(fig)

    note = [
        "# Slide 10 Metric Reconciliation",
        "",
        f"- Comparison used for plots: `{comparison_label}`",
        f"- Best GRU run used here: `{gru_sig}`",
        f"- GRU RMSE weighted (H1-16): {gru_best['RMSE_weighted_mean']:.6f} m",
        f"- GRU RMSE at H16: {gru_best['RMSE_h16']:.6f} m",
        "",
        f"- Best TFT run used here: `{tft_best_run}`",
        f"- TFT RMSE weighted (H1-16): {float(tft_rank.iloc[0]['RMSE_weighted_h1_16']):.6f} m",
        "",
        "Interpretation for slides:",
        "1. These comparison plots now use TFT runs from `full_merged` only, so GRU and TFT are on the same spatial split.",
        "2. The ~0.068 values are weighted H1-16 horizon summaries, not H16.",
        "3. H16 RMSE is substantially higher than 0.068 for both models.",
        "4. If you use per-well median RMSE on slide 10 as well, label it explicitly as `median per-well RMSE`.",
    ]
    (OUT_TXT / "slide10_metric_reconciliation.md").write_text("\n".join(note), encoding="utf-8")

    nse_note = [
        "# Slide 12 NSE Plot Note",
        "",
        f"- Comparison used for plots: `{comparison_label}`",
        f"- Best GRU overall: `{gru_sig}`",
        f"- Best TFT: `{tft_best_run}`",
        "",
        "Interpretation:",
        "- The plot now shows one GRU line and one TFT line only.",
        "- This comparison is restricted to `full_merged`, i.e. the spatial split only.",
        f"- Against the best `full_merged` TFT run, GRU wins {win_counts['RMSE']}/16 horizons on RMSE, {win_counts['MAE']}/16 on MAE, {win_counts['nRMSE']}/16 on nRMSE, and {win_counts['NSE']}/16 on NSE.",
        "- If another slide says TFT wins overall, that slide is almost certainly mixing in `full_raw` TFT runs or using a different aggregation.",
    ]
    (OUT_TXT / "slide12_nse_plot_note.md").write_text("\n".join(nse_note), encoding="utf-8")


def build_spatial_split_plots(coords: pd.DataFrame) -> None:
    rings = _load_boundary_rings(BOUNDARY_FILE, {"Brandenburg"})

    split_cluster = pd.read_csv(SPLITS_DIR / "spatial_split_full_merged_spf0p8_sc20_ss42.csv")
    split_maxdist = pd.read_csv(SPLITS_DIR / "random_max_dist_90.csv")

    ids = coords["id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(42)
    n_holdout = int((split_maxdist["spatial_split"] == "spatial_holdout").sum())
    holdout = set(rng.choice(ids, size=n_holdout, replace=False).tolist())
    split_random = pd.DataFrame({
        "id": ids,
        "cluster": 0,
        "spatial_split": np.where(pd.Series(ids).isin(holdout), "spatial_holdout", "spatial_train"),
    })
    split_random.to_csv(OUT_MET / "random_split_90_seed42.csv", index=False)

    split_map = {
        "cluster": split_cluster,
        "random": split_random,
        "maxdist": split_maxdist,
    }

    def _single_plot(df_split: pd.DataFrame, title: str, out_name: str) -> None:
        df = df_split.merge(coords, on="id", how="left")
        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_boundary(ax, rings)
        cmap = df["spatial_split"].map({"spatial_train": "#1f77b4", "spatial_holdout": "#ff7f0e"})
        ax.scatter(df["x_25833"], df["y_25833"], c=cmap, s=12, alpha=0.85)
        n_tr = int((df["spatial_split"] == "spatial_train").sum())
        n_ho = int((df["spatial_split"] == "spatial_holdout").sum())
        ax.set_title(f"{title} (train={n_tr}, holdout={n_ho})")
        ax.set_xlabel("x_25833")
        ax.set_ylabel("y_25833")
        ax.grid(True, alpha=0.2)
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(color="#1f77b4", label="spatial_train"), Patch(color="#ff7f0e", label="spatial_holdout")],
            frameon=False,
        )
        fig.tight_layout()
        fig.savefig(OUT_FIG / out_name, dpi=220)
        plt.close(fig)

    _single_plot(split_cluster, "Cluster-Based Split", "split_train_holdout_cluster.png")
    _single_plot(split_random, "Random Split (seed=42)", "split_train_holdout_random.png")
    _single_plot(split_maxdist, "Max-Distance-Constrained Split", "split_train_holdout_maxdist.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, (name, sp) in zip(axes, split_map.items()):
        df = sp.merge(coords, on="id", how="left")
        _plot_boundary(ax, rings)
        cmap = df["spatial_split"].map({"spatial_train": "#1f77b4", "spatial_holdout": "#ff7f0e"})
        ax.scatter(df["x_25833"], df["y_25833"], c=cmap, s=10, alpha=0.8)
        n_tr = int((df["spatial_split"] == "spatial_train").sum())
        n_ho = int((df["spatial_split"] == "spatial_holdout").sum())
        ax.set_title(f"{name}\ntrain={n_tr}, holdout={n_ho}")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x_25833")
    axes[0].set_ylabel("y_25833")
    fig.suptitle("Spatial Split Strategies: Train vs Holdout")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "split_train_holdout_comparison.png", dpi=220)
    plt.close(fig)


def build_distance_vs_rmse_plot(coords: pd.DataFrame) -> None:
    gp_summary_path = ROOT / "reports" / "metrics" / "gp" / "gp_metrics_summary.csv"
    gp_summary = pd.read_csv(gp_summary_path)
    gp_summary = gp_summary[gp_summary["model_prefix"] == "GRU_FCOV"].copy()
    if gp_summary.empty:
        raise RuntimeError("No GRU_FCOV rows found in gp_metrics_summary.csv")
    best = gp_summary.loc[gp_summary["NSE_pooled"].idxmax()]
    run_dir = ROOT / "outputs" / "gp" / str(best["dir_name"])
    pred = pq.read_table(run_dir / "gp_pred.parquet").to_pandas()

    split_path = SPLITS_DIR / "spatial_split_full_merged_spf0p8_sc20_ss42.csv"
    split = pd.read_csv(split_path)
    train_ids = set(split.loc[split["spatial_split"] == "spatial_train", "id"])
    holdout_ids = set(split.loc[split["spatial_split"] == "spatial_holdout", "id"])

    pred_h = pred[pred["id"].isin(holdout_ids)].copy()
    rmse_rows = []
    for wid, g in pred_h.groupby("id"):
        y = g["gws_true"].to_numpy(dtype=float)
        p = g["gws_forecast"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        rmse_rows.append({"id": wid, "rmse": rmse})
    rmse_df = pd.DataFrame(rmse_rows)

    train_xy = coords[coords["id"].isin(train_ids)][["x_25833", "y_25833"]].to_numpy(dtype=float)
    hold_df = coords[coords["id"].isin(rmse_df["id"])][["id", "x_25833", "y_25833"]].copy().reset_index(drop=True)
    hold_xy = hold_df[["x_25833", "y_25833"]].to_numpy(dtype=float)

    dmat = np.sqrt(((hold_xy[:, None, :] - train_xy[None, :, :]) ** 2).sum(axis=2))
    k_list = [1, 3, 5, 10]
    for k in k_list:
        idx = np.argpartition(dmat, kth=k - 1, axis=1)[:, :k]
        hold_df[f"d_k{k}_mean_m"] = np.array([dmat[i, idx[i]].mean() for i in range(len(hold_df))], dtype=float)

    merged = hold_df.merge(rmse_df, on="id", how="left")
    merged.to_csv(OUT_MET / "rmse_vs_train_distance_per_well.csv", index=False)

    def _distance_figure(out_name: str, title: str, *, log_y: bool, trim_q: float | None = None) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes = axes.flatten()
        for ax, k in zip(axes, k_list):
            x_all = merged[f"d_k{k}_mean_m"].to_numpy(dtype=float) / 1000.0
            y_all = merged["rmse"].to_numpy(dtype=float)
            mask = np.isfinite(x_all) & np.isfinite(y_all) & (y_all > 0)
            x = x_all[mask]
            y = y_all[mask]
            if trim_q is not None:
                y_cut = float(np.quantile(y, trim_q))
                keep = y <= y_cut
                x = x[keep]
                y = y[keep]
            ax.scatter(x, y, s=22, alpha=0.75, color="#1f77b4")
            if len(x) > 1:
                xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                if log_y:
                    m, b = np.polyfit(x, np.log10(y), 1)
                    ax.plot(xx, np.power(10.0, m * xx + b), color="#d62728", linewidth=1.6)
                else:
                    m, b = np.polyfit(x, y, 1)
                    ax.plot(xx, m * xx + b, color="#d62728", linewidth=1.6)
            spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
            ax.set_title(f"k={k} | rho={spearman:.2f}")
            ax.set_xlabel(f"Average distance to k={k} nearest train wells (km)")
            ax.set_ylabel("Per-well RMSE (m)")
            if log_y:
                ax.set_yscale("log")
                ax.set_ylabel("Per-well RMSE (m, log scale)")
            ax.grid(True, alpha=0.25)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(OUT_FIG / out_name, dpi=220)
        plt.close(fig)

    _distance_figure(
        "rmse_vs_avg_distance_k_1_3_5_10.png",
        "Holdout Per-well RMSE vs Distance to Nearest Training Wells",
        log_y=True,
    )
    _distance_figure(
        "rmse_vs_avg_distance_k_1_3_5_10_trimmed_p98.png",
        "Holdout Per-well RMSE vs Distance to Nearest Training Wells (trimmed at P98 RMSE)",
        log_y=False,
        trim_q=0.98,
    )

    def _poly_fit_figure(out_name: str) -> None:
        fit_colors = {"deg1": "#d62728", "deg2": "#2ca02c", "deg3": "#ff7f0e", "power": "#9467bd"}
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()
        r2_table = []

        for ax, k in zip(axes, k_list):
            x_all = merged[f"d_k{k}_mean_m"].to_numpy(dtype=float) / 1000.0
            y_all = merged["rmse"].to_numpy(dtype=float)
            mask = np.isfinite(x_all) & np.isfinite(y_all) & (y_all > 0)
            x = x_all[mask]
            y = y_all[mask]
            y_cut = float(np.quantile(y, 0.98))
            keep = y <= y_cut
            x, y = x[keep], y[keep]

            ax.scatter(x, y, s=22, alpha=0.6, color="#1f77b4", zorder=1)
            xx = np.linspace(x.min(), x.max(), 300)
            row = {"k": k}

            for deg, col, lbl in [
                (1, fit_colors["deg1"], "Linear"),
                (2, fit_colors["deg2"], "Quadratic"),
                (3, fit_colors["deg3"], "Cubic"),
            ]:
                coeffs = np.polyfit(x, y, deg)
                y_fit = np.polyval(coeffs, x)
                ss_res = float(np.sum((y - y_fit) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                ax.plot(xx, np.polyval(coeffs, xx), color=col, linewidth=1.6,
                        label=f"{lbl} R²={r2:.3f}")
                row[f"r2_deg{deg}"] = r2

            pos = x > 0.05
            if pos.sum() > 5:
                lx = np.log10(x[pos])
                ly = np.log10(y[pos])
                m_p, b_p = np.polyfit(lx, ly, 1)
                xx_pos = np.linspace(x[pos].min(), x[pos].max(), 300)
                y_pow_fit = np.power(10.0, m_p * np.log10(x[pos]) + b_p)
                ss_res_p = float(np.sum((y[pos] - y_pow_fit) ** 2))
                ss_tot_p = float(np.sum((y[pos] - y[pos].mean()) ** 2))
                r2_pow = 1.0 - ss_res_p / ss_tot_p if ss_tot_p > 0 else float("nan")
                ax.plot(xx_pos, np.power(10.0, m_p * np.log10(xx_pos) + b_p),
                        color=fit_colors["power"], linewidth=1.6, linestyle="--",
                        label=f"Power-law R²={r2_pow:.3f}")
                row["r2_power"] = r2_pow
            else:
                row["r2_power"] = float("nan")

            r2_table.append(row)
            spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
            ax.set_title(f"k={k}  |  Spearman ρ={spearman:.2f}")
            ax.set_xlabel(f"Avg dist to k={k} nearest train wells (km)")
            ax.set_ylabel("Per-well RMSE (m)")
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.25)

        fig.suptitle("RMSE vs Distance: Polynomial / Power-law Fit Comparison (P98 trimmed)")
        fig.tight_layout()
        fig.savefig(OUT_FIG / out_name, dpi=220)
        plt.close(fig)

        print("\n── Polynomial fit R² summary ──")
        header = f"{'k':>4}  {'Linear R²':>10}  {'Quadratic R²':>13}  {'Cubic R²':>9}  {'Power-law R²':>13}"
        print(header)
        for row in r2_table:
            print(f"{row['k']:>4}  {row.get('r2_deg1', float('nan')):>10.4f}"
                  f"  {row.get('r2_deg2', float('nan')):>13.4f}"
                  f"  {row.get('r2_deg3', float('nan')):>9.4f}"
                  f"  {row.get('r2_power', float('nan')):>13.4f}")

    _poly_fit_figure("rmse_vs_distance_poly_comparison.png")

    p98 = float(np.quantile(merged["rmse"].to_numpy(dtype=float), 0.98))
    note = [
        f"GP run used: {run_dir.name}",
        f"Split used: {split_path.name}",
        f"Holdout wells: {len(rmse_df)}",
        "Figure titles now report only Spearman rho, not Pearson r.",
        "rho is the rank correlation: it measures whether more isolated wells tend to have worse RMSE, without assuming a linear relationship.",
        f"Default figure uses log-y scale so all wells remain visible, including RMSE outliers up to {float(merged['rmse'].max()):.2f} m.",
        f"Linear backup figure trims wells above the 98th percentile RMSE ({p98:.2f} m).",
        "Recommendation: use the log-scale figure on the slide and keep the trimmed plot only as a backup.",
    ]
    (OUT_TXT / "slide14_distance_plot_notes.txt").write_text("\n".join(note), encoding="utf-8")


def build_maxdist_feasibility_note() -> None:
    split_files = sorted(SPLITS_DIR.glob("*max_dist*.csv"))
    gp_dirs = [p.name for p in (ROOT / "outputs" / "gp").iterdir() if p.is_dir()]
    maxdist_runs = [d for d in gp_dirs if "max_dist" in d.lower() or "rmd" in d.lower()]

    lines = [
        "# Max-Distance Constraint: Performance-vs-d Feasibility",
        "",
        f"- Max-distance split files found: {len(split_files)}",
        f"- GP run directories with max-distance-like tags found: {len(maxdist_runs)}",
    ]
    if len(split_files) <= 1 or len(maxdist_runs) <= 1:
        lines += [
            "",
            "Conclusion: Not feasible from current local artifacts.",
            "Reason: Need multiple trained/evaluated runs at different distance constraints `d`.",
            "Current repo has only one explicit max-distance split (`random_max_dist_90.csv`) and no comparable run grid over `d`.",
        ]
    else:
        lines += [
            "",
            "Potentially feasible: enough artifacts found to build a performance-vs-d curve.",
            "Implementing this requires parsing each run's effective split constraint.",
        ]
    (OUT_TXT / "max_distance_performance_feasibility.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    paths = _resolve_data_paths()
    merged_path = paths["full_merged_path"]
    meta_path = paths["metadata_path"]

    coords = _load_coords(merged_path)

    model_ids = set(coords["id"].astype(str).unique().tolist())
    build_metadata_assets(meta_path, model_ids=model_ids)
    build_gp_feature_importance_assets()
    build_gru_tft_horizon_plots()
    build_spatial_split_plots(coords)
    build_distance_vs_rmse_plot(coords)
    build_maxdist_feasibility_note()

    print(f"Done. Assets written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
