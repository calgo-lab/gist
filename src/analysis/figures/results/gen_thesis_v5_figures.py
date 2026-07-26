from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mcm
import matplotlib.font_manager as _fm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from scipy.ndimage import gaussian_filter


ROOT        = Path(__file__).resolve().parents[4]
SHARED_DATA = ROOT.parent / "data"
ORACLE_DIR  = ROOT / "outputs" / "oracle"
REPORTS_DIR = ROOT / "reports" / "tables"
BOUNDARY_F  = ROOT / "data" / "boundaries" / "geoBoundaries-DEU-ADM1_simplified.geojson"
FIG_DATA     = ROOT / "reports" / "figures" / "data"
FIG_ABLATION = ROOT / "reports" / "figures" / "feature_ablation_gp"
FIG_CALIB    = ROOT / "reports" / "figures" / "calibration"
FIG_PRED     = ROOT / "reports" / "figures" / "predictions"
for _d in (FIG_DATA, FIG_ABLATION, FIG_CALIB, FIG_PRED):
    _d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

HYDRORAUM_EN = {
    "Speisungsgebiete": "Recharge",
    "Transitgebiete":   "Transit",
    "Entlastungsgebiete": "Discharge",
}
HYDRORAUM_COLORS = {
    "Recharge":   "#2166ac",
    "Transit":    "#4dac26",
    "Discharge":  "#d01c8b",
}


def _make_black_turbo() -> LinearSegmentedColormap:
    turbo = mcm.get_cmap("turbo", 256)
    turbo_colors = turbo(np.linspace(0.12, 1.0, 220))
    black_ramp = np.zeros((36, 4))
    black_ramp[:, 3] = 1.0
    for i, t in enumerate(np.linspace(0, 1, 36)):
        black_ramp[i] = (1 - t) * np.array([0, 0, 0, 1]) + t * turbo_colors[0]
    return LinearSegmentedColormap.from_list("black_turbo", np.vstack([black_ramp, turbo_colors]))

BLACK_TURBO = _make_black_turbo()


def gen_mean_gwl_map():
    print("Generating mean_gwl_map.png ...")

    meta = pd.read_csv(SHARED_DATA / "meta_LFU_info.csv", sep=";")
    merged_ids = pd.read_parquet(SHARED_DATA / "merged.parquet", columns=["id"])["id"].unique()
    meta = meta[meta["id"].isin(merged_ids)].dropna(subset=["x_25833", "y_25833"]).copy()

    df_full = pd.read_parquet(
        SHARED_DATA / "gws_bb_complete_hyras_1000.parquet",
        columns=["id", "datum", "gws"],
    )
    df_full["datum"] = pd.to_datetime(df_full["datum"])
    df_2008 = df_full[df_full["datum"].dt.year >= 2008]
    mean_gwl = df_2008.groupby("id")["gws"].mean().rename("mean_gwl")
    meta = meta.join(mean_gwl, on="id")
    meta = meta.dropna(subset=["mean_gwl"])

    gdf = gpd.GeoDataFrame(
        meta,
        geometry=gpd.points_from_xy(meta["x_25833"], meta["y_25833"]),
        crs="EPSG:25833",
    )

    boundary = gpd.read_file(BOUNDARY_F)
    boundary = boundary[boundary["shapeName"] == "Brandenburg"].to_crs("EPSG:25833")

    vmin, vmax = meta["mean_gwl"].quantile(0.02), meta["mean_gwl"].quantile(0.98)
    norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 9))
    boundary.plot(ax=ax, color="#f5f5f5", edgecolor="0.35", linewidth=0.9, zorder=1)

    sc = ax.scatter(
        meta["x_25833"], meta["y_25833"],
        c=meta["mean_gwl"],
        cmap=BLACK_TURBO,
        norm=norm,
        s=16, alpha=0.85, linewidths=0, zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.032, pad=0.02, shrink=0.75)
    cbar.set_label("Mean GWL (m a.s.l.)", fontsize=11)
    ax.set_axis_off()
    ax.set_aspect("equal")

    scalebar = AnchoredSizeBar(
        ax.transData,
        50_000,
        "50 km",
        loc="lower right",
        pad=0.5,
        color="black",
        frameon=False,
        size_vertical=3_000,
        fontproperties=_fm.FontProperties(size=10),
    )
    ax.add_artist(scalebar)

    plt.tight_layout()
    out = ROOT / "reports" / "figures" / "data" / "mean_gwl_map.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ mean_gwl_map.png")


def gen_gwl_timeseries():
    print("Generating gwl_timeseries_hydroraum.png ...")

    meta = pd.read_csv(SHARED_DATA / "meta_LFU_info.csv", sep=";")
    merged_ids = pd.read_parquet(SHARED_DATA / "merged.parquet", columns=["id"])["id"].unique()
    meta = meta[meta["id"].isin(merged_ids)].copy()
    meta = meta.dropna(subset=["hydroraum", "parde_seasonality"])
    meta["hydroraum_en"] = meta["hydroraum"].map(HYDRORAUM_EN)
    meta = meta[meta["hydroraum_en"].notna()]

    df = pd.read_parquet(SHARED_DATA / "merged.parquet", columns=["id", "datum", "gws"])
    df["datum"] = pd.to_datetime(df["datum"])

    t_start = pd.Timestamp("2021-01-01")
    t_end   = pd.Timestamp("2023-12-31")
    df_win = df[(df["datum"] >= t_start) & (df["datum"] <= t_end)]
    well_counts = df_win.groupby("id").size()

    chosen = {}
    for zone_de, zone_en in HYDRORAUM_EN.items():
        zone_ids = meta[meta["hydroraum"] == zone_de]["id"].tolist()
        zone_meta = meta[meta["id"].isin(zone_ids) & meta["id"].isin(well_counts[well_counts > 100].index)]
        if len(zone_meta) == 0:
            zone_meta = meta[meta["hydroraum"] == zone_de]
        target = zone_meta["parde_seasonality"].median()
        idx = (zone_meta["parde_seasonality"] - target).abs().idxmin()
        chosen[zone_en] = zone_meta.loc[idx, "id"]

    print(f"  Chosen wells: {chosen}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    zone_order = ["Recharge", "Transit", "Discharge"]

    for ax, zone in zip(axes, zone_order):
        well_id = chosen[zone]
        well_data = df[(df["id"] == well_id) & (df["datum"] >= t_start) & (df["datum"] <= t_end)]
        well_data = well_data.sort_values("datum")
        color = HYDRORAUM_COLORS[zone]
        ax.plot(well_data["datum"], well_data["gws"], color=color, lw=1.2, alpha=0.9)
        ax.set_ylabel("GWL (m a.s.l.)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", alpha=0.2, ls="--")
        ax.annotate(
            f"{zone} zone · {well_id}",
            xy=(0.02, 0.92), xycoords="axes fraction",
            fontsize=9, color=color, fontweight="bold",
        )

    axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(matplotlib.dates.YearLocator())
    axes[-1].set_xlabel("Year", fontsize=9)
    fig.align_ylabels(axes)
    plt.tight_layout(h_pad=0.5)
    fig.savefig(FIG_DATA / "gwl_timeseries_hydroraum.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ gwl_timeseries_hydroraum.png")


FEAT_LABELS = {
    "coords_only":             "Coordinates only (baseline)",
    "hydroraum_only":          "Hydrological zone only",
    "gespannt_only":           "Aquifer confinement only",
    "hydroraum_gespannt":      "Zone + confinement",
    "siwa_only":               "Residence time (SIWA) only",
    "siwa_gespannt":           "SIWA + confinement",
    "siwa_hydroraum":          "SIWA + zone",
    "siwa_hydroraum_gespannt": "SIWA + zone + confinement",
    "fok_only":                "Screen top depth only",
    "fuk_only":                "Screen bottom depth only",
    "fok_fuk":                 "Both screen depths",
    "best_fok":                "Best combo + screen top",
    "best_fuk":                "Best combo + screen bottom",
    "best_fok_fuk":            "Best combo + both screen depths",
    "siwa_fok":                "SIWA + screen top",
    "siwa_fuk":                "SIWA + screen bottom",
    "siwa_fok_fuk":            "SIWA + both screen depths",
    "pr1lag_only":             "Precip 1-week lag only",
    "pr52lag_only":            "Precip 52-week lag only",
    "pr52xcorr_only":          "Precip 52-week xcorr only",
    "best_pr1lag":             "Best combo + 1-week lag",
    "best_pr52lag":            "Best combo + 52-week lag",
    "best_pr52xcorr":          "Best combo + 52-week xcorr",
    "best_pr52lag_xcorr":      "Best combo + lag + xcorr",
    "best_pr_all":             "Best combo + all precip",
    "acf_only":                "Autocorrelation class only",
    "acf_class_only":          "ACF class only",
    "best_acf_class":          "Best combo + ACF class",
    "gfa_hydroraum":           "GFA + zone",
    "hr_gesp_acf":             "Zone + confinement + ACF",
    "siwa_hr_gesp_acf":        "SIWA + zone + confinement + ACF",
    "siwa_acf":                "SIWA + autocorrelation",
    "siwa_acf_class":          "SIWA + ACF class",
    "gwlk_cont_only":          "GWL class (continuous) only",
    "best_gwlk_cont":          "Best combo + GWL class (cont.)",
    "best_gwlk_onehot":        "Best combo + GWL class (one-hot)",
}

FEAT_GROUPS = {
    "Baseline":         {"coords_only"},
    "Hydrogeological":  {"hydroraum_only", "gespannt_only", "hydroraum_gespannt",
                         "siwa_only", "siwa_gespannt", "siwa_hydroraum",
                         "siwa_hydroraum_gespannt", "gfa_hydroraum"},
    "Screen depths":    {"fok_only", "fuk_only", "fok_fuk",
                         "best_fok", "best_fuk", "best_fok_fuk",
                         "siwa_fok", "siwa_fuk", "siwa_fok_fuk"},
    "Precipitation":    {"pr1lag_only", "pr52lag_only", "pr52xcorr_only",
                         "best_pr1lag", "best_pr52lag", "best_pr52xcorr",
                         "best_pr52lag_xcorr", "best_pr_all"},
    "ACF / GWL class":  {"acf_only", "acf_class_only", "best_acf_class",
                         "hr_gesp_acf", "siwa_hr_gesp_acf",
                         "siwa_acf", "siwa_acf_class",
                         "gwlk_cont_only", "best_gwlk_cont", "best_gwlk_onehot"},
}

GROUP_COLORS = {
    "Baseline":        "#808080",
    "Hydrogeological": "#2166ac",
    "Screen depths":   "#f4a261",
    "Precipitation":   "#4dac26",
    "ACF / GWL class": "#d01c8b",
}

SELECTED_FEAT = "hydroraum_only"


def gen_gp_ablation_bar():
    print("Generating gp_ablation_bar.png ...")

    df = pd.read_csv(REPORTS_DIR / "ablation_bar_data.csv")
    rf95 = df[df["split"] == "rand_f95"].copy()
    baseline = float(rf95.loc[rf95["feat_config"] == "coords_only", "nRMSE_pw_mean"].iloc[0])

    rf95["delta"] = rf95["nRMSE_pw_mean"] - baseline
    rf95 = rf95.sort_values("nRMSE_pw_mean", ascending=True)

    def feat_color(fc):
        for grp, members in FEAT_GROUPS.items():
            if fc in members:
                return GROUP_COLORS[grp]
        return "#aaaaaa"

    colors = [feat_color(fc) for fc in rf95["feat_config"]]
    labels = [FEAT_LABELS.get(fc, fc) for fc in rf95["feat_config"]]

    fig, ax = plt.subplots(figsize=(9, 12))
    y = np.arange(len(rf95))
    bars = ax.barh(y, rf95["nRMSE_pw_mean"], xerr=rf95["nRMSE_pw_std"],
                   color=colors, alpha=0.85, height=0.7,
                   error_kw={"elinewidth": 0.8, "capsize": 2})

    ax.axvline(baseline, color="#333333", lw=1.4, ls="--", zorder=5, label="Baseline (coords only)")

    sel_idx = rf95["feat_config"].tolist().index(SELECTED_FEAT)
    ax.scatter([rf95.iloc[sel_idx]["nRMSE_pw_mean"]], [sel_idx],
               marker="*", s=220, color="gold", edgecolors="#333", linewidths=0.5,
               zorder=10, label=f"Selected: {FEAT_LABELS[SELECTED_FEAT]}")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Median nRMSE$_{pw}$ (range-normalised, lower = better)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(left=0)

    patches = [mpatches.Patch(color=c, label=g, alpha=0.85)
               for g, c in GROUP_COLORS.items()]
    patches.append(plt.Line2D([0], [0], color="#333333", ls="--", lw=1.4, label="Baseline"))
    patches.append(plt.Line2D([0], [0], marker="*", color="w",
                               markerfacecolor="gold", markersize=11,
                               markeredgecolor="#333", label="Selected config"))
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    ax.grid(True, axis="x", alpha=0.2, ls="--")
    plt.tight_layout()
    fig.savefig(FIG_ABLATION / "gp_ablation_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ gp_ablation_bar.png")


def gen_calibration_reliability():
    print("Generating calibration_reliability.png ...")

    df = pd.read_csv(REPORTS_DIR / "calibration_curve.csv")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Perfect calibration")

    ax.plot(df["nominal_level"], df["oracle_coverage"],
            "o-", color="#2166ac", lw=1.8, ms=6, label="Oracle pipeline")
    ax.plot(df["nominal_level"], df["decoupled_coverage"],
            "s--", color="#d01c8b", lw=1.8, ms=6, label="Decoupled pipeline")

    ax.fill_between(df["nominal_level"], df["nominal_level"], df["oracle_coverage"],
                    alpha=0.07, color="#2166ac")

    ax.set_xlabel("Nominal coverage level", fontsize=11)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2, ls="--")
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(FIG_CALIB / "calibration_reliability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ calibration_reliability.png")


def gen_example_predictions():
    print("Generating example_predictions.png ...")

    oracle_file = ORACLE_DIR / "oracle_rand_f95_ss47_gps1_pred.parquet"
    df = pd.read_parquet(oracle_file)
    df["datum"] = pd.to_datetime(df["datum"])

    df1 = df[df["horizon"] == 1].copy()

    perwell_rmse = (
        df1.groupby("id")
        .apply(lambda g: np.sqrt(((g["gws_true"] - g["gws_forecast"]) ** 2).mean()),
               include_groups=False)
        .sort_values()
    )

    n = len(perwell_rmse)
    selected = {
        "Low RMSE":    perwell_rmse.index[0],
        "Median RMSE": perwell_rmse.index[n // 2],
        "High RMSE":   perwell_rmse.index[-1],
    }

    t_start = pd.Timestamp("2022-01-01")
    t_end   = pd.Timestamp("2022-12-31")

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    colors = {"Low RMSE": "#2166ac", "Median RMSE": "#f4a261", "High RMSE": "#d73027"}

    for ax, (label, well_id) in zip(axes, selected.items()):
        wd = df1[(df1["id"] == well_id) & (df1["datum"] >= t_start) & (df1["datum"] <= t_end)].sort_values("datum")
        rmse = perwell_rmse[well_id]
        color = colors[label]

        sigma = wd["gws_forecast_std"].values
        mu    = wd["gws_forecast"].values
        t     = wd["datum"].values

        ax.fill_between(t, mu - sigma, mu + sigma, alpha=0.25, color=color, label="±1σ interval")
        ax.plot(t, mu,             color=color,   lw=1.6, label="GP prediction")
        ax.plot(t, wd["gws_true"], color="black", lw=1.2, ls="-", label="Observed GWL", alpha=0.85)

        ax.set_ylabel("GWL (m a.s.l.)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", alpha=0.2, ls="--")
        ax.annotate(
            f"{label} · {well_id} · RMSE = {rmse:.2f} m",
            xy=(0.02, 0.90), xycoords="axes fraction",
            fontsize=8.5, color=color, fontweight="bold",
        )

    axes[0].legend(fontsize=8, loc="upper right", ncol=3)
    axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=25, ha="right")
    axes[-1].set_xlabel("Date (2022)", fontsize=9)
    fig.align_ylabels(axes)
    plt.tight_layout(h_pad=0.5)
    fig.savefig(FIG_PRED / "example_predictions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ example_predictions.png")


def main():
    print(f"Output directory: {ROOT / 'reports' / 'figures'}\n")
    gen_mean_gwl_map()
    gen_gwl_timeseries()
    gen_gp_ablation_bar()
    gen_calibration_reliability()
    gen_example_predictions()
    print("\nAll figures done.")


if __name__ == "__main__":
    main()
