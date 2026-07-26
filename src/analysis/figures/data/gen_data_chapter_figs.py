from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
import shapely
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[4]
SHARED_DATA = ROOT.parent / "data"
THESIS_FIGS = ROOT / "reports" / "figures" / "data"

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


def _bar_annotated(ax, labels, counts, color, rotation=0):
    bars = ax.bar(labels, counts, color=color)
    ax.set_ylabel("Wells")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=rotation)
    ymax = max(counts) if counts else 0
    pad = max(1.0, 0.02 * ymax) if ymax > 0 else 1.0
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + pad,
            f"{int(cnt)}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylim(top=ymax * 1.15)


def _mapped_counts(series, label_map, col_name):
    counts = (
        series.fillna("NaN").astype(str)
        .value_counts().rename_axis(col_name).reset_index(name="count")
    )
    counts["label_en"] = counts[col_name].map(lambda v: label_map.get(v, v))
    return counts


def gen_temporal_coverage_figs(df):
    time_col, id_col = "datum", "id"
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    start_years = df.groupby(id_col)[time_col].min().dt.year.value_counts().sort_index()
    start_years = start_years.copy()
    start_years.index = start_years.index.astype(int)
    y0, y1 = int(start_years.index.min()), int(start_years.index.max())
    start_years = start_years.reindex(range(y0, y1 + 1), fill_value=0).astype(int)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    start_years.plot(kind="bar", ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    years = list(start_years.index)
    pos = [i for i, y in enumerate(years) if (y - years[0]) % 5 == 0]
    ax.set_xticks(pos)
    ax.set_xticklabels([years[i] for i in pos], rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(THESIS_FIGS / "wells_by_start_year.png", dpi=200)
    plt.close(fig)
    print("✓ wells_by_start_year.png")

    end_years = df.groupby(id_col)[time_col].max().dt.year.value_counts().sort_index()
    end_years = end_years.copy()
    end_years.index = end_years.index.astype(int)
    y0, y1 = int(end_years.index.min()), int(end_years.index.max())
    end_years = end_years.reindex(range(y0, y1 + 1), fill_value=0).astype(int)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    end_years.plot(kind="bar", ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    years = list(end_years.index)
    pos = [i for i, y in enumerate(years) if (y - years[0]) % 5 == 0]
    ax.set_xticks(pos)
    ax.set_xticklabels([years[i] for i in pos], rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(THESIS_FIGS / "wells_by_end_year.png", dpi=200)
    plt.close(fig)
    print("✓ wells_by_end_year.png")

    weeks = df[time_col].dt.to_period("W-MON").dt.start_time
    active = df.groupby(weeks)[id_col].nunique().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    active.plot(ax=ax)
    ax.set_xlabel("Week (Mon-start)")
    ax.set_ylabel("Number of Active Wells")
    plt.tight_layout()
    fig.savefig(THESIS_FIGS / "active_wells_per_week.png", dpi=200)
    plt.close(fig)
    print("✓ active_wells_per_week.png")


def gen_all_wells_map():
    GRID_CELLS = 400
    BANDWIDTH_M = 22_000

    meta = pd.read_csv(SHARED_DATA / "meta_LFU_info.csv", sep=";")
    merged_ids = pq.read_table(SHARED_DATA / "merged.parquet", columns=["id"]).to_pandas()["id"].unique()
    meta = meta[meta["id"].isin(merged_ids)].dropna(subset=["x_25833", "y_25833"])

    gdf_wells = gpd.GeoDataFrame(
        meta,
        geometry=gpd.points_from_xy(meta["x_25833"], meta["y_25833"]),
        crs="EPSG:25833",
    )

    boundary = gpd.read_file(ROOT / "data" / "boundaries" / "geoBoundaries-DEU-ADM1_simplified.geojson")
    boundary = boundary[boundary["shapeName"] == "Brandenburg"].to_crs("EPSG:25833")
    boundary_union = boundary.union_all()

    xmin, ymin, xmax, ymax = boundary_union.bounds
    pad = 5_000
    xs = np.linspace(xmin - pad, xmax + pad, GRID_CELLS + 1)
    ys = np.linspace(ymin - pad, ymax + pad, GRID_CELLS + 1)
    hist, _, _ = np.histogram2d(meta["x_25833"].values, meta["y_25833"].values, bins=[xs, ys])
    hist = hist.T
    cell_size_m = (xs[-1] - xs[0]) / GRID_CELLS
    sigma_px = BANDWIDTH_M / cell_size_m
    zz = gaussian_filter(hist.astype(float), sigma=sigma_px)
    xs = (xs[:-1] + xs[1:]) / 2
    ys = (ys[:-1] + ys[1:]) / 2
    xx, yy = np.meshgrid(xs, ys)
    pts = shapely.points(xx.ravel(), yy.ravel())
    mask = shapely.contains(boundary_union, pts).reshape(xx.shape)
    zz_masked = np.where(mask, zz, np.nan)

    fig, ax = plt.subplots(figsize=(10, 10))
    boundary.plot(ax=ax, color="white", edgecolor="none")
    im = ax.imshow(
        zz_masked, origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap="Reds", aspect="equal", interpolation="bilinear", zorder=2,
    )
    boundary.plot(ax=ax, color="none", edgecolor="0.35", linewidth=0.9, zorder=3)
    gdf_wells.plot(ax=ax, color="steelblue", markersize=5, alpha=0.6, zorder=4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Well density (KDE)", fontsize=11)
    cbar.set_ticks([])
    ax.set_axis_off()
    fig.savefig(THESIS_FIGS / "all_wells_map_kde.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ all_wells_map_kde.png")


def gen_categorical_figs(meta):
    meta_w = meta.drop_duplicates("id").copy()

    if "hydroraum" in meta_w.columns:
        counts_df = _mapped_counts(meta_w["hydroraum"], HYDRORAUM_LABELS_EN, "hydroraum")
        labels = counts_df["label_en"].tolist()
        counts = counts_df["count"].astype(int).tolist()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _bar_annotated(ax, labels, counts, color="#4C78A8", rotation=15)
        fig.tight_layout()
        fig.savefig(THESIS_FIGS / "hydroraum_dist.png", dpi=220)
        plt.close(fig)
        print("✓ hydroraum_dist.png")

    if "gw_gespannt" in meta_w.columns:
        counts_df = _mapped_counts(meta_w["gw_gespannt"], CONFINEMENT_LABELS_EN, "gw_gespannt")
        labels = counts_df["label_en"].tolist()
        counts = counts_df["count"].astype(int).tolist()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _bar_annotated(ax, labels, counts, color="#59A14F")
        fig.tight_layout()
        fig.savefig(THESIS_FIGS / "aquifer_dist.png", dpi=220)
        plt.close(fig)
        print("✓ aquifer_dist.png")

    if "siwa_verweilzeit_j" in meta_w.columns:
        siwa_nonmissing = (
            meta_w.loc[meta_w["siwa_verweilzeit_j"].notna(), "siwa_verweilzeit_j"]
            .astype(float).value_counts().sort_index()
            .rename_axis("siwa_verweilzeit_j").reset_index(name="count")
        )
        siwa_missing_n = int(meta_w["siwa_verweilzeit_j"].isna().sum())
        siwa_counts = siwa_nonmissing.copy()
        siwa_counts["label_en"] = siwa_counts["siwa_verweilzeit_j"].map(lambda v: f"{int(v)} years")
        if siwa_missing_n > 0:
            siwa_counts = pd.concat([
                siwa_counts,
                pd.DataFrame({"siwa_verweilzeit_j": [np.nan], "count": [siwa_missing_n], "label_en": ["Missing"]}),
            ], ignore_index=True)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        colors = ["#4C78A8" if label != "Missing" else "#B8B8B8" for label in siwa_counts["label_en"]]
        bars = ax.bar(siwa_counts["label_en"], siwa_counts["count"], color=colors)
        ax.set_xlabel("Residence time category")
        ax.set_ylabel("Wells")
        ax.grid(True, axis="y", alpha=0.25)
        ymax = int(siwa_counts["count"].max()) if len(siwa_counts) else 0
        pad = max(1.0, 0.02 * ymax) if ymax > 0 else 1.0
        for bar, cnt in zip(bars, siwa_counts["count"].astype(int).tolist()):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + pad,
                    f"{cnt}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(top=ymax * 1.15)
        fig.tight_layout()
        fig.savefig(THESIS_FIGS / "siwa_dist.png", dpi=220)
        plt.close(fig)
        print("✓ siwa_dist.png")


def main():
    print(f"Saving figures to: {THESIS_FIGS}")
    THESIS_FIGS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SHARED_DATA / "merged.parquet")
    gen_temporal_coverage_figs(df)

    gen_all_wells_map()

    meta = pd.read_csv(SHARED_DATA / "meta_LFU_info.csv", sep=";")
    merged_ids = set(df["id"].astype(str).unique())
    meta = meta[meta["id"].astype(str).isin(merged_ids)].copy()
    gen_categorical_figs(meta)

    print("\nDone.")


if __name__ == "__main__":
    main()
