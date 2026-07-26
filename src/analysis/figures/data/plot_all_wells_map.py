from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SHARED_DATA = ROOT.parent / "data"
REPO_DATA = ROOT / "data"


def main():
    meta = pd.read_csv(SHARED_DATA / "meta_LFU_info.csv", sep=";")
    merged_ids = pq.read_table(SHARED_DATA / "merged.parquet", columns=["id"]).to_pandas()["id"].unique()
    meta = meta[meta["id"].isin(merged_ids)].dropna(subset=["x_25833", "y_25833"])

    gdf_wells = gpd.GeoDataFrame(
        meta,
        geometry=gpd.points_from_xy(meta["x_25833"], meta["y_25833"]),
        crs="EPSG:25833",
    )

    boundary = gpd.read_file(REPO_DATA / "boundaries" / "geoBoundaries-DEU-ADM1_simplified.geojson")
    boundary = boundary[boundary["shapeName"] == "Brandenburg"].to_crs("EPSG:25833")
    boundary_union = boundary.union_all()

    fig, ax = plt.subplots(figsize=(10, 10))

    boundary.plot(ax=ax, color="white", edgecolor="0.35", linewidth=0.9)

    gdf_wells.plot(
        ax=ax,
        color="steelblue",
        markersize=20,
        alpha=0.6,
        zorder=2,
    )

    ax.set_title(
        f"Spatial distribution of all {len(gdf_wells)} groundwater monitoring wells\nBrandenburg",
        fontsize=17,
    )
    ax.set_axis_off()

    area_km2 = boundary_union.area / 1e6
    ax.text(
        0.03, 0.10,
        f"Brandenburg: {area_km2:,.0f} km²",
        transform=ax.transAxes,
        fontsize=14,
        va="bottom",
    )

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    bar_len = 50_000
    x0 = xlim[0] + 0.03 * xspan
    x1 = x0 + bar_len
    y_bar = ylim[0] + 0.035 * yspan
    y_text = ylim[0] + 0.055 * yspan
    ax.plot([x0, x1], [y_bar, y_bar], color="k", lw=2.5, zorder=6, solid_capstyle="butt")
    ax.text(
        (x0 + x1) / 2, y_text,
        "50 km  (31 mi)",
        ha="center", va="bottom", fontsize=14, zorder=6,
    )

    out_path = ROOT / "reports" / "figures" / "data" / "all_wells_map.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
