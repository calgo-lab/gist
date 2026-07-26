from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
SHARED_DATA = ROOT.parent / "data"
REPO_DATA = ROOT / "data"

COLORS = {
    "Speisungsgebiete": "#2166ac",
    "Transitgebiete":   "#f4a582",
    "Entlastungsgebiete": "#d6604d",
}
LABELS = {
    "Speisungsgebiete":   "Recharge zones",
    "Transitgebiete":     "Transit zones",
    "Entlastungsgebiete": "Discharge zones",
}

def main():
    meta = pd.read_csv(SHARED_DATA / "meta_LFU_info.csv", sep=";")
    merged_ids = pq.read_table(SHARED_DATA / "merged.parquet", columns=["id"]).to_pandas()["id"].unique()
    meta = meta[meta["id"].isin(merged_ids)].dropna(subset=["x_25833", "y_25833", "hydroraum"])

    gdf_wells = gpd.GeoDataFrame(
        meta,
        geometry=gpd.points_from_xy(meta["x_25833"], meta["y_25833"]),
        crs="EPSG:25833",
    )

    boundary = gpd.read_file(REPO_DATA / "boundaries" / "geoBoundaries-DEU-ADM1_simplified.geojson")
    boundary = boundary[boundary["shapeName"] == "Brandenburg"].to_crs("EPSG:25833")

    fig, ax = plt.subplots(figsize=(10, 10))
    boundary.plot(ax=ax, color="whitesmoke", edgecolor="gray", linewidth=0.5)

    for hydroraum, group in gdf_wells.groupby("hydroraum"):
        group.plot(
            ax=ax,
            color=COLORS.get(hydroraum, "black"),
            markersize=8,
            alpha=0.8,
            label=LABELS.get(hydroraum, hydroraum),
        )

    legend_patches = [
        mpatches.Patch(color=COLORS[h], label=LABELS[h])
        for h in COLORS if h in gdf_wells["hydroraum"].unique()
    ]
    ax.legend(handles=legend_patches, fontsize=11, loc="upper left")
    ax.set_title("Well distribution by hydroraum type", fontsize=14)
    ax.set_axis_off()

    out_path = ROOT / "reports" / "figures" / "data" / "hydroraum_map.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
