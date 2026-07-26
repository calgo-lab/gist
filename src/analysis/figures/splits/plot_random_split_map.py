from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import pyarrow.parquet as pq
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from pyproj import Transformer

ROOT       = Path(__file__).resolve().parents[4]
DATA_PATH  = ROOT.parent / "data" / "merged.parquet"
SPLIT_FILE = ROOT / "splits" / "spatial_split_full_merged_coloc_rand90_f90_ss42.csv"
BOUNDARY   = ROOT / "data" / "boundaries" / "geoBoundaries-DEU-ADM1_simplified.geojson"
OUT_PATH   = ROOT / "reports" / "figures" / "splits" / "spatial_split_rand90_ss42_presentation.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TRAIN_COLOR     = "#4C72B0"
TEST_COLOR      = "#F4915A"

LEGEND_FONTSIZE = 12


def load_boundary_rings() -> list:
    data = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    tr = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
    rings = []
    for feat in data.get("features", []):
        if feat.get("properties", {}).get("shapeName") != "Brandenburg":
            continue
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            for ring in poly:
                xs, ys = zip(*ring)
                xx, yy = tr.transform(xs, ys)
                rings.append((list(xx), list(yy)))
    return rings


def main():
    coords = (
        pq.read_table(DATA_PATH, columns=["id", "x_25833", "y_25833"])
        .to_pandas()
        .drop_duplicates("id")
        .dropna(subset=["x_25833", "y_25833"])
    )

    split = pd.read_csv(SPLIT_FILE)
    n_train = int((split["spatial_split"] == "spatial_train").sum())
    n_test  = int((split["spatial_split"] == "spatial_holdout").sum())
    n_total = n_train + n_test
    train_pct = round(100 * n_train / n_total)
    test_pct  = round(100 * n_test  / n_total)

    df = split.merge(coords, on="id", how="left")

    rings = load_boundary_rings()

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    for xr, yr in rings:
        ax.plot(xr, yr, color="0.35", linewidth=0.9, alpha=0.9, zorder=2)

    train_mask = df["spatial_split"] == "spatial_train"
    test_mask  = df["spatial_split"] == "spatial_holdout"

    ax.scatter(
        df.loc[train_mask, "x_25833"], df.loc[train_mask, "y_25833"],
        s=22, color=TRAIN_COLOR, alpha=0.75, linewidths=0, zorder=3,
    )
    ax.scatter(
        df.loc[test_mask, "x_25833"], df.loc[test_mask, "y_25833"],
        s=20, color=TEST_COLOR, alpha=0.9, linewidths=0, zorder=4,
    )

    legend_handles = [
        mpatches.Patch(color=TRAIN_COLOR, label=f"Training ({train_pct}%, n={n_train})"),
        mpatches.Patch(color=TEST_COLOR,  label=f"Test ({test_pct}%, n={n_test})"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
    )

    scalebar = AnchoredSizeBar(
        ax.transData,
        50_000,
        "50 km",
        loc="lower left",
        pad=0.5,
        color="black",
        frameon=False,
        size_vertical=3_000,
        fontproperties=_fm.FontProperties(size=10),
    )
    ax.add_artist(scalebar)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
