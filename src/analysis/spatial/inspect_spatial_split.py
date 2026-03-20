from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import yaml
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from pyproj import Transformer
from libs.spatial_split import resolve_split_path


def _load_boundary_rings(path, names):
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = [
        f for f in data.get("features", [])
        if f.get("properties", {}).get("shapeName") in names
    ]
    if not feats:
        return []
    tr = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
    rings = []
    for feat in feats:
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        polys = [coords] if gtype == "Polygon" else coords if gtype == "MultiPolygon" else []
        for poly in polys:
            for ring in poly:
                xs_r, ys_r = zip(*ring)
                xx_r, yy_r = tr.transform(xs_r, ys_r)
                rings.append((xx_r, yy_r))
    return rings


def _plot_admin_boundaries(ax, rings, color="0.35", linewidth=1.1, alpha=0.9):
    for xr, yr in rings:
        ax.plot(xr, yr, color=color, linewidth=linewidth, alpha=alpha, zorder=1)


def main():
    DATASET = "full_merged"

    tft_cfg = yaml.safe_load((ROOT / "configs" / "tft.yaml").read_text()) or {}
    spatial_cfg = tft_cfg.get("spatial_split", {}) if isinstance(tft_cfg.get("spatial_split", {}), dict) else {}
    split_path = resolve_split_path(ROOT / "splits", DATASET, spatial_cfg)
    split = pd.read_csv(split_path)

    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    data_path = (ROOT / cfg["full_merged_path"]).resolve()

    coords = pq.read_table(data_path, columns=["id", "x_25833", "y_25833"]).to_pandas()
    coords = coords.drop_duplicates("id")

    df = split.merge(coords, on="id", how="left")
    boundary_path = ROOT / "data" / "boundaries" / "geoBoundaries-DEU-ADM1_simplified.geojson"
    boundary_rings = _load_boundary_rings(boundary_path, {"Brandenburg"}) if boundary_path.exists() else []

    figures_dir = ROOT / "reports" / "spatial_split" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_admin_boundaries(ax, boundary_rings)
    clusters = sorted(df["cluster"].dropna().unique())
    colors = "#ff4fa3", "#56B4E9", "#4daf4a", "#ff7f00", "#000000"]
    markers = ["o", "x", "*", "^"]

    for i, c in enumerate(clusters):
        sub = df[df["cluster"] == c]
        color = colors[(i // len(markers)) % len(colors)]
        marker = markers[i % len(markers)]
        if marker in ["o", "^"]:
            ax.scatter(
                sub["x_25833"], sub["y_25833"],
                s=30, alpha=0.9,
                facecolors="none", edgecolors=color, linewidths=1.2,
                marker=marker, label=str(int(c)),
            )
        else:
            ax.scatter(
                sub["x_25833"], sub["y_25833"],
                s=30, alpha=0.9,
                color=color, marker=marker, label=str(int(c)),
            )

    ax.set_title("Spatial clusters (KMeans)")
    ax.set_xlabel("x_25833")
    ax.set_ylabel("y_25833")
    ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(figures_dir / "spatial_clusters_plot.png", dpi=200)
    plt.close(fig)
    print(f"Saved {figures_dir / 'spatial_clusters_plot.png'}")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    _plot_admin_boundaries(ax2, boundary_rings)
    color_map = df["spatial_split"].map({"spatial_train": "tab:blue", "spatial_holdout": "tab:orange"})
    ax2.scatter(df["x_25833"], df["y_25833"], c=color_map, s=10, alpha=0.8)
    from matplotlib.patches import Patch
    ax2.legend(
        handles=[
            Patch(color="tab:blue", label="spatial_train"),
            Patch(color="tab:orange", label="spatial_holdout"),
        ],
        frameon=False,
    )
    ax2.set_title("Spatial split: train vs holdout")
    ax2.set_xlabel("x_25833")
    ax2.set_ylabel("y_25833")
    fig2.tight_layout()
    fig2.savefig(figures_dir / "spatial_split_train_holdout.png", dpi=200)
    plt.close(fig2)
    print(f"Saved {figures_dir / 'spatial_split_train_holdout.png'}")


if __name__ == "__main__":
    main()
