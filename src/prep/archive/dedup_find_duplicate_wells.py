import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

DATA_FILE  = ROOT.parent / "data" / "merged.parquet"
SPLITS_DIR = ROOT / "splits"
OUT_FILE   = SPLITS_DIR / "dedup_well_ids.csv"

DIST_M_THRESHOLD   = 50.0
GWL_DIFF_THRESHOLD = 0.5


def deduplicate_wells(gws: pd.DataFrame) -> list[str]:

    meta = (
        gws.groupby("id")
           .agg(
               x=("x_25833", "first"),
               y=("y_25833", "first"),
               mean_gwl=("gws", "mean"),
               n_obs=("gws", "count"),
           )
           .reset_index()
    )

    coords = meta[["x", "y"]].values
    tree   = cKDTree(coords)
    pairs  = list(tree.query_pairs(r=DIST_M_THRESHOLD))

    rows, cols = [], []
    for i, j in pairs:
        if abs(meta.iloc[i]["mean_gwl"] - meta.iloc[j]["mean_gwl"]) <= GWL_DIFF_THRESHOLD:
            rows += [i, j]
            cols += [j, i]

    n = len(meta)
    if rows:
        graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(n)

    meta["_comp"] = labels
    keep_ids = (
        meta.sort_values("n_obs", ascending=False)
            .groupby("_comp")["id"]
            .first()
            .tolist()
    )

    n_removed = n - len(keep_ids)
    print(
        f"  {n} → {len(keep_ids)} wells "
        f"({n_removed} duplicates removed, "
        f"dist≤{DIST_M_THRESHOLD}m & GWL_diff≤{GWL_DIFF_THRESHOLD}m)"
    )
    return keep_ids


def main():
    if OUT_FILE.exists():
        ids = pd.read_csv(OUT_FILE)["id"].tolist()
        print(f"Already exists: {OUT_FILE.name} ({len(ids)} wells) — skipping.")
        return

    print(f"Loading {DATA_FILE} ...")
    gws = pd.read_parquet(DATA_FILE)
    print(f"  {gws['id'].nunique()} wells, {len(gws)} rows")

    print("Deduplicating ...")
    keep_ids = deduplicate_wells(gws)

    SPLITS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"id": keep_ids}).to_csv(OUT_FILE, index=False)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
