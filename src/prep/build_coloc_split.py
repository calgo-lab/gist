import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

X_COL, Y_COL = "x_25833", "y_25833"


def load_well_coords(data_path):
    tbl = pq.read_table(data_path, columns=["id", X_COL, Y_COL]).to_pandas()
    return tbl.dropna(subset=[X_COL, Y_COL]).drop_duplicates("id").reset_index(drop=True)


def colocated_ids(meta, radius_m):
    xy = meta[[X_COL, Y_COL]].to_numpy(float)
    pairs = cKDTree(xy).query_pairs(r=radius_m)
    idx = {i for pair in pairs for i in pair}
    return set(meta.loc[sorted(idx), "id"])


def density_pool(eligible, percentile, k):
    if percentile is None:
        return eligible["id"].to_numpy()
    xy = eligible[[X_COL, Y_COL]].to_numpy(float)
    dists, _ = cKDTree(xy).query(xy, k=k + 1)
    mean_nn = dists[:, 1:].mean(axis=1)
    keep = mean_nn <= np.percentile(mean_nn, percentile)
    return eligible.loc[keep, "id"].to_numpy()


def build_split(meta, pool, mode, n_holdout, n_val, seed):
    if len(pool) < n_holdout:
        raise ValueError(f"pool has {len(pool)} wells, need >= {n_holdout} for holdout")
    rng = np.random.default_rng(seed)
    split = pd.DataFrame({"id": meta["id"].tolist(), "spatial_split": "spatial_train"})

    if mode == "two-way":
        holdout = rng.choice(pool, size=n_holdout, replace=False)
        split.loc[split["id"].isin(holdout), "spatial_split"] = "spatial_holdout"
    elif mode == "three-way":
        test = rng.choice(pool, size=n_holdout, replace=False)
        remaining = np.setdiff1d(pool, test, assume_unique=False)
        if len(remaining) < n_val:
            raise ValueError(f"{len(remaining)} wells left, need >= {n_val} for val")
        val = rng.choice(remaining, size=n_val, replace=False)
        split.loc[split["id"].isin(test), "spatial_split"] = "spatial_test"
        split.loc[split["id"].isin(val), "spatial_split"] = "spatial_val"
    else:
        raise ValueError(f"unknown mode: {mode}")
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/storage/data/merged.parquet"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--coloc-radius", type=float, default=8.0)
    ap.add_argument("--density-percentile", default="none")
    ap.add_argument("--max-dist-k", type=int, default=3)
    ap.add_argument("--n-holdout", type=int, default=104)
    ap.add_argument("--n-val", type=int, default=52)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["two-way", "three-way"], default="two-way")
    args = ap.parse_args()

    if args.output.exists():
        existing = pd.read_csv(args.output)
        print(f"{args.output} already exists ({len(existing)} wells) -> leaving as-is")
        print(existing["spatial_split"].value_counts().to_string())
        return

    pct = None if str(args.density_percentile).lower() == "none" else float(args.density_percentile)

    meta = load_well_coords(args.input)
    coloc = colocated_ids(meta, args.coloc_radius)
    eligible = meta[~meta["id"].isin(coloc)].reset_index(drop=True)
    pool = density_pool(eligible, pct, args.max_dist_k)

    split = build_split(meta, pool, args.mode, args.n_holdout, args.n_val, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
