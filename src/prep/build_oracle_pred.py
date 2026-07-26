import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def select_well_ids(split_file, subset):
    split = pd.read_csv(split_file)
    if subset == "spatial-train":
        return set(split.loc[split["spatial_split"] == "spatial_train", "id"])
    if subset == "all-ids":
        return set(split["id"])
    raise ValueError(f"unknown subset: {subset}")


def build_oracle(data_path, well_ids, in_len, out_len, val_cutoff):
    gws = pq.read_table(data_path, columns=["datum", "id", "gws"]).to_pandas()
    gws["datum"] = pd.to_datetime(gws["datum"])
    gws = gws[gws["id"].isin(well_ids)].sort_values(["id", "datum"])
    cutoff = np.datetime64(val_cutoff)

    rows = []
    for gid, g in gws.groupby("id"):
        g = g.sort_values("datum").reset_index(drop=True)
        times = g["datum"].values
        vals = g["gws"].values
        for i in range(in_len, len(g) - out_len + 1):
            if times[i + out_len - 1] <= cutoff:
                continue
            for h in range(out_len):
                v = float(vals[i + h])
                rows.append({
                    "id": gid,
                    "datum": pd.Timestamp(times[i + h]),
                    "horizon": h + 1,
                    "gws": v if np.isfinite(v) else float("nan"),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/storage/data/merged.parquet"))
    ap.add_argument("--split-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--well-subset", choices=["spatial-train", "all-ids"], default="spatial-train")
    ap.add_argument("--in-len", type=int, default=52)
    ap.add_argument("--out-len", type=int, default=16)
    ap.add_argument("--val-cutoff", default="2020-01-01")
    args = ap.parse_args()

    well_ids = select_well_ids(args.split_file, args.well_subset)
    df = build_oracle(args.input, well_ids, args.in_len, args.out_len,
                      pd.Timestamp(args.val_cutoff))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), args.output)


if __name__ == "__main__":
    main()
