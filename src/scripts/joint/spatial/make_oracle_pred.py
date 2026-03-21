"""
Creates an oracle pred.parquet where gws_pred = true observed GWL at training wells.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from libs.spatial_split import resolve_split_path


def _load_yaml(path):
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gp/gp.yaml")
    parser.add_argument("--horizons", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    gp_cfg = _load_yaml(ROOT / args.config)
    data_cfg = _load_yaml(ROOT / "configs" / "data.yaml")
    gru_cfg = _load_yaml(ROOT / "configs" / "gru" / "gru.yaml")

    for key in ["full_merged_path", "full_raw_path", "sample_path"]:
        if key in data_cfg and data_cfg[key]:
            p = Path(data_cfg[key])
            data_cfg[key] = str(p if p.is_absolute() else (ROOT / p).resolve())

    dataset = str(gp_cfg.get("dataset", "full_merged"))
    spatial_cfg = gp_cfg.get("spatial_split", gru_cfg.get("spatial_split", {}))
    split_path = resolve_split_path(ROOT / "splits", dataset, spatial_cfg)
    split = pd.read_csv(split_path)
    train_ids = set(split.loc[split["spatial_split"] == "spatial_train", "id"])

    if dataset == "full_merged":
        data_path = Path(data_cfg["full_merged_path"])
    elif dataset == "full_raw":
        data_path = Path(data_cfg["full_raw_path"])
    else:
        data_path = Path(data_cfg["sample_path"])

    cols = ["datum", "id", "gws"]
    gws = (
        pd.read_csv(data_path, usecols=cols, low_memory=False)
        if str(data_path).lower().endswith(".csv")
        else pq.read_table(data_path, columns=cols).to_pandas()
    )
    gws["datum"] = pd.to_datetime(gws["datum"])

    train_obs = gws[gws["id"].isin(train_ids)].dropna(subset=["gws"]).copy()

    horizons = list(range(1, args.horizons + 1))
    rows = []
    for h in horizons:
        chunk = train_obs.copy()
        chunk["horizon"] = h
        rows.append(chunk)

    out_df = pd.concat(rows, ignore_index=True)
    out_df = out_df.rename(columns={"gws": "gws"})

    if args.out:
        out_path = Path(args.out)
    else:
        tag = gp_cfg.get("gru_run_sig", "oracle")
        out_path = ROOT / "outputs" / "oracle" / f"oracle_{tag}" / "predictions" / "pred.parquet"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(out_df), out_path)
    print(f"Saved oracle pred.parquet to {out_path}")
    print(f"  Train wells: {out_df['id'].nunique()}")
    print(f"  Dates: {out_df['datum'].nunique()}")
    print(f"  Horizons: {sorted(out_df['horizon'].unique())}")


if __name__ == "__main__":
    main()
