import sys, pandas as pd, yaml
from pathlib import Path

def load_table(path: str, sep: str | None = None) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parq", ".parquet")):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=sep)

def main(cfg):
    raw_path   = cfg["full_raw_path"]
    meta_path  = cfg.get("metadata_path")
    merged_out = cfg["full_merged_path"]
    sample_out = cfg["sample_path"]

    dk, mk = cfg.get("data_join_key"), cfg.get("meta_join_key")
    tcol   = cfg.get("time_col")
    data_sep = cfg.get("data_sep", ",")
    meta_sep = cfg.get("meta_sep", ";")

    # load
    data = load_table(raw_path, sep=data_sep)
    meta = load_table(meta_path, sep=meta_sep) if meta_path else None

    # time to datetime
    if tcol and tcol in data.columns:
        data[tcol] = pd.to_datetime(data[tcol], errors="coerce")

    # merge metadata if present
    merged = data
    if meta is not None and dk and mk and dk in data.columns and mk in meta.columns:
        # drop right-side dup cols except key
        drop_cols = [c for c in meta.columns if c != mk and c in data.columns]
        right = meta.drop(columns=drop_cols)
        merged = data.merge(right, left_on=dk, right_on=mk, how="left")

    # write outputs (PVC paths)
    Path(merged_out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(merged_out, index=False)

    # sample from merged so statics are present
    Path(sample_out).parent.mkdir(parents=True, exist_ok=True)
    merged.head(5000).to_csv(sample_out, index=False)

    # small preview for convenience (local)
    Path("data").mkdir(parents=True, exist_ok=True)
    merged.head(20).to_csv("data/head_preview.csv", index=False)

    print(f"wrote merged={merged_out}, sample={sample_out}, rows={len(merged):,}, cols={merged.shape[1]}")
    return 0

if __name__ == "__main__":
    with open("configs/data.yaml","r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sys.exit(main(cfg))
