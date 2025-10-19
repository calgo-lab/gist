import os
import sys
import pandas as pd
import yaml


def load_table(path: str, sep: str | None = None) -> pd.DataFrame:
    pl = path.lower()
    if pl.endswith((".parq", ".parquet")):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=sep)


def main(cfg):
    data_path = cfg["data_path"]
    meta_path = cfg.get("metadata_path")
    dk = cfg.get("data_join_key")
    mk = cfg.get("meta_join_key")
    tcol = cfg.get("time_col")
    vcol = cfg.get("value_col")


    data = load_table(data_path)
    meta = load_table(meta_path, sep=";")

    # convert time column
    if tcol:
        data[tcol] = pd.to_datetime(data[tcol], errors="coerce")
        nulls = data[tcol].isna().sum()

    # create a head preview
    os.makedirs("data", exist_ok=True)
    head_path = "data/head_preview.csv"
    data.head(20).to_csv(head_path, index=False)
    
    # also create a sample of first 5000 rows
    sample_path = "data/sample.csv"
    data.head(5000).to_csv(sample_path, index=False)

    return 0


if __name__ == "__main__":
    cfg_path = "configs/data.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    sys.exit(main(cfg))

