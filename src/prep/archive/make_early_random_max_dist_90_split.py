from pathlib import Path
import yaml
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve()
for p in [ROOT, *ROOT.parents]:
    if (p / "configs" / "data.yaml").exists():
        ROOT = p
        break

import sys
sys.path.insert(0, str(ROOT / "src"))
from libs.spatial_split import spatial_split_random_max_dist

data_cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
data_path = Path(data_cfg["full_merged_path"])
if not data_path.is_absolute():
    data_path = (ROOT / data_path).resolve()

coords = (
    pq.read_table(data_path, columns=["id", "x_25833", "y_25833"])
    .to_pandas()
    .drop_duplicates("id")
    .dropna(subset=["x_25833", "y_25833"])
    .reset_index(drop=True)
)
print(f"Total wells: {len(coords)}")

save_path = ROOT / "splits" / "random_max_dist_90.csv"

result = spatial_split_random_max_dist(
    coords_df=coords,
    train_fraction=0.9,
    k=3,
    d_percentile=25,
    rng_seed=42,
    save_path=save_path,
)

n_train = (result["spatial_split"] == "spatial_train").sum()
n_holdout = (result["spatial_split"] == "spatial_holdout").sum()
print(f"train: {n_train}  holdout: {n_holdout}  total: {len(result)}")
print(f"Saved → {save_path}")
