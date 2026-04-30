import pandas as pd
import glob
import os

# Find decoupled rmd90 GP output
gp_dirs = glob.glob("outputs/gp/GRU_FCOV_in52_out16_ep50_bs4096_seed40_full_merged_rmd90_decoupled__*")
print("GP dirs found:", gp_dirs)
if not gp_dirs:
    print("Listing outputs/gp:")
    for d in sorted(os.listdir("outputs/gp")):
        print(" ", d)
    raise SystemExit(1)

gp_dir = gp_dirs[0]
df = pd.read_parquet(os.path.join(gp_dir, "gp_pred.parquet"))
print(f"GP pred: {len(df)} rows, {df['id'].nunique()} wells")
print("Columns:", df.columns.tolist())

# Load the main dataset to get GOK and gw_gespannt_bin
# path from configs/data.yaml: ../data/merged.parquet
import yaml
data_cfg = yaml.safe_load(open("configs/data.yaml"))
data_path = data_cfg.get("full_merged_path", "../data/merged.parquet")
print(f"\nLoading dataset from: {data_path}")
meta = pd.read_parquet(data_path)
print("Meta shape:", meta.shape)
print("Meta columns sample:", [c for c in meta.columns if any(k in c.lower() for k in ["gespannt", "gok", "id"])])

# Get one row per well for static features
id_col = "id" if "id" in meta.columns else meta.columns[0]
gespannt_col = next((c for c in ["gw_gespannt_bin", "gw_gespannt"] if c in meta.columns), None)
print(f"Confined aquifer column: {gespannt_col}")
static_cols = [id_col] + [c for c in ["gok", gespannt_col] if c is not None and c in meta.columns]
print("Static cols available:", static_cols)

well_meta = meta[static_cols].drop_duplicates(id_col)
print(f"Well meta: {len(well_meta)} wells")

# Merge
merged = df.merge(well_meta, on="id", how="left")
print(f"Merged: {len(merged)} rows, gespannt null: {merged['gw_gespannt_bin'].isna().sum() if 'gw_gespannt_bin' in merged.columns else 'N/A'}")

if "gok" in merged.columns:
    merged["over_gok"] = merged["gws_forecast"] > merged["gok"]
    print(f"\nRows over GOK: {merged['over_gok'].sum()} / {len(merged)} ({merged['over_gok'].mean():.3%})")
    print(f"Wells with any over-GOK: {merged.groupby('id')['over_gok'].any().sum()} / {merged['id'].nunique()}")

    gespannt_col_merged = next((c for c in ["gw_gespannt_bin", "gw_gespannt"] if c in merged.columns), None)
    if gespannt_col_merged:
        print(f"\n--- Clipping rate by confined/unconfined ({gespannt_col_merged}) ---")
        print(f"Unique values: {sorted(merged[gespannt_col_merged].dropna().unique())}")
        result = merged.groupby(gespannt_col_merged)["over_gok"].agg(["mean", "sum", "count"])
        result.columns = ["clip_rate", "clipped_rows", "total_rows"]
        print(result.to_string())

        # Per-well clipping rate
        per_well = merged.groupby(["id", gespannt_col_merged])["over_gok"].mean().reset_index()
        print(f"\n--- Per-well mean clip rate by {gespannt_col_merged} ---")
        print(per_well.groupby(gespannt_col_merged)["over_gok"].describe())
    else:
        print("No confined aquifer column found in merged data")
else:
    print("No GOK column found in merged data")
    print("Available columns:", merged.columns.tolist())
