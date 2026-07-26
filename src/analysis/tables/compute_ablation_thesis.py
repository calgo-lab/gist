from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parents[3]
GP_OUT_DIR  = ROOT / "outputs" / "gp"
MERGED_PATH = ROOT.parent / "data" / "merged.parquet"
OUT_CSV     = ROOT / "reports" / "tables" / "ablation_bar_data.csv"

SS = 47
GP_SEEDS = [1, 2, 3]
SPLITS = ["rand_f95", "rand_f80"]

def compute_range_nrmse(pred_df: pd.DataFrame, per_well_range: pd.Series) -> float:
    per_well = (
        pred_df.groupby("id")
        .apply(lambda g: np.sqrt(((g["gws_true"] - g["gws_forecast"]) ** 2).mean()),
               include_groups=False)
        .rename("rmse")
        .to_frame()
        .join(per_well_range)
    )
    per_well = per_well[per_well["range"] > 0]
    per_well["nrmse_range"] = per_well["rmse"] / per_well["range"]
    return float(per_well["nrmse_range"].median())

def main():
    print("Loading merged.parquet for per-well range...")
    df_full = pd.read_parquet(MERGED_PATH, columns=["id", "gws"])
    per_well_range = (
        df_full.groupby("id")["gws"]
        .agg(lambda x: x.max() - x.min())
        .rename("range")
    )
    print(f"  {len(per_well_range)} wells.")

    pat = re.compile(
        r"ORACLE_oracle_ablat_(?P<split>rand_f95|rand_f80)_ss(?P<ss>\d+)"
        r"__feat_(?P<feat>.+?)_gps(?P<gps>\d+)__predobstrain"
    )
    feat_configs = set()
    for d in GP_OUT_DIR.iterdir():
        m = pat.match(d.name)
        if m and int(m["ss"]) == SS:
            feat_configs.add(m["feat"])
    feat_configs = sorted(feat_configs)
    print(f"Found {len(feat_configs)} feature configs: {feat_configs}")

    rows = []
    for split in SPLITS:
        for feat in feat_configs:
            nrmse_vals = []
            for gps in GP_SEEDS:
                dir_name = f"ORACLE_oracle_ablat_{split}_ss{SS}__feat_{feat}_gps{gps}__predobstrain"
                pred_path = GP_OUT_DIR / dir_name / "gp_pred.parquet"
                if not pred_path.exists():
                    print(f"  MISSING: {dir_name}")
                    continue
                pred = pd.read_parquet(pred_path, columns=["id", "gws_true", "gws_forecast"])
                val = compute_range_nrmse(pred, per_well_range)
                nrmse_vals.append(val)
            if nrmse_vals:
                rows.append({
                    "split": split,
                    "feat_config": feat,
                    "nRMSE_pw_mean": np.mean(nrmse_vals),
                    "nRMSE_pw_std": np.std(nrmse_vals),
                    "n_gp_seeds": len(nrmse_vals),
                })
                print(f"  {split} {feat}: nRMSE={np.mean(nrmse_vals):.4f} ± {np.std(nrmse_vals):.4f}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}")

if __name__ == "__main__":
    main()
