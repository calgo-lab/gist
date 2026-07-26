from pathlib import Path
import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parents[3]
GP_OUT_DIR  = ROOT / "outputs" / "gp"
MERGED_PATH = ROOT.parent / "data" / "merged.parquet"
OUT_CSV     = ROOT / "reports" / "tables" / "oracle_ablat_iqr_nrmse_f90.csv"

SPLIT_SEEDS = list(range(42, 52))
GP_SEEDS    = [1]
SPLIT       = "rand"

FEATURE_CONFIGS = [
    "coords_only",
    "hydroraum_only",
    "gespannt_only",
    "hydroraum_gespannt",
    "siwa_only",
    "siwa_gespannt",
    "siwa_hydroraum",
    "siwa_hydroraum_gespannt",
    "fok_only",
    "fuk_only",
    "fok_fuk",
    "pr1lag_only",
    "pr52lag_only",
    "pr52xcorr_only",
    "best_fok_fuk",
    "acf_class_only",
    "siwa_acf_class",
    "best_acf_class",
    "gwlk_cont_only",
    "best_gwlk_cont",
    "best_gwlk_onehot",
    "siwa_fok",
    "siwa_fuk",
    "siwa_fok_fuk",
    "best_fok",
    "best_fuk",
    "best_pr1lag",
    "best_pr52lag",
    "best_pr52xcorr",
    "best_pr_all",
]


def per_well_rmse(pred_df: pd.DataFrame) -> pd.Series:
    return (
        pred_df.groupby("id")
        .apply(
            lambda g: float(np.sqrt(((g["gws_true"] - g["gws_forecast"]) ** 2).mean())),
            include_groups=False,
        )
        .rename("rmse")
    )


def nrmse_pw(pred_df: pd.DataFrame, per_well_iqr: pd.Series) -> float:
    pw = per_well_rmse(pred_df).to_frame().join(per_well_iqr)
    pw["nrmse"] = pw["rmse"] / pw["iqr"]
    return float(pw["nrmse"].median())


def rmse_pw(pred_df: pd.DataFrame) -> float:
    return float(per_well_rmse(pred_df).median())


def load_pred(ss, feat, gps):
    dir_name  = f"ORACLE_oracle_ablat_rand_f90_ss{ss}__feat_{feat}_gps{gps}__predobstrain"
    pred_path = GP_OUT_DIR / dir_name / "gp_pred.parquet"
    if not pred_path.exists():
        return None
    return pd.read_parquet(pred_path, columns=["id", "gws_true", "gws_forecast"])


def main():
    print("Loading merged.parquet for per-well IQR...")
    df_full = pd.read_parquet(MERGED_PATH, columns=["id", "gws"])
    per_well_iqr = (
        df_full.groupby("id")["gws"]
        .apply(lambda x: float(np.percentile(x.dropna(), 75) - np.percentile(x.dropna(), 25)))
        .rename("iqr")
    )
    print(f"  {len(per_well_iqr)} wells with IQR.\n")

    rows = []
    missing = 0
    for feat in FEATURE_CONFIGS:
        for ss in SPLIT_SEEDS:
            for gps in GP_SEEDS:
                pred = load_pred(ss, feat, gps)
                if pred is None:
                    missing += 1
                    continue
                rows.append({
                    "split":     SPLIT,
                    "feat":      feat,
                    "nRMSE_iqr": nrmse_pw(pred, per_well_iqr),
                    "rmse_pw":   rmse_pw(pred),
                })

    if missing:
        print(f"WARNING: {missing} missing prediction files.\n")

    df_out = pd.DataFrame(rows, columns=["split", "feat", "nRMSE_iqr", "rmse_pw"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df_out)} rows → {OUT_CSV}")

    agg = df_out.groupby("feat")["rmse_pw"].agg(mean="mean", std="std", n="count")
    agg = agg.sort_values("mean")
    baseline = float(agg.loc["coords_only", "mean"])
    print(f"\nBaseline coords_only RMSE_pw: {baseline:.4f} m")
    print(agg.to_string())


if __name__ == "__main__":
    main()
