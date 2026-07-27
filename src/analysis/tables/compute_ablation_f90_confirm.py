from pathlib import Path
import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parents[3]
GP_OUT_DIR  = ROOT / "outputs" / "gp"
MERGED_PATH = ROOT.parent / "data" / "merged.parquet"

SPLIT_SEEDS = list(range(42, 52))
GP_SEEDS    = [1]

FEATURE_CONFIGS = [
    "coords_only",
    "hydroraum_only",
    "gespannt_only",
    "siwa_only",
    "pr52lag_only",
]


def nrmse_pw(pred_df: pd.DataFrame, per_well_iqr: pd.Series) -> float:
    per_well = (
        pred_df.groupby("id")
        .apply(
            lambda g: float(np.sqrt(((g["gws_true"] - g["gws_forecast"]) ** 2).mean())),
            include_groups=False,
        )
        .rename("rmse")
        .to_frame()
        .join(per_well_iqr)
    )
    per_well["nrmse"] = per_well["rmse"] / per_well["iqr"]
    return float(per_well["nrmse"].median())


def rmse_pw(pred_df: pd.DataFrame) -> float:
    per_well = (
        pred_df.groupby("id")
        .apply(
            lambda g: float(np.sqrt(((g["gws_true"] - g["gws_forecast"]) ** 2).mean())),
            include_groups=False,
        )
    )
    return float(per_well.median())


def load_pred(ss, feat, gps):
    dir_name  = f"ORACLE_oracle_ablat_rand_f90_ss{ss}__feat_{feat}_gps{gps}__predobstrain"
    pred_path = GP_OUT_DIR / dir_name / "gp_pred.parquet"
    if not pred_path.exists():
        return None
    return pd.read_parquet(pred_path, columns=["id", "gws_true", "gws_forecast"])


def aggregate(feat, per_well_iqr):
    seed_nrmse, seed_rmse = [], []
    missing = 0
    for ss in SPLIT_SEEDS:
        gps_nrmse, gps_rmse = [], []
        for gps in GP_SEEDS:
            pred = load_pred(ss, feat, gps)
            if pred is None:
                missing += 1
                continue
            gps_nrmse.append(nrmse_pw(pred, per_well_iqr))
            gps_rmse.append(rmse_pw(pred))
        if gps_nrmse:
            seed_nrmse.append(float(np.median(gps_nrmse)))
            seed_rmse.append(float(np.median(gps_rmse)))
    return {
        "nRMSE_pw": float(np.mean(seed_nrmse)) if seed_nrmse else float("nan"),
        "RMSE_pw":  float(np.mean(seed_rmse))  if seed_rmse  else float("nan"),
        "nRMSE_std": float(np.std(seed_nrmse)) if seed_nrmse else float("nan"),
        "RMSE_std":  float(np.std(seed_rmse))  if seed_rmse  else float("nan"),
        "n_seeds": len(seed_nrmse),
        "missing": missing,
    }


def main():
    print("Loading merged.parquet for per-well IQR...")
    df_full = pd.read_parquet(MERGED_PATH, columns=["id", "gws"])
    per_well_iqr = (
        df_full.groupby("id")["gws"]
        .apply(lambda x: float(np.percentile(x.dropna(), 75) - np.percentile(x.dropna(), 25)))
        .rename("iqr")
    )
    print(f"  {len(per_well_iqr)} wells with IQR computed.\n")

    rows = []
    for feat in FEATURE_CONFIGS:
        r = aggregate(feat, per_well_iqr)
        rows.append({"feat": feat, **r})
        if r["missing"]:
            print(f"  WARNING: {r['missing']} missing files for {feat}")

    baseline_nrmse = next(r for r in rows if r["feat"] == "coords_only")["nRMSE_pw"]
    rows_sorted = sorted(rows, key=lambda r: r["nRMSE_pw"])

    print(f"### RAND f90 confirmatory ablation — oracle GP")
    print(f"### {len(SPLIT_SEEDS)} split seeds × {len(GP_SEEDS)} GP seed, 104 holdout wells (siwa/pr52lag: 3 seeds only)")
    print(f"### Baseline coords_only: nRMSE_pw = {baseline_nrmse:.3f}\n")

    print(f"| Feature config | nRMSE_pw | ±std | RMSE_pw (m) | ±std | vs baseline (nRMSE) | n_seeds |")
    print(f"|---|---|---|---|---|---|---|")
    for r in rows_sorted:
        delta = r["nRMSE_pw"] - baseline_nrmse
        sign  = "−" if delta < 0 else "+"
        print(
            f"| {r['feat']} "
            f"| {r['nRMSE_pw']:.3f} | {r['nRMSE_std']:.3f} "
            f"| {r['RMSE_pw']:.3f} | {r['RMSE_std']:.3f} "
            f"| {sign}{abs(delta):.3f} "
            f"| {r['n_seeds']} |"
        )


if __name__ == "__main__":
    main()
