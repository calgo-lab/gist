from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT      = Path(__file__).resolve().parents[4]
DATA_PATH = ROOT.parent / "data" / "merged.parquet"
GP        = ROOT / "outputs" / "gp"
OUT       = ROOT / "reports" / "tables"

SPLIT_SEEDS = list(range(42, 52))
FEAT_CONFIGS = [
    {"tag": "fuk_only",           "gp_tag": "dec_feat_alt_fuk_only_gps1"},
    {"tag": "hydroraum_gespannt", "gp_tag": "dec_feat_alt_hydroraum_gespannt_gps1"},
]


def metrics_pw(pred_path, iqr_dict):
    df = pq.read_table(pred_path, columns=["id", "gws_true", "gws_forecast"]).to_pandas().dropna()
    per_well = []
    for wid, g in df.groupby("id"):
        t = g["gws_true"].to_numpy(float)
        p = g["gws_forecast"].to_numpy(float)
        rmse  = float(np.sqrt(np.mean((p - t) ** 2)))
        iqr   = iqr_dict.get(wid, float("nan"))
        rng   = float(t.max() - t.min())
        nrmse_iqr   = rmse / iqr if (np.isfinite(iqr) and iqr > 0) else float("nan")
        nrmse_range = rmse / rng if rng > 0 else float("nan")
        per_well.append({"rmse": rmse, "nrmse_iqr": nrmse_iqr, "nrmse_range": nrmse_range})
    pw = pd.DataFrame(per_well)
    return (float(pw["nrmse_iqr"].median()),
            float(pw["nrmse_range"].median()),
            float(pw["rmse"].median()),
            len(pw))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading full-history IQR ...")
    gws = pq.read_table(DATA_PATH, columns=["id", "gws"]).to_pandas().dropna(subset=["gws"])
    q = gws.groupby("id")["gws"].quantile([0.25, 0.75]).unstack()
    q.columns = ["q25", "q75"]
    iqr_dict = (q["q75"] - q["q25"]).to_dict()
    print(f"  {len(iqr_dict)} wells with IQR")

    rows = []
    for fc in FEAT_CONFIGS:
        for ss in SPLIT_SEEDS:
            run_sig = f"in52_out16_ep50_bs4096_seed40_coloc_rand_f95_ss{ss}"
            gp_tag  = fc["gp_tag"]
            d = GP / f"GRU_FCOV_{run_sig}__{gp_tag}__predobstrain" / "gp_pred.parquet"
            if not d.exists():
                print(f"  MISSING: {fc['tag']} ss={ss}  ({d})")
                continue
            nrmse_iqr, nrmse_range, rmse, n = metrics_pw(d, iqr_dict)
            rows.append({"feat": fc["tag"], "ss": ss,
                          "nRMSE_iqr": nrmse_iqr, "nRMSE_range": nrmse_range,
                          "RMSE_pw": rmse, "n_wells": n})
            print(f"  {fc['tag']} ss={ss}: nRMSE_iqr={nrmse_iqr:.4f}  nRMSE_range={nrmse_range:.4f}  RMSE={rmse:.4f}")

    df_alt = pd.DataFrame(rows)

    RE_HR = re.compile(
        r"^GRU_FCOV_in\d+_out\d+_ep\d+_bs\d+_seed\d+_coloc_"
        r"rand_f95_ss(?P<ss>\d+)"
        r"__dec_prod_hydroraum_gps\d+__predobstrain$"
    )
    hr_rows = []
    for d in sorted(GP.iterdir()):
        m = RE_HR.match(d.name)
        if not m:
            continue
        ss = int(m["ss"])
        pred_path = d / "gp_pred.parquet"
        if not pred_path.exists():
            print(f"  MISSING hydroraum_only ss={ss}")
            continue
        nrmse_iqr, nrmse_range, rmse, n = metrics_pw(pred_path, iqr_dict)
        hr_rows.append({"feat": "hydroraum_only", "ss": ss,
                        "nRMSE_iqr": nrmse_iqr, "nRMSE_range": nrmse_range,
                        "RMSE_pw": rmse, "n_wells": n})
        print(f"  hydroraum_only ss={ss}: nRMSE_iqr={nrmse_iqr:.4f}  nRMSE_range={nrmse_range:.4f}  RMSE={rmse:.4f}")
    df_hr = pd.DataFrame(hr_rows)[["feat", "ss", "nRMSE_iqr", "nRMSE_range", "RMSE_pw"]]

    all_df = pd.concat([df_alt[["feat", "ss", "nRMSE_iqr", "nRMSE_range", "RMSE_pw"]], df_hr], ignore_index=True)

    summary = (
        all_df.groupby("feat")
        .agg(
            n               =("nRMSE_iqr",   "count"),
            nRMSE_iqr_mean  =("nRMSE_iqr",   "mean"),
            nRMSE_iqr_std   =("nRMSE_iqr",   "std"),
            nRMSE_range_mean=("nRMSE_range", "mean"),
            nRMSE_range_std =("nRMSE_range", "std"),
            RMSE_mean       =("RMSE_pw",     "mean"),
            RMSE_std        =("RMSE_pw",     "std"),
        )
        .reset_index()
        .sort_values("nRMSE_iqr_mean")
    )

    print("\n=== DECOUPLED RAND f95 — FEAT-ALT COMPARISON (mean over split seeds) ===")
    print(f"{'Feature':<25} {'n':>3}  {'nRMSE_iqr':>10} {'±':>7}  {'nRMSE_range':>12} {'±':>7}  {'RMSE_pw':>8} {'±':>7}")
    print("-" * 95)
    for _, r in summary.iterrows():
        print(f"  {r['feat']:<23} {int(r['n']):>3}"
              f"  {r['nRMSE_iqr_mean']:>10.4f} {r['nRMSE_iqr_std']:>7.4f}"
              f"  {r['nRMSE_range_mean']:>12.4f} {r['nRMSE_range_std']:>7.4f}"
              f"  {r['RMSE_mean']:>8.4f} {r['RMSE_std']:>7.4f}")

    all_df.to_csv(OUT / "feat_alt_decoupled_raw.csv", index=False)
    summary.to_csv(OUT / "feat_alt_decoupled_summary.csv", index=False)
    print(f"\nSaved -> {OUT}/feat_alt_decoupled_raw.csv")
    print(f"Saved -> {OUT}/feat_alt_decoupled_summary.csv")


if __name__ == "__main__":
    main()
