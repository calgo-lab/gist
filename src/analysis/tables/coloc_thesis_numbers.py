import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT      = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT.parent / "data" / "merged.parquet"
GP_DIR    = ROOT / "outputs" / "gp"
JOINT_DIR = ROOT / "outputs" / "GRU_GP_JOINT"
GRU_DIR   = ROOT / "outputs" / "GRU_FCOV"
REG_CSV   = ROOT / "outputs" / "run_registry.csv"
OUT_DIR   = ROOT / "reports" / "tables"

VAL_CUTOFF = pd.Timestamp("2020-01-01")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading IQR ...")
    gws_raw = pq.read_table(DATA_PATH, columns=["id", "gws"]).to_pandas().dropna(subset=["gws"])
    q = gws_raw.groupby("id")["gws"].quantile([0.25, 0.75]).unstack()
    q.columns = ["q25", "q75"]
    iqr_dict = (q["q75"] - q["q25"]).to_dict()
    print(f"  {len(iqr_dict)} wells with IQR")

    def rmse_pw(df, true_col="gws_true", pred_col="gws_forecast", filter_cutoff=True):
        if filter_cutoff and "datum" in df.columns:
            df = df.copy()
            df["datum"] = pd.to_datetime(df["datum"])
            df = df[df["datum"] > VAL_CUTOFF]
        df = df.dropna(subset=[true_col, pred_col])
        rows = []
        for wid, g in df.groupby("id"):
            t = g[true_col].to_numpy(float)
            p = g[pred_col].to_numpy(float)
            m = np.isfinite(t) & np.isfinite(p)
            if m.sum() == 0:
                continue
            rmse = float(np.sqrt(np.mean((p[m] - t[m]) ** 2)))
            iqr  = iqr_dict.get(wid, np.nan)
            rows.append({"id": wid, "rmse": rmse,
                         "nrmse": rmse / iqr if (np.isfinite(iqr) and iqr > 0) else np.nan})
        pw = pd.DataFrame(rows)
        return {
            "n_wells": len(pw),
            "RMSE_pw_mean": float(pw["rmse"].mean()),
            "RMSE_pw_med":  float(pw["rmse"].median()),
            "nRMSE_pw_mean": float(pw["nrmse"].mean()),
            "nRMSE_pw_med":  float(pw["nrmse"].median()),
        }

    registry = pd.read_csv(REG_CSV) if REG_CSV.exists() else pd.DataFrame()

    def lookup_run_dir(model_type, run_sig):
        if registry.empty:
            return None
        sub = registry[
            (registry["model_type"] == model_type) &
            (registry["run_sig"] == run_sig)
        ]
        if sub.empty:
            return None
        run_id = int(sub.iloc[-1]["run_id"])
        return ROOT / "outputs" / model_type / f"{model_type}_{run_id:04d}"

    results = {}

    print("\n=== 1. GRU temporal model ===")
    GRU_SIGS = {
        "rand": "in52_out16_ep50_bs4096_seed40_coloc_rand_f95_ss42",
        "md50": "in52_out16_ep50_bs4096_seed40_coloc_md50_f95_ss42",
    }
    for split, sig in GRU_SIGS.items():
        d = lookup_run_dir("GRU_FCOV", sig)
        if d is None:
            print(f"  GRU {split}: run not found in registry")
            continue
        pred_path = d / "predictions" / "pred.parquet"
        if not pred_path.exists():
            pred_path = d / "predictions" / "spatial_holdout" / "pred.parquet"
        if not pred_path.exists():
            print(f"  GRU {split}: pred.parquet not found in {d}")
            continue
        df = pq.read_table(pred_path, columns=["id", "datum", "gws_forecast", "gws"]).to_pandas()
        r = rmse_pw(df, true_col="gws", pred_col="gws_forecast")
        print(f"  GRU {split}: RMSE_pw_med={r['RMSE_pw_med']:.4f}  n={r['n_wells']}")
        results[f"gru_{split}"] = r

    print("\n=== 2. Oracle ===")
    oracle_patterns = [
        "ORACLE_oracle_clean_rand_ss42__oracle_clean_wk__predobstrain",
        "ORACLE_oracle_clean_rand_ss42__oracle_clean_me__predobstrain",
        "ORACLE_oracle_clean_md50_ss42__oracle_clean_wk__predobstrain",
        "ORACLE_oracle_clean_md50_ss42__oracle_clean_me__predobstrain",
    ]
    for pat in oracle_patterns:
        d = GP_DIR / pat
        pred_path = d / "gp_pred.parquet"
        if not pred_path.exists():
            matches = list(GP_DIR.glob(f"ORACLE_oracle_clean*{pat.split('__')[1]}*"))
            if not matches:
                print(f"  Oracle: not found: {pat}")
                continue
            pred_path = matches[0] / "gp_pred.parquet"
            if not pred_path.exists():
                print(f"  Oracle: pred not found: {matches[0].name}")
                continue
        df = pq.read_table(pred_path, columns=["id", "datum", "gws_true", "gws_forecast"]).to_pandas()
        r = rmse_pw(df)
        tag = pat.split("__")[1]
        print(f"  Oracle [{tag}]: RMSE_pw_med={r['RMSE_pw_med']:.4f}  n={r['n_wells']}")
        results[f"oracle_{tag}"] = r

    for d in sorted(GP_DIR.glob("ORACLE_oracle_clean_*")):
        pred_path = d / "gp_pred.parquet"
        if not pred_path.exists():
            continue
        if d.name in oracle_patterns:
            continue
        df = pq.read_table(pred_path, columns=["id", "datum", "gws_true", "gws_forecast"]).to_pandas()
        r = rmse_pw(df)
        print(f"  Oracle extra [{d.name}]: RMSE_pw_med={r['RMSE_pw_med']:.4f}  n={r['n_wells']}")
        results[f"oracle_extra_{d.name}"] = r

    print("\n=== 3. Decoupled GP+GRU (pilot, single seed) ===")
    DECOUPLED_DIRS = {
        "rand": "GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_rand_f95_ss42__coloc_rand_f95__predobstrain",
        "md50": "GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_md50_f95_ss42__coloc_md50_f95__predobstrain",
    }
    for split, dname in DECOUPLED_DIRS.items():
        pred_path = GP_DIR / dname / "gp_pred.parquet"
        if not pred_path.exists():
            print(f"  Decoupled {split}: not found: {dname}")
            continue
        df = pq.read_table(pred_path, columns=["id", "datum", "gws_true", "gws_forecast"]).to_pandas()
        r = rmse_pw(df)
        print(f"  Decoupled {split}: RMSE_pw_med={r['RMSE_pw_med']:.4f}  n={r['n_wells']}")
        results[f"decoupled_{split}"] = r

    print("\n=== 4. Joint GRU+GP ===")
    JOINT_SIGS = {
        "rand": "joint_in52_out16_ep50_bs8192_seed40_coloc_rand_f95_ss42",
        "md30": "joint_in52_out16_ep50_bs8192_seed40_coloc_md30_f95_ss42",
    }
    for split, sig in JOINT_SIGS.items():
        d = lookup_run_dir("GRU_GP_JOINT", sig)
        if d is None:
            print(f"  Joint {split}: run not found in registry (sig={sig})")
            continue
        for subdir in ["eval/test", "eval/holdout", "eval/spatial_holdout"]:
            pred_path = d / subdir / "gp_pred.parquet"
            if pred_path.exists():
                break
        else:
            print(f"  Joint {split}: gp_pred.parquet not found in {d}")
            continue
        df = pq.read_table(pred_path, columns=["id", "datum", "gws_true", "gws_forecast"]).to_pandas()
        r = rmse_pw(df)
        print(f"  Joint {split}: RMSE_pw_med={r['RMSE_pw_med']:.4f}  n={r['n_wells']}")
        results[f"joint_{split}"] = r

    print("\n=== 5. Decoupled best spatial split (multiseed) ===")
    ms_csv = OUT_DIR / "coloc_multiseed_results.csv"
    if ms_csv.exists():
        ms = pd.read_csv(ms_csv)
        best = ms.sort_values("RMSE_mean").iloc[0]
        print(f"  Best split: {best['split']}_f{best['frac']}  "
              f"RMSE_mean={best['RMSE_mean']:.4f}  RMSE_med={best['RMSE_med']:.4f}  n={best['n']}")
        results["decoupled_best_multiseed"] = {
            "split": f"{best['split']}_f{int(best['frac'])}",
            "n_runs": int(best["n"]),
            "RMSE_pw_mean": float(best["RMSE_mean"]),
            "RMSE_pw_med": float(best["RMSE_med"]),
            "nRMSE_pw_mean": float(best["nRMSE_mean"]),
            "nRMSE_pw_med": float(best["nRMSE_med"]),
        }
    else:
        print("  coloc_multiseed_results.csv not found")

    print("\n\n=== THESIS TABLE (coloc splits, median RMSE per well) ===")
    print(f"{'Pipeline':<40} {'Split':<10} {'RMSE_pw_med':>12} {'nRMSE_pw_med':>13} {'n_wells':>8}")
    print("-" * 85)
    order = [
        ("gru_rand",                "GRU temporal (rand)"),
        ("gru_md50",                "GRU temporal (md50)"),
        ("oracle_oracle_clean_wk",  "Oracle GP (wk, rand)"),
        ("oracle_oracle_clean_me",  "Oracle GP (ME, rand)"),
        ("decoupled_rand",          "Decoupled GP+GRU (rand)"),
        ("decoupled_md50",          "Decoupled GP+GRU (md50)"),
        ("joint_rand",              "Joint GRU+GP (rand)"),
        ("joint_md30",              "Joint GRU+GP (md30)"),
        ("decoupled_best_multiseed", "Decoupled best (multiseed)"),
    ]
    for key, label in order:
        if key not in results:
            print(f"  {label:<38} {'---':>10} {'---':>12} {'---':>13} {'---':>8}  [NOT FOUND]")
            continue
        r = results[key]
        split_info = r.get("split", key.split("_")[-1])
        rmse = r.get("RMSE_pw_med", float("nan"))
        nrmse = r.get("nRMSE_pw_med", float("nan"))
        n = r.get("n_wells", r.get("n_runs", "?"))
        print(f"  {label:<38} {split_info:<10} {rmse:>12.4f} {nrmse:>13.4f} {n:>8}")

    (OUT_DIR / "coloc_thesis_numbers.json").write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved -> {OUT_DIR}/coloc_thesis_numbers.json")


if __name__ == "__main__":
    main()
