from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
GP   = ROOT / "outputs" / "gp"
GRU  = ROOT / "outputs" / "GRU_FCOV"
REG  = pd.read_csv(ROOT / "outputs" / "run_registry.csv")
OUT_DIR = ROOT / "reports" / "tables"


def rmse_pw_gp(path):
    df = pq.read_table(path, columns=["id", "gws_true", "gws_forecast"]).to_pandas().dropna()
    rows = []
    for wid, g in df.groupby("id"):
        t = g["gws_true"].to_numpy(float)
        p = g["gws_forecast"].to_numpy(float)
        m = np.isfinite(t) & np.isfinite(p)
        if m.sum() == 0:
            continue
        rows.append(float(np.sqrt(np.mean((p[m] - t[m]) ** 2))))
    return float(np.median(rows)), len(rows)


def rmse_pw_gru(path):
    df = pq.read_table(path, columns=["id", "gws_forecast", "gws"]).to_pandas().dropna()
    rows = []
    for wid, g in df.groupby("id"):
        t = g["gws"].to_numpy(float)
        p = g["gws_forecast"].to_numpy(float)
        m = np.isfinite(t) & np.isfinite(p)
        if m.sum() == 0:
            continue
        rows.append(float(np.sqrt(np.mean((p[m] - t[m]) ** 2))))
    return float(np.median(rows)), len(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_rows = []

    print("=== ORACLE CLEAN (2-way coloc split, daily) ===")
    for tag, d in [
        ("rand", "ORACLE_oracle_clean_rand_ss42__oracle_clean_wk__predobstrain"),
        ("md30", "ORACLE_oracle_clean_md30_ss42__oracle_clean_wk__predobstrain"),
        ("md50", "ORACLE_oracle_clean_md50_ss42__oracle_clean_wk__predobstrain"),
    ]:
        p = GP / d / "gp_pred.parquet"
        if p.exists():
            med, n = rmse_pw_gp(p)
            print(f"  oracle_{tag}: {med:.4f}  (n={n})")
            out_rows.append({"metric": f"oracle_clean_{tag}", "RMSE_pw_med": med, "n_wells": n})
        else:
            print(f"  oracle_{tag}: NOT FOUND")
            out_rows.append({"metric": f"oracle_clean_{tag}", "RMSE_pw_med": np.nan, "n_wells": 0})

    print("\n=== GLOBAL GRU (all 1040 wells, temporal holdout) ===")
    global_runs = REG[REG["run_sig"].str.contains("_global", na=False)].sort_values("created_at")
    latest = global_runs.groupby("run_sig").last().reset_index()
    all_rmse = []
    for _, row in latest.iterrows():
        rid = int(row["run_id"])
        p = GRU / f"GRU_FCOV_{rid:04d}" / "predictions" / "pred.parquet"
        if p.exists():
            med, n = rmse_pw_gru(p)
            all_rmse.append(med)
            print(f"  {row['run_sig']} (id={rid}): {med:.4f}  n={n}")
        else:
            print(f"  {row['run_sig']} (id={rid}): pred not found")
    if all_rmse:
        med_over = float(np.median(all_rmse))
        mean_over = float(np.mean(all_rmse))
        print(f"  --> median over seeds: {med_over:.4f}  mean: {mean_over:.4f}")
        out_rows.append({"metric": "global_gru_median_over_seeds", "RMSE_pw_med": med_over,
                         "n_wells": len(all_rmse)})
        out_rows.append({"metric": "global_gru_mean_over_seeds", "RMSE_pw_med": mean_over,
                         "n_wells": len(all_rmse)})

    print("\n=== SUMMARY TABLE ===")
    df = pd.DataFrame(out_rows)
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "collect_thesis_numbers.csv", index=False)
    print(f"\nSaved -> {OUT_DIR}/collect_thesis_numbers.csv")


if __name__ == "__main__":
    main()
