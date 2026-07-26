import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT      = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT.parent / "data" / "merged.parquet"
GP_DIR    = ROOT / "outputs" / "gp"
JOINT_DIR = ROOT / "outputs" / "GRU_GP_JOINT"
GRU_DIR   = ROOT / "outputs" / "GRU_FCOV"
OUT_DIR   = ROOT / "reports" / "tables"

VAL_CUTOFF = pd.Timestamp("2020-01-01")

RE_COLOC_DECOUPLED = re.compile(
    r"in\d+_out\d+_ep\d+_bs\d+_seed(?P<ms>\d+)_coloc"
    r"_(?P<split>rand|km|md\d+)_f(?P<frac>\d+)_ss(?P<ss>\d+)"
)
RE_ORACLE_COLOC = re.compile(
    r"oracle_coloc_(?P<split>rand|km|md\d+)_f(?P<frac>\d+)_ss(?P<ss>\d+)"
)
RE_ORACLE_DEDUP = re.compile(
    r"oracle_dedup_(?P<split>rand|km|md\d+)_f(?P<frac>\d+)_ss(?P<ss>\d+)"
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading merged.parquet for IQR ...")
    gws = pq.read_table(DATA_PATH, columns=["id", "gws"]).to_pandas().dropna(subset=["gws"])
    iqr_dict = (
        gws.groupby("id")["gws"]
        .agg(q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .eval("iqr = q75 - q25")["iqr"].to_dict()
    )
    print(f"  {len(iqr_dict)} wells with IQR")

    def rmse_pw(pred_path, cutoff=VAL_CUTOFF):
        df = pq.read_table(
            pred_path, columns=["id", "datum", "gws_true", "gws_forecast"]
        ).to_pandas()
        df["datum"] = pd.to_datetime(df["datum"])
        df = df[df["datum"] > cutoff].dropna(subset=["gws_true", "gws_forecast"])
        rows = []
        for wid, g in df.groupby("id"):
            t = g["gws_true"].to_numpy(float)
            p = g["gws_forecast"].to_numpy(float)
            m = np.isfinite(t) & np.isfinite(p)
            if m.sum() == 0:
                continue
            rmse = float(np.sqrt(np.mean((p[m] - t[m]) ** 2)))
            iqr  = iqr_dict.get(wid, np.nan)
            rows.append({"id": wid, "rmse": rmse, "iqr": iqr,
                          "nrmse": rmse / iqr if (np.isfinite(iqr) and iqr > 0) else np.nan})
        return pd.DataFrame(rows)

    print("\n=== Scanning GP output dir ===")
    oracle_rows = []
    decoupled_rows = []

    for d in sorted(GP_DIR.iterdir()):
        pred_path = d / "gp_pred.parquet"
        if not pred_path.exists():
            continue
        name = d.name

        m = RE_ORACLE_COLOC.search(name) or RE_ORACLE_DEDUP.search(name)
        if m and ("coloc" in name or "dedup" in name):
            gp_tag = name.split("__")[1] if "__" in name else ""
            try:
                pw = rmse_pw(pred_path)
                oracle_rows.append({
                    "run_name": name, "gp_tag": gp_tag,
                    "split": m["split"], "frac": int(m["frac"]), "ss": int(m["ss"]),
                    "RMSE_pw": float(pw["rmse"].median()),
                    "nRMSE_pw": float(pw["nrmse"].median()),
                    "n_wells": len(pw),
                })
                print(f"  ORACLE  {name[:60]}  RMSE_pw={oracle_rows[-1]['RMSE_pw']:.4f}")
            except Exception as e:
                print(f"  SKIP oracle {name}: {e}")
            continue

        run_sig = name.split("__")[0]
        if "GRU_FCOV_" in name:
            run_sig = name[len("GRU_FCOV_"):name.index("__")] if "__" in name else name[len("GRU_FCOV_"):]
        m2 = RE_COLOC_DECOUPLED.search(run_sig)
        if m2:
            gp_tag = name.split("__")[1] if "__" in name else ""
            try:
                pw = rmse_pw(pred_path)
                decoupled_rows.append({
                    "run_name": name, "gp_tag": gp_tag,
                    "split": m2["split"], "frac": int(m2["frac"]),
                    "ss": int(m2["ss"]), "ms": int(m2["ms"]),
                    "RMSE_pw": float(pw["rmse"].median()),
                    "nRMSE_pw": float(pw["nrmse"].median()),
                    "n_wells": len(pw),
                })
            except Exception as e:
                print(f"  SKIP decoupled {name}: {e}")

    print(f"\nOracle coloc runs found: {len(oracle_rows)}")
    print(f"Decoupled coloc runs found: {len(decoupled_rows)}")

    print("\n=== Scanning GRU_GP_JOINT dir (via registry) ===")
    joint_rows = []
    RE_JOINT_COLOC_SIG = re.compile(
        r"joint_in\d+_out\d+_ep\d+_bs\d+_seed\d+"
        r"_coloc_(?P<split>rand|md\d+)_f\d+_ss(?P<ss>\d+)"
    )
    reg_path = ROOT / "outputs" / "run_registry.csv"
    if reg_path.exists() and JOINT_DIR.exists():
        reg = pd.read_csv(reg_path)
        joint_reg = reg[reg["model_type"] == "GRU_GP_JOINT"]
        for _, row in joint_reg.iterrows():
            m = RE_JOINT_COLOC_SIG.search(str(row.get("run_sig", "")))
            if not m:
                continue
            run_id = int(row["run_id"])
            d = JOINT_DIR / f"GRU_GP_JOINT_{run_id:04d}"
            pred_path = None
            for sub in ["eval/test", "eval/holdout", "eval/spatial_holdout"]:
                p = d / sub / "gp_pred.parquet"
                if p.exists():
                    pred_path = p
                    break
            if pred_path is None:
                print(f"  NO PRED: {d.name} (sig={row.get('run_sig','')})")
                continue
            try:
                pw = rmse_pw(pred_path)
                joint_rows.append({
                    "run_name": d.name,
                    "run_sig": str(row.get("run_sig", "")),
                    "split": m["split"], "ss": int(m["ss"]),
                    "RMSE_pw": float(pw["rmse"].median()),
                    "nRMSE_pw": float(pw["nrmse"].median()),
                    "n_wells": len(pw),
                })
                print(f"  JOINT   {d.name}  split={m['split']} ss={m['ss']}  RMSE_pw={joint_rows[-1]['RMSE_pw']:.4f}")
            except Exception as e:
                print(f"  SKIP joint {d.name}: {e}")
    else:
        print("  run_registry.csv or GRU_GP_JOINT dir not found")
    print(f"Joint coloc runs found: {len(joint_rows)}")

    print("\n=== Scanning for global GRU run ===")
    global_rows = []
    RE_GLOBAL_SIG = re.compile(r"in\d+_out\d+_ep\d+_bs\d+_seed(?P<ms>\d+)_global")
    reg_path2 = ROOT / "outputs" / "run_registry.csv"
    if reg_path2.exists() and GRU_DIR.exists():
        reg2 = pd.read_csv(reg_path2)
        gru_reg = reg2[reg2["model_type"] == "GRU_FCOV"]
        for _, row in gru_reg.iterrows():
            m = RE_GLOBAL_SIG.search(str(row.get("run_sig", "")))
            if not m:
                continue
            run_id = int(row["run_id"])
            d = GRU_DIR / f"GRU_FCOV_{run_id:04d}"
            pred_path = d / "predictions" / "pred.parquet"
            if not pred_path.exists():
                print(f"  NO PRED: {d.name}")
                continue
            try:
                df_pred = pq.read_table(pred_path, columns=["id", "datum", "gws", "gws_forecast"]).to_pandas()
                df_pred["datum"] = pd.to_datetime(df_pred["datum"])
                df_pred = df_pred[df_pred["datum"] > VAL_CUTOFF].dropna(subset=["gws", "gws_forecast"])
                pw_rows = []
                for wid, g in df_pred.groupby("id"):
                    t = g["gws"].to_numpy(float)
                    p = g["gws_forecast"].to_numpy(float)
                    msk = np.isfinite(t) & np.isfinite(p)
                    if msk.sum() == 0:
                        continue
                    rmse = float(np.sqrt(np.mean((p[msk] - t[msk]) ** 2)))
                    iqr  = iqr_dict.get(wid, np.nan)
                    pw_rows.append({"id": wid, "rmse": rmse,
                                    "nrmse": rmse / iqr if (np.isfinite(iqr) and iqr > 0) else np.nan})
                pw = pd.DataFrame(pw_rows)
                global_rows.append({
                    "run_name": d.name, "run_sig": str(row.get("run_sig", "")),
                    "ms": int(m["ms"]),
                    "RMSE_pw": float(pw["rmse"].median()),
                    "nRMSE_pw": float(pw["nrmse"].median()),
                    "n_wells": len(pw),
                })
                print(f"  GLOBAL  {d.name}  RMSE_pw={global_rows[-1]['RMSE_pw']:.4f}  n={len(pw)}")
            except Exception as e:
                print(f"  SKIP global {d.name}: {e}")
    else:
        print("  run_registry.csv or GRU_FCOV dir not found")
    print(f"Global GRU runs found: {len(global_rows)}")

    if oracle_rows:
        pd.DataFrame(oracle_rows).to_csv(OUT_DIR / "oracle_coloc_results.csv", index=False)
        print(f"\nSaved oracle_coloc_results.csv ({len(oracle_rows)} rows)")
        df_o = pd.DataFrame(oracle_rows)
        for split, g in df_o.groupby("split"):
            print(f"  oracle {split}: RMSE_pw mean={g.RMSE_pw.mean():.4f} std={g.RMSE_pw.std():.4f} median={g.RMSE_pw.median():.4f}")
    else:
        print("\nNo oracle coloc runs found.")

    if joint_rows:
        pd.DataFrame(joint_rows).to_csv(OUT_DIR / "joint_coloc_results.csv", index=False)
        print(f"Saved joint_coloc_results.csv ({len(joint_rows)} rows)")
        df_j = pd.DataFrame(joint_rows)
        for split, g in df_j.groupby("split"):
            print(f"  joint {split}: RMSE_pw mean={g.RMSE_pw.mean():.4f} std={g.RMSE_pw.std():.4f} median={g.RMSE_pw.median():.4f}")
    else:
        print("No joint coloc runs found.")

    if global_rows:
        pd.DataFrame(global_rows).to_csv(OUT_DIR / "global_gru_coloc_results.csv", index=False)
        print(f"Saved global_gru_coloc_results.csv ({len(global_rows)} rows)")
        df_g = pd.DataFrame(global_rows)
        print(f"  global GRU: RMSE_pw mean={df_g.RMSE_pw.mean():.4f}")
    else:
        print("No global GRU eval runs found.")

    if decoupled_rows:
        df_d = pd.DataFrame(decoupled_rows)
        df_d = df_d[df_d["gp_tag"].str.startswith("multiseed_")]
        summary = []
        for (split, frac), g in df_d.groupby(["split", "frac"]):
            seed_rmse  = g.groupby("ss")["RMSE_pw"].mean()
            seed_nrmse = g.groupby("ss")["nRMSE_pw"].mean()
            summary.append({
                "split": split, "frac": frac,
                "n_seeds": len(seed_rmse),
                "RMSE_mean": seed_rmse.mean(), "RMSE_std": seed_rmse.std(),
                "RMSE_med": seed_rmse.median(),
                "nRMSE_mean": seed_nrmse.mean(), "nRMSE_std": seed_nrmse.std(),
            })
        df_sum = pd.DataFrame(summary).sort_values("RMSE_mean")
        df_sum.to_csv(OUT_DIR / "coloc_multiseed_results_recomputed.csv", index=False)
        print(f"\nRecomputed coloc summary (multiseed only, {len(df_d)} runs):")
        print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
