import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT      = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT.parent / "data" / "merged.parquet"
GP_DIR    = ROOT / "outputs" / "gp"
OUT_DIR   = ROOT / "reports" / "tables"

RE_MS = re.compile(
    r"in\d+_out\d+_ep\d+_bs\d+_seed(?P<ms>\d+)_dedup"
    r"_(?P<split>rand|km|md\d+)"
    r"_f(?P<frac>\d+)"
    r"_ss(?P<ss>\d+)"
)

SPLIT_LABELS = {
    ("md50",  95): "MaxDist-50  f=0.95",
    ("md50",  90): "MaxDist-50  f=0.90",
    ("rand",  90): "Random      f=0.90",
    ("rand",  95): "Random      f=0.95",
    ("md90",  95): "MaxDist-90  f=0.95",
    ("md10",  95): "MaxDist-10  f=0.95",
    ("km",    95): "K-Means     f=0.95",
    ("rand",  80): "Random      f=0.80",
    ("md100", 95): "MaxDist-100 f=0.95",
}


def main() -> None:
    print("Loading full-history IQR ...")
    gws = pq.read_table(DATA_PATH, columns=["id", "gws"]).to_pandas().dropna(subset=["gws"])
    q = gws.groupby("id")["gws"].quantile([0.25, 0.75]).unstack()
    q.columns = ["q25", "q75"]
    iqr_dict = (q["q75"] - q["q25"]).to_dict()
    print(f"  {len(iqr_dict)} wells with full-history IQR")

    rows = []
    for gp_dir in sorted(p for p in GP_DIR.iterdir() if p.is_dir()):
        pred_path = gp_dir / "gp_pred.parquet"
        if not pred_path.exists():
            continue
        run_name = gp_dir.name
        if "__ms__" not in run_name or not run_name.startswith("GRU_FCOV_"):
            continue
        run_sig = run_name[len("GRU_FCOV_"):run_name.index("__ms__")]
        m = RE_MS.search(run_sig)
        if not m:
            continue

        split = m["split"]
        frac  = int(m["frac"])
        ss    = int(m["ss"])
        ms    = int(m["ms"])

        try:
            pred = pq.read_table(
                pred_path, columns=["id", "gws_true", "gws_forecast"]
            ).to_pandas().dropna(subset=["gws_true", "gws_forecast"])
        except Exception as e:
            print(f"  SKIP {run_name}: {e}")
            continue

        per_well = []
        for well_id, g in pred.groupby("id"):
            t = g["gws_true"].to_numpy(float)
            p = g["gws_forecast"].to_numpy(float)
            rmse  = float(np.sqrt(np.mean((p - t) ** 2)))
            denom = float(np.sum((t - t.mean()) ** 2))
            nse   = float(1.0 - np.sum((p - t) ** 2) / denom) if denom > 0 else np.nan
            iqr   = iqr_dict.get(well_id, np.nan)
            nrmse = rmse / iqr if (np.isfinite(iqr) and iqr > 0) else np.nan
            per_well.append({"rmse": rmse, "nse": nse, "nrmse": nrmse})

        if not per_well:
            continue
        pw = pd.DataFrame(per_well)
        rows.append({
            "split": split, "frac": frac, "ss": ss, "ms": ms,
            "RMSE_pw":  float(pw["rmse"].median()),
            "NSE_pw":   float(pw["nse"].median()),
            "nRMSE_pw": float(pw["nrmse"].median()),
        })

    df = pd.DataFrame(rows)
    grp = (
        df.groupby(["split", "frac"])
        .agg(
            n=("RMSE_pw", "count"),
            RMSE_mean =("RMSE_pw",  "mean"),
            RMSE_med  =("RMSE_pw",  "median"),
            NSE_mean  =("NSE_pw",   "mean"),
            NSE_med   =("NSE_pw",   "median"),
            nRMSE_mean=("nRMSE_pw", "mean"),
            nRMSE_med =("nRMSE_pw", "median"),
        )
        .reset_index()
        .sort_values("RMSE_mean")
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grp.to_csv(OUT_DIR / "decoupled_multiseed_results_recomputed.csv", index=False)

    header = (
        "| Split | n "
        "| RMSE_pw mean | RMSE_pw median "
        "| NSE_pw mean | NSE_pw median "
        "| nRMSE_pw mean | nRMSE_pw median |"
    )
    sep = "|-------|---|-------------|----------------|------------|---------------|--------------|-----------------|"
    lines = [header, sep]
    for _, r in grp.iterrows():
        label = SPLIT_LABELS.get((r["split"], int(r["frac"])), f"{r['split']}_f{int(r['frac'])}")
        lines.append(
            f"| {label} | {int(r['n'])} "
            f"| {r['RMSE_mean']:.3f} | {r['RMSE_med']:.3f} "
            f"| {r['NSE_mean']:.3f} | {r['NSE_med']:.3f} "
            f"| {r['nRMSE_mean']:.3f} | {r['nRMSE_med']:.3f} |"
        )
    md = "\n".join(lines) + "\n"
    (OUT_DIR / "decoupled_multiseed_results_recomputed.md").write_text(md)

    print("\n" + md)
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
