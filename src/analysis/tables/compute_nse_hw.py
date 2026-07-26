import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT    = Path(__file__).resolve().parents[3]
GP_DIR  = ROOT / "outputs" / "gp"
OUT_DIR = ROOT / "reports" / "tables"

RE = re.compile(
    r"GRU_FCOV_.*?_coloc_(?P<split>[^_]+)_f(?P<frac>\d+)_ss(?P<ss>\d+)"
    r"__(?P<tag>[^_].+?)__predobstrain$"
)


def nse(p, r, y_bar=None):
    if y_bar is None:
        y_bar = r.mean()
    d = np.sum((r - y_bar) ** 2)
    return float(1 - np.sum((p - r) ** 2) / d) if d > 0 else np.nan


def nse_hw(pred_df):
    rows = []
    for (wid, h), g in pred_df.groupby(["id", "horizon"]):
        p = g["gws_forecast"].to_numpy(float)
        r = g["gws_true"].to_numpy(float)
        mask = np.isfinite(p) & np.isfinite(r)
        if mask.sum() < 2:
            continue
        rows.append({"horizon": int(h), "nse": nse(p[mask], r[mask])})
    if not rows:
        return np.nan
    df = pd.DataFrame(rows).groupby("horizon")["nse"].median()
    return float(df.mean())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for d in sorted(GP_DIR.iterdir()):
        m = RE.match(d.name)
        if not m:
            continue
        pred_path = d / "gp_pred.parquet"
        if not pred_path.exists():
            print(f"  MISSING pred: {d.name}")
            continue
        try:
            df = pq.read_table(
                pred_path, columns=["id", "horizon", "gws_forecast", "gws_true"]
            ).to_pandas()
            val = nse_hw(df)
            results.append({
                "split":  m["split"],
                "frac":   int(m["frac"]),
                "ss":     int(m["ss"]),
                "tag":    m["tag"],
                "NSE_hw": val,
                "run":    d.name,
            })
            print(f"  {m['split']:8s} f{m['frac']} ss{m['ss']}  NSE_hw={val:.4f}  ({d.name[:60]})")
        except Exception as e:
            print(f"  ERROR {d.name}: {e}")

    if not results:
        print("No matching runs found.")
        return

    df_all = pd.DataFrame(results)
    df_all.to_csv(OUT_DIR / "nse_hw_interpolation.csv", index=False)
    print(f"\nSaved {len(df_all)} rows to reports/tables/nse_hw_interpolation.csv")

    print("\n=== Summary (mean +/- std of NSE_hw across split seeds) ===")
    for (split, frac), g in df_all.groupby(["split", "frac"]):
        vals = g["NSE_hw"].dropna()
        print(f"  {split:8s} f{frac}  n={len(vals):2d}  "
              f"mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")


if __name__ == "__main__":
    main()
