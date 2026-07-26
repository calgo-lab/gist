from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT       = Path(__file__).resolve().parents[4]
GP_RAW     = ROOT / "outputs" / "gp"
SPLITS_DIR = ROOT / "splits"
OUT_DIR    = ROOT / "reports" / "figures" / "spatial_distance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

all_coords = pd.read_parquet(ROOT.parent / "data" / "merged.parquet",
                             columns=["id", "x_25833", "y_25833"]).drop_duplicates("id")

dirs_by_seed = {}
for d in GP_RAW.iterdir():
    if ("rand90_f90" in d.name
            and "dec_prod_hr_gps7" in d.name
            and (d / "gp_pred.parquet").exists()):
        for s in [f"ss{i}" for i in range(42, 82)]:
            if f"_{s}__" in d.name:
                dirs_by_seed[s] = d
                break

print(f"Found {len(dirs_by_seed)} seed directories")

rows = []
for seed in sorted(dirs_by_seed):
    d = dirs_by_seed[seed]
    split_f = SPLITS_DIR / f"spatial_split_full_merged_coloc_rand90_f90_{seed}.csv"
    if not split_f.exists():
        print(f"  No split file for {seed}")
        continue
    split_df = pd.read_csv(split_f)
    train_ids = set(split_df[split_df["spatial_split"] == "spatial_train"]["id"])

    pred = pd.read_parquet(d / "gp_pred.parquet")
    h_coords = pred[["id", "x_25833", "y_25833"]].drop_duplicates("id")[["x_25833", "y_25833"]].values
    t_coords = all_coords[all_coords["id"].isin(train_ids)][["x_25833", "y_25833"]].values
    dists_m = cKDTree(t_coords).query(h_coords, k=1)[0]

    pw_rmse = np.array([
        float(np.sqrt(((wdf["gws_forecast"].values - wdf["gws_true"].values) ** 2).mean()))
        for _, wdf in pred.groupby("id")
    ])
    rows.append({
        "seed": seed,
        "median_nn_km": np.median(dists_m) / 1000,
        "RMSE_median": np.median(pw_rmse),
    })

df = pd.DataFrame(rows)
rho, p = spearmanr(df["median_nn_km"], df["RMSE_median"])
print(f"Spearman rho={rho:.4f}  p={p:.6f}  (n={len(df)})")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(df["median_nn_km"], df["RMSE_median"], color="#4477AA", s=35, zorder=3)
ax.set_xlabel("Median NTN distance (km)")
ax.set_ylabel("Median per-well RMSE (m)")
p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
ax.annotate(f"Spearman $\\rho$ = {rho:.3f}, {p_txt}",
            xy=(0.03, 0.94), xycoords="axes fraction", fontsize=10, va="top")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"seed_distance_scatter.{ext}", dpi=300, bbox_inches="tight")
print(f"Wrote {OUT_DIR}/seed_distance_scatter.png + .pdf")
