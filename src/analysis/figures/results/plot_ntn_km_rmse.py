import os
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.spatial import cKDTree

from pathlib import Path

BASE = str(Path(__file__).resolve().parents[4])
GP_DIR    = os.path.join(BASE, "outputs", "gp")
JOINT_DIR = os.path.join(BASE, "outputs", "main_runs", "joint")
SPLITS_DIR = os.path.join(BASE, "splits")
MERGED_PARQUET = str(Path(BASE).parent / "data" / "merged.parquet")
OUT_DIR = os.path.join(BASE, "reports", "figures", "joint_vs_two_stage")

SPLIT_SEEDS = list(range(42, 82))
TWO_STAGE_PATTERN = (
    "GRU_FCOV_in52_out16_ep50_bs4096_seed47_coloc_rand90_f90_ss{ss}"
    "__dec_prod_hr_gps7__predobstrain"
)
SPLIT_FILE_PATTERN = "spatial_split_full_merged_coloc_rand90_f90_joint_ss{ss}.csv"

print("Loading well coordinates …")
coord_lookup = (
    pd.read_parquet(MERGED_PARQUET, columns=["id", "x_25833", "y_25833"])
    .drop_duplicates("id")
    .set_index("id")[["x_25833", "y_25833"]]
    .to_dict("index")
)

print("Mapping joint model directories …")
joint_ss_to_dir = {}
for d in os.listdir(JOINT_DIR):
    meta_path = os.path.join(JOINT_DIR, d, "meta.yaml")
    if not os.path.exists(meta_path):
        continue
    with open(meta_path) as f:
        m = yaml.safe_load(f)
    sig = m.get("run_sig", "")
    if "seed44" not in sig or "rand90" not in sig or "f90" not in sig or "valcheck" in sig:
        continue
    match = re.search(r"ss(\d+)$", sig)
    if match:
        joint_ss_to_dir[int(match.group(1))] = d

records = []
for ss in SPLIT_SEEDS:
    split_df = pd.read_csv(os.path.join(SPLITS_DIR, SPLIT_FILE_PATTERN.format(ss=ss)))
    train_ids = split_df[split_df["spatial_split"] == "spatial_train"]["id"].values
    test_ids  = split_df[split_df["spatial_split"] == "spatial_holdout"]["id"].values

    train_xy = np.array([[coord_lookup[i]["x_25833"], coord_lookup[i]["y_25833"]]
                          for i in train_ids if i in coord_lookup])
    tree = cKDTree(train_xy)
    ntn = {}
    for wid in test_ids:
        if wid not in coord_lookup:
            continue
        d_, _ = tree.query([coord_lookup[wid]["x_25833"], coord_lookup[wid]["y_25833"]])
        ntn[wid] = d_ / 1000.0

    ts_path = os.path.join(GP_DIR, TWO_STAGE_PATTERN.format(ss=ss), "gp_pred.parquet")
    if not os.path.exists(ts_path) or ss not in joint_ss_to_dir:
        continue
    ts_pred = pd.read_parquet(ts_path, columns=["id", "horizon", "gws_true", "gws_forecast"])
    ts_pred = ts_pred[ts_pred["horizon"] == 16]
    ts_rmse = (ts_pred.assign(sq_err=(ts_pred["gws_true"] - ts_pred["gws_forecast"]) ** 2)
               .groupby("id")["sq_err"].mean().apply(np.sqrt))

    jp_path = os.path.join(JOINT_DIR, joint_ss_to_dir[ss], "eval", "test", "gp_pred.parquet")
    jt_pred = pd.read_parquet(jp_path, columns=["id", "horizon", "gws_true", "gws_forecast"])
    jt_pred = jt_pred[jt_pred["horizon"] == 16]
    jt_rmse = (jt_pred.assign(sq_err=(jt_pred["gws_true"] - jt_pred["gws_forecast"]) ** 2)
               .groupby("id")["sq_err"].mean().apply(np.sqrt))

    for wid in test_ids:
        if wid not in ntn or wid not in ts_rmse.index or wid not in jt_rmse.index:
            continue
        records.append(dict(ntn_km=ntn[wid], rmse_ts=ts_rmse[wid], rmse_jt=jt_rmse[wid]))

    if ss % 10 == 2:
        print(f"  processed ss={ss} … {len(records)} records so far")

df = pd.DataFrame(records)
print(f"\nTotal (well, seed) pairs: {len(df)}")

bin_edges  = list(range(0, 9)) + [np.inf]
bin_labels = [f"{i}–{i+1}" for i in range(8)] + ["≥8"]
df["km_bin"] = pd.cut(df["ntn_km"], bins=bin_edges, labels=bin_labels, right=False)

binned = (
    df.groupby("km_bin", observed=True)
    .agg(
        rmse_ts = ("rmse_ts", "median"),
        rmse_jt = ("rmse_jt", "median"),
        ntn_med = ("ntn_km",  "median"),
        n       = ("ntn_km",  "count"),
    )
    .reset_index()
)

print(f"\n{'Bin':>6}  {'n':>5}  {'NTN med':>8}  {'Two-stage':>10}  {'Joint':>8}  {'Rel gap %':>10}")
print("-" * 60)
for _, row in binned.iterrows():
    rel = (row.rmse_jt - row.rmse_ts) / row.rmse_ts * 100
    print(f"{row.km_bin:>6}  {int(row.n):>5}  {row.ntn_med:8.2f}  "
          f"{row.rmse_ts:10.4f}  {row.rmse_jt:8.4f}  {rel:+10.2f}%")

x = binned["ntn_med"].values

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(x, binned["rmse_ts"], color="#d87f3a", lw=1.8, marker="o", ms=5,
        label="Two-stage")
ax.plot(x, binned["rmse_jt"], color="#4c72b0", lw=1.8, marker="o", ms=5,
        label="Jointly trained")

ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.set_xlabel("Nearest-training-well distance (km)", fontsize=11)
ax.set_ylabel("Median per-well RMSE (m)", fontsize=11)
ax.set_ylim(0, None)
ax.grid(axis="both", ls=":", lw=0.7, alpha=0.7)

ax.legend(fontsize=10, framealpha=0.9)

plt.tight_layout()

out_pdf = os.path.join(OUT_DIR, "ntn_km_rmse.pdf")
out_png = os.path.join(OUT_DIR, "ntn_km_rmse.png")
fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
