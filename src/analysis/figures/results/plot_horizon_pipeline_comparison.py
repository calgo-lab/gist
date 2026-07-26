import os
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pathlib import Path

GIST = str(Path(__file__).resolve().parents[4])
GP_BASE = os.path.join(GIST, "outputs", "gp")
JT_BASE = os.path.join(GIST, "outputs", "main_runs", "joint")
OUT_DIR = os.path.join(GIST, "reports", "figures", "joint_vs_two_stage")

SPLIT_SEEDS = list(range(42, 82))
DEC_PATTERN = (
    "GRU_FCOV_in52_out16_ep50_bs4096_seed47_coloc_rand90_f90_ss{ss}"
    "__dec_prod_hr_gps7__predobstrain"
)
ORC_PATTERN = (
    "ORACLE_oracle_coloc_rand90_f90_ss{ss}"
    "__oracle_prod_hr_gps7__predobstrain"
)

HORIZONS = list(range(1, 17))

print("Mapping joint model directories …")
joint_ss_to_dir = {}
for d in os.listdir(JT_BASE):
    meta_path = os.path.join(JT_BASE, d, "meta.yaml")
    if not os.path.exists(meta_path):
        continue
    with open(meta_path) as f:
        m = yaml.safe_load(f)
    sig = m.get("run_sig", "")
    if "seed44" not in sig or "rand90" not in sig or "f90" not in sig or "valcheck" in sig:
        continue
    match = re.search(r"ss(\d+)$", sig)
    if match:
        joint_ss_to_dir[int(match.group(1))] = os.path.join(JT_BASE, d)

print(f"  Found joint runs for seeds: {sorted(joint_ss_to_dir)}")


def horizon_pw_medians(pred_path: str) -> dict:
    df = pd.read_parquet(pred_path, columns=["id", "horizon", "gws_true", "gws_forecast"])
    out = {}
    for h, grp in df.groupby("horizon"):
        pw = grp.assign(sq=(grp["gws_true"] - grp["gws_forecast"]) ** 2).groupby("id")["sq"].mean().apply(np.sqrt)
        out[int(h)] = float(pw.median())
    return out


dec_by_hz = {h: [] for h in HORIZONS}
orc_by_hz = {h: [] for h in HORIZONS}
jt_by_hz  = {h: [] for h in HORIZONS}

for ss in SPLIT_SEEDS:
    dec_path = os.path.join(GP_BASE, DEC_PATTERN.format(ss=ss), "gp_pred.parquet")
    orc_path = os.path.join(GP_BASE, ORC_PATTERN.format(ss=ss), "gp_pred.parquet")

    if ss not in joint_ss_to_dir:
        print(f"  ss={ss}: no joint dir, skipping")
        continue
    jt_path = os.path.join(joint_ss_to_dir[ss], "eval", "test", "gp_pred.parquet")

    missing = [p for p in [dec_path, orc_path, jt_path] if not os.path.exists(p)]
    if missing:
        print(f"  ss={ss}: missing {[os.path.basename(p) for p in missing]}, skipping")
        continue

    dec_m = horizon_pw_medians(dec_path)
    orc_m = horizon_pw_medians(orc_path)
    jt_m  = horizon_pw_medians(jt_path)

    for h in HORIZONS:
        if h in dec_m: dec_by_hz[h].append(dec_m[h])
        if h in orc_m: orc_by_hz[h].append(orc_m[h])
        if h in jt_m:  jt_by_hz[h].append(jt_m[h])

    if ss % 10 == 2:
        print(f"  processed ss={ss} … {len(dec_by_hz[16])} seeds so far")

n_seeds = len(dec_by_hz[16])
print(f"\nSeeds with complete data: {n_seeds}")

dec_vals = [np.mean(dec_by_hz[h]) for h in HORIZONS]
orc_vals = [np.mean(orc_by_hz[h]) for h in HORIZONS]
jt_vals  = [np.mean(jt_by_hz[h])  for h in HORIZONS]

print(f"\n{'H':>3}  {'Two-stage':>10}  {'Jointly tr.':>12}  {'Baseline':>10}")
print("-" * 42)
for h, d, j, o in zip(HORIZONS, dec_vals, jt_vals, orc_vals):
    print(f"{h:>3}  {d:10.4f}  {j:12.4f}  {o:10.4f}")


fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(HORIZONS, dec_vals, "o-", color="#d87f3a", lw=2, ms=5, label="Two-stage pipeline")
ax.plot(HORIZONS, jt_vals,  "o-", color="#4c72b0", lw=2, ms=5, label="Jointly trained")
ax.plot(HORIZONS, orc_vals, "o--", color="#2d7a4f", lw=1.5, ms=5, alpha=0.7,
        label="Interpolation baseline")

ax.set_xlabel("Forecast horizon (weeks)", fontsize=11)
ax.set_ylabel("Mean per-seed median per-well RMSE (m)", fontsize=11)
ax.set_xticks(HORIZONS)
ax.set_xticklabels([f"H{h}" for h in HORIZONS], fontsize=8)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.set_ylim(1.0, 1.3)
fig.tight_layout()

out_path = os.path.join(OUT_DIR, "horizon_pipeline_comparison.pdf")
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"\nSaved: {out_path}")
