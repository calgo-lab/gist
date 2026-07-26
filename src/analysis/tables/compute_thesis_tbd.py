import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import cKDTree
import yaml

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[3]
GP_FINAL   = ROOT / "outputs" / "gp"
GRU_FINAL  = ROOT / "outputs" / "01_final_results" / "GRU_FCOV"
JOINT_JUST = ROOT / "outputs" / "02_justification" / "GRU_GP_JOINT"
SPLITS_DIR = ROOT / "splits"


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    r     = pearsonr(obs, sim)[0]
    alpha = sim.std() / obs.std()
    beta  = sim.mean() / obs.mean()
    return float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))


def find_gp_dirs(must_contain: list[str], gp_root: Path) -> list[Path]:
    out = []
    for d in sorted(gp_root.iterdir()):
        if all(p in d.name for p in must_contain) and (d / "gp_metrics.parquet").exists():
            out.append(d)
    return out


def per_well_stats(dirs: list[Path]) -> dict:
    pw_rmse_all, gl_rmse, gl_nse, gl_kge = [], [], [], []
    range_nrmse_seeds, iqr_nrmse_seeds = [], []

    for d in dirs:
        pred  = pd.read_parquet(d / "gp_pred.parquet")
        gm    = pd.read_parquet(d / "gp_metrics.parquet").set_index("metric")["value"]

        gl_rmse.append(float(gm["RMSE"]))
        gl_nse.append(float(gm["NSE_pooled"]))

        obs = pred["gws_true"].values
        sim = pred["gws_forecast"].values
        gl_kge.append(kge(obs, sim))

        rows = []
        for wid, wdf in pred.groupby("id"):
            o = wdf["gws_true"].values
            s = wdf["gws_forecast"].values
            rmse = np.sqrt(((s - o)**2).mean())
            iqr  = float(np.diff(np.quantile(o, [0.25, 0.75]))[0])
            rng  = float(o.max() - o.min())
            rows.append({"rmse": rmse, "iqr": iqr, "range": rng})
        pw = pd.DataFrame(rows)
        pw_rmse_all.append(pw["rmse"])

        valid = pw[(pw["iqr"] > 0) & (pw["range"] > 0)]
        iqr_nrmse_seeds.append((valid["rmse"] / valid["iqr"]).median())
        range_nrmse_seeds.append((valid["rmse"] / valid["range"]).median())

    all_rmse = pd.concat(pw_rmse_all, ignore_index=True)
    return {
        "P10":            round(all_rmse.quantile(0.10), 4),
        "P25":            round(all_rmse.quantile(0.25), 4),
        "Median":         round(all_rmse.median(), 4),
        "P75":            round(all_rmse.quantile(0.75), 4),
        "P90":            round(all_rmse.quantile(0.90), 4),
        "Gl_RMSE":        round(np.mean(gl_rmse), 4),
        "Gl_NSE":         round(np.mean(gl_nse), 4),
        "KGE":            round(np.mean(gl_kge), 4),
        "nRMSE_range":    f"{np.mean(range_nrmse_seeds):.4f} ± {np.std(range_nrmse_seeds):.4f}",
        "nRMSE_IQR":      f"{np.mean(iqr_nrmse_seeds):.4f} ± {np.std(iqr_nrmse_seeds):.4f}",
    }


def print_block(label: str, dirs: list[Path]):
    print(f"\n  {label}: {len(dirs)} runs")
    if not dirs:
        print("    NO RUNS FOUND")
        return
    s = per_well_stats(dirs)
    for k, v in s.items():
        print(f"    {k}: {v}")


print("\n" + "="*70)
print("Decoupled GP (tab:multiseed_results + tab:multiseed_nrmse)")
print("="*70)
for split, parts in [("rand90_f90", ["rand90_f90", "dec_prod_hydroraum"]),
                     ("md50_f90",   ["md50_f90",   "dec_prod_hydroraum"]),
                     ("km_f90",     ["km_f90",     "dec_prod_hydroraum"])]:
    print_block(split, find_gp_dirs(parts, GP_FINAL))


print("\n" + "="*70)
print("Oracle GP (tab:oracle_gap + tab:pipeline_md30)")
print("="*70)
for split, parts in [("oracle rand90_f90", ["rand90_f90", "oracle_prod_hr_gps1"]),
                     ("oracle md50_f90",   ["md50_f90",   "oracle_prod_hr_gps1"])]:
    print_block(split, find_gp_dirs(parts, GP_FINAL))


print("\n" + "="*70)
print("Joint GP (tab:pipeline_main)")
print("="*70)
joint_dirs = []
for run_dir in sorted(JOINT_JUST.iterdir()):
    eval_test = run_dir / "eval" / "test"
    if not eval_test.exists():
        continue
    meta_f = run_dir / "meta.yaml"
    if not meta_f.exists():
        continue
    with open(meta_f) as f:
        sig = yaml.safe_load(f).get("run_sig", "")
    if "rand_f90" in sig or "rand90_f90" in sig:
        if (eval_test / "gp_metrics.parquet").exists():
            joint_dirs.append(eval_test)
print_block("joint rand_f90", joint_dirs)


print("\n" + "="*70)
print("Global GRU — rand90_f90 holdout (tab:pipeline_main, first row)")
print("="*70)

seed_to_run: dict[str, Path] = {}
for run_dir in sorted(GRU_FINAL.iterdir()):
    meta_f = run_dir / "meta.yaml"
    if not meta_f.exists():
        continue
    with open(meta_f) as f:
        sig = yaml.safe_load(f).get("run_sig", "")
    if "rand90_f90" not in sig:
        continue
    for s in [f"ss{i}" for i in range(42, 52)]:
        if sig.endswith(s) and s not in seed_to_run:
            seed_to_run[s] = run_dir
            break

print(f"  Found {len(seed_to_run)} GRU seeds")
gru_pw_all, gru_pred_all, gru_rmse_seeds = [], [], []
for seed, run_dir in sorted(seed_to_run.items()):
    split_f = SPLITS_DIR / f"spatial_split_full_merged_coloc_rand90_f90_{seed}.csv"
    if not split_f.exists():
        print(f"  No split file for {seed}")
        continue
    holdout_ids = set(
        pd.read_csv(split_f).query("spatial_split == 'spatial_holdout'")["id"]
    )
    pred_f = run_dir / "predictions" / "pred.parquet"
    if not pred_f.exists():
        continue
    pred = pd.read_parquet(pred_f)
    pred = pred[pred["id"].isin(holdout_ids)].rename(columns={"gws": "gws_true"})

    pw_rmse_vals = []
    for wid, wdf in pred.groupby("id"):
        o = wdf["gws_true"].values
        s = wdf["gws_forecast"].values
        pw_rmse_vals.append(float(np.sqrt(((s - o)**2).mean())))
    pw_rmse = pd.Series(pw_rmse_vals, dtype=float)
    gru_pw_all.append(pw_rmse)
    gru_pred_all.append(pred[["gws_true", "gws_forecast"]])
    gru_rmse_seeds.append(float(pw_rmse.median()))

if gru_pw_all and any(len(x) > 0 for x in gru_pw_all):
    all_pw = pd.concat([x for x in gru_pw_all if len(x) > 0], ignore_index=True)
    all_pred = pd.concat([x for x in gru_pred_all if len(x) > 0], ignore_index=True)
    if len(all_pred) > 1:
        obs, sim = all_pred["gws_true"].values, all_pred["gws_forecast"].values
        gl_rmse = np.sqrt(((sim - obs)**2).mean())
        gl_nse  = 1 - ((obs - sim)**2).sum() / ((obs - obs.mean())**2).sum()
        seeds_ok = [v for v in gru_rmse_seeds if not np.isnan(v)]
        print(f"  Mean±Std (per-seed medians): {np.mean(seeds_ok):.4f} ± {np.std(seeds_ok):.4f}")
        for p, q in [("P10",0.10),("P25",0.25),("Median",0.50),("P75",0.75),("P90",0.90)]:
            print(f"  {p}: {all_pw.quantile(q):.4f}")
        print(f"  Gl. RMSE: {gl_rmse:.4f}")
        print(f"  Gl. NSE:  {gl_nse:.4f}")
        print(f"  KGE:      {kge(obs, sim):.4f}")
    else:
        print("  GRU pred.parquet contains only training wells — holdout predictions not available.")
        print("  Global GRU row requires running GRU inference at holdout locations separately.")
else:
    print("  GRU pred.parquet contains only training wells — holdout predictions not available.")
    print("  Global GRU row requires running GRU inference at holdout locations separately.")


print("\n" + "="*70)
print("tab:seed_distance — rand_f95, per seed")
print("="*70)

all_coords = pd.read_parquet(ROOT.parent / "data" / "merged.parquet",
                              columns=["id","x_25833","y_25833"]).drop_duplicates("id")

rand95_dirs_by_seed = {}
for d in GP_FINAL.iterdir():
    if "rand_f95" in d.name and "dec_prod_hydroraum" in d.name and (d / "gp_metrics_by_id.csv").exists():
        for s in [f"ss{i}" for i in range(42, 52)]:
            if s in d.name:
                rand95_dirs_by_seed[s] = d
                break

seed_rows = []
for seed in sorted(rand95_dirs_by_seed):
    d = rand95_dirs_by_seed[seed]
    split_f = SPLITS_DIR / f"spatial_split_full_merged_coloc_rand_f95_{seed}.csv"
    if not split_f.exists():
        print(f"  No split file for {seed}")
        continue
    split_df = pd.read_csv(split_f)
    holdout_ids = set(split_df[split_df["spatial_split"] == "spatial_holdout"]["id"])
    train_ids   = set(split_df[split_df["spatial_split"] == "spatial_train"]["id"])

    h_coords = all_coords[all_coords["id"].isin(holdout_ids)][["x_25833","y_25833"]].values
    t_coords = all_coords[all_coords["id"].isin(train_ids)][["x_25833","y_25833"]].values

    dists = cKDTree(t_coords).query(h_coords, k=1)[0]
    pred = pd.read_parquet(d / "gp_pred.parquet")
    pw_rmse = []
    for wid, wdf in pred.groupby("id"):
        o = wdf["gws_true"].values
        s = wdf["gws_forecast"].values
        pw_rmse.append(float(np.sqrt(((s - o)**2).mean())))
    pw_rmse = np.array(pw_rmse)
    seed_rows.append({
        "seed": seed,
        "median_nn_km": round(np.median(dists) / 1000, 3),
        "RMSE_mean":    round(pw_rmse.mean(), 4),
        "RMSE_median":  round(np.median(pw_rmse), 4),
    })

if seed_rows:
    df_dist = pd.DataFrame(seed_rows).sort_values("median_nn_km")
    print(df_dist.to_string(index=False))
    rho, pval = spearmanr(df_dist["median_nn_km"], df_dist["RMSE_mean"])
    print(f"\n  Spearman rho (dist vs RMSE_mean): {rho:.4f}  p={pval:.4f}")


print("\n" + "="*70)
print("GRU KGE — temporal (tab:gru_vs_tft)")
print("="*70)

gru_kge_chunks = []
seen_seeds: set[str] = set()
for run_dir in sorted(GRU_FINAL.iterdir()):
    meta_f = run_dir / "meta.yaml"
    if not meta_f.exists():
        continue
    with open(meta_f) as f:
        sig = yaml.safe_load(f).get("run_sig", "")
    if "rand_f95" not in sig:
        continue
    seed = next((f"ss{i}" for i in range(42, 52) if sig.endswith(f"ss{i}")), None)
    if seed is None or seed in seen_seeds:
        continue
    pred_f = run_dir / "predictions" / "pred.parquet"
    if not pred_f.exists():
        continue
    pred = pd.read_parquet(pred_f)
    gru_kge_chunks.append(pred[["gws_forecast", "gws"]].rename(columns={"gws": "gws_true"}))
    seen_seeds.add(seed)

if gru_kge_chunks:
    gru_all = pd.concat(gru_kge_chunks, ignore_index=True)
    obs = gru_all["gws_true"].values
    sim = gru_all["gws_forecast"].values
    print(f"  GRU Global KGE (pooled, {len(seen_seeds)} seeds): {kge(obs, sim):.4f}")
    pw_kge = []
    for wid, wdf in gru_all.groupby(gru_all.index // (len(gru_all) // (len(seen_seeds) * 1000) + 1)):
        pass
    last_chunk = gru_kge_chunks[-1]
    for run_dir in sorted(GRU_FINAL.iterdir()):
        meta_f = run_dir / "meta.yaml"
        if not meta_f.exists(): continue
        with open(meta_f) as f:
            sig = yaml.safe_load(f).get("run_sig", "")
        if "rand_f95" not in sig or not sig.endswith("ss51"): continue
        pred_f = run_dir / "predictions" / "pred.parquet"
        if not pred_f.exists(): continue
        p = pd.read_parquet(pred_f)
        pw_kge = []
        for wid, wdf in p.groupby("id"):
            o = wdf["gws"].values
            s = wdf["gws_forecast"].values
            if len(np.unique(o)) > 1 and len(np.unique(s)) > 1:
                pw_kge.append(kge(o, s))
        print(f"  GRU per-well median KGE (ss51, {len(pw_kge)} wells): {np.median(pw_kge):.4f}")
        break
    print(f"  GRU Global RMSE: {np.sqrt(((sim-obs)**2).mean()):.4f}")
    ss_tot = ((obs - obs.mean())**2).sum()
    print(f"  GRU Global NSE:  {1 - ((obs-sim)**2).sum()/ss_tot:.4f}")


print("\n" + "="*70)
print("tab:seed_distance — rand90_f90, 40 seeds (dec_prod_hr_gps7)")
print("="*70)

GP_RAW = ROOT / "outputs" / "gp"

rand90_dirs_by_seed: dict[str, Path] = {}
for d in GP_RAW.iterdir():
    if ("rand90_f90" in d.name
            and "dec_prod_hr_gps7" in d.name
            and (d / "gp_pred.parquet").exists()):
        for s in [f"ss{i}" for i in range(42, 82)]:
            if f"_{s}__" in d.name:
                rand90_dirs_by_seed[s] = d
                break

print(f"  Found {len(rand90_dirs_by_seed)} seed directories")

seed_rows_90 = []
for seed in sorted(rand90_dirs_by_seed):
    d = rand90_dirs_by_seed[seed]
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

    seed_rows_90.append({
        "seed":         seed,
        "median_nn_km": round(np.median(dists_m) / 1000, 3),
        "RMSE_mean":    round(pw_rmse.mean(), 3),
        "RMSE_median":  round(np.median(pw_rmse), 3),
    })

if seed_rows_90:
    df90 = pd.DataFrame(seed_rows_90).sort_values("median_nn_km")
    print(df90.to_string(index=False))
    rho_med, p_med = spearmanr(df90["median_nn_km"], df90["RMSE_median"])
    rho_mn,  p_mn  = spearmanr(df90["median_nn_km"], df90["RMSE_mean"])
    print(f"\n  Spearman rho (dist vs RMSE_median): {rho_med:.4f}  p={p_med:.6f}")
    print(f"  Spearman rho (dist vs RMSE_mean):   {rho_mn:.4f}  p={p_mn:.6f}")
