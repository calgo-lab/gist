from pathlib import Path
import numpy as np, pandas as pd

ROOT       = Path(__file__).resolve().parents[3]
GP_OUT_DIR = ROOT / "outputs" / "gp"
MERGED     = ROOT.parent / "data" / "merged.parquet"

SEEDS = list(range(42, 52))
GP_SEEDS = [1, 2, 3]


def pool_tag(dirs_all, tag_substr, split_substr):
    matching = [d for d in dirs_all
                if f'__{tag_substr}__' in d and split_substr in d and 'ablat' not in d]
    frames, missing = [], []
    for d in matching:
        p = GP_OUT_DIR / d / "gp_pred.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p, columns=["id","gws_true","gws_forecast","gws_forecast_std"]))
        else:
            missing.append(d)
    return frames, missing

def stats(frames, tag):
    if not frames:
        print(f"  {tag}: NO DATA"); return
    df = pd.concat(frames, ignore_index=True)
    err = np.abs(df["gws_true"] - df["gws_forecast"])
    sig = df["gws_forecast_std"]
    picp = {lvl: float((err <= z*sig).mean())
            for lvl, z in [(50,0.674),(90,1.645),(95,1.960)]}
    rmse_pw = (df.groupby("id")
               .apply(lambda g: np.sqrt(((g["gws_true"]-g["gws_forecast"])**2).mean()),
                      include_groups=False)
               .median())
    sample = pd.read_parquet(GP_OUT_DIR / frames[0].index[0] if False else
                             GP_OUT_DIR / "dummy", columns=None) if False else None
    n_rows = sum(len(f) for f in frames)
    n_files = len(frames)
    print(f"  {tag}: files={n_files}, rows={n_rows:,}, "
          f"RMSE_pw={rmse_pw:.3f}m | "
          f"PICP@50%={picp[50]:.1%} @90%={picp[90]:.1%} @95%={picp[95]:.1%}")

dirs_all = [d.name for d in GP_OUT_DIR.iterdir() if d.is_dir()]

print("=" * 70)
print("ORACLE pipeline candidates (rand f95)")
print("=" * 70)
print("  [Thesis table: RMSE~1.337m | PICP@50=27.1% @90=49.0% @95=54.9%]")
for tag in ["oracle_ms_gps1","oracle_ms_gps2","oracle_ms_gps3"]:
    frames, miss = pool_tag(dirs_all, tag, "rand_f95")
    stats(frames, tag)
print("--- pooled oracle_ms (all 3 GP seeds) ---")
all_frames, _ = pool_tag(dirs_all, "oracle_ms_gps", "rand_f95")
stats(all_frames, "oracle_ms ALL GPS")

print()
print("  --- newer oracle with hydroraum-only feature ---")
for tag in ["oracle_prod_hr_gps1","oracle_prod_hr_gps2","oracle_prod_hr_gps3"]:
    frames, miss = pool_tag(dirs_all, tag, "rand_f95")
    stats(frames, tag)
print("--- pooled oracle_prod_hr (all 3 GPS) ---")
all_frames, _ = pool_tag(dirs_all, "oracle_prod_hr_gps", "rand_f95")
stats(all_frames, "oracle_prod_hr ALL GPS")

print()
print("=" * 70)
print("DECOUPLED pipeline candidates (rand f95)")
print("=" * 70)
print("  [Thesis table: RMSE~? | PICP@50=26.8% @90=48.7% @95=54.8%]")
for tag in ["multiseed_rand_gps1","multiseed_rand_gps2","multiseed_rand_gps3"]:
    frames, miss = pool_tag(dirs_all, tag, "rand_f95")
    if miss: print(f"    WARNING: {len(miss)} missing dirs")
    stats(frames, tag)
print("--- pooled multiseed_rand (all 3 GPS) ---")
all_frames, _ = pool_tag(dirs_all, "multiseed_rand_gps", "rand_f95")
stats(all_frames, "multiseed_rand ALL GPS")

print()
print("  --- newer decoupled with hydroraum-only feature ---")
for tag in ["dec_prod_hydroraum_gps1","dec_prod_hydroraum_gps2","dec_prod_hydroraum_gps3"]:
    frames, miss = pool_tag(dirs_all, tag, "rand_f95")
    if miss: print(f"    WARNING: {len(miss)} missing dirs")
    stats(frames, tag)
print("--- pooled dec_prod_hydroraum (all GPS) ---")
all_frames, _ = pool_tag(dirs_all, "dec_prod_hydroraum_gps", "rand_f95")
stats(all_frames, "dec_prod_hydroraum ALL GPS")

print()
print("=" * 70)
print("ROW COUNT CHECK (to debug earlier loading bug)")
print("=" * 70)
for tag in ["oracle_ms_gps1", "multiseed_rand_gps1"]:
    matching = [d for d in dirs_all if f'__{tag}__' in d and 'rand_f95' in d and 'ablat' not in d]
    for d in sorted(matching)[:3]:
        p = GP_OUT_DIR / d / "gp_pred.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["id","horizon"])
            print(f"  {d[-60:]}: {len(df):,} rows, horizons={sorted(df['horizon'].unique())[:4]}...")

print()
print("=" * 70)
print("GP ABLATION: verify ss47 is the right seed")
print("=" * 70)
df_full = pd.read_parquet(MERGED, columns=["id","gws"])
per_well_range = df_full.groupby("id")["gws"].agg(lambda x: x.max()-x.min()).rename("range")

tag = "feat_coords_only_gps1"
for ss in [42, 45, 47, 48, 50]:
    d = f"ORACLE_oracle_ablat_rand_f95_ss{ss}__feat_coords_only_gps1__predobstrain"
    p = GP_OUT_DIR / d / "gp_pred.parquet"
    if p.exists():
        pred = pd.read_parquet(p, columns=["id","gws_true","gws_forecast"])
        pw = pred.groupby("id").apply(lambda g: np.sqrt(((g["gws_true"]-g["gws_forecast"])**2).mean()), include_groups=False).rename("rmse").to_frame().join(per_well_range)
        pw["nrmse"] = pw["rmse"] / pw["range"]
        print(f"  ss{ss} coords_only: nRMSE_pw={pw['nrmse'].median():.4f}")
    else:
        print(f"  ss{ss}: MISSING {d}")
