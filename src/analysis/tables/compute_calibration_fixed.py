from pathlib import Path
import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parents[3]
GP_OUT_DIR = ROOT / "outputs" / "gp"
OUT_CSV    = ROOT / "reports" / "tables" / "calibration_curve.csv"


def load_pred_files(tag_prefix: str, split_substr: str = "rand_f95") -> pd.DataFrame:
    dirs_all = [d for d in GP_OUT_DIR.iterdir() if d.is_dir()]
    matching = [
        d for d in dirs_all
        if f'__{tag_prefix}' in d.name and split_substr in d.name and 'ablat' not in d.name
    ]
    frames = []
    for d in sorted(matching):
        p = d / "gp_pred.parquet"
        if p.exists():
            frames.append(pd.read_parquet(
                p, columns=["id", "gws_true", "gws_forecast", "gws_forecast_std"]
            ))
        else:
            print(f"  MISSING: {d.name}")
    if not frames:
        raise RuntimeError(f"No files found for tag_prefix={tag_prefix!r}")
    df = pd.concat(frames, ignore_index=True)
    print(f"  {tag_prefix}: {len(matching)} dirs, {len(df):,} rows")
    return df


def compute_picp_pooled(df: pd.DataFrame, alphas: list[float]) -> list[float]:
    err = np.abs(df["gws_true"] - df["gws_forecast"])
    sig = df["gws_forecast_std"]
    coverages = []
    for alpha in alphas:
        z = float(np.quantile(np.random.default_rng(0).standard_normal(10_000_000).__abs__(), alpha)) \
            if False else _alpha_to_z(alpha)
        coverages.append(float((err <= z * sig).mean()))
    return coverages


def _alpha_to_z(alpha: float) -> float:
    try:
        from scipy.stats import norm
        return norm.ppf((1 + alpha) / 2)
    except ImportError:
        table = {0.50: 0.6745, 0.60: 0.8416, 0.70: 1.0364, 0.80: 1.2816,
                 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
        return table[round(alpha, 2)]


def compute_picp_per_well_avg(df: pd.DataFrame, alphas: list[float]) -> list[float]:
    err = np.abs(df["gws_true"] - df["gws_forecast"])
    sig = df["gws_forecast_std"]
    df2 = df.copy()
    df2["err"] = err
    df2["sig"] = sig
    coverages = []
    for alpha in alphas:
        z = _alpha_to_z(alpha)
        per_well = df2.groupby("id").apply(
            lambda g: float((g["err"] <= z * g["sig"]).mean()), include_groups=False
        )
        coverages.append(float(per_well.mean()))
    return coverages


ALPHAS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

print("=== Oracle pipeline (oracle_ms, siwa+hr+gesp) ===")
df_oracle = load_pred_files("oracle_ms_gps")

print("\n=== Decoupled pipeline (multiseed_rand, siwa+hr+gesp) ===")
df_dec = load_pred_files("multiseed_rand_gps")

print("\n=== Computing calibration curves ===")
oracle_pooled  = compute_picp_pooled(df_oracle, ALPHAS)
dec_pooled     = compute_picp_pooled(df_dec, ALPHAS)
oracle_pw      = compute_picp_per_well_avg(df_oracle, ALPHAS)
dec_pw         = compute_picp_per_well_avg(df_dec, ALPHAS)

print("\nNominal | Oracle(pooled) | Oracle(per-well) | Dec(pooled) | Dec(per-well)")
for i, alpha in enumerate(ALPHAS):
    print(f"  {alpha:.0%}  |   {oracle_pooled[i]:.1%}    |    {oracle_pw[i]:.1%}      "
          f"|   {dec_pooled[i]:.1%}  |   {dec_pw[i]:.1%}")

print("\n[Thesis table reference: oracle PICP@50=27.1% @90=49.0% @95=54.9%]")
print("[Thesis table reference: decoupled PICP@50=26.8% @90=48.7% @95=54.8%]")

out = pd.DataFrame({
    "nominal_level":     ALPHAS,
    "oracle_coverage":   oracle_pooled,
    "decoupled_coverage": dec_pooled,
    "oracle_coverage_pw": oracle_pw,
    "decoupled_coverage_pw": dec_pw,
})
out.to_csv(OUT_CSV, index=False)
print(f"\nSaved to {OUT_CSV}")
