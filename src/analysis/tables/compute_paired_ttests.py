import glob
import re

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t as tdist

RUN_METRICS = "reports/tables/run_metrics_from_logs.csv"
JOINT_RUNS = "outputs/GRU_GP_JOINT/*/meta.yaml"


def paired_ttest(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = d.size
    dbar = d.mean()
    s_d = d.std(ddof=1)
    se = s_d / np.sqrt(n)
    tstat = dbar / se
    df = n - 1
    p = 2.0 * tdist.sf(abs(tstat), df)
    return dict(n=n, dbar=dbar, s_d=s_d, se=se, t=tstat, df=df, p=p)


def report(name, a, b):
    r = paired_ttest(a, b)
    print(f"\n{name}")
    print(f"  n            = {r['n']}")
    print(f"  mean A       = {np.mean(a):.4f}")
    print(f"  mean B       = {np.mean(b):.4f}")
    print(f"  mean diff    = {r['dbar']:+.4f} m  (A - B)")
    print(f"  std diff     = {r['s_d']:.4f}")
    print(f"  std error    = {r['se']:.4f}")
    print(f"  t_{r['df']:<2d}        = {r['t']:.3f}")
    print(f"  p (2-sided)  = {r['p']:.4f}")
    if r['n'] != 40:
        print(f"  !! expected n = 40 split seeds, got {r['n']} -- check the run filter")
    return r


def per_seed(df, run_contains, run_excludes=(), value_col="RMSE_pw"):
    m = pd.Series(True, index=df.index)
    for s in run_contains:
        m &= df["run"].str.contains(s, case=False, na=False)
    for s in run_excludes:
        m &= ~df["run"].str.contains(s, case=False, na=False)
    sub = df[m].copy()
    sub["ss"] = sub["run"].str.extract(r"_ss(\d+)").astype(float)
    sub = sub.dropna(subset=["ss", value_col])
    return sub.groupby("ss")[value_col].mean().sort_index()


def perwell_median_rmse(pred_path, horizon=None):
    df = pd.read_parquet(pred_path, columns=["id", "horizon", "gws_true", "gws_forecast"])
    if horizon is not None:
        df = df[df["horizon"] == horizon]
    pw = np.sqrt(((df["gws_forecast"] - df["gws_true"]) ** 2).groupby(df["id"]).mean())
    return pw.median()


def joint_variant(run_sig):
    run_sig = re.sub(r"_ss\d+", "", run_sig)
    if "coloc_rand90_f90" not in run_sig or "valcheck" in run_sig:
        return None
    if run_sig.endswith("_hr_gru"):
        return "joint35"
    if run_sig.endswith("_joint_nohr"):
        return None
    if "seed44" in run_sig and run_sig.endswith("_joint"):
        return "joint32"
    return None


def joint_per_seed(variant, horizon=None):
    out = {}
    for meta_path in glob.glob(JOINT_RUNS):
        meta = yaml.safe_load(open(meta_path))
        sig = meta.get("run_sig", "")
        if joint_variant(sig) != variant:
            continue
        ss = re.search(r"_ss(\d+)", sig)
        if not ss:
            continue
        pred = meta_path.replace("meta.yaml", "eval/test/gp_pred.parquet")
        try:
            out[int(ss.group(1))] = perwell_median_rmse(pred, horizon)
        except FileNotFoundError:
            pass
    return out


def main():
    df = pd.read_csv(RUN_METRICS)

    pipe = per_seed(df, ["coloc_rand90_f90", "dec_prod"], run_excludes=["oracle", "ORACLE"])
    oracle = per_seed(df, ["coloc_rand90_f90", "oracle"])
    common = pipe.index.intersection(oracle.index)
    if len(common) >= 2:
        report("Comparison 1 -- pipeline vs oracle interpolation baseline (expect t~0.68, p~0.50)",
               pipe.loc[common].values, oracle.loc[common].values)
    else:
        print("\nComparison 1: could not align pipeline/oracle seeds -- adjust filters "
              f"(pipe n={pipe.size}, oracle n={oracle.size}).")

    j32 = joint_per_seed("joint32", horizon=None)
    j35 = joint_per_seed("joint35", horizon=None)
    common = sorted(set(j32) & set(j35))
    if len(common) >= 2:
        report("Comparison 2 -- joint 32 vs 35 static, hr_gru (expect dRMSE~-0.011, p~0.311)",
               [j35[s] for s in common], [j32[s] for s in common])
    else:
        print("\nComparison 2: could not align joint 32/35 seeds "
              f"(j32 n={len(j32)}, j35 n={len(j35)}) -- check outputs/GRU_GP_JOINT.")


if __name__ == "__main__":
    main()
