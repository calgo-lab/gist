from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT    = Path(__file__).resolve().parents[4]
GP_DIR  = ROOT / "outputs" / "gp"
JT_DIR  = ROOT / "outputs" / "01_final_results" / "GRU_GP_JOINT"
OUT_DIR = ROOT / "reports" / "figures" / "joint_vs_two_stage"
ORACLE_DIR = ROOT / "reports" / "figures" / "trueobs_baseline_vs_two_stage"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ORACLE_DIR.mkdir(parents=True, exist_ok=True)

TEST_CUTOFF = pd.Timestamp("2020-01-01")
CLIP_RMSE   = 5.0

JOINT_RAND_F90_BY_SS = {42: 1344, 43: 1347, 44: 1354, 45: 1352, 46: 1349,
                         47: 1345, 48: 1351, 49: 1348, 50: 1346, 51: 1353}
JOINT_RAND_F95_BY_SS = {42: 1329, 43: 1331, 44: 1328, 45: 1330, 46: 1332,
                         47: 1295, 48: 1275, 49: 1280, 50: 1281, 51: 1277}
JOINT_MD50_F90_BY_SS = {42: 1350, 43: 1362, 44: 1360, 45: 1357, 46: 1358,
                         47: 1364, 48: 1359, 49: 1363, 50: 1361, 51: 1365}
JOINT_KM_F90_BY_SS   = {42: 1380, 43: 1382, 44: 1381, 45: 1375, 46: 1374,
                         47: 1377, 48: 1379, 49: 1373, 50: 1376, 51: 1378}
JOINT_MD50_F95_BY_SS = {42: 1279, 43: 1278, 44: 1283, 45: 1292, 46: 1294,
                         47: 1273, 48: 1276, 49: 1274, 50: 1282, 51: 1286}
JOINT_KM_F95_BY_SS   = {42: 1474, 43: 1480, 44: 1481, 45: 1477, 46: 1475,
                         47: 1476, 48: 1473, 49: 1478, 50: 1482, 51: 1479}
JOINT_RAND90_F90_BY_SS = {42: 1485, 43: 1490, 44: 1483, 45: 1492, 46: 1488,
                           47: 1484, 48: 1491, 49: 1486, 50: 1487, 51: 1489}
JOINT_FAR80_F90_BY_SS  = {42: 1503, 43: 1504, 44: 1508, 45: 1506, 46: 1507,
                           47: 1509, 48: 1505, 49: 1510, 50: 1511, 51: 1512}


def _rmse_per_well(df: pd.DataFrame) -> pd.Series:
    df = df[df["datum"] > TEST_CUTOFF].dropna(subset=["gws_true", "gws_forecast"])
    return (
        df.groupby("id")
        .apply(lambda g: np.sqrt(np.mean((g["gws_true"] - g["gws_forecast"]) ** 2)))
        .rename("rmse")
    )


def _load_joint(ss_to_run_id: dict[int, int]) -> dict[int, pd.Series]:
    result = {}
    for ss, run_id in ss_to_run_id.items():
        pred = JT_DIR / f"GRU_GP_JOINT_{run_id}" / "eval" / "test" / "gp_pred.parquet"
        if not pred.exists():
            print(f"  SKIP joint ss{ss} run_id={run_id}: no pred file")
            continue
        df = pq.read_table(pred, columns=["id", "datum", "gws_true", "gws_forecast"]).to_pandas()
        df["datum"] = pd.to_datetime(df["datum"])
        result[ss] = _rmse_per_well(df)
    return result


def _load_dec_gp(pattern: str) -> dict[int, pd.Series]:
    result = {}
    for ss in range(42, 52):
        p = GP_DIR / pattern.format(ss=ss) / "gp_pred.parquet"
        if not p.exists():
            print(f"  SKIP dec ss{ss}: {p.name}")
            continue
        df = pq.read_table(p, columns=["id", "datum", "gws_true", "gws_forecast"]).to_pandas()
        df["datum"] = pd.to_datetime(df["datum"])
        result[ss] = _rmse_per_well(df)
    return result


def _scatter(x_vals, y_vals, xlabel, ylabel, title, out_path,
             med_x_override=None, med_y_override=None, clip_rmse=None):
    clip = clip_rmse if clip_rmse is not None else CLIP_RMSE
    xc = np.clip(x_vals, 0, clip)
    yc = np.clip(y_vals, 0, clip)
    n_clipped = int(np.sum((x_vals > clip) | (y_vals > clip)))
    med_x = med_x_override if med_x_override is not None else float(np.median(x_vals))
    med_y = med_y_override if med_y_override is not None else float(np.median(y_vals))
    x_better = int(np.sum(x_vals < y_vals))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(xc, yc, s=28, alpha=0.55, color="#4C72B0", edgecolors="none", zorder=3)
    ax.plot([0, clip], [0, clip], "k--", lw=1.2, alpha=0.5, zorder=2)
    ax.axvline(min(med_x, clip), color="#C44E52", lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.axhline(min(med_y, clip), color="#55A868", lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.set_xlim(0, clip)
    ax.set_ylim(0, clip)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)

    line1 = f"n={len(x_vals)}  x better: {x_better}/{len(x_vals)}"
    if n_clipped:
        line1 += f"  {n_clipped} clipped at {clip:g} m"
    line2 = f"median x: {med_x:.3f} m   median y: {med_y:.3f} m"
    txt1 = TextArea(line1, textprops=dict(fontsize=8))
    txt2 = TextArea(line2, textprops=dict(fontsize=8, fontweight="bold"))
    packed = VPacker(children=[txt1, txt2], pad=0, sep=2)
    ab = AnchoredOffsetbox(loc="upper left", child=packed, pad=0.3, frameon=True,
                           bbox_to_anchor=(0.03, 0.97), bbox_transform=ax.transAxes,
                           borderpad=0.4)
    ab.patch.set(facecolor="white", alpha=0.8)
    ax.add_artist(ab)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")
    return med_x, med_y


def _pair_and_collect(dec_by_ss, jt_by_ss):
    x_vals, y_vals = [], []
    x_seed_meds, y_seed_meds = [], []
    for ss in sorted(set(dec_by_ss) & set(jt_by_ss)):
        d = dec_by_ss[ss]
        j = jt_by_ss[ss]
        common = sorted(set(d.index) & set(j.index))
        if len(common) == 0:
            print(f"  WARNING ss{ss}: 0 common wells (dec={len(d)}, jt={len(j)})")
            continue
        dx = d.loc[common].values
        jy = j.loc[common].values
        x_vals.extend(dx.tolist())
        y_vals.extend(jy.tolist())
        x_seed_meds.append(float(np.median(dx)))
        y_seed_meds.append(float(np.median(jy)))
        print(f"  ss{ss}: {len(common)} paired wells, dec_median={x_seed_meds[-1]:.3f}, other_median={y_seed_meds[-1]:.3f}")
    mean_med_x = float(np.mean(x_seed_meds))
    mean_med_y = float(np.mean(y_seed_meds))
    print(f"  → mean-of-medians: x={mean_med_x:.4f} m  y={mean_med_y:.4f} m")
    return np.array(x_vals), np.array(y_vals), mean_med_x, mean_med_y


def mean_of_medians(label: str, by_ss: dict[int, pd.Series]) -> float:
    seed_meds = [float(np.median(s.values)) for s in by_ss.values()]
    mom = float(np.mean(seed_meds))
    print(f"  {label}: n_seeds={len(by_ss)}, mean-of-medians={mom:.4f} m  (seed medians: {[f'{v:.3f}' for v in seed_meds]})")
    return mom


def main():
    print("\n=== Scatter: joint vs decoupled — rand 90/10 (matched holdout sets) ===\n")
    dec_f90 = _load_dec_gp(
        "GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_rand90_f90_ss{ss}__dec_prod_hydroraum_gps1__predobstrain"
    )
    jt_f90 = _load_joint(JOINT_RAND90_F90_BY_SS)
    print(f"  Loaded: {len(dec_f90)} dec seeds, {len(jt_f90)} joint seeds")
    x, y, mom_dec_f90, mom_jt_f90 = _pair_and_collect(dec_f90, jt_f90)
    print(f"  Total paired points: {len(x)}")
    _scatter(
        x, y,
        xlabel="Two-stage model (GRU → GP) — per-well RMSE [m]",
        ylabel="End-to-end model (GRU + GP) — per-well RMSE [m]",
        title="Two-stage vs end-to-end spatiotemporal model\n(Random split · 90/10 · 10 seeds · 104 holdout wells/seed)",
        out_path=OUT_DIR / "scatter_joint_vs_dec_rand_f90_corrected.png",
        med_x_override=mom_dec_f90,
        med_y_override=mom_jt_f90,
    )

    print("\n=== Scatter: decoupled vs oracle — rand 90/10 ===\n")
    orc_f90 = _load_dec_gp(
        "ORACLE_oracle_coloc_rand90_f90_ss{ss}__oracle_prod_hr_gps1__predobstrain"
    )
    print(f"  Loaded: {len(dec_f90)} dec seeds, {len(orc_f90)} oracle seeds")
    x, y, mom_dec_f90_orc, mom_orc_f90 = _pair_and_collect(dec_f90, orc_f90)
    print(f"  Total paired points: {len(x)}")
    _scatter(
        x, y,
        xlabel="Two-stage model (GRU → GP) — per-well RMSE [m]",
        ylabel="Upper bound (true GWL + GP) — per-well RMSE [m]",
        title="Two-stage model vs upper bound\n(Random split · 90/10 · 10 seeds · 104 holdout wells/seed)",
        out_path=ORACLE_DIR / "scatter_oracle_vs_dec_rand_f90.png",
        med_x_override=mom_dec_f90_orc,
        med_y_override=mom_orc_f90,
    )

    print("\n=== Scatter: joint vs decoupled — rand 95/5 ===\n")
    dec_f95 = _load_dec_gp(
        "GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_rand_f95_ss{ss}__dec_prod_hydroraum_gps1__predobstrain"
    )
    jt_f95 = _load_joint(JOINT_RAND_F95_BY_SS)
    print(f"  Loaded: {len(dec_f95)} dec seeds, {len(jt_f95)} joint seeds")
    x, y, mom_dec_f95, mom_jt_f95 = _pair_and_collect(dec_f95, jt_f95)
    print(f"  Total paired points: {len(x)}")
    _scatter(
        x, y,
        xlabel="Two-stage model (GRU → GP) — per-well RMSE [m]",
        ylabel="End-to-end model (GRU + GP) — per-well RMSE [m]",
        title="Two-stage vs end-to-end spatiotemporal model\n(Random split · 95/5 · 10 seeds · 52 holdout wells/seed)",
        out_path=OUT_DIR / "scatter_joint_vs_dec_rand_f95.png",
        med_x_override=mom_dec_f95,
        med_y_override=mom_jt_f95,
    )

    print("\n=== Scatter: joint vs decoupled — md50 f90 (dense wells) ===\n")
    dec_md50_f90_s = _load_dec_gp(
        "GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_md50_f90_ss{ss}__dec_prod_hydroraum_gps1__predobstrain"
    )
    jt_md50_f90_s = _load_joint(JOINT_MD50_F90_BY_SS)
    print(f"  Loaded: {len(dec_md50_f90_s)} dec seeds, {len(jt_md50_f90_s)} joint seeds")
    x, y, mom_dec_md50, mom_jt_md50 = _pair_and_collect(dec_md50_f90_s, jt_md50_f90_s)
    print(f"  Total paired points: {len(x)}")
    _scatter(
        x, y,
        xlabel="Two-stage model (GRU → GP) — per-well RMSE [m]",
        ylabel="End-to-end model (GRU + GP) — per-well RMSE [m]",
        title="Two-stage vs end-to-end spatiotemporal model\n(Dense split · 90/10 · 10 seeds · 104 holdout wells/seed)",
        out_path=OUT_DIR / "scatter_joint_vs_dec_md50_f90.png",
        med_x_override=mom_dec_md50,
        med_y_override=mom_jt_md50,
        clip_rmse=5.0,
    )

    print("\n=== Scatter: joint vs decoupled — far80 f90 (isolated wells) ===\n")
    dec_far80 = _load_dec_gp(
        "GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_far80_f90_ss{ss}__dec_prod_hydroraum_gps1__predobstrain"
    )
    jt_far80 = _load_joint(JOINT_FAR80_F90_BY_SS)
    print(f"  Loaded: {len(dec_far80)} dec seeds, {len(jt_far80)} joint seeds")
    x, y, mom_dec_far80, mom_jt_far80 = _pair_and_collect(dec_far80, jt_far80)
    print(f"  Total paired points: {len(x)}")
    _scatter(
        x, y,
        xlabel="Two-stage model (GRU → GP) — per-well RMSE [m]",
        ylabel="End-to-end model (GRU + GP) — per-well RMSE [m]",
        title="Two-stage vs end-to-end spatiotemporal model\n(Far-80 split · 90/10 · 10 seeds · 104 holdout wells/seed)",
        out_path=OUT_DIR / "scatter_joint_vs_dec_far80_f90.png",
        med_x_override=mom_dec_far80,
        med_y_override=mom_jt_far80,
        clip_rmse=15.0,
    )

    print("\n=== Loading remaining runs ===\n")
    jt_md50_f90 = _load_joint(JOINT_MD50_F90_BY_SS)
    jt_km_f90   = _load_joint(JOINT_KM_F90_BY_SS)
    jt_md50_f95 = _load_joint(JOINT_MD50_F95_BY_SS)

    dec_md50_f90 = _load_dec_gp(
        "GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_md50_f90_ss{ss}__dec_prod_hydroraum_gps1__predobstrain"
    )
    dec_md50_f95 = _load_dec_gp(
        "GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_md50_f95_ss{ss}__dec_prod_hydroraum_gps1__predobstrain"
    )
    dec_km_f90 = _load_dec_gp(
        "GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_km_f90_ss{ss}__dec_prod_hydroraum_gps1__predobstrain"
    )
    dec_km_f95 = _load_dec_gp(
        "GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_km_f95_ss{ss}__dec_prod_hydroraum_gps1__predobstrain"
    )

    orc_f95     = _load_dec_gp(
        "ORACLE_oracle_coloc_rand_f95_ss{ss}__oracle_prod_hr_gps1__predobstrain"
    )
    orc_md50_f90 = _load_dec_gp(
        "ORACLE_oracle_coloc_md50_f90_ss{ss}__oracle_prod_hr_gps1__predobstrain"
    )
    orc_md50_f95 = _load_dec_gp(
        "ORACLE_oracle_coloc_md50_f95_ss{ss}__oracle_prod_hr_gps1__predobstrain"
    )
    orc_km_f90   = _load_dec_gp(
        "ORACLE_oracle_coloc_km_f90_ss{ss}__oracle_ms_gps1__predobstrain"
    )

    def mom(by_ss):
        if not by_ss:
            return None
        return float(np.mean([np.median(s.values) for s in by_ss.values()]))

    rows = [
        ("rand",  "90/10", mom(dec_f90),      mom(orc_f90),      mom(jt_f90)),
        ("rand",  "95/5",  mom(dec_f95),      mom(orc_f95),      mom(jt_f95)),
        ("md50",  "90/10", mom(dec_md50_f90), mom(orc_md50_f90), mom(jt_md50_f90)),
        ("md50",  "95/5",  mom(dec_md50_f95), mom(orc_md50_f95), mom(jt_md50_f95)),
        ("km",    "90/10", mom(dec_km_f90),   mom(orc_km_f90),   mom(jt_km_f90)),
        ("km",    "95/5",  mom(dec_km_f95),   None,              None),
    ]

    def _fmt(v):
        return f"{v:.4f} m" if v is not None else "     —"

    print("\n=== Full table: mean-of-per-seed-medians per-well RMSE ===\n")
    hdr = f"{'Split':<6}  {'Holdout':<7}  {'Decoupled':>10}  {'Oracle':>10}  {'Joint':>10}"
    print(hdr)
    print("-" * len(hdr))
    for split, holdout, d, o, j in rows:
        print(f"{split:<6}  {holdout:<7}  {_fmt(d):>10}  {_fmt(o):>10}  {_fmt(j):>10}")


if __name__ == "__main__":
    main()
