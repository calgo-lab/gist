from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

REPO       = Path(__file__).resolve().parents[4]
METRICS_PQ = REPO / "data" / "metrics_aggregated.parquet"
DEC_BASE   = REPO / "outputs" / "gp"
ORC_BASE   = REPO / "outputs" / "gp"
GRU_BASE   = REPO / "outputs" / "GRU_FCOV"
OUT_DIR    = REPO / "reports" / "figures" / "joint_vs_two_stage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(42, 82))

GRU_RUN_IDS = list(range(1513, 1523))

DEC_PATTERN = (
    "GRU_FCOV_in52_out16_ep50_bs4096_seed47_coloc_rand90_f90_ss{ss}"
    "__dec_prod_hr_gps7__predobstrain"
)
ORC_PATTERN = (
    "ORACLE_oracle_coloc_rand90_f90_ss{ss}"
    "__oracle_prod_hr_gps7__predobstrain"
)

COLOR      = "#4C72B0"
TEST_START = pd.Timestamp("2020-01-01")


def scatter_rmse(
    xv: np.ndarray,
    yv: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    clip: float = 5.0,
) -> None:
    xc = np.clip(xv, 0, clip)
    yc = np.clip(yv, 0, clip)
    n = len(xv)
    x_better = int(np.sum(xv < yv))
    n_clipped = int(np.sum((xv > clip) | (yv > clip)))
    med_x = float(np.median(xv))
    med_y = float(np.median(yv))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(xc, yc, s=28, alpha=0.65, color=COLOR, edgecolors="none", zorder=3)
    ax.plot([0, clip], [0, clip], "k--", lw=1.2, alpha=0.5, zorder=2)
    ax.axvline(min(med_x, clip), color="#C44E52", lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.axhline(min(med_y, clip), color="#55A868", lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.set_xlim(0, clip)
    ax.set_ylim(0, clip)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)

    line1 = f"n={n}   x better: {x_better}/{n}"
    if n_clipped:
        line1 += f"   {n_clipped} clipped at {clip:g} m"
    line2 = f"median x: {med_x:.3f} m   median y: {med_y:.3f} m"
    _add_annotation(ax, line1, line2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def scatter_nse(
    xv: np.ndarray,
    yv: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    clip_lo: float = -100.0,
    clip_hi: float = 1.0,
) -> None:
    xc = np.clip(xv, clip_lo, clip_hi)
    yc = np.clip(yv, clip_lo, clip_hi)
    n = len(xv)
    x_better = int(np.sum(xv > yv))
    n_clipped = int(np.sum((xv < clip_lo) | (yv < clip_lo)))
    med_x = float(np.median(xv))
    med_y = float(np.median(yv))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(xc, yc, s=28, alpha=0.65, color=COLOR, edgecolors="none", zorder=3)
    ax.plot([clip_lo, clip_hi], [clip_lo, clip_hi], "k--", lw=1.2, alpha=0.5, zorder=2)
    ax.axvline(np.clip(med_x, clip_lo, clip_hi), color="#C44E52",
               lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.axhline(np.clip(med_y, clip_lo, clip_hi), color="#55A868",
               lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.set_xlim(clip_lo, clip_hi)
    ax.set_ylim(clip_lo, clip_hi)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)

    line1 = f"n={n}   x better: {x_better}/{n}"
    if n_clipped:
        line1 += f"   {n_clipped} clipped at {clip_lo:g}"
    line2 = f"median x: {med_x:.3f}   median y: {med_y:.3f}"
    _add_annotation(ax, line1, line2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def _add_annotation(ax: plt.Axes, line1: str, line2: str) -> None:
    txt1 = TextArea(line1, textprops=dict(fontsize=8))
    txt2 = TextArea(line2, textprops=dict(fontsize=8, fontweight="bold"))
    packed = VPacker(children=[txt1, txt2], pad=0, sep=2)
    ab = AnchoredOffsetbox(
        loc="upper left", child=packed, pad=0.3, frameon=True,
        bbox_to_anchor=(0.03, 0.97), bbox_transform=ax.transAxes, borderpad=0.4,
    )
    ab.patch.set(facecolor="white", alpha=0.8)
    ax.add_artist(ab)


def _load_gru_run_h16(run_id: int) -> pd.DataFrame:
    pred_path = GRU_BASE / f"GRU_FCOV_{run_id}" / "predictions" / "pred.parquet"
    if not pred_path.exists():
        print(f"  MISSING GRU_FCOV_{run_id}")
        return pd.DataFrame(columns=["id", "RMSE", "NSE"])

    pred = pq.read_table(pred_path).to_pandas()
    pred = pred[(pred["datum"] >= TEST_START) & (pred["horizon"] == 16)].copy()

    def _rmse(g):
        return np.sqrt(((g["gws_forecast"] - g["gws"]) ** 2).mean())

    def _nse(g):
        ss_res = ((g["gws_forecast"] - g["gws"]) ** 2).sum()
        ss_tot = ((g["gws"] - g["gws"].mean()) ** 2).sum()
        return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot

    rmse_s = pred.groupby("id").apply(_rmse, include_groups=False).rename("RMSE")
    nse_s  = pred.groupby("id").apply(_nse,  include_groups=False).rename("NSE")
    return pd.concat([rmse_s, nse_s], axis=1).reset_index()


def _load_gp_h16(base: Path, pattern: str, ss: int, metric: str) -> pd.Series:
    p = base / pattern.format(ss=ss) / "gp_metrics_by_id_horizon.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(p)
    h16 = df[df["horizon"] == 16][["id", metric]].set_index("id")[metric]
    return h16


def gen_gru_tft(met: pd.DataFrame) -> None:
    print("\nLoading correct 10-seed GRU runs (GRU_FCOV_1513–1522) …")
    gru_frames = []
    for rid in GRU_RUN_IDS:
        gru_frames.append(_load_gru_run_h16(rid))
    gru_all = pd.concat(gru_frames, ignore_index=True)

    gru_rmse = gru_all.groupby("id")["RMSE"].median()
    gru_nse  = gru_all.groupby("id")["NSE"].median()
    print(f"  GRU: {len(gru_rmse)} wells  |  median RMSE={gru_rmse.median():.4f} m  "
          f"median NSE={gru_nse.median():.4f}")

    for metric, clip_lo, clip_hi, suffix, fmt in [
        ("RMSE", 0.0,  0.5,  "rmse", ".pdf"),
        ("NSE",  -2.0, 1.0,  "nse",  ".pdf"),
    ]:
        tft = (
            met[(met["architecture"] == "TFT_dyn_stat")
                & (met["metric"] == metric)
                & (met["horizon"] == 16.0)]
            .set_index("id")["value"]
        )
        gru_series = gru_rmse if metric == "RMSE" else gru_nse
        common = sorted(set(gru_series.index) & set(tft.index))
        xv = tft.loc[common].values
        yv = gru_series.loc[common].values
        print(f"\nGRU vs TFT {metric} h=16: {len(common)} wells")

        xlabel = f"TFT — per-well {metric} at h=16"
        ylabel = f"GRU — per-well {metric} at h=16"
        title = f"GRU vs TFT: per-well {metric} at h=16\n(1040 wells, 10 seeds)"

        out = OUT_DIR / f"scatter_{suffix}_gru_tft{fmt}"
        if metric == "RMSE":
            scatter_rmse(xv, yv, xlabel, ylabel, title, out, clip=clip_hi)
        else:
            scatter_nse(xv, yv, xlabel, ylabel, title, out,
                        clip_lo=clip_lo, clip_hi=clip_hi)


def gen_oracle_vs_decoupled() -> None:
    print("\nCollecting true-obs baseline vs two-stage NSE/RMSE h=16 (40 seeds) …")
    dec_nse, orc_nse = [], []
    dec_rmse, orc_rmse = [], []

    for ss in SEEDS:
        dec_n = _load_gp_h16(DEC_BASE, DEC_PATTERN, ss, "NSE")
        orc_n = _load_gp_h16(ORC_BASE, ORC_PATTERN, ss, "NSE")
        dec_r = _load_gp_h16(DEC_BASE, DEC_PATTERN, ss, "RMSE")
        orc_r = _load_gp_h16(ORC_BASE, ORC_PATTERN, ss, "RMSE")
        if dec_n.empty or orc_n.empty:
            print(f"  MISSING ss={ss}")
            continue
        common_n = sorted(set(dec_n.index) & set(orc_n.index))
        common_r = sorted(set(dec_r.index) & set(orc_r.index))
        dec_nse.extend(dec_n.loc[common_n].values.tolist())
        orc_nse.extend(orc_n.loc[common_n].values.tolist())
        dec_rmse.extend(dec_r.loc[common_r].values.tolist())
        orc_rmse.extend(orc_r.loc[common_r].values.tolist())

    print(f"  NSE: {len(dec_nse)} (well, seed) pairs")
    print(f"  RMSE: {len(dec_rmse)} (well, seed) pairs")

    TITLE = ("Two-stage spatiotemporal pipeline vs true-observation interpolation baseline\n"
             "(Random 90/10 · 40 seeds · 104 test wells/seed)")

    scatter_nse(
        np.array(dec_nse), np.array(orc_nse),
        xlabel="Two-stage spatiotemporal pipeline (GRU→GP) — per-well NSE at h=16",
        ylabel="Interpolation baseline (true obs.+GP) — per-well NSE at h=16",
        title=TITLE,
        out_path=OUT_DIR / "scatter_nse_oracle_vs_decoupled.pdf",
        clip_lo=-100.0,
        clip_hi=1.0,
    )

    scatter_rmse(
        np.array(dec_rmse), np.array(orc_rmse),
        xlabel="Two-stage spatiotemporal pipeline (GRU→GP) — per-well RMSE [m] at h=16",
        ylabel="Interpolation baseline (true obs.+GP) — per-well RMSE [m] at h=16",
        title=TITLE,
        out_path=OUT_DIR / "scatter_rmse_oracle_vs_decoupled.pdf",
        clip=5.0,
    )


def gen_rmse_rand() -> None:
    print("\nLoading correct 10-seed GRU runs for two-stage vs GRU RMSE …")
    gru_frames = []
    for rid in GRU_RUN_IDS:
        gru_frames.append(_load_gru_run_h16(rid))
    gru_all = pd.concat(gru_frames, ignore_index=True)
    gru_rmse_all = gru_all.groupby("id")["RMSE"].median()

    print("Collecting two-stage vs GRU RMSE h=16 (40 seeds) …")
    gp_vals, gru_vals = [], []
    for ss in SEEDS:
        dec = _load_gp_h16(DEC_BASE, DEC_PATTERN, ss, "RMSE")
        if dec.empty:
            print(f"  MISSING dec ss={ss}")
            continue
        common = sorted(set(dec.index) & set(gru_rmse_all.index))
        gp_vals.extend(dec.loc[common].values.tolist())
        gru_vals.extend(gru_rmse_all.loc[common].values.tolist())

    xv = np.array(gp_vals)
    yv = np.array(gru_vals)
    print(f"  {len(xv)} (well, seed) pairs")

    scatter_rmse(
        xv, yv,
        xlabel="Two-stage spatiotemporal pipeline (GRU→GP) — per-well RMSE [m] at h=16",
        ylabel="GRU temporal model — per-well RMSE [m] at h=16",
        title="Two-stage spatiotemporal pipeline vs GRU direct forecast\n"
              "(Random 90/10 · 40 seeds · 104 test wells/seed)",
        out_path=OUT_DIR / "scatter_rmse_rand.png",
        clip=2.0,
    )


def main() -> None:
    print("Loading metrics_aggregated.parquet …")
    met = pq.read_table(METRICS_PQ).to_pandas()

    print("\n=== GRU vs TFT (RMSE and NSE at h=16) ===")
    gen_gru_tft(met)

    print("\n=== True-obs baseline vs Two-stage (NSE and RMSE at h=16) ===")
    gen_oracle_vs_decoupled()

    print("\n=== Two-stage vs GRU RMSE at h=16 ===")
    gen_rmse_rand()

    print("\nDone.")


if __name__ == "__main__":
    main()
