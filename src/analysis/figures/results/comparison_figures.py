import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT      = Path(__file__).resolve().parents[4]
DATA_PATH = ROOT.parent / "data" / "merged.parquet"
GP_DIR    = ROOT / "outputs" / "gp"
OUT_DIR   = ROOT / "reports" / "figures" / "_dead_ends"

VAL_CUTOFF = pd.Timestamp("2020-01-01")

RE_MS = re.compile(
    r"in\d+_out\d+_ep\d+_bs\d+_seed(?P<ms>\d+)_coloc"
    r"_(?P<split>rand|km|md\d+)"
    r"_f(?P<frac>\d+)"
    r"_ss(?P<ss>\d+)"
)
RE_JOINT = re.compile(
    r"joint_in\d+_out\d+_ep\d+_seed\d+_exact3w_(?P<split>rand|md\d+)_f\d+_ss(?P<ss>\d+)"
)



def load_decoupled(split: str, frac: int) -> pd.DataFrame:
    rows = []
    for d in GP_DIR.iterdir():
        pred_path = d / "gp_pred.parquet"
        if not pred_path.exists():
            continue
        run = d.name
        if not run.startswith("GRU_FCOV_"):
            continue
        run_sig = run[len("GRU_FCOV_"):].split("__")[0]
        m = RE_MS.search(run_sig)
        if not m or m["split"] != split or int(m["frac"]) != frac:
            continue
        try:
            df = pq.read_table(
                pred_path, columns=["id", "datum", "horizon", "gws_true", "gws_forecast"]
            ).to_pandas()
            df["datum"] = pd.to_datetime(df["datum"])
            df = df[df["datum"] > VAL_CUTOFF].dropna(subset=["gws_true", "gws_forecast"])
            rows.append(df)
        except Exception as e:
            print(f"  SKIP {run}: {e}")
    if not rows:
        raise RuntimeError(f"No decoupled preds for {split}_f{frac}")
    return pd.concat(rows, ignore_index=True)


def load_joint(split: str) -> pd.DataFrame:
    joint_root = ROOT / "outputs" / "GRU_GP_JOINT"
    if not joint_root.exists():
        raise RuntimeError(f"Joint output dir not found: {joint_root}")
    rows = []
    for d in joint_root.iterdir():
        pred_path = d / "eval" / "test" / "gp_pred.parquet"
        if not pred_path.exists():
            continue
        m = RE_JOINT.search(d.name)
        if not m or m["split"] != split:
            continue
        try:
            df = pq.read_table(
                pred_path, columns=["id", "datum", "horizon", "gws_true", "gws_forecast"]
            ).to_pandas()
            df["datum"] = pd.to_datetime(df["datum"])
            df = df[df["datum"] > VAL_CUTOFF].dropna(subset=["gws_true", "gws_forecast"])
            rows.append(df)
        except Exception as e:
            print(f"  SKIP {d.name}: {e}")
    if not rows:
        raise RuntimeError(f"No joint preds for {split}")
    return pd.concat(rows, ignore_index=True)


def per_well_rmse(df: pd.DataFrame) -> dict:
    result = {}
    for well_id, g in df.groupby("id"):
        t = g["gws_true"].to_numpy(float)
        p = g["gws_forecast"].to_numpy(float)
        mask = np.isfinite(t) & np.isfinite(p)
        if mask.sum() == 0:
            continue
        result[well_id] = float(np.sqrt(np.mean((p[mask] - t[mask]) ** 2)))
    return result


def median_rmse_by_horizon(df: pd.DataFrame) -> pd.Series:
    rows = []
    for h, hg in df.groupby("horizon"):
        pw = []
        for _, wg in hg.groupby("id"):
            t = wg["gws_true"].to_numpy(float)
            p = wg["gws_forecast"].to_numpy(float)
            mask = np.isfinite(t) & np.isfinite(p)
            if mask.sum() == 0:
                continue
            pw.append(float(np.sqrt(np.mean((p[mask] - t[mask]) ** 2))))
        rows.append({"horizon": int(h), "median_rmse": float(np.median(pw))})
    return pd.DataFrame(rows).sort_values("horizon").set_index("horizon")["median_rmse"]



def scatter_comparison(
    rmse_x: dict, rmse_y: dict,
    xlabel: str, ylabel: str, title: str,
    out_path: Path,
    clip: float = 5.0,
    diagonal: bool = True,
):
    common = sorted(set(rmse_x) & set(rmse_y))
    if not common:
        print(f"  WARNING: no common wells for {out_path.name}")
        return
    xv = np.array([rmse_x[w] for w in common])
    yv = np.array([rmse_y[w] for w in common])
    xc = np.clip(xv, 0, clip)
    yc = np.clip(yv, 0, clip)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(xc, yc, s=30, alpha=0.7, color="#4C72B0", zorder=3)
    if diagonal:
        ax.plot([0, clip], [0, clip], "k--", lw=1, alpha=0.5, zorder=2)
    ax.set_xlim(0, clip)
    ax.set_ylim(0, clip)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)

    n = len(common)
    x_better = int(np.sum(xv < yv))
    clipped = int(np.sum((xv > clip) | (yv > clip)))
    info = f"n = {n}   x better: {x_better}/{n}   {clipped} clipped"
    ax.text(0.03, 0.97, info, transform=ax.transAxes, va="top",
            fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path.name}")



COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
STYLES = ["-", "--", "-.", ":", "-"]


def line_plot(
    series_dict: dict,
    ylabel: str,
    title: str,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, s) in enumerate(series_dict.items()):
        ax.plot(s.index, s.values,
                color=COLORS[i % len(COLORS)],
                ls=STYLES[i % len(STYLES)],
                lw=2, marker="o", ms=4, label=label)
    ax.set_xlabel("Forecast horizon (weeks)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(1, 17))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path.name}")



def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading md30_f95 ...")
    md30 = load_decoupled("md30", 95)

    print("Loading rand_f95 ...")
    rand95 = load_decoupled("rand", 95)

    print("Loading rand_f90 ...")
    rand90 = load_decoupled("rand", 90)

    print("Loading rand_f80 ...")
    try:
        rand80 = load_decoupled("rand", 80)
    except RuntimeError as e:
        print(f"  WARNING: {e}")
        rand80 = None

    print("Loading joint rand ...")
    try:
        joint = load_joint("rand")
    except RuntimeError as e:
        print(f"  WARNING: {e}")
        joint = None

    print("\nScatter: md30_f95 vs rand_f95 ...")
    rmse_md30 = per_well_rmse(md30)
    rmse_rand95 = per_well_rmse(rand95)
    scatter_comparison(
        rmse_md30, rmse_rand95,
        xlabel="MaxDist-30 f=0.95 — per-well RMSE [m]",
        ylabel="Random f=0.95 — per-well RMSE [m]",
        title="Split comparison: MaxDist-30 vs Random (rand_f95)\nper-well mean RMSE averaged over all horizons & seeds",
        out_path=OUT_DIR / "scatter_split_md30_vs_rand95.png",
        clip=5.0,
    )

    print("Scatter: rand_f90 vs rand_f95 ...")
    rmse_rand90 = per_well_rmse(rand90)
    scatter_comparison(
        rmse_rand95, rmse_rand90,
        xlabel="Random f=0.95 — per-well RMSE [m]",
        ylabel="Random f=0.90 — per-well RMSE [m]",
        title="Split ratio comparison: Random f=0.95 vs f=0.90\nper-well mean RMSE averaged over all horizons & seeds",
        out_path=OUT_DIR / "scatter_split_rand95_vs_rand90.png",
        clip=5.0,
    )

    if joint is not None:
        print("Scatter: decoupled vs joint ...")
        rmse_joint = per_well_rmse(joint)
        scatter_comparison(
            rmse_rand95, rmse_joint,
            xlabel="Decoupled GRU+GP (rand f=0.95) — per-well RMSE [m]",
            ylabel="Joint GRU+GP (rand f=0.95) — per-well RMSE [m]",
            title="Decoupled vs Joint pipeline: per-well mean RMSE\n(rand f=0.95 holdout wells, all seeds)",
            out_path=OUT_DIR / "scatter_decoupled_vs_joint.png",
            clip=5.0,
        )

    print("\nLine plot: pipeline comparison by horizon ...")
    series = {"Decoupled (rand f=0.95)": median_rmse_by_horizon(rand95)}
    if joint is not None:
        series["Joint (rand f=0.95)"] = median_rmse_by_horizon(joint)
    line_plot(
        series,
        ylabel="Median per-well RMSE [m]",
        title="Pipeline comparison by forecast horizon (rand f=0.95 split)",
        out_path=OUT_DIR / "line_rmse_by_horizon_pipeline.png",
    )

    print("Line plot: split type comparison by horizon ...")
    line_plot(
        {
            "MaxDist-30 f=0.95": median_rmse_by_horizon(md30),
            "Random f=0.95":     median_rmse_by_horizon(rand95),
        },
        ylabel="Median per-well RMSE [m]",
        title="Spatial split comparison by forecast horizon",
        out_path=OUT_DIR / "line_rmse_by_horizon_splits.png",
    )

    print("Line plot: split fraction comparison by horizon ...")
    fracs_dict = {
        "Random f=0.95": median_rmse_by_horizon(rand95),
        "Random f=0.90": median_rmse_by_horizon(rand90),
    }
    if rand80 is not None:
        fracs_dict["Random f=0.80"] = median_rmse_by_horizon(rand80)
    line_plot(
        fracs_dict,
        ylabel="Median per-well RMSE [m]",
        title="Train fraction comparison by forecast horizon (random split)",
        out_path=OUT_DIR / "line_rmse_by_horizon_fracs.png",
    )

    print("Line plot: all split types by horizon ...")
    try:
        md10 = load_decoupled("md10", 95)
        md90 = load_decoupled("md90", 95)
        multi_series = {
            "MaxDist-10 f=0.95": median_rmse_by_horizon(md10),
            "MaxDist-30 f=0.95": median_rmse_by_horizon(md30),
            "MaxDist-90 f=0.95": median_rmse_by_horizon(md90),
            "Random f=0.95":     median_rmse_by_horizon(rand95),
        }
        line_plot(
            multi_series,
            ylabel="Median per-well RMSE [m]",
            title="All split strategies by forecast horizon",
            out_path=OUT_DIR / "line_rmse_by_horizon_all_splits.png",
        )
    except RuntimeError as e:
        print(f"  WARNING (all-splits line): {e}")

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
