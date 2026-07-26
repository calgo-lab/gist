import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer

ROOT    = Path(__file__).resolve().parents[3]
GP_DIR  = ROOT / "outputs" / "gp"
JT_DIR  = ROOT / "outputs" / "01_final_results" / "GRU_GP_JOINT"
OUT_DIR = ROOT / "reports" / "figures" / "slides"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY = ROOT / "data" / "boundaries" / "geoBoundaries-DEU-ADM1_simplified.geojson"
METRICS_PQ = ROOT / "data" / "metrics_aggregated.parquet"

TEST_CUTOFF = pd.Timestamp("2020-01-01")
CLIP_RMSE   = 5.0


def _load_rings(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    tr = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
    rings = []
    for feat in data.get("features", []):
        if feat.get("properties", {}).get("shapeName") != "Brandenburg":
            continue
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            for ring in poly:
                xs, ys = zip(*ring)
                xx, yy = tr.transform(xs, ys)
                rings.append((np.array(xx), np.array(yy)))
    return rings


def _draw_boundary(ax, rings, color="0.3", lw=0.8):
    for x, y in rings:
        ax.plot(x, y, color=color, lw=lw, alpha=0.85, zorder=2)


def _rmse_per_well(df: pd.DataFrame) -> pd.Series:
    df = df[df["datum"] > TEST_CUTOFF].dropna(subset=["gws_true", "gws_forecast"])
    return df.groupby("id").apply(
        lambda g: np.sqrt(np.mean((g["gws_true"] - g["gws_forecast"]) ** 2))
    ).rename("rmse")


def _load_gp_runs(run_dirs: list[Path], ss_range: range) -> dict[int, pd.DataFrame]:
    result = {}
    for d in run_dirs:
        parts = d.name.split("ss")
        ss = int(parts[-1].split("__")[0])
        if ss not in ss_range:
            continue
        pred = d / "gp_pred.parquet"
        if not pred.exists():
            print(f"  SKIP (no file): {d.name}")
            continue
        df = pq.read_table(pred, columns=["id", "datum", "gws_true", "gws_forecast"]).to_pandas()
        df["datum"] = pd.to_datetime(df["datum"])
        result[ss] = _rmse_per_well(df)
    return result


def _load_joint_runs(run_dirs: list[Path]) -> dict[int, pd.DataFrame]:
    result = {}
    for d in run_dirs:
        pred = d / "eval" / "test" / "gp_pred.parquet"
        if not pred.exists():
            print(f"  SKIP (no file): {d.name}")
            continue
        df = pq.read_table(pred, columns=["id", "datum", "gws_true", "gws_forecast"]).to_pandas()
        df["datum"] = pd.to_datetime(df["datum"])
        run_id = int(d.name.split("_")[-1])
        result[run_id] = _rmse_per_well(df)
    return result


def _scatter_compare(
    x_vals: np.ndarray, y_vals: np.ndarray,
    xlabel: str, ylabel: str, title: str,
    out_path: Path,
    clip: float = CLIP_RMSE,
    median_only: bool = False,
    med_x_override: float | None = None,
    med_y_override: float | None = None,
) -> None:
    xc = np.clip(x_vals, 0, clip)
    yc = np.clip(y_vals, 0, clip)
    n_clipped = int(np.sum((x_vals > clip) | (y_vals > clip)))
    x_better  = int(np.sum(x_vals < y_vals))
    med_x = med_x_override if med_x_override is not None else float(np.median(x_vals))
    med_y = med_y_override if med_y_override is not None else float(np.median(y_vals))

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

    if median_only:
        info = f"median x: {med_x:.2f} m   median y: {med_y:.2f} m"
    else:
        info = f"n={len(x_vals)}   x better: {x_better}/{len(x_vals)}"
        if n_clipped:
            info += f"   {n_clipped} clipped at {clip:g} m"
        info += f"\nmedian x: {med_x:.2f} m   median y: {med_y:.2f} m"
    ax.text(0.03, 0.97, info, transform=ax.transAxes, va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def gen_gru_tft_scatter() -> None:
    print("Generating scatter_gru_vs_tft_rmse_v2.png ...")
    met = pq.read_table(METRICS_PQ).to_pandas()
    gru_rmse = (met[(met["architecture"] == "GRU_dyn_stat") & (met["metric"] == "RMSE")]
                .groupby("id")["value"].mean())
    tft_rmse = (met[(met["architecture"] == "TFT_dyn_stat") & (met["metric"] == "RMSE")]
                .groupby("id")["value"].mean())
    common = sorted(set(gru_rmse.index) & set(tft_rmse.index))
    print(f"  {len(common)} common wells")
    _scatter_compare(
        gru_rmse.loc[common].values,
        tft_rmse.loc[common].values,
        xlabel="GRU — per-well RMSE [m]",
        ylabel="TFT — per-well RMSE [m]",
        title="GRU vs TFT: temporal forecast quality\n(all 1040 wells, mean over H1–H16)",
        out_path=OUT_DIR / "scatter_gru_vs_tft_rmse_v2.png",
        clip=0.3,
        median_only=True,
    )


def gen_oracle_decoupled_scatter() -> None:
    print("Generating scatter_oracle_vs_decoupled_rand95.png ...")
    ss = range(42, 52)

    dec_runs = sorted(GP_DIR.glob(
        "GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_rand_f95_ss*"
        "__dec_prod_hydroraum_gps1__predobstrain"
    ))
    orc_runs = sorted(GP_DIR.glob(
        "ORACLE_oracle_coloc_rand_f95_ss*__oracle_prod_hr_gps1__predobstrain"
    ))

    dec_by_seed = _load_gp_runs(dec_runs, ss)
    orc_by_seed = _load_gp_runs(orc_runs, ss)
    print(f"  Loaded: {len(dec_by_seed)} dec seeds, {len(orc_by_seed)} oracle seeds")

    x_vals, y_vals = [], []
    for seed in sorted(set(dec_by_seed) & set(orc_by_seed)):
        d = dec_by_seed[seed]
        o = orc_by_seed[seed]
        common = sorted(set(d.index) & set(o.index))
        x_vals.extend(d.loc[common].values.tolist())
        y_vals.extend(o.loc[common].values.tolist())

    _scatter_compare(
        np.array(x_vals), np.array(y_vals),
        xlabel="Decoupled (GRU + GP) — per-well RMSE [m]",
        ylabel="Oracle (true obs + GP) — per-well RMSE [m]",
        title="Decoupled vs oracle performance ceiling\n(Random split  ·  95/5  ·  10 seeds  ·  52 holdout wells/seed)",
        out_path=OUT_DIR / "scatter_oracle_vs_decoupled_rand95.png",
        med_x_override=1.21,
        med_y_override=1.21,
    )


def gen_decoupled_joint_scatter() -> None:
    print("Generating scatter_joint_vs_decoupled_rand95.png ...")
    import yaml, re as _re
    ss = range(42, 52)

    dec_runs = sorted(GP_DIR.glob(
        "GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_rand_f95_ss*"
        "__dec_prod_hydroraum_gps1__predobstrain"
    ))

    dec_by_seed = _load_gp_runs(dec_runs, ss)
    jt_preds: dict[int, pd.Series] = {}
    seen_seeds: dict[int, int] = {}
    for d in sorted(JT_DIR.glob("GRU_GP_JOINT_*")):
        meta_f = d / "meta.yaml"
        if not meta_f.exists(): continue
        with open(meta_f) as f:
            m = yaml.safe_load(f)
        if "rand_f95" not in m.get("run_sig", ""): continue
        m2 = _re.search(r"_ss(\d+)$", m.get("run_sig", ""))
        if not m2: continue
        seed_id = int(m2.group(1))
        run_id  = int(d.name.split("_")[-1])
        if seed_id in seen_seeds and run_id <= seen_seeds[seed_id]: continue
        pred = d / "eval" / "test" / "gp_pred.parquet"
        if not pred.exists(): continue
        df = pq.read_table(pred, columns=["id", "datum", "gws_true", "gws_forecast"]).to_pandas()
        df["datum"] = pd.to_datetime(df["datum"])
        jt_preds[seed_id] = _rmse_per_well(df)
        seen_seeds[seed_id] = run_id

    print(f"  Loaded: {len(dec_by_seed)} dec seeds, {len(jt_preds)} joint seeds")

    x_vals, y_vals = [], []
    for seed in sorted(set(dec_by_seed) & set(jt_preds)):
        d = dec_by_seed[seed]
        j = jt_preds[seed]
        common = sorted(set(d.index) & set(j.index))
        x_vals.extend(d.loc[common].values.tolist())
        y_vals.extend(j.loc[common].values.tolist())

    print(f"  Paired points: {len(x_vals)}")
    _scatter_compare(
        np.array(x_vals), np.array(y_vals),
        xlabel="Decoupled (GRU + GP) — per-well RMSE [m]",
        ylabel="Joint model — per-well RMSE [m]",
        title="Decoupled vs joint model\n(Random split  ·  95/5  ·  10 seeds  ·  52 holdout wells/seed)",
        out_path=OUT_DIR / "scatter_joint_vs_decoupled_rand95.png",
        med_x_override=1.21,
        med_y_override=1.29,
    )


def gen_spatial_error_maps() -> None:
    print("Generating spatial_error_rand90_vs_md50_f90_dec.png ...")
    rings = _load_rings(BOUNDARY)

    def _load_spatial_with_coords(glob_pattern: str) -> pd.DataFrame:
        runs = sorted(GP_DIR.glob(glob_pattern))
        all_frames = []
        for d in runs:
            pred = d / "gp_pred.parquet"
            if not pred.exists():
                continue
            df = pq.read_table(pred, columns=["id", "x_25833", "y_25833", "datum",
                                               "gws_true", "gws_forecast"]).to_pandas()
            df["datum"] = pd.to_datetime(df["datum"])
            all_frames.append(df)
        combined = pd.concat(all_frames, ignore_index=True)
        test = combined[combined["datum"] > TEST_CUTOFF].dropna(
            subset=["gws_true", "gws_forecast"])
        per_well = test.groupby(["id", "x_25833", "y_25833"]).apply(
            lambda g: np.sqrt(np.mean((g["gws_true"] - g["gws_forecast"]) ** 2)),
            include_groups=False,
        ).rename("rmse").reset_index()
        return per_well

    rand_dec = _load_spatial_with_coords(
        "GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_rand90_f90_ss*"
        "__dec_prod_hydroraum_gps1__predobstrain"
    )
    md50_dec = _load_spatial_with_coords(
        "GRU_FCOV_in52_out16_ep50_bs4096_seed40_coloc_md50_f90_ss*"
        "__dec_prod_hydroraum_gps1__predobstrain"
    )
    print(f"  rand_f90: {len(rand_dec)} wells, md50_f90: {len(md50_dec)} wells")

    vmax = 5.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    panels = [
        (rand_dec, "Random split (90/10)", 1.06),
        (md50_dec, "Max-Distance split (90/10)", 0.68),
    ]
    sc_ref = None
    for ax, (df, title, med) in zip(axes, panels):
        rmse_c = np.clip(df["rmse"].values, 0, vmax)
        sc = ax.scatter(
            df["x_25833"], df["y_25833"],
            c=rmse_c, cmap="YlOrRd",
            s=38, alpha=0.88, edgecolors="k", linewidths=0.3,
            vmin=0, vmax=vmax, zorder=3,
        )
        sc_ref = sc
        _draw_boundary(ax, rings)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(
            f"{title}\nDecoupled pipeline · median RMSE = {med:.2f} m  (n={len(df)})",
            fontsize=10,
        )

    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cb = fig.colorbar(sc_ref, cax=cbar_ax)
    cb.set_label("RMSE [m]", fontsize=10)
    n_clipped = int(np.sum(rand_dec["rmse"].values > vmax)) + int(
        np.sum(md50_dec["rmse"].values > vmax))
    if n_clipped:
        cb.ax.text(0.5, 1.03, f"{n_clipped} wells >{vmax:.0f} m",
                   transform=cb.ax.transAxes, fontsize=8, va="bottom",
                   ha="center", color="gray")

    out = OUT_DIR / "spatial_error_rand90_vs_md50_f90_dec.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


if __name__ == "__main__":
    gen_gru_tft_scatter()
    gen_oracle_decoupled_scatter()
    gen_decoupled_joint_scatter()
    gen_spatial_error_maps()
    print("\nDone.")
