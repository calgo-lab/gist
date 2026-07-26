import re
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree

REPO      = Path(__file__).resolve().parents[4]
META_PATH = REPO.parent / "data" / "merged.parquet"
DEC_BASE  = REPO / "outputs/main_runs/decoupled"
JNT_BASE  = REPO / "outputs/main_runs/joint"
OUT_DIR   = REPO / "reports/figures/joint_vs_two_stage"

HORIZON = 16

def load_coords():
    import pyarrow.parquet as pq
    t = pq.read_table(META_PATH, columns=["id", "x_25833", "y_25833"])
    return t.to_pandas().drop_duplicates("id").set_index("id")

def rmse_h16_from_pred(pred_path: Path) -> dict:
    df = pd.read_parquet(pred_path, columns=["id", "horizon", "gws_true", "gws_forecast"])
    df = df[df["horizon"] == HORIZON]
    err = df.groupby("id").apply(
        lambda g: np.sqrt(((g["gws_true"] - g["gws_forecast"]) ** 2).mean()),
        include_groups=False,
    )
    return err.to_dict()

def rmse_h16_from_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path, usecols=["id", "horizon", "RMSE"])
    return df[df["horizon"] == HORIZON].set_index("id")["RMSE"].to_dict()

def build_joint_map() -> dict:
    mapping = {}
    for d in sorted(JNT_BASE.iterdir()):
        meta = d / "meta.yaml"
        if not meta.exists():
            continue
        with open(meta) as f:
            m = yaml.safe_load(f)
        sig = m.get("run_sig", "")
        match = re.search(r"_ss(\d+)$", sig)
        if match:
            mapping[int(match.group(1))] = d
    return mapping

def collect_all(coords: pd.DataFrame) -> pd.DataFrame:
    joint_map = build_joint_map()
    rows = []

    for ss, jnt_dir in sorted(joint_map.items()):
        split_df = pd.read_csv(jnt_dir / "split_info.csv")
        train_ids = set(split_df.loc[split_df["spatial_split"] == "spatial_train", "id"])
        test_ids  = set(split_df.loc[split_df["spatial_split"] == "spatial_test",  "id"])

        train_coords = coords.loc[coords.index.isin(train_ids), ["x_25833", "y_25833"]].dropna()
        test_coords  = coords.loc[coords.index.isin(test_ids),  ["x_25833", "y_25833"]].dropna()
        if len(train_coords) == 0 or len(test_coords) == 0:
            continue

        tree = cKDTree(train_coords.values)
        dists_m, _ = tree.query(test_coords.values, k=1)
        ntn = {wid: dist / 1000.0 for wid, dist in zip(test_coords.index, dists_m)}

        dec_pattern = f"*_coloc_rand90_f90_ss{ss}__dec_prod_hr_gps7__predobstrain"
        dec_matches = list(DEC_BASE.glob(dec_pattern))
        if not dec_matches:
            continue
        dec_dir = dec_matches[0]
        horizon_csv = dec_dir / "gp_metrics_by_id_horizon.csv"
        if horizon_csv.exists():
            dec_rmse = rmse_h16_from_csv(horizon_csv)
        else:
            pred_path = dec_dir / "gp_pred.parquet"
            if not pred_path.exists():
                continue
            dec_rmse = rmse_h16_from_pred(pred_path)

        jnt_pred = jnt_dir / "eval" / "test" / "gp_pred.parquet"
        if not jnt_pred.exists():
            continue
        jnt_rmse = rmse_h16_from_pred(jnt_pred)

        for wid in test_ids:
            if wid not in ntn or wid not in dec_rmse or wid not in jnt_rmse:
                continue
            rows.append({"ntn_km": ntn[wid], "rmse_ts": dec_rmse[wid], "rmse_jt": jnt_rmse[wid]})

        if ss % 10 == 2:
            print(f"  processed ss={ss} … {len(rows)} records so far")

    return pd.DataFrame(rows)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading coordinates …")
    coords = load_coords()

    print("Collecting all seeds (h=16) …")
    df = collect_all(coords)
    print(f"  {len(df)} (well, seed) pairs total")

    df["decile"] = pd.qcut(df["ntn_km"], q=10, labels=range(1, 11))

    binned = (
        df.groupby("decile", observed=True)
        .agg(
            rmse_ts = ("rmse_ts", "median"),
            rmse_jt = ("rmse_jt", "median"),
            ntn_med = ("ntn_km",  "median"),
            ntn_max = ("ntn_km",  "max"),
            n       = ("ntn_km",  "count"),
        )
        .reset_index()
    )

    print(f"\n{'Decile':>7}  {'n':>6}  {'NTN med':>8}  {'NTN max':>8}  {'Two-stage':>10}  {'Joint':>8}  {'Gap':>8}  {'Rel%':>8}")
    print("-" * 80)
    for _, row in binned.iterrows():
        gap = row.rmse_jt - row.rmse_ts
        rel = gap / row.rmse_ts * 100
        print(f"{int(row.decile):>7}  {int(row.n):>6}  {row.ntn_med:8.2f}  {row.ntn_max:8.2f}  "
              f"{row.rmse_ts:10.4f}  {row.rmse_jt:8.4f}  {gap:+8.4f}  {rel:+7.1f}%")

    p10_row = binned[binned["decile"] == 1].iloc[0]
    p90_row = binned[binned["decile"] == 10].iloc[0]
    print(f"\nDecile 1  (closest 10%,  all within {p10_row.ntn_max:.2f} km):")
    print(f"  Two-stage {p10_row.rmse_ts:.3f} m  Joint {p10_row.rmse_jt:.3f} m  "
          f"gap {p10_row.rmse_ts - p10_row.rmse_jt:+.3f} m  (two-stage better)")
    print(f"\nDecile 10 (farthest 10%, all beyond {binned[binned['decile']==9].iloc[0].ntn_max:.2f} km):")
    print(f"  Two-stage {p90_row.rmse_ts:.3f} m  Joint {p90_row.rmse_jt:.3f} m  "
          f"gap {p90_row.rmse_jt - p90_row.rmse_ts:+.3f} m  (joint better)")

    x = binned["decile"].astype(int).values
    x2 = binned["ntn_med"].values

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax2 = ax.twiny()

    ax.plot(x, binned["rmse_ts"], color="#d87f3a", lw=1.8, marker="o", ms=5, label="Two-stage")
    ax.plot(x, binned["rmse_jt"], color="#4c72b0", lw=1.8, marker="o", ms=5, label="Jointly trained")

    ax.set_xlabel("NTN-distance decile  (1 = closest to training)", fontsize=11)
    ax.set_ylabel("Median per-well RMSE (m)", fontsize=11)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, None)
    ax.grid(axis="both", ls=":", lw=0.7, alpha=0.7)
    ax.legend(fontsize=10, framealpha=0.9)

    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{v:.1f}" for v in x2], fontsize=8)
    ax2.set_xlabel("Median NTN distance (km)", fontsize=10)

    ax.set_title("RMSE by distance to nearest training well\n(decile 1 = closest)", fontsize=11)

    plt.tight_layout()

    out_png = OUT_DIR / "ntn_decile_rmse_closest.png"
    out_pdf = OUT_DIR / "ntn_decile_rmse_closest.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=180, bbox_inches="tight")
    print(f"\nSaved:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()
