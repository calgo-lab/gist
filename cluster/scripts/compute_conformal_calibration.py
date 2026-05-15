import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT       = Path("/storage/gwl-interpolation")
GP_OUT_DIR = ROOT / "outputs" / "gp"
GP_TAG     = "dec_prod_hydroraum_gps1"
MODEL_SEED = 40
SPLIT_SEEDS = list(range(42, 52))
ALPHA_LEVEL = 0.95    # target coverage
Z_NAIVE     = 1.96

SPLITS = {
    "rand_f95":  {"label_tpl": "coloc_rand_f95_ss{ss}",  "n_holdout": 52},
    "rand90_f90": {"label_tpl": "coloc_rand90_f90_ss{ss}", "n_holdout": 104},
    "md50_f95":  {"label_tpl": "coloc_md50_f95_ss{ss}",  "n_holdout": 52},
}


def pred_path(split_label: str) -> Path:
    run_sig = f"in52_out16_ep50_bs4096_seed{MODEL_SEED}_{split_label}"
    dir_name = f"GRU_FCOV_{run_sig}__{GP_TAG}__predobstrain"
    return GP_OUT_DIR / dir_name / "gp_pred.parquet"


def loo_conformal(df: pd.DataFrame) -> dict:
    """
    LOO conformal calibration over a set of test wells.

    df must have columns: id, gws_true, gws_forecast, gws_forecast_std
    Returns a dict of scalar metrics.
    """
    well_ids = df["id"].unique()
    n_wells  = len(well_ids)

    q_per_well        = []
    coverage_per_well = []
    pi_width_naive    = []
    pi_width_calib    = []

    for i, wid in enumerate(well_ids):
        mask_test = df["id"] == wid
        mask_cal  = ~mask_test

        cal   = df[mask_cal]
        test  = df[mask_test]

        scores_cal = (np.abs(cal["gws_true"].to_numpy() - cal["gws_forecast"].to_numpy())
                      / cal["gws_forecast_std"].to_numpy())

        n_cal = len(scores_cal)
        # finite-sample conformal quantile
        q_idx = math.ceil((n_cal + 1) * ALPHA_LEVEL) - 1   # 0-based
        q_idx = min(q_idx, n_cal - 1)
        scores_sorted = np.sort(scores_cal)
        q_i = scores_sorted[q_idx]

        test_std   = test["gws_forecast_std"].to_numpy()
        test_score = (np.abs(test["gws_true"].to_numpy() - test["gws_forecast"].to_numpy())
                      / test_std)

        coverage_i = float((test_score <= q_i).mean())

        q_per_well.append(q_i)
        coverage_per_well.append(coverage_i)
        pi_width_naive.extend((2 * Z_NAIVE   * test_std).tolist())
        pi_width_calib.extend((2 * q_i       * test_std).tolist())

    scores_all  = (np.abs(df["gws_true"].to_numpy() - df["gws_forecast"].to_numpy())
                   / df["gws_forecast_std"].to_numpy())
    picp_naive  = float((scores_all <= Z_NAIVE).mean())

    return {
        "alpha_median":      float(np.median(q_per_well)),
        "alpha_std":         float(np.std(q_per_well)),
        "alpha_min":         float(np.min(q_per_well)),
        "alpha_max":         float(np.max(q_per_well)),
        "picp_calibrated":   float(np.mean(coverage_per_well)),
        "picp_naive":        picp_naive,
        "pi_width_naive_med":  float(np.median(pi_width_naive)),
        "pi_width_calib_med":  float(np.median(pi_width_calib)),
        "n_wells":           n_wells,
        "n_obs":             len(df),
    }


def main() -> None:
    for split_name, cfg in SPLITS.items():
        print(f"\n{'='*60}")
        print(f"Split: {split_name}  ({cfg['n_holdout']} holdout wells)")
        print(f"{'='*60}")

        seed_results = []
        missing = []

        for ss in SPLIT_SEEDS:
            label = cfg["label_tpl"].format(ss=ss)
            p     = pred_path(label)
            if not p.exists():
                missing.append(ss)
                continue
            df = pd.read_parquet(p, columns=["id", "gws_true", "gws_forecast", "gws_forecast_std"])
            res = loo_conformal(df)
            res["ss"] = ss
            seed_results.append(res)

        if missing:
            print(f"  WARNING: missing seeds {missing}")
        if not seed_results:
            print("  No data — skipping.")
            continue

        print(f"  Seeds with data: {[r['ss'] for r in seed_results]}")

        alpha_vals = [r["alpha_median"] for r in seed_results]
        picp_c     = [r["picp_calibrated"] for r in seed_results]
        picp_n     = [r["picp_naive"] for r in seed_results]
        pw_n       = [r["pi_width_naive_med"] for r in seed_results]
        pw_c       = [r["pi_width_calib_med"] for r in seed_results]

        print()
        print(f"  Calibration factor α (median LOO, per seed):")
        for r in seed_results:
            print(f"    ss{r['ss']:02d}: α={r['alpha_median']:.3f}  "
                  f"(range {r['alpha_min']:.3f}–{r['alpha_max']:.3f}  "
                  f"std_within={r['alpha_std']:.3f})")
        print()
        print(f"  {'Metric':<35} {'Median over seeds':>18} {'Std':>8}")
        print(f"  {'-'*63}")
        rows = [
            ("α (median LOO)",              alpha_vals),
            ("Calibrated PICP @95%",        [v * 100 for v in picp_c]),
            ("Naive PICP (z=1.96)",         [v * 100 for v in picp_n]),
            ("Naive PI width (median, m)",  pw_n),
            ("Calibrated PI width (med, m)", pw_c),
        ]
        for label, vals in rows:
            med = np.median(vals)
            std = np.std(vals)
            unit = "%" if "PICP" in label else ""
            print(f"  {label:<35} {med:>17.3f}{unit} {std:>7.3f}")


if __name__ == "__main__":
    main()
