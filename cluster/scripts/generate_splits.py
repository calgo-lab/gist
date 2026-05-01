
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from libs.spatial_split import (
    spatial_split_random,
    spatial_split_kmeans,
    spatial_split_random_max_dist,
)

DATA_FILE       = ROOT.parent / "data" / "merged.parquet"
SPLITS_DIR      = ROOT / "splits"
DEDUP_IDS_FILE  = SPLITS_DIR / "dedup_well_ids.csv"
DATASET         = "full_merged_dedup"

STATIC_REGEX = (
    "eumohp_(.+)_(.+)_(.*[1])"
    "|shannongeom10kmsha"
    "|entgeom10kment"
    "|unigeom10kmuni"
    "|gwn"
    "|huek250.+_(kf).+"
    "|corine"
    "|^TWI_dgm50_r1000m$"
    "|^gok$"
    "|^parde_seasonality$"
    "|^GW_recharge_r1000m$"
    "|^siwa_verweilzeit_j$"
    "|^gw_gespannt_bin$"
    "|^hydroraum_Entlastungsgebiete$"
    "|^hydroraum_Transitgebiete$"
    "|^hydroraum_Speisungsgebiete$"
)
EXCLUDE_TERMS = ["geometry", "x_25833", "y_25833"]

RANDOM_FRAC_SEEDS = {
    0.95: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    0.90: [42, 43, 44, 45, 46],
    0.80: [42, 43, 44, 45, 46],
}

KM_FRAC  = 0.95
KM_SEEDS = [42, 43, 44, 45, 46]

MD_FRAC      = 0.95
MD_PCT_SEEDS = {
    10:  [42],
    50:  [42, 43, 44, 45, 46],
    90:  [42, 43, 44, 45, 46],
    100: [42, 43, 44, 45, 46],
}


def _fs(f: float) -> str:
    return str(f).replace(".", "p")


def _skip_or_make(path: Path, fn, *args, **kwargs):
    if path.exists():
        print(f"  skip  {path.name}")
        return
    fn(*args, **kwargs, save_path=path)
    print(f"  wrote {path.name}")


def main():
    if not DEDUP_IDS_FILE.exists():
        raise FileNotFoundError(
            f"{DEDUP_IDS_FILE} not found — run preprocess_dedup.py first."
        )

    SPLITS_DIR.mkdir(exist_ok=True)
    kept_ids = pd.read_csv(DEDUP_IDS_FILE)["id"].tolist()
    print(f"Loading {DATA_FILE} ...")
    gws_full = pd.read_parquet(DATA_FILE)
    gws = gws_full[gws_full["id"].isin(kept_ids)]
    print(f"  {gws['id'].nunique()} wells after dedup (from {gws_full['id'].nunique()} total)")

    print("\n--- random ---")
    for frac, seeds in RANDOM_FRAC_SEEDS.items():
        for seed in seeds:
            path = SPLITS_DIR / f"spatial_split_{DATASET}_random_f{_fs(frac)}_ss{seed}.csv"
            _skip_or_make(path, spatial_split_random, gws, train_fraction=frac, rng_seed=seed)

    print("\n--- kmeans ---")
    for seed in KM_SEEDS:
        path = SPLITS_DIR / f"spatial_split_{DATASET}_kmeans_f{_fs(KM_FRAC)}_sc10_ss{seed}.csv"
        _skip_or_make(
            path, spatial_split_kmeans, gws,
            static_regex=STATIC_REGEX, train_fraction=KM_FRAC,
            n_clusters=10, rng_seed=seed, exclude_terms=EXCLUDE_TERMS,
        )

    print("\n--- max_dist ---")
    for pct, seeds in MD_PCT_SEEDS.items():
        for seed in seeds:
            path = SPLITS_DIR / f"spatial_split_{DATASET}_max_dist_f{_fs(MD_FRAC)}_k3_p{pct}_ss{seed}.csv"
            _skip_or_make(
                path, spatial_split_random_max_dist, gws,
                train_fraction=MD_FRAC, k=3, d_percentile=pct, rng_seed=seed,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
