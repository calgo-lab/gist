import argparse
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
SPLITS = ROOT / "splits"

SOURCE = SPLITS / "spatial_split_full_merged_coloc_rand90_f90_ss42.csv"
TRAIN_FRACTION = 0.9
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def make_split(train_wells, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_wells))
    n_train = round(len(train_wells) * TRAIN_FRACTION)
    n_holdout = len(train_wells) - n_train
    arr = np.array(train_wells)
    return pd.DataFrame({
        "id": arr[idx[:n_train]].tolist() + arr[idx[n_train:]].tolist(),
        "spatial_split": ["spatial_train"] * n_train + ["spatial_holdout"] * n_holdout,
    }).sort_values("id").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds (default: 42-46)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS

    source = pd.read_csv(SOURCE)
    train_wells = source.loc[source["spatial_split"] == "spatial_train", "id"].tolist()
    print(f"Production training wells: {len(train_wells)}")

    for seed in seeds:
        split = make_split(train_wells, seed)
        out = SPLITS / f"spatial_split_full_merged_coloc_rand90_f90_ss42_hpo90_rs{seed}.csv"
        split.to_csv(out, index=False)
        counts = split["spatial_split"].value_counts()
        print(f"  seed={seed}: {counts['spatial_train']} train / {counts['spatial_holdout']} holdout → {out.name}")


if __name__ == "__main__":
    main()
