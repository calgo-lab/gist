from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
for p in [ROOT, *ROOT.parents]:
    if (p / "splits").is_dir():
        ROOT = p
        break

SPLITS_DIR = ROOT / "splits"
SEEDS      = list(range(42, 52))
N_VAL      = 52


def _source_2way(split_type: str, ss: int) -> Path:
    """Return the canonical 2-way split file for a given type and seed."""
    if split_type == "rand":
        p = SPLITS_DIR / f"spatial_split_full_merged_coloc_rand_f90_ss{ss}.csv"
        if not p.exists():
            p = SPLITS_DIR / f"spatial_split_full_merged_coloc_rand90_f90_ss{ss}.csv"
    elif split_type == "md50":
        p = SPLITS_DIR / f"spatial_split_full_merged_coloc_md50_f90_ss{ss}.csv"
    elif split_type == "km":
        p = SPLITS_DIR / f"spatial_split_full_merged_coloc_km_f90_ss{ss}.csv"
    else:
        raise ValueError(f"Unknown split_type: {split_type}")
    if not p.exists():
        raise FileNotFoundError(f"2-way source not found: {p}")
    return p


def fix_km(ss: int, dry_run: bool = False) -> dict:
    """km_f90: relabel spatial_holdout → spatial_test in the joint file."""
    joint_path = SPLITS_DIR / f"spatial_split_full_merged_coloc_km_f90_joint_ss{ss}.csv"
    two_way_path = _source_2way("km", ss)

    df_joint = pd.read_csv(joint_path)
    df2 = pd.read_csv(two_way_path)

    holdout_2way = set(df2.loc[df2["spatial_split"] == "spatial_holdout", "id"])
    holdout_joint = set(df_joint.loc[df_joint["spatial_split"] == "spatial_holdout", "id"])

    overlap = len(holdout_2way & holdout_joint)
    assert overlap == len(holdout_2way) == len(holdout_joint), (
        f"km_ss{ss}: holdout mismatch! 2way={len(holdout_2way)}, "
        f"joint={len(holdout_joint)}, overlap={overlap}"
    )

    new_df = df_joint.copy()
    new_df.loc[new_df["spatial_split"] == "spatial_holdout", "spatial_split"] = "spatial_test"

    out_path = SPLITS_DIR / f"spatial_split_full_merged_coloc_km_f90_joint_ss{ss}_v2.csv"
    if not dry_run:
        new_df.to_csv(out_path, index=False)

    counts = new_df["spatial_split"].value_counts().to_dict()
    return {
        "seed": ss, "type": "km", "out": str(out_path),
        "counts": counts, "test_matches_2way": overlap,
        "n_2way_holdout": len(holdout_2way),
    }


def fix_rand_md50(split_type: str, ss: int, dry_run: bool = False) -> dict:
    """rand_f90 / md50_f90: replace holdout wells with 2-way holdout, pick fresh val."""
    two_way_path = _source_2way(split_type, ss)
    df2 = pd.read_csv(two_way_path)

    holdout_ids = set(df2.loc[df2["spatial_split"] == "spatial_holdout", "id"])
    train_ids   = sorted(df2.loc[df2["spatial_split"] == "spatial_train", "id"].tolist())

    rng     = np.random.default_rng(ss)
    val_ids = set(rng.choice(train_ids, size=N_VAL, replace=False).tolist())

    assert val_ids.isdisjoint(holdout_ids), f"{split_type}_ss{ss}: val ∩ holdout non-empty!"

    new_df = pd.DataFrame({"id": df2["id"].tolist()})
    new_df["spatial_split"] = "spatial_train"
    new_df.loc[new_df["id"].isin(holdout_ids), "spatial_split"] = "spatial_test"
    new_df.loc[new_df["id"].isin(val_ids),     "spatial_split"] = "spatial_val"

    n_train = (new_df["spatial_split"] == "spatial_train").sum()
    n_val   = (new_df["spatial_split"] == "spatial_val").sum()
    n_test  = (new_df["spatial_split"] == "spatial_test").sum()
    assert n_test  == len(holdout_ids), f"{split_type}_ss{ss}: test count mismatch"
    assert n_val   == N_VAL,            f"{split_type}_ss{ss}: val count mismatch"
    assert n_train == len(train_ids) - N_VAL, f"{split_type}_ss{ss}: train count mismatch"

    label  = "rand" if split_type == "rand" else "md50"
    out_path = SPLITS_DIR / (
        f"spatial_split_full_merged_coloc_{label}_f90_joint_ss{ss}_v2.csv"
    )
    if not dry_run:
        new_df.to_csv(out_path, index=False)

    return {
        "seed": ss, "type": split_type, "out": str(out_path),
        "counts": {"spatial_train": n_train, "spatial_val": n_val, "spatial_test": n_test},
        "test_matches_2way": n_test,
        "n_2way_holdout": len(holdout_ids),
    }


def verify_v2_vs_original(split_type: str, ss: int) -> dict:
    """Compare the _v2 file's spatial_test against the 2-way spatial_holdout."""
    label = split_type
    v2_path = SPLITS_DIR / f"spatial_split_full_merged_coloc_{label}_f90_joint_ss{ss}_v2.csv"
    two_way_path = _source_2way(split_type, ss)

    if not v2_path.exists():
        return {"seed": ss, "type": split_type, "status": "MISSING _v2 file"}

    dfv2 = pd.read_csv(v2_path)
    df2  = pd.read_csv(two_way_path)

    test_v2     = set(dfv2.loc[dfv2["spatial_split"] == "spatial_test",     "id"])
    holdout_2   = set(df2.loc[df2["spatial_split"]   == "spatial_holdout",  "id"])
    val_v2      = set(dfv2.loc[dfv2["spatial_split"] == "spatial_val",       "id"])
    train_2     = set(df2.loc[df2["spatial_split"]   == "spatial_train",     "id"])

    overlap_test = len(test_v2 & holdout_2)
    val_from_train = len(val_v2 & train_2)
    val_from_holdout = len(val_v2 & holdout_2)

    ok = (overlap_test == len(holdout_2) == len(test_v2)
          and val_from_holdout == 0
          and len(val_v2) == N_VAL)

    return {
        "seed": ss, "type": split_type,
        "status": "OK" if ok else "FAIL",
        "test_v2": len(test_v2), "holdout_2way": len(holdout_2),
        "overlap_test": overlap_test,
        "val_from_train": val_from_train, "val_from_holdout": val_from_holdout,
        "counts_v2": dfv2["spatial_split"].value_counts().to_dict(),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Don't write files, just report")
    p.add_argument("--verify-only", action="store_true", help="Only verify existing _v2 files")
    args = p.parse_args()

    results = []

    if not args.verify_only:
        print("=== Generating corrected _v2 joint split files ===\n")
        for ss in SEEDS:
            r = fix_km(ss, dry_run=args.dry_run)
            results.append(r)
            tag = "[dry]" if args.dry_run else "wrote"
            print(f"  km   ss{ss}: {r['counts']}  test_matches_2way={r['test_matches_2way']}  → {tag}")
        print()
        for split_type in ["rand", "md50"]:
            for ss in SEEDS:
                r = fix_rand_md50(split_type, ss, dry_run=args.dry_run)
                results.append(r)
                tag = "[dry]" if args.dry_run else "wrote"
                print(f"  {split_type:4s} ss{ss}: {r['counts']}  → {tag}")
        print()

    print("=== Verification (spatial_test == 2-way spatial_holdout) ===\n")
    all_ok = True
    for split_type in ["km", "rand", "md50"]:
        for ss in SEEDS:
            v = verify_v2_vs_original(split_type, ss)
            status = v.get("status", "?")
            if status != "OK":
                all_ok = False
            overlap = v.get("overlap_test", "?")
            n_test  = v.get("test_v2", "?")
            n_2way  = v.get("holdout_2way", "?")
            vfh     = v.get("val_from_holdout", "?")
            print(
                f"  {split_type:4s} ss{ss}  {status}  "
                f"test={n_test}  2way_holdout={n_2way}  overlap={overlap}  "
                f"val_from_holdout={vfh}"
            )

    print()
    if all_ok:
        print("All checks passed.")
        print()
        print("To replace originals, run (from the splits directory):")
        print("  for f in *_v2.csv; do mv \"$f\" \"${f/_v2/}\"; done")
    else:
        print("Some checks FAILED — do not rename until fixed.")


if __name__ == "__main__":
    main()
