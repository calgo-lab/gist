"""
prepare_90_10_split.py

Creates a spatial split CSV where the joint training's spatial_val wells
are promoted to spatial_train, giving a 90/10 train/test split that exactly
matches what the joint model sees as train+val vs test.

Run once (locally or on cluster):
    python src/scripts/prepare_90_10_split.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
for p in [ROOT, *ROOT.parents]:
    if (p / 'configs' / 'data.yaml').exists():
        ROOT = p
        break

IN_SPLIT  = ROOT / 'splits' / 'spatial_split_full_merged_spf0p8_sc20_ss42.csv'
OUT_SPLIT = ROOT / 'splits' / 'spatial_split_full_merged_spf0p8_sc20_ss42_90_10.csv'

VAL_FRACTION = 0.5   # same as joint training's spatial_val_fraction
RNG_SEED     = 42    # same as spatial split_seed


def _make_three_way_split(split_df, val_fraction, rng_seed):
    rng = np.random.default_rng(rng_seed)
    split_df = split_df.copy()
    holdout = split_df[split_df['spatial_split'] == 'spatial_holdout']
    val_ids = set()
    for _, group in holdout.groupby('cluster'):
        ids = group['id'].to_numpy()
        n_val = max(1, int(round(len(ids) * val_fraction)))
        chosen = rng.choice(ids, size=n_val, replace=False)
        val_ids.update(chosen.tolist())
    split_df.loc[
        split_df['id'].isin(val_ids) & (split_df['spatial_split'] == 'spatial_holdout'),
        'spatial_split'
    ] = 'spatial_val'
    split_df.loc[split_df['spatial_split'] == 'spatial_holdout', 'spatial_split'] = 'spatial_test'
    return split_df


split_df = pd.read_csv(IN_SPLIT)
three_way = _make_three_way_split(split_df, VAL_FRACTION, RNG_SEED)

# Promote spatial_val → spatial_train, spatial_test → spatial_holdout
# so gp_eval.py (which looks for spatial_holdout) works correctly
out = three_way.copy()
out.loc[out['spatial_split'] == 'spatial_val', 'spatial_split'] = 'spatial_train'
out.loc[out['spatial_split'] == 'spatial_test', 'spatial_split'] = 'spatial_holdout'

train_n = (out['spatial_split'] == 'spatial_train').sum()
test_n  = (out['spatial_split'] == 'spatial_holdout').sum()
print(f'train: {train_n}  test: {test_n}  total: {len(out)}')

out.to_csv(OUT_SPLIT, index=False)
print(f'Saved → {OUT_SPLIT}')
