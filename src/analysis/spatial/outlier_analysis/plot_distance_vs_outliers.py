"""
Plot median distance to k closest training wells vs outlier group size.

For each k in [2, 5, 10], creates a subplot showing:
  - x-axis: size of the outlier group (top-N worst RMSE holdout wells)
  - y-axis: median distance to the k closest training wells
  - two lines: outlier group vs non-outlier group
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve()
for p in [ROOT, *ROOT.parents]:
    if (p / 'configs' / 'data.yaml').exists():
        ROOT = p
        break

RUN_TAG = 'TFT_in52_out16_ep50_bs4096_stat1_seed40_full_merged__predobstrain'
K_VALUES = [2, 5, 10]

cfg = yaml.safe_load((ROOT / 'configs' / 'data.yaml').read_text())
data_path = Path(cfg['full_merged_path'])
if not data_path.is_absolute():
    data_path = (ROOT / data_path).resolve()

run_dir = ROOT / 'reports' / 'outlier_analysis' / 'kriging'
profile = pd.read_csv(run_dir / 'outlier_well_profile.csv')

data = pq.read_table(data_path, columns=['id', 'x_25833', 'y_25833']).to_pandas()
coords = data[['id', 'x_25833', 'y_25833']].dropna().drop_duplicates('id')

holdout_ids = set(profile['id'])
all_ids = set(coords['id'])
train_ids = all_ids - holdout_ids

train_xy = coords[coords['id'].isin(train_ids)][['x_25833', 'y_25833']].to_numpy(float)
holdout_meta = coords[coords['id'].isin(holdout_ids)].copy()
holdout_xy = holdout_meta[['x_25833', 'y_25833']].to_numpy(float)

print(f'Computing distance matrix ({len(holdout_xy)} holdout × {len(train_xy)} train)…')
dist = np.sqrt(((holdout_xy[:, None, :] - train_xy[None, :, :]) ** 2).sum(axis=2))

dist_by_k = {}
for k in K_VALUES:
    idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    mean_dists = np.array([dist[i, idx[i]].mean() for i in range(len(holdout_xy))])
    dist_by_k[k] = mean_dists

holdout_meta = holdout_meta.reset_index(drop=True)
for k in K_VALUES:
    holdout_meta[f'mean_dist_k{k}'] = dist_by_k[k]

df = profile[['id', 'rmse']].merge(holdout_meta, on='id', how='inner')
df = df.sort_values('rmse', ascending=False).reset_index(drop=True)

n_holdout = len(df)
MAX_OUTLIER_GROUP = 190
top_n_range = range(1, MAX_OUTLIER_GROUP + 1)

results = {}
for k in K_VALUES:
    col = f'mean_dist_k{k}'
    outlier_medians = []
    non_outlier_medians = []
    for top_n in top_n_range:
        outlier_group = df.iloc[:top_n][col]
        non_outlier_group = df.iloc[top_n:][col]
        outlier_medians.append(np.median(outlier_group))
        non_outlier_medians.append(np.median(non_outlier_group) if len(non_outlier_group) > 0 else np.nan)
    results[k] = {
        'outlier': np.array(outlier_medians),
        'non_outlier': np.array(non_outlier_medians),
    }

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
x = np.array(list(top_n_range))

for ax, k in zip(axes, K_VALUES):
    outlier_med = results[k]['outlier'] / 1000
    non_outlier_med = results[k]['non_outlier'] / 1000

    ax.plot(x, outlier_med, color='#d62728', linewidth=1.8, label='Outlier wells')
    ax.plot(x, non_outlier_med, color='#1f77b4', linewidth=1.8, label='Non-outlier wells')

    ax.set_title(f'k = {k}', fontsize=13)
    ax.set_xlabel(f'Outlier group size\nTop N wells sorted descending by RMSE\nNon-outlier group size: {n_holdout}−N wells', fontsize=10)
    ax.set_ylabel('Median distance to k closest\ntraining wells (km)', fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    'Distance to nearest training wells: outlier vs non-outlier holdout wells',
    fontsize=13, y=1.01,
)
plt.tight_layout()

out_path = ROOT / 'reports' / 'outlier_analysis' / 'kriging' / 'outlier_dist_train.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved → {out_path}')
