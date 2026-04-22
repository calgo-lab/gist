import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

ROOT = Path('/storage/gwl-interpolation')
OUT = Path('/tmp/consistent_slide_assets')
OUT.mkdir(parents=True, exist_ok=True)

split = pd.read_csv(ROOT / 'splits/rmd90_test52.csv')
col = 'spatial_split' if 'spatial_split' in split.columns else 'split'
test52_ids = set(split.loc[split[col].isin(['spatial_test', 'spatial_holdout']), 'id'])

RUNS = {
    'Global GRU': ROOT / 'outputs/GRU_FCOV/GRU_FCOV_in52_out16_ep50_bs4096_seed40_full_merged_all_train/predictions/pred.parquet',
    'GP on true obs': ROOT / 'outputs/gp/GRU_FCOV_oracle_rmd90_test52__oracle_hpo_t020_ni128_ps200_j1e-03__predobstrain/gp_pred.parquet',
    'GP on GRU preds': ROOT / 'outputs/gp/GRU_FCOV_in52_out16_ep50_bs8192_seed40_full_merged_test52_hpo_sep_t118__hpo_sep_t118__predobstrain/gp_pred.parquet',
    'Joint training GRU + GP': ROOT / 'outputs/GRU_GP_JOINT/GRU_GP_JOINT_0260/eval/test/gp_pred.parquet',
}


def normalize_pred(path: Path) -> pd.DataFrame:
    df = pq.read_table(path).to_pandas()
    cols = set(df.columns)
    pred_col = 'gws_forecast' if 'gws_forecast' in cols else 'pred' if 'pred' in cols else 'y_mean' if 'y_mean' in cols else None
    true_col = 'gws' if 'gws' in cols else 'gws_true' if 'gws_true' in cols else 'target' if 'target' in cols else 'y' if 'y' in cols else None
    if pred_col is None or true_col is None:
        raise ValueError(f'Could not infer pred/true columns for {path}: {df.columns.tolist()}')
    out = df.rename(columns={pred_col: 'pred', true_col: 'true'}).copy()
    out = out[['id', 'datum', 'horizon', 'pred', 'true']]
    out['datum'] = pd.to_datetime(out['datum'])
    out = out[out['id'].isin(test52_ids)].copy()
    out['datum_s'] = out['datum'].astype(str)
    return out

frames = {name: normalize_pred(path) for name, path in RUNS.items()}
common = None
for df in frames.values():
    keys = set(zip(df['id'], df['datum_s'], df['horizon']))
    common = keys if common is None else common & keys
common_df = pd.DataFrame(sorted(common), columns=['id', 'datum_s', 'horizon'])

summary_rows = []
horizon_rows = []
for model, df in frames.items():
    sub = df.merge(common_df, on=['id', 'datum_s', 'horizon'], how='inner').copy()
    per_well = []
    for wid, g in sub.groupby('id'):
        pred = g['pred'].to_numpy(dtype=float)
        true = g['true'].to_numpy(dtype=float)
        mask = np.isfinite(pred) & np.isfinite(true)
        if mask.sum() < 2:
            continue
        pred = pred[mask]
        true = true[mask]
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        iqr = float(np.quantile(true, 0.75) - np.quantile(true, 0.25))
        per_well.append({'id': wid, 'rmse': rmse, 'iqr': iqr, 'ratio': rmse / iqr if iqr > 0 else np.nan})
    s = pd.DataFrame(per_well)
    summary_rows.append({
        'model': model,
        'n_wells': int(s['id'].nunique()),
        'common_keys': int(len(common_df)),
        'median_nrmse_per_well': float(s['ratio'].median()),
        'median_rmse_per_well_m': float(s['rmse'].median()),
    })

    rmse_pw_h = (
        sub.groupby(['id', 'horizon'], as_index=False)
        .apply(lambda g: pd.Series({'rmse': float(np.sqrt(np.mean((g['pred'] - g['true']) ** 2)))}))
        .reset_index(drop=True)
    )
    by_h = (
        rmse_pw_h.groupby('horizon', as_index=False)['rmse']
        .median()
        .rename(columns={'rmse': 'median_rmse_per_well'})
        .sort_values('horizon')
    )
    by_h['model'] = model
    by_h['n_wells'] = int(s['id'].nunique())
    horizon_rows.append(by_h[['model', 'horizon', 'median_rmse_per_well', 'n_wells']])

summary = pd.DataFrame(summary_rows).sort_values('model')
horizon = pd.concat(horizon_rows, ignore_index=True)
summary.to_csv(OUT / 'slide_table_consistent_metrics.csv', index=False)
horizon.to_csv(OUT / 'slide_table_rmse_per_horizon_consistent.csv', index=False)
print(summary.to_string(index=False))
print('\ncommon_keys', len(common_df))
