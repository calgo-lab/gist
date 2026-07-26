from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve()
for p in [ROOT, *ROOT.parents]:
    if (p / 'configs' / 'data.yaml').exists():
        ROOT = p
        break

GRU_ROOT = ROOT / 'outputs' / 'GRU_FCOV'
OUT = ROOT / 'reports' / 'metrics' / 'gru' / 'bs_ablation_metrics.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)

HORIZON_WEIGHTS = np.linspace(1.0, 2.0, 16)

RUNS = [
    ('in52_out16_ep50_bs2048_seed40_full_merged_r0_s0_spf0p8_sc20_ss42', 2048, 2),
    ('in52_out16_ep50_bs8192_seed40_full_merged_r0_s0_spf0p8_sc20_ss42', 8192, 3),
]

HP_BASE = {
    'hp.model.use_revin': '',
    'hp.training.lr_scheduler.enabled': '',
    'hp.model.gru_dropout': 0.2,
    'hp.model.gru_hidden': 192.0,
    'hp.model.gru_layers': 4.0,
    'hp.training.lr': 0.0004,
}


def nse(p, r):
    d = np.sum((r - r.mean()) ** 2)
    return float(1 - np.sum((p - r) ** 2) / d) if d > 0 else np.nan


def compute(run_sig, batch_size, trial):
    pred_path = GRU_ROOT / f'GRU_FCOV_{run_sig}' / 'predictions' / 'pred.parquet'
    metrics_path = GRU_ROOT / f'GRU_FCOV_{run_sig}' / 'metrics.parquet'

    if not pred_path.exists():
        print(f'[SKIP] {run_sig}')
        return None

    df = pq.read_table(pred_path).to_pandas()

    objective = np.nan
    if metrics_path.exists():
        m = pd.read_parquet(metrics_path)
        row = m[m['horizon'] == 16]
        if not row.empty:
            objective = float(row['RMSE'].iloc[0])

    rows = []
    for (_, h), g in df.groupby(['id', 'horizon']):
        p = g['gws_forecast'].to_numpy()
        r = g['gws'].to_numpy()
        rows.append({'horizon': int(h),
                     'NSE': nse(p, r),
                     'RMSE': float(np.sqrt(np.mean((p - r) ** 2))),
                     'MAE': float(np.mean(np.abs(p - r)))})

    hz = pd.DataFrame(rows).groupby('horizon')[['NSE', 'RMSE', 'MAE']].median()

    def wmean(col):
        return float(np.average(hz[col], weights=HORIZON_WEIGHTS))

    def smean(col, h1, h2):
        return float(hz.loc[h1:h2, col].mean())

    def at16(col):
        return float(hz.loc[16, col])

    return {
        'hpo_name': 'bs_ablation',
        'trial': trial,
        'run_sig': run_sig,
        'status': 'ok',
        'objective': objective,
        'weighted_mean_nse_1_16': wmean('NSE'),
        'mean_nse_1_16': smean('NSE', 1, 16),
        'mean_nse_13_16': smean('NSE', 13, 16),
        'nse_h16': at16('NSE'),
        'weighted_mean_rmse_1_16': wmean('RMSE'),
        'mean_rmse_1_16': smean('RMSE', 1, 16),
        'mean_rmse_13_16': smean('RMSE', 13, 16),
        'rmse_h16': at16('RMSE'),
        'weighted_mean_mae_1_16': wmean('MAE'),
        'mean_mae_1_16': smean('MAE', 1, 16),
        'mean_mae_13_16': smean('MAE', 13, 16),
        'mae_h16': at16('MAE'),
        'hp.training.batch_size': batch_size,
        **HP_BASE,
    }


rows = [r for run_sig, bs, trial in RUNS if (r := compute(run_sig, bs, trial))]
if rows:
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'Saved → {OUT}')
    print(pd.DataFrame(rows)[['run_sig', 'weighted_mean_nse_1_16', 'mean_nse_1_16', 'nse_h16', 'mean_rmse_1_16', 'rmse_h16']].to_string(index=False))
else:
    print('No completed runs found.')
