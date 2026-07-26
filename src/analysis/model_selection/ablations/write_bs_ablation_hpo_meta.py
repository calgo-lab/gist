from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve()
for p in [ROOT, *ROOT.parents]:
    if (p / 'configs' / 'data.yaml').exists():
        ROOT = p
        break

OUT = ROOT / 'reports' / 'metrics' / 'gru' / 'hpo' / 'bs_ablation_results.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = [
    {
        'trial': 1,
        'run_sig': 'in52_out16_ep50_bs1024_seed40_full_merged_r0_s0_spf0p8_sc20_ss42',
        'status': 'ok',
        'objective': None,
        'hp.model.gru_layers': 4,
        'hp.model.gru_hidden': 192,
        'hp.model.gru_dropout': 0.2,
        'hp.training.lr': 4e-4,
        'hp.training.batch_size': 1024,
        'hp.model.use_revin': False,
        'hp.training.lr_scheduler.enabled': False,
    },
    {
        'trial': 2,
        'run_sig': 'in52_out16_ep50_bs2048_seed40_full_merged_r0_s0_spf0p8_sc20_ss42',
        'status': 'ok',
        'objective': None,
        'hp.model.gru_layers': 4,
        'hp.model.gru_hidden': 192,
        'hp.model.gru_dropout': 0.2,
        'hp.training.lr': 4e-4,
        'hp.training.batch_size': 2048,
        'hp.model.use_revin': False,
        'hp.training.lr_scheduler.enabled': False,
    },
    {
        'trial': 3,
        'run_sig': 'in52_out16_ep50_bs8192_seed40_full_merged_r0_s0_spf0p8_sc20_ss42',
        'status': 'ok',
        'objective': None,
        'hp.model.gru_layers': 4,
        'hp.model.gru_hidden': 192,
        'hp.model.gru_dropout': 0.2,
        'hp.training.lr': 4e-4,
        'hp.training.batch_size': 8192,
        'hp.model.use_revin': False,
        'hp.training.lr_scheduler.enabled': False,
    },
]

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f'Wrote → {OUT}')
