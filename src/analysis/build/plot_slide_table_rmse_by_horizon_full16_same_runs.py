from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_MET = ROOT / 'reports' / 'presentation_assets' / 'metrics'
OUT_FIG = ROOT / 'reports' / 'presentation_assets' / 'figures'
OUT_TXT = ROOT / 'reports' / 'presentation_assets' / 'notes'

MODEL_ORDER = [
    'Global GRU',
    'GP on true obs',
    'GP on GRU preds',
    'Joint training GRU + GP',
]
COLORS = {
    'Global GRU': '#1f77b4',
    'GP on true obs': '#2ca02c',
    'GP on GRU preds': '#ff7f0e',
    'Joint training GRU + GP': '#d62728',
}


def _make_plot(df: pd.DataFrame, labels: list[str], title: str, out_path: Path, legend_loc: str, legend_anchor: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    for label in labels:
        sub = df.loc[df['model'] == label].sort_values('horizon')
        ax.plot(
            sub['horizon'],
            sub['median_rmse_per_well'],
            marker='o',
            linewidth=2.2,
            markersize=4.8,
            color=COLORS[label],
            label=label,
        )
    ax.set_xlabel('Forecast horizon')
    ax.set_ylabel('Median RMSE per well (m)')
    ax.set_title(title)
    ax.set_xticks(range(1, 17))
    ax.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax.legend(frameon=False, loc=legend_loc, bbox_to_anchor=legend_anchor)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TXT.mkdir(parents=True, exist_ok=True)

    horizon_df = pd.read_csv(OUT_MET / 'slide_table_rmse_per_horizon_same_runs_full16.csv')
    metric_df = pd.read_csv(OUT_MET / 'slide_table_same_runs_full16_metrics.csv')
    common_df = pd.read_csv(OUT_MET / 'slide_table_consistent_metrics.csv')

    horizon_df['model'] = pd.Categorical(horizon_df['model'], categories=MODEL_ORDER, ordered=True)
    horizon_df = horizon_df.sort_values(['model', 'horizon']).reset_index(drop=True)

    fig_all = OUT_FIG / 'slide_table_rmse_per_horizon_all16_same_runs.png'
    _make_plot(
        df=horizon_df,
        labels=MODEL_ORDER,
        title='Per-Well RMSE by Horizon for the Same Four Runs (All 16 Horizons)',
        out_path=fig_all,
        legend_loc='center left',
        legend_anchor=(1.02, 0.5),
    )

    fig_no_gru = OUT_FIG / 'slide_table_rmse_per_horizon_all16_same_runs_no_global_gru.png'
    _make_plot(
        df=horizon_df,
        labels=[label for label in MODEL_ORDER if label != 'Global GRU'],
        title='Per-Well RMSE by Horizon Without the Global GRU Baseline (All 16 Horizons)',
        out_path=fig_no_gru,
        legend_loc='upper left',
        legend_anchor=(1.02, 1.0),
    )

    full_map = {
        row['model']: (row['median_nrmse_per_well'], row['median_rmse_per_well_m'], row['min_date'], row['max_date'])
        for _, row in metric_df.iterrows()
    }
    common_map = {
        row['model']: (row['median_nrmse_per_well'], row['median_rmse_per_well_m'])
        for _, row in common_df.iterrows()
    }

    note = '\n'.join([
        'Figures: all-16-horizon RMSE curves for the exact same four runs used in the table.',
        'These plots use the same run identities and the same 52 rmd90_test52 wells as the table.',
        'Difference from the table:',
        '- The table uses the strict common-row slice across all four runs (11,232 shared keys).',
        '- These plots use each run\'s full saved test52 rows so that all 16 horizons remain visible.',
        '- Therefore the plot lines are run-matched to the table, but not row-by-row aligned to the table slice.',
        'Full-slice aggregate metrics for the exact plotted runs:',
        f"- Global GRU: nRMSE_pw={full_map['Global GRU'][0]:.4f}, RMSE_pw={full_map['Global GRU'][1]:.4f} m, range={full_map['Global GRU'][2]} to {full_map['Global GRU'][3]}",
        f"- GP on true obs: nRMSE_pw={full_map['GP on true obs'][0]:.4f}, RMSE_pw={full_map['GP on true obs'][1]:.4f} m, range={full_map['GP on true obs'][2]} to {full_map['GP on true obs'][3]}",
        f"- GP on GRU preds: nRMSE_pw={full_map['GP on GRU preds'][0]:.4f}, RMSE_pw={full_map['GP on GRU preds'][1]:.4f} m, range={full_map['GP on GRU preds'][2]} to {full_map['GP on GRU preds'][3]}",
        f"- Joint training GRU + GP: nRMSE_pw={full_map['Joint training GRU + GP'][0]:.4f}, RMSE_pw={full_map['Joint training GRU + GP'][1]:.4f} m, range={full_map['Joint training GRU + GP'][2]} to {full_map['Joint training GRU + GP'][3]}",
        'Strict common-slice table metrics for reference:',
        f"- Global GRU: nRMSE_pw={common_map['Global GRU'][0]:.4f}, RMSE_pw={common_map['Global GRU'][1]:.4f} m",
        f"- GP on true obs: nRMSE_pw={common_map['GP on true obs'][0]:.4f}, RMSE_pw={common_map['GP on true obs'][1]:.4f} m",
        f"- GP on GRU preds: nRMSE_pw={common_map['GP on GRU preds'][0]:.4f}, RMSE_pw={common_map['GP on GRU preds'][1]:.4f} m",
        f"- Joint training GRU + GP: nRMSE_pw={common_map['Joint training GRU + GP'][0]:.4f}, RMSE_pw={common_map['Joint training GRU + GP'][1]:.4f} m",
    ])
    note_path = OUT_TXT / 'slide_table_rmse_per_horizon_all16_same_runs_note.md'
    note_path.write_text(note + '\n', encoding='utf-8')

    print(fig_all)
    print(fig_no_gru)
    print(note_path)


if __name__ == '__main__':
    main()
