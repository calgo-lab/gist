"""
Creates a table plot: one well per row, columns are the 1st and 2nd
observation of each chosen year (e.g. 2009-01, 2009-02, 2010-01, ...).
Values shown are the GWS measurement and the observation date.

Change YEARS, N_WELLS, and SEED at the top to adjust the output.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Settings — edit these ─────────────────────────────────────────────────────
YEARS   = [1960, 1970, 1980, 1990, 2000, 2005, 2010, 2012, 2014, 2018, 2022]   # years to inspect
N_WELLS = 25                   # how many wells to sample
SEED    = 1                   # change for a different random draw

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
STORAGE   = HERE.parents[2]
DATA_PATH = STORAGE / "data" / "gws_bb_complete_hyras_1000.parquet"
OUT_PATH  = STORAGE / "gwl_interpolation" / "reports" / "sample_well_observations.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Load & sort ───────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH, columns=['id', 'datum', 'gws'])
df = df.sort_values(['id', 'datum']).reset_index(drop=True)
df['year'] = df['datum'].dt.year

# ── Random sample of wells ────────────────────────────────────────────────────
rng      = np.random.default_rng(SEED)
selected = rng.permutation(df['id'].unique())[:N_WELLS]

# ── Build wide table: rows = wells, columns = YYYY ───────────────────────────
columns = [str(y) for y in YEARS]
records = []
for well in selected:
    wdf  = df[df['id'] == well]
    row  = {'well': well}
    for year in YEARS:
        year_obs = wdf[wdf['year'] == year]
        obs = year_obs.sample(n=1, random_state=rng) if len(year_obs) else year_obs
        if len(obs):
            r = obs.iloc[0]
            row[str(year)] = f"{r['gws']:.2f}\n({r['datum'].strftime('%d.%m.')})"
        else:
            row[str(year)] = '—'
    records.append(row)

table_df = pd.DataFrame(records, columns=['well'] + columns)

# ── Plot as a matplotlib table ────────────────────────────────────────────────
n_rows = len(table_df)
n_cols = len(table_df.columns)

fig_w = 2.2 + n_cols * 1.4
fig_h = 0.5 + n_rows * 0.55
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis('off')

cell_text = table_df.values.tolist()
col_labels = table_df.columns.tolist()

tbl = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 2.2)

# Style header
for j in range(n_cols):
    cell = tbl[0, j]
    cell.set_facecolor('#1565C0')
    cell.set_text_props(color='white', fontweight='bold')

# Style year group headers with alternating shades
year_colors = ['#E3F2FD', '#BBDEFB']
for j, col in enumerate(col_labels):
    if col == 'well':
        continue
    year = int(col)
    shade = year_colors[YEARS.index(year) % len(year_colors)]
    for i in range(1, n_rows + 1):
        tbl[i, j].set_facecolor(shade)

# Alternating row shading for the well column
for i in range(1, n_rows + 1):
    tbl[i, 0].set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')
    tbl[i, 0].set_text_props(fontsize=7)

ax.set_title(
    f'{N_WELLS} randomly sampled wells, 1 random observation per year',
    fontsize=20, fontweight='bold', pad=6
)

fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved → {OUT_PATH}')
