import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths (relative to this script file, so the script works from any cwd) ───
HERE      = Path(__file__).resolve().parent          # src/analysis/
STORAGE   = HERE.parents[2]                          # storage/
DATA_PATH = STORAGE / "data" / "gws_bb_complete_hyras_1000.parquet"
OUT_PATH  = STORAGE / "gwl_interpolation" / "reports" / "data_completeness_proof.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Split boundaries ──────────────────────────────────────────────────────────
# Nick's split
NICK_TRAIN_END = 2010
NICK_VAL_END   = 2013
NICK_TEST_END  = 2016

# Robert's split
ROBERT_TRAIN_END  = 2016
ROBERT_VAL_END    = 2020

# ── Active well criteria ──────────────────────────────────────────────────────
EXACT_OBS    = 52
PARTIAL_YEAR = 2024

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
N  = df['id'].nunique()

df['year'] = df['datum'].dt.year
df = df.sort_values(['id', 'datum'])

# ── Compute per-well-year observation counts ──────────────────────────────────
print('Computing per-well-year stats...')
stats = df.groupby(['id', 'year'])['datum'].count().rename('n_obs').reset_index()

# Active = exactly 52 observations (53-week years also accepted)
stats['active_strict'] = stats['n_obs'].isin([52, 53])

active_strict = stats.groupby('year')['active_strict'].sum()
years = sorted(stats['year'].unique())

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                gridspec_kw={'height_ratios': [1, 1]},
                                constrained_layout=True)

# ── Plot 1: full history ───────────────────────────────────
full_years = [y for y in years if y < PARTIAL_YEAR]
bar_colors = ['#90CAF9' if active_strict.get(y, 0) < N else '#1565C0' for y in full_years]
ax1.bar(full_years, [active_strict.get(y, 0) for y in full_years],
        color=bar_colors, width=0.85, edgecolor='white', linewidth=0.3)

first_full_year_all = next(y for y in full_years if active_strict.get(y, 0) == N)

ax1.set_xlim(1950, 2025)
ax1.set_ylim(0, N + 80)
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel(f'Wells with {EXACT_OBS} observations', fontsize=11)
ax1.set_title(
    f'Active wells per year — full record (1951–{PARTIAL_YEAR - 1})',
    fontsize=12, fontweight='bold')
ax1.legend(handles=[
    *ax1.get_legend_handles_labels()[0],
    mpatches.Patch(facecolor='#90CAF9', label='Not yet all wells active'),
    mpatches.Patch(facecolor='#1565C0', label='All wells active for entire year'),
], fontsize=10, loc='upper left')
ax1.grid(True, axis='y', alpha=0.3)

# ── Plot 2: zoom 2000–2023 ─────────────────────────────────
zoom_years  = [y for y in years if 2000 <= y < PARTIAL_YEAR]
zoom_strict = [int(active_strict.get(y, 0)) for y in zoom_years]

ax2.bar(zoom_years, zoom_strict, color=['#90CAF9' if v < N else '#1565C0' for v in zoom_strict],
        width=0.75, edgecolor='white', linewidth=0.3, zorder=2)
ax2.axhline(N, color='black', linestyle=':', linewidth=1.4)
ax2.set_facecolor('white')

# ── Period label row y-positions ───────────────
LABEL_Y_NICK   = N + 20
LABEL_Y_SEP    = N + 38
LABEL_Y_ROBERT = N + 55
TOP            = N + 90

# ── Vertical split boundary lines ────────────────────
LINE_KW = dict(colors='black', linestyle='--', linewidth=1.5, zorder=5)

# Nick inner boundaries: stop at separator so they don't cut through Robert's band
ax2.vlines([NICK_TRAIN_END - 0.5, NICK_VAL_END - 0.5],
           ymin=780, ymax=LABEL_Y_SEP, **LINE_KW)

# Lines at Robert's own boundaries (and Nick's test end): span full height
ax2.vlines([NICK_TEST_END - 0.5, ROBERT_TRAIN_END - 0.5, ROBERT_VAL_END - 0.5],
           ymin=780, ymax=TOP, **LINE_KW)

# ── Period label rows — Rectangle patches give each row its own clean color ──

def add_band(ax, x0, x1, y0, y1, color):
    """Fully opaque rectangle — no alpha blending with layers below."""
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        facecolor=color, alpha=1.0, zorder=3, clip_on=True
    ))

# Nick label band — pastel warm tones (opaque so no bleed from axvspan below)
add_band(ax2, 1999.5,               NICK_TRAIN_END - 0.5, N, LABEL_Y_SEP, "#FE776A")  # pastel red
add_band(ax2, NICK_TRAIN_END - 0.5, NICK_VAL_END   - 0.5, N, LABEL_Y_SEP, "#FCAA62")  # pastel orange
add_band(ax2, NICK_VAL_END   - 0.5, NICK_TEST_END  - 0.5, N, LABEL_Y_SEP, "#FDDC64")  # pastel yellow
add_band(ax2, NICK_TEST_END  - 0.5, PARTIAL_YEAR   - 0.5, N, LABEL_Y_SEP, "#73879B")  # light grey

# Robert label band — pastel cool tones, fully opaque
add_band(ax2, 1999.5,              ROBERT_TRAIN_END - 0.5, LABEL_Y_SEP, TOP, "#6FBEF6")  # pastel blue
add_band(ax2, ROBERT_TRAIN_END - 0.5, ROBERT_VAL_END   - 0.5, LABEL_Y_SEP, TOP, "#43C47B")  # pastel green
add_band(ax2, ROBERT_VAL_END   - 0.5, PARTIAL_YEAR  - 0.5, LABEL_Y_SEP, TOP, "#AE6BCF")  # pastel purple

# Horizontal separator across the full x-axis
ax2.axhline(LABEL_Y_SEP, color='#333333', linewidth=1.0, linestyle='-', zorder=6)

# Nick's labels (lower row)
ax2.text(2004.5, LABEL_Y_NICK, "Nick: train",
         color='#7b0f05', fontsize=8, ha='center', fontweight='bold', va='center', zorder=7)
ax2.text(2011.0, LABEL_Y_NICK, "Nick: val",
         color='#7a3200', fontsize=8, ha='center', fontweight='bold', va='center', zorder=7)
ax2.text(2014.0, LABEL_Y_NICK, "Nick: test",
         color='#5c4a00', fontsize=8, ha='center', fontweight='bold', va='center', zorder=7)

# Robert's labels (upper row) — same color label for train across all three sub-spans
ax2.text(2009, LABEL_Y_ROBERT, "Robert: train",
         color="#292962", fontsize=8, ha='center', fontweight='bold', va='center', zorder=7)
ax2.text(2017.5, LABEL_Y_ROBERT, "Robert: val",
         color="#203520", fontsize=8, ha='center', fontweight='bold', va='center', zorder=7)
ax2.text(2021.5, LABEL_Y_ROBERT, "Robert: test",
         color="#2C1B34", fontsize=8, ha='center', fontweight='bold', va='center', zorder=7)

# Data labels for years where not all wells pass the strict criterion
for y, v in zip(zoom_years, zoom_strict):
    if v < N:
        ax2.text(y, v + 5, str(v), ha='center', va='bottom', fontsize=8, color='#1565C0')

ax2.set_xlim(1999.5, PARTIAL_YEAR - 0.5)
ax2.set_ylim(780, N + 90)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel(f'Wells with {EXACT_OBS} observations', fontsize=11)
ax2.set_title(
    f'Zoom 2000–{PARTIAL_YEAR - 1}',
    fontsize=12, fontweight='bold')
ax2.grid(True, axis='y', alpha=0.3)

first_full_year = next(y for y, v in zip(zoom_years, zoom_strict) if v == N)
assert first_full_year == first_full_year_all
fig.suptitle(
    'Brandenburg Groundwater Dataset Completeness Proof',
    fontsize=13, fontweight='bold')

fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved → {OUT_PATH}')
