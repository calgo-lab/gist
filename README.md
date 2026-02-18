# Groundwater Level Interpolation 
This project is building a pipeline for spatiotemporal interpolation of groundwater levels

Last update: February 08th, 2026

The first step is recreating the baseline from Kunz et al. (2024), accessible here:
https://doi.org/10.5194/egusphere-2024-3484

That first step is forecasting of groundwater levels.

## What is done in this repo
- Temporal forecasting of groundwater levels (TFT)
- Spatial interpolation of forecasts (kriging) or observed values
- End-to-end pipeline runner that does both

## Setup
### Quickstart
On local machine:
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### Requirements
- Python 3.10+ recommended (GPU optional; CPU works but slower)
- No system packages required unless your local PyArrow build needs them

### Data
Configure local dataset file paths in `configs/data.yaml`. `data/sample.csv` is a small sample of the dataset.
Expected columns (minimum):
- `id`, `datum`, `gws`
- dynamic covariates: `tas_5km`, `hurs_5km`, `pr_5km`, `tag_sin`, `tag_cos`
- for spatial steps: `x_25833`, `y_25833`

### Checksums
Store hashes for locally saved data:
```bash
shasum -a 256 /path/to/main_data.parquet
shasum -a 256 /path/to/metadata.csv
```
And compare to hashes in `checksums/data.sha256` (datasets are identical if hashes are identical)

### Reproduce outputs
If you want to reproduce the contents of data and reports, run this; if checksum test (next section) is successful, it will recreate the exact same files though
```bash
python src/prep/ingest.py
python src/prep/summary.py
```

## Choose your run path
You have two options:
1. Temporal forecasting first, then spatial interpolation
2. Run both at once with `global_run.py`

## Temporal forecasting
### Training
Basic training (single run):
```bash
DATASET=full_raw SEED=40 IN_LEN=52 OUT_LEN=16 STATICS=1 EPOCHS=50 \
python src/scripts/separate/temporal/kunz_darts/tft_train.py
```

Outputs:
- `outputs/<MODEL>/<RUN_NAME>/`: model artifacts and checkpoints
- `splits/spatial_split_<DATASET>.csv`: spatial train/holdout IDs

### Evaluation
Evaluation generates predictions and metrics:
```bash
DATASET=full_raw SEED=40 IN_LEN=52 OUT_LEN=16 STATICS=1 \
python src/scripts/separate/temporal/kunz_darts/tft_eval.py
```

Outputs per run:
- `predictions/pred.parquet`: historical forecasts
- `metrics.parquet`: evaluation metrics by horizon
- `skill_by_horizon.parquet`: skill vs persistence by horizon

### Temporal predictions: how they work
TFT produces historical forecasts over the evaluation period using the date-based split.
The output `predictions/pred.parquet` contains per-series forecasts with horizons in weeks.

### Sweeps
To run multi-seed sweeps and log metrics:
```bash
python src/scripts/separate/temporal/kunz_darts/tft_runs.py
```

This writes `reports/metrics/metrics.csv` with timing and selected summary values.

For multi-seed runs on several GPUs in parallel, run the contents of jobs/training.

### Metrics summary
Aggregate run metrics into a summary table:
```bash
python src/analysis/build_tft_metrics_summary.py
```

Outputs:
- `reports/metrics/tft_metrics_summary.csv`
- `reports/metrics/tft_metrics_summary_sparse.csv`

## Spatial interpolation (kriging)
Pick one of these:
1. Script-only run:
```bash
python src/scripts/separate/spatial/kriging.py
```
2. Notebook run (includes plots/visuals and extra metrics):
`kriging_predicted_seed40.ipynb`

### Notebook vs script
The plan is to move plots and visualizations into the script and eventually remove the notebook.

## Global runs
One command to run the end-to-end pipeline (temporal + spatial):
```bash
python src/scripts/global_run.py
```

## Project layout
- `configs/*.yaml`: all run configuration
- `src/scripts/separate/temporal/kunz_darts/*`: TFT train/eval/sweeps
- `src/scripts/separate/spatial/kriging.py`: spatial interpolation
- `src/scripts/global_run.py`: end-to-end pipeline runner
- `outputs/`, `splits/`, `reports/`: generated artifacts
