# Groundwater Level Interpolation 
This project is building a pipeline for spatiotemporal interpolation of groundwater levels.

This Readme is always gonna show the current state of the project.

The first step is recreating the baseline from Kunz et al. (2024), accessible here:
https://doi.org/10.5194/egusphere-2024-3484

That first step is forecasting of groundwater levels.

## Groundwater Forecasting

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

### Data
Configure local dataset file paths in `configs/data.yaml`. `data/sample.csv` is a small sample of the dataset.


### Reproduce outputs
If you want to reproduce the contents of data and reports, run this; if checksum test (next section) is successful, it will recreate the exact same files though
```bash
python -m src.data.ingest
python -m src.data.summary
```

### Training
Basic training (single run):
```bash
DATASET=full_raw SEED=40 IN_LEN=52 OUT_LEN=16 STATICS=1 EPOCHS=50 \
python src/models/kunz_darts/train.py
```


Outputs:
- `outputs/<MODEL>/<RUN_NAME>/`: model artifacts and checkpoints
- `outputs/spatial_split_<DATASET>.csv`: spatial train/holdout IDs

### Evaluation
Evaluation generates predictions and metrics:
```bash
DATASET=full_raw SEED=40 IN_LEN=52 OUT_LEN=16 STATICS=1 \
python src/models/kunz_darts/eval.py
```

Outputs per run:
- `predictions/pred.parquet`: historical forecasts
- `metrics.parquet`: evaluation metrics by horizon
- `skill_by_horizon.parquet`: skill vs persistence by horizon

### Sweeps
To run multi-seed sweeps and log metrics:
```bash
python src/models/kunz_darts/runs.py
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

### Spatial interpolation notebook
`kriging_predicted_seed40.ipynb` builds spatial interpolation maps from:
- model predictions at a chosen horizon, and
- true values at the same horizon,
then compares against spatial holdout wells


### Checksums
Store hashes for locally saved data:
```bash
shasum -a 256 /path/to/main_data.parquet
shasum -a 256 /path/to/metadata.csv
```
And compare to hashes in `checksums/data.sha256` (datasets are identical if hashes are identical)
