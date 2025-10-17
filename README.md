# Groundwater Level Interpolation 
This project is building a pipeline for spatiotemporal interpolation of groundwater levels.

The first step is recreating the baseline from Kunz et al. (2024), accessible here:
https://doi.org/10.5194/egusphere-2024-3484

# Groundwater Forecasting

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Data
Configure local dataset file paths in `configs/data.yaml`. `data/sample.csv` is a small sample of the dataset.

## Reproduce outputs (optional)
If you want to reproduce the contents of data and reports, run this; if checksum test (next section) is successful, it will recreate the exact same files though
```bash
python -m src.data.ingest
python -m src.data.summary
```

## Checksums
Store hashes for locally saved data:
```bash
shasum -a 256 /path/to/main_data.parquet
shasum -a 256 /path/to/metadata.csv
```
And compare to hashes in `checksums/data.sha256` (datasets are identical if hashes are identical)