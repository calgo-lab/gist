# GIST: Groundwater Interpolation in Space and Time

#### Hybrid Spatiotemporal Modeling for Forecast-Anywhere Groundwater Level Prediction

Two modelling approaches:
1. **Separate**:
    Baselines:
        - TFT temporal forecasting based on Kunz et al. (2024), models from darts
        - Kriging spatial interpolation, models from sklearn
    Own implementation:
        - GRU temporal forecasting
        - GP spatial interpolation
2. **Joint**:
    Temporal GRU + GP spatial layer (from separate), trained/backpropped end-to-end

Reference: Kunz et al. (2024): https://doi.org/10.5194/egusphere-2024-3484

Last update: March 15, 2026

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          
# Windows: .\.venv\Scripts\Activate.ps1
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Requirements: Python 3.10+, GPU recommended (A100 for cluster runs)

### Data

Configure dataset paths in `configs/data.yaml`. A small preview file is at `data/head_preview.csv`.

Required columns: `id`, `datum`, `gws`, `tas_5km`, `hurs_5km`, `pr_5km`, `tag_sin`, `tag_cos`, `x_25833`, `y_25833`

Verify data integrity against `checksums/data.txt`:
```bash
shasum -a 256 /path/to/main_data.parquet
```

### Data prep

```bash
python src/prep/ingest.py
python src/prep/summary.py
```

---

## Separate models (baseline)

### TFT — temporal forecasting

```bash
# Train
python src/scripts/separate/temporal/kunz_darts/tft_train.py

# Evaluate
python src/scripts/separate/temporal/kunz_darts/tft_eval.py

# Multi-seed sweeps
python src/scripts/separate/temporal/kunz_darts/tft_runs.py
```

Config: `configs/baselines/tft.yaml`

Outputs: `outputs/TFT/<RUN_NAME>/` — predictions, metrics, skill scores

### Kriging — spatial interpolation

```bash
python src/scripts/separate/spatial/kriging.py
```

Config: `configs/baselines/kriging.yaml`

Outputs: `outputs/gp/<RUN_TAG>/` — `gp_pred.parquet`, `gp_metrics.parquet`, `gp_grid.npz`

### End-to-end (separate with baselines)

```bash
python src/scripts/separate/global_run.py
```

---

## Joint model (GRU + GP)

### GRU — temporal encoder

```bash
# Train
python src/scripts/joint/temporal/gru_train.py

# Evaluate
python src/scripts/joint/temporal/gru_eval.py
```

Config: `configs/gru/gru.yaml` — dataset, in/out length, epochs, hidden size, layers, spatial split

Outputs: `outputs/GRU_FCOV/<RUN_NAME>/` — model checkpoints, predictions

### GP — spatial layer (on GRU outputs)

```bash
python src/scripts/joint/spatial/gp_eval.py
```

Config: `configs/gp/gp.yaml`

Outputs: `outputs/gp/<RUN_TAG>/` — `gp_pred.parquet`, `gp_metrics.parquet`

### Joint training (GRU + GP end-to-end)

```bash
# Train
python src/scripts/joint/gru_gp_train.py

# Evaluate
python src/scripts/joint/gru_gp_eval.py
```

Config: `configs/joint/gru_gp.yaml`

Outputs: `outputs/GRU_GP_JOINT/<RUN_NAME>/`

---

## HPO

```bash
# GRU HPO
# Submit: cluster/jobs/gw-gru-hpo-a100.yaml
# Config: configs/gru/hpo_gru.yaml

# GP HPO
# Submit: cluster/jobs/gw-gp-hpo-a100.yaml
# Config: configs/gp/hpo_gp.yaml
```

HPO results land in `reports/gru/hpo/` and `reports/gp/hpo/`.

---

## Spatial splits

Three splits are stored in `splits/`:
- `spatial_split_full_merged_spf0p8_sc20_ss42.csv` — 80/20 train/holdout (used by separate models and GP)
- `spatial_split_full_merged_spf0p8_sc20_ss42_90_10.csv` — 90/10 train/holdout (for fair comparison: promotes the joint model's spatial_val into train)

---

## Analysis & reports

All report-building scripts are in `src/analysis/build/`:

```bash
python src/analysis/build/gru_report.py         # reports/gru/metrics/gru_metrics_summary.csv
python src/analysis/build/gru_hpo_tables.py     # reports/gru/hpo/ tables
python src/analysis/build/gp_report.py          # reports/gp/metrics/gp_metrics_summary.csv
python src/analysis/build/kriging_report.py     # reports/kriging/metrics/kriging_metrics_summary.csv
python src/analysis/build/joint_report.py       # reports/gru_gp_joint/metrics/ + figures
python src/analysis/build/tft_report.py         # reports/tft/metrics/tft_metrics_summary.csv
python src/analysis/build/cross_model_table.py  # reports/cross_model/cross_model_table.csv
```

Spatial analysis scripts in `src/analysis/spatial/`:
- `inspect_spatial_split.py` — plots train/holdout split map → `reports/spatial_split/figures/`
- `evaluate_kriging.py` — kriging surface plots → `reports/kriging/figures/`
- `outlier_analysis/analyze_outlier_wells.py` — outlier well characterization → `reports/outlier_analysis/kriging/`
- `outlier_analysis/plot_distance_vs_outliers.py` — distance-to-training analysis → `reports/outlier_analysis/kriging/`

---

## Cluster (Kubernetes)

Jobs are in `cluster/jobs/`. Submit with:
```bash
kubectl apply -f cluster/jobs/<job>.yaml
```

Available jobs:
- `gw-gru-l2-seed40-ep50-a100.yaml` — single GRU run
- `gw-gru-hpo-a100.yaml` — GRU HPO
- `gw-gp-hpo-a100.yaml` — GP HPO
- `gw-gp-spatial-a100.yaml` — GP on temporal model predictions (set `MODEL_PREFIX` + `TEMPORAL_RUN_SIG`)
- `gw-gru-gp-gpytorch-seed40-ep50-a100.yaml` — joint GRU+GP run

Docker image: `row56/gw-pred:py312-cu124`. Storage mounted at `/storage` via PVC `gw-pred-rwx`.

---

## Project layout

```
configs/
  data.yaml                  dataset paths and feature config
  gru/                       GRU model config and HPO
  gp/                        GP model config and HPO
  joint/                     joint GRU+GP config
  baselines/                 TFT and kriging configs
src/
  libs/                      shared utilities (run_sig, spatial_split, utils)
  prep/                      data ingestion, summary, split preparation
  scripts/
    joint/                   GRU+GP joint model (train/eval + sub-models)
    separate/                TFT + kriging baseline (train/eval + global runner)
  analysis/
    build/                   report and metrics summary scripts
    spatial/                 spatial split inspection and kriging   evaluation
        outlier_analysis     analysis of outlier wells in kriging/gp predictive performance
splits/                      spatial split CSVs
outputs/                     model outputs (gitignored)
reports/                     metrics CSVs, HPO results, figures
cluster/                     Kubernetes job YAMLs and Docker configs
```
