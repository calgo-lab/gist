# Forecast Anywhere Groundwater Interpolation (FAGI): Spatiotemporal Groundwater Level Prediction with Neural Temporal Models and Spatial Interpolation

## Administrative Context
This thesis is written and submitted in the M.Sc. Data Science program at Berliner Hochschule fuer Technik (BHT), with submission target date May 11, 2026. The work is conducted in collaboration with the Cognitive Algorithms Lab under Prof. Felix Biessmann (Berlin, Germany) and the Earth AI and Data program at Lawrence Berkeley National Laboratory under Charuleka Varadharajan (Berkeley, CA, USA).

## 1. Introduction
Groundwater level forecasting is operationally relevant for drought monitoring, water resource management, and climate adaptation planning. In practice, measurements are sparse in space but dense in time at instrumented wells. This creates a core challenge: temporal forecasting is available at measured wells, but decision support requires predictions across unmonitored locations.

The goal of this thesis is a "forecast anywhere" pipeline: generate time-series forecasts at monitoring wells and then estimate values at unobserved locations for a fixed geographic region. The spatial step is primarily interpolation in geostatistical terms.

This work starts from replication and extension of the TFT-based groundwater forecasting pipeline by Kunz et al. (2025) and then evaluates GRU alternatives, spatial interpolation setups, split sensitivity, and robust metric design.

## 1.1 Research Question
Is it possible to build a robust groundwater "forecast anywhere" pipeline by combining neural temporal forecasting (e.g., TFT or GRU) with spatial interpolation, and how can this pipeline be improved through fair split design, separate-vs-joint training comparisons, and robust evaluation metrics?

## 1.2 Contributions (Current Target)
1. A modular end-to-end spatiotemporal pipeline from well-level forecasting to spatial interpolation (separate pipeline completed).
2. A reproducible framework for comparing separate vs joint training (joint in progress).
3. A split-sensitivity protocol across multiple spatial split strategies (S1 fixed, S2-S4 in progress).
4. A metric framework covering mean accuracy and high-error tail behavior.

## 1.3 Scope and Non-Claims
This thesis does not claim novelty of all individual components. The contribution is rigorous integration, reproducible evaluation, and systematic analysis in the groundwater setting.

## 2. Related Work and Technical Background
### 2.1 Groundwater Time-Series Forecasting
Kunz et al. (2025) show strong groundwater forecasting performance in Brandenburg with deep sequence models using hydro-meteorological covariates and periodic calendar features (`tag_sin`, `tag_cos`).

### 2.2 Spatial Interpolation
Kriging/GP-style methods are natural spatial heads when future values are available at sparse well locations. The two-stage pipeline is:
1. forecast at observed wells,
2. interpolate to unobserved locations.

### 2.3 Separate vs Joint Training
A central question is whether separate optimization (temporal then spatial) is sufficient or whether joint backpropagation improves overall performance, worst-case behavior, and robustness under harder spatial splits.

## 3. Data and Models
### 3.1 Dataset Snapshot
Current Brandenburg dataset in this repo:
- Rows: 2,142,687
- Wells: 1,040
- Time span: 1951-01-01 to 2024-06-03

### 3.2 Temporal Split (Code-Backed)
- Train: up to 2016-01-01
- Validation: (2016-01-01, 2020-01-01]
- Test: after 2020-01-01

### 3.3 Spatial Split S1 (Current Reference)
- train_fraction = 0.8
- cluster_count = 20
- cluster_method = kmeans
- split_seed = 42
- counts: 841 train wells, 199 holdout wells

### 3.4 Implemented Models
- TFT: `darts.models.TFTModel` (`src/scripts/separate/temporal/kunz_darts/tft_train.py`)
- GRU: custom `GRUSeq2Seq` (`src/scripts/joint/temporal/gru_model.py`)
- Spatial interpolation: `sklearn.gaussian_process.GaussianProcessRegressor` with Matern + White kernels (`src/scripts/separate/spatial/kriging.py`)
- Joint GRU+custom differentiable GP head: planned, not fully completed

## 4. Experimental Protocol
### 4.1 Primary Metrics
- MAE
- RMSE
- NSE (with explicit baseline definition)
- Skill vs persistence
- RMSE p95 across holdout wells
- Failure rate above thresholds

### 4.2 Failure Rate Definition
`FailureRate@tau = (# holdout wells with RMSE_well > tau) / (# holdout wells)`

Initial S1-informed thresholds:
- `FailureRate@2.0` (moderate)
- `FailureRate@5.0` (severe)

### 4.3 Outlier Definition Policy
Primary rule for this draft: top-10% wells by RMSE.

Sensitivity checks (planned/optional in reporting):
- IQR rule: `RMSE > Q3 + 1.5*IQR`
- absolute threshold rule

### 4.4 Baseline Fairness
To avoid leakage, baselines must use only training-available information. Planned baseline families:
1. global train-only baseline
2. k-nearest-neighbor train-well baseline

### 4.5 Planned Split Suite
- S1: clustered split (current)
- S2: random split
- S3: blocked split (20 km x 20 km grid)
- S4: buffered-blocked split (anti-leakage margin)

## 5. Experiments So Far
### 5.1 Completed Runs
1. TFT replication runs (including full 10-seed aligned configuration).
2. S1 spatial interpolation on TFT predictions.
3. S1 outlier analysis (per-well error profiling).
4. GRU pipeline implementation and stabilized evaluation runs on S1.

### 5.2 What Is Not Completed Yet
1. Final split expansion to S2-S4.
2. Full separate-vs-joint benchmark matrix on all final splits.
3. End-to-end joint GRU + custom differentiable GP spatial head experiments.
4. Final metric freeze across all experiments.

### 5.3 Reproducibility Artifacts Already Present
- Config-driven runs (`configs/*.yaml`)
- Split files in `splits/`
- Metrics and figures in `reports/`
- Cluster job specs in `cluster/jobs/`

## 6. Preliminary Results (Current)
### 6.1 TFT Replication
- Kunz reference (horizon 16): NSE approx 0.74, RMSE approx 0.093
- replicated run (horizon 16): NSE approx 0.76, RMSE approx 0.091

Interpretation: replication-level match with slight improvement.

Most plausible contributors:
1. Cleaner scaling/preprocessing (fit only on train)
2. Explicit short-series filtering before training
3. More consistent run protocol and split artifacts
4. Hardware/runtime stack differences

### 6.2 GRU Status
GRU started unstable in early runs, then improved after architecture/pipeline alignment and evaluation fixes. Current S1 runs indicate competitive potential in selected configurations.

### 6.3 Outlier Behavior (S1)
S1 shows clear heavy-tail behavior: median holdout errors are moderate, but a small subset of wells carries very high errors. This motivates tail-risk metrics and proximity-aware analysis.

## 7. Discussion (Draft Direction)
The expected scientific value is protocol quality and robust comparative evidence rather than architectural novelty claims. The analysis focuses on when and why models fail under spatial constraints.

## 8. Project Context and Chronology
### 8.1 Phase 1: Setup and Replication (Dec 2025 - Jan 2026)
- Topic definition: multi-horizon groundwater forecasting plus spatial forecast-anywhere extension.
- Reading and planning based on Kunz et al. and additional literature.
- Initial misunderstanding of replication target and missing Brandenburg-only reference metrics.
- Early sample-first experiments led to misleading intermediate conclusions.
- Multiple crashes and fixes: short-sequence filtering and sample hyperparameter corrections.
- Infrastructure setup and instability: Docker rebuild cycles, GPU environment issues, PVC mode limitations, pod/job interruptions, and cluster contention during long runs.
- Recovery path: obtain Brandenburg metrics reference, align config to Kunz setup, run full 10-seed TFT replication.

### 8.2 Phase 2: Consolidation, Kriging, and GRU (Jan-Feb 2026)
- Repository cleanup and restructuring.
- Spatial interpolation evaluation and outlier analysis for S1.
- GRU implementation matured from unstable early behavior to competitive runs under aligned setup.
- Key fixes beyond the eval-period bug:
  - static-covariate transform usage aligned to non-leaky transform pattern,
  - handling of short series and split-path consistency,
  - evaluation correction to test period,
  - batching-based inference in GRU eval to avoid OOM,
  - run signature standardization and analysis tooling expansion,
  - cluster/deployment hardening and orchestration improvements.

### 8.3 Phase 3: Joint Training and Final Evaluation (Mar-May 2026)
- Full joint GRU plus custom GP-style differentiable spatial head still pending.
- Split sensitivity across S2-S4 still pending.
- Final metric set and threshold policy still being frozen.
- Final separate-vs-joint benchmark matrix still to be executed.