# GIST: Groundwater Interpolation in Space and Time

#### Hybrid Spatiotemporal Modeling for Forecast-Anywhere Groundwater Level Prediction

**Robert Wienröder**, Master's thesis, supervised by Prof. Dr. Felix Bießmann (BHT Berlin,
Cognitive Algorithms Lab) and Dr. Charuleka Varadharajan (Earth and Environmental Sciences
Area, Lawrence Berkeley National Laboratory)

Master's thesis code. The point is to predict groundwater levels at wells that were never
trained on, by combining a temporal model with a spatial one. I explored two ways of doing that:

1. **Two-stage**: a GRU forecasts each well on its own, then a GP interpolates those
   forecasts in space. There's also an "oracle" variant where the GP interpolates real
   observations instead of forecasts, wich basically tells you how good the GP could be at best.
2. **Joint**: the GRU and the GP spatial layer trained end-to-end in one go.

Baseline is a TFT (from darts), following Kunz et al. (2025):
https://doi.org/10.5194/hess-29-3405-2025

The thesis PDF is in the repo root, the LaTeX source is in `thesis/`.

Last update: July 27, 2026

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows: .\.venv\Scripts\Activate.ps1
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Python 3.12, and you really want a GPU (cluster runs used an A100).

**Run everything from the repo root.** A lot of scripts resolve their configs and `../data`
relative to the working directory, so if you cd into a subfolder first they just break.

### Data

The dataset isnt in here, its way too big. You put the paths into `configs/data.yaml`, and
there's a small preview at `data/head_preview.csv` so you can atleast see what it looks like.

Columns you need: `id`, `datum`, `gws`, `tas_5km`, `hurs_5km`, `pr_5km`, `tag_sin`,
`tag_cos`, `x_25833`, `y_25833`

To check you have the same data I did:

```bash
shasum -a 256 /path/to/main_data.parquet   # compare against checksums/data.txt
```

Then:

```bash
python src/prep/ingest.py
python src/prep/summary.py
```

---

## How it fits together

```
data (configs/data.yaml -> merged.parquet)
  -> splits        src/prep/build_coloc_split.py            -> splits/*.csv
  -> experiments   configs/experiments.yaml (60 entries)
                   resolved by src/libs/experiment.py
  -> launch        python cluster/scripts/submit.py <name>   (cluster)
                   python src/scripts/... --experiment <name> (local)
  -> train/eval    src/scripts/pipelines/ + baselines/       -> outputs/
  -> analysis      src/analysis/                             -> reports/
  -> thesis        reports/tables/ + reports/figures/
```

Pretty much everything I report in the thesis goes through the registry in
`configs/experiments.yaml` (the TFT baseline is the one thing that doesnt, more on that
below). An entry just names a base config, a script, and whatever it overrides on top, so an
experiment is basically "base config plus these changes" and I dont have to keep 60 copies
of the same yaml lying around. `src/libs/experiment.py` is what resolves that, and it also
writes the fully merged config and the git commit into every run folder, so if you look at
some result months later you can still find out what exactly produced it.

```bash
python cluster/scripts/submit.py --list
python cluster/scripts/submit.py <entry from registry> --dry-run
python cluster/scripts/submit.py --analysis src/analysis/tables/compute_nse_hw.py
```

Entries that have a `seeds:` field (e.g. `42-81`) turn into one job per seed. And if an
entry needs something built before it can run, like a split or an oracle prediction file,
that gets listed under `prebuild:` and then runs as its own step before the training starts.
The builders check if the output is already there and skip themselves in that case, so
re-running a job doesnt hurt.

---

## Models

You just need to hand over the name from the registry as an argument to the submit script,
then it finds the correct script automatically. `submit.py <entry>` looks up the `script:`
field, tacks the eval step on after training, and runs any `prebuild:` steps first. So GRU,
GP and joint all launch the same way:

```bash
python cluster/scripts/submit.py <entry from registry>
```

If you'd rather run something locally than on the cluster, you can also just call the script
that the entry points at and hand over the same name, like
`python src/scripts/pipelines/temporal/gru_train.py --experiment <entry from registry>`.
Thats basically the only reason you'd need to know the paths below.

The TFT is a bit different, its older than the registry so there is no entry for it at all
and the scripts dont even take an `--experiment` argument, they just read
`configs/baselines/tft.yaml`. So there you run the scripts directly.

### GRU (temporal)

`src/scripts/pipelines/temporal/` (`gru_train.py`, `gru_eval.py`), config
`configs/gru/gru.yaml`, outputs in `outputs/GRU_FCOV/<RUN_NAME>/`.

### GP (spatial)

`src/scripts/pipelines/spatial/gp_eval.py`, config `configs/gp/gp.yaml`. Matérn-3/2 kernel
by default. For the true-observation baseline the entry points it at an oracle prediction
file that gets built by `src/prep/build_oracle_pred.py` and sets `model_prefix: ORACLE`.
Outputs in `outputs/gp/<RUN_TAG>/` as `gp_pred.parquet` and `gp_metrics.parquet`.

### Joint

`src/scripts/pipelines/joint/` (`gru_gp_train.py`, `gru_gp_eval.py`), config
`configs/joint/gru_gp.yaml`, outputs in `outputs/GRU_GP_JOINT/<RUN_NAME>/`.

### TFT baseline

```bash
python src/scripts/baselines/tft_train.py
python src/scripts/baselines/tft_eval.py
python src/scripts/baselines/tft_runs.py   # multi-seed
```

Config `configs/baselines/tft.yaml`, outputs in `outputs/TFT/<RUN_NAME>/`.

### Running it without the cluster

`submit.py` only builds the Kubernetes jobs, you dont need it for anything else. Without
cluster access you just call the scripts directly. With `--experiment <name>` you get the
exact setup one of my reported runs used, and without it the script falls back to `--config`
and just runs the plain base config.

The only thing you lose is the `prebuild:` steps, those only run through `submit.py`. So if
a split is missing you build it yourself first, the entry tells you which builder and which
arguments.

---

## HPO

The HPO runs are in the registry too, but they work a bit differently. Instead of a
`script:` field they have a `command:` field with the full command line in it, wich
`submit.py` then just runs as it is. So they dont go through the config resolving, they get
their search space handed over with `--hpo-config`. Launching is the same though:

```bash
python cluster/scripts/submit.py gru_hpo             # configs/gru/hpo_gru.yaml
python cluster/scripts/submit.py gp_hpo_rounds1_3    # configs/gp/hpo_gp_rounds1-3.yaml
python cluster/scripts/submit.py gp_hpo_round4       # configs/gp/hpo_gp_round4.yaml
python cluster/scripts/submit.py gp_lr_check_lr01    # lr robustness check
```

`gp_lr_check_lr01` is the exception here, thats a normal entry with a `base:` and a
`script:` and a seed range, so it goes the usual way.

Round 4 is the re-validation on the leakage-free hpo90 split and it's self-contained, so it
doesnt depend on the earlier rounds. The results land in `reports/metrics/gru/hpo/` and
`reports/metrics/gp/hpo/`.

---

## Splits

`splits/` has one CSV per split, mapping the well id to train/holdout (the three-way ones
also have a val set). The filenames matter here, because the registry points at them
directly:

```
spatial_split_full_merged_<type>_f<trainfrac>_ss<seed>[_joint].csv
```

- main experiment family: `coloc_rand90_f90_ss42..82` (+ `_joint` mirrors), 104 holdout wells
- `md<NN>` = max-distance-percentile splits, `km` = k-means, `far80` = far holdout
- all of them drop wells sitting within 8 m of another well, those are duplicates

Splits never get generated during training anymore. If a split file is missing then the run
just stops and tells you to build it first. I did that on purpose, because before that a
config mismatch between train and eval could quietly hand you two different holdout sets
and you'd never even notice. So you build them yourself:

```bash
python src/prep/build_coloc_split.py --output splits/<name>.csv \
    --density-percentile 30 --mode two-way --n-holdout 52
```

The `--input` defaults to the cluster path, so locally you have to add
`--input ../data/merged.parquet`.

Why the density threshold is what it is, thats all explained in
`src/analysis/constrained_split_design.ipynb`. The one-off scripts that made the older
splits are parked in `src/prep/archive/`, so you can still see where those CSVs came from.

---

## Analysis & reports

The analysis scripts are sorted by what they spit out:

```
src/analysis/
  tables/            thesis numbers, stats, per-model reports  -> reports/tables, reports/metrics
  figures/           all figure generators                     -> reports/figures
    architecture/    flowcharts and the pipeline diagram
    data/            data chapter figures, well maps
    results/         result scatters, horizon plots
    splits/          split maps
  model_selection/   hpo tables + ablations                    -> reports/metrics
  unused/            kept around for reference, feeds nothing in the thesis
```

The ones you'd actually run:

```bash
python src/analysis/tables/collect_thesis_numbers.py    # the reported numbers
python src/analysis/tables/compute_paired_ttests.py     # the significance tests
python src/analysis/tables/log_metrics_harvest.py       # scrapes run logs -> reports/tables/
python src/analysis/tables/gru_report.py                # per-model summaries
python src/analysis/figures/results/gen_thesis_v5_figures.py
```

Most of them just print everything to stdout, they dont write any files.

And `reports/` mirrors that:

```
reports/
  dataset/    coverage, metadata counts, the data summary
  tables/     result tables the thesis quotes
  metrics/    per-model metrics + hpo results (gp/, gru/, tft/)
  figures/    by topic: architectures, data, gru, splits, predictions,
              spatial_distance, joint_vs_two_stage, temporal_gru_vs_tft,
              trueobs_baseline_vs_two_stage, _dead_ends
```

`thesis/src/figures/` is the frozen set that the submitted PDF was built from, the
generators never write in there. So if you regenerate something it goes into
`reports/figures/` and then you copy it over by hand.

---

## Cluster (Kubernetes)

```bash
python cluster/scripts/submit.py <experiment> [--seeds 42-81] [--set key=value] [--dry-run]
```

`submit.py` takes one of the two templates in `cluster/jobs/` (`gw-run.yaml` for training,
`gw-analysis.yaml` for analysis scripts), stamps the actual commands into it and applies it.
There are no per-experiment job files anymore, every experiment, HPO run and robustness
check is just a registry entry, and anything you want to sweep besides the seed you hand
over with `--set`.

Image `row56/gw-pred:py312-cu124`, storage mounted at `/storage` via PVC `gw-pred-rwx`.
The Dockerfile and the yaml files for the PVC and the deployment are in `cluster/docker/`,
`cluster/pvc/` and `cluster/deployments/`.

---

## Layout

```
configs/          data.yaml, experiments.yaml (the registry), per-model configs
src/
  libs/           experiment resolver, run registry, spatial split, metrics
  prep/           ingest, summary, split + oracle builders, archive/ of one-offs
  scripts/
    pipelines/    temporal/ = GRU, spatial/ = GP, joint/ = end-to-end
    baselines/    TFT
  analysis/       tables/, figures/, model_selection/, unused/
splits/           split CSVs (the names matter)
reports/          dataset/, tables/, metrics/, figures/
cluster/          submit.py, the two job templates, docker stuff + k8s yamls
thesis/           LaTeX source of the thesis
outputs/          model outputs, gitignored (run dir names matter)
```
