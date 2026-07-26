import math
import os
import re
import gc
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = str(ROOT / "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import numpy as np
import pandas as pd
import pickle
import pyarrow.parquet as pq
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging, LearningRateMonitor
from pytorch_lightning import seed_everything

from sklearn.preprocessing import StandardScaler
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from darts.models import TFTModel, TiDEModel
from darts.utils.likelihood_models import QuantileRegression
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, R2Score
import yaml

import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

from libs.spatial_split import load_split, resolve_split_path
from libs.run_sig_tft import resolve_tft_run_sig

DATA_CFG = yaml.safe_load(Path("configs/data.yaml").read_text(encoding="utf-8"))
if not isinstance(DATA_CFG, dict):
    DATA_CFG = {}

TFT_CFG = yaml.safe_load(Path("configs/baselines/tft.yaml").read_text(encoding="utf-8"))
if not isinstance(TFT_CFG, dict):
    TFT_CFG = {}

DATASET = TFT_CFG.get("dataset", "full_raw")
if DATASET == "sample":
    DATA_FILE = DATA_CFG["sample_path"]
elif DATASET == "full_raw":
    DATA_FILE = DATA_CFG["full_raw_path"]
elif DATASET == "full_merged":
    DATA_FILE = DATA_CFG["full_merged_path"]
else:
    raise ValueError(f"Unknown DATASET={DATASET}")

OUTPUT_PATH = ROOT / "outputs"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
SPLITS_ROOT = ROOT / "splits"
SPLITS_ROOT.mkdir(parents=True, exist_ok=True)

STATIC_FEATURE_REGEX = (
    'eumohp_(.+)_(.+)_(.*[1])'
    '|shannongeom10kmsha'
    '|entgeom10kment'
    '|unigeom10kmuni'
    '|gwn'
    '|huek250.+_(kf).+'
    '|corine'
    '|twi'
)

if __name__ == '__main__':

    training_cfg = TFT_CFG.get("training", {}) if isinstance(TFT_CFG.get("training", {}), dict) else {}
    data_cfg = TFT_CFG.get("data", {}) if isinstance(TFT_CFG.get("data", {}), dict) else {}
    model_cfg = TFT_CFG.get("model", {}) if isinstance(TFT_CFG.get("model", {}), dict) else {}
    spatial_cfg = TFT_CFG.get("spatial_split", {}) if isinstance(TFT_CFG.get("spatial_split", {}), dict) else {}
    SEED = int(training_cfg.get("seed", 40))
    seeds = [SEED]
    use_static_features = bool(data_cfg.get("statics", True))
    n_epochs = int(training_cfg.get("epochs", 50))
    batch_size = int(training_cfg.get("batch_size", 4096))
    num_workers = int(training_cfg.get("num_workers", 10))
    early_stop_cfg = training_cfg.get("early_stopping", {}) if isinstance(training_cfg.get("early_stopping", {}), dict) else {}
    dropout = float(model_cfg.get("dropout", 0.2))

    spatial_fraction = float(spatial_cfg.get("train_fraction", 0.8))
    spatial_clusters = int(spatial_cfg.get("cluster_count", 10))
    spatial_seed = int(spatial_cfg.get("split_seed", 42))
    _exclude_raw = spatial_cfg.get("exclude_terms", "geometry,x_,y_,lon,lat,koord")
    spatial_excludes = tuple(
        t.strip().lower() for t in _exclude_raw.split(",") if t.strip()
    )
    torch.multiprocessing.freeze_support()

    gws_full = (
        pd.read_csv(DATA_FILE, low_memory=False)
        if str(DATA_FILE).lower().endswith(".csv")
        else pq.read_table(DATA_FILE).to_pandas()
    )
    if 'datum' in gws_full.columns:
        gws_full['datum'] = pd.to_datetime(gws_full['datum']).astype('datetime64[ns]')

    spatial_split_path = resolve_split_path(SPLITS_ROOT, DATASET, spatial_cfg)
    gws_spatial, _spatial_info = load_split(
        gws_full,
        static_regex=STATIC_FEATURE_REGEX,
        train_fraction=spatial_fraction,
        n_clusters=spatial_clusters,
        rng_seed=spatial_seed,
        exclude_terms=spatial_excludes,
        save_path=spatial_split_path,
    )
    kept = gws_spatial['id'].nunique()
    total = gws_full['id'].nunique()
    share = 0.0 if total == 0 else kept / total
    print(f"Spatial clustering kept {kept} / {total} wells ({share:.1%} of ids).")

    gws_bb = gws_spatial.copy()
    gws_bb['datum'] = pd.to_datetime(gws_bb['datum']).astype('datetime64[ns]')

    if use_static_features:
        static_features = list(gws_bb.filter(regex=STATIC_FEATURE_REGEX))
    else:
        static_features = None

    gws_series = TimeSeries.from_group_dataframe(
        df=gws_bb,
        group_cols='id',
        time_col='datum',
        value_cols='gws',
        static_cols=static_features if use_static_features else None,
        drop_group_cols='id'
    )

    cov_series = TimeSeries.from_group_dataframe(
        df=gws_bb,
        group_cols='id',
        time_col='datum',
        value_cols=['tas_5km', 'hurs_5km', 'pr_5km', 'tag_sin', 'tag_cos'],
        drop_group_cols='id'
    )

    train_cutoff = pd.Timestamp('20160101')
    val_cutoff = pd.Timestamp('20200101')

    train_gws = []
    val_gws = []
    for data in gws_series:
        _train, _val = data.split_after(train_cutoff)
        _val, _ = _val.split_after(val_cutoff)
        train_gws.append(_train)
        val_gws.append(_val)

    train_cov = []
    val_cov = []
    for data in cov_series:
        _train, _val = data.split_after(train_cutoff)
        _val, _ = _val.split_after(val_cutoff)
        train_cov.append(_train)
        val_cov.append(_val)

    standard_scaler = StandardScaler()

    transformer = Scaler(standard_scaler)
    train_gws = transformer.fit_transform(train_gws)
    val_gws = transformer.transform(val_gws)
    gws_series = transformer.transform(gws_series)

    if use_static_features:
        transfomer_categorical = StaticCovariatesTransformer(transformer_num=standard_scaler)
        train_gws = transfomer_categorical.fit_transform(train_gws)
        val_gws = transfomer_categorical.transform(val_gws)
        gws_series = transfomer_categorical.transform(gws_series)

    transformer_cov = Scaler(standard_scaler)
    train_cov = transformer_cov.fit_transform(train_cov)
    val_cov = transformer_cov.transform(val_cov)

    IN_LEN = int(data_cfg.get("in_len", 52))
    OUT_LEN = int(data_cfg.get("out_len", 16))
    min_train_len = IN_LEN + OUT_LEN + (OUT_LEN - 1)
    keep = [
        i for i in range(len(train_gws))
        if len(train_gws[i]) >= min_train_len and len(val_gws[i]) >= OUT_LEN
    ]
    if not keep:
        raise ValueError(f"No series long enough. Need ≥ {min_train_len} in train and ≥ {OUT_LEN} in val.")
    sel = lambda L: [L[i] for i in keep]
    train_gws, val_gws = sel(train_gws), sel(val_gws)
    train_cov, val_cov = sel(train_cov), sel(val_cov)

    for seed in seeds:
        run_sig = resolve_tft_run_sig(TFT_CFG, model="TFT")

        for model_arch, model_class in [('TFT', TFTModel)]:

            seed_everything(seed, workers=True)

            if use_static_features:
                model_arch_seed = f'{model_arch}_dyn_stat_5km_{seed}'
            else:
                model_arch_seed = f'{model_arch}_dyn_5km_{seed}'

            MODEL_PATH_MAIN = os.path.join(OUTPUT_PATH, model_arch)
            os.makedirs(MODEL_PATH_MAIN, exist_ok=True)

            torch_metrics = MetricCollection(
                [MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()]
            )

            lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
            lr_scheduler_kwargs = {
                'monitor': 'train_loss',
                'mode': 'min',
                'factor': 0.999,
                'patience': 2
            }
            lr_logger = LearningRateMonitor(logging_interval='step')
            early_stop = EarlyStopping(
                monitor=early_stop_cfg.get("monitor", "val_loss"),
                patience=int(early_stop_cfg.get("patience", 5)),
                min_delta=float(early_stop_cfg.get("min_delta", 0.0)),
                mode=early_stop_cfg.get("mode", "min"),
            )

            optimizer_kwargs = {
                'lr': 3e-4
            }

            pl_trainer_kwargs = {
                'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
                'devices': 1,
                'val_check_interval': 1.0,
                'log_every_n_steps': 10,
                'enable_model_summary': True,
                'enable_checkpointing': True,
                'callbacks': [lr_logger, early_stop],
                'gradient_clip_val': 1,
                'num_nodes': 1
            }

            params = {
                'input_chunk_length': IN_LEN,
                'output_chunk_length': OUT_LEN,
                'use_reversible_instance_norm': True,
                'pl_trainer_kwargs': pl_trainer_kwargs,
                'optimizer_kwargs': optimizer_kwargs,
                'likelihood': None,
                'loss_fn': torch.nn.MSELoss(),
                'save_checkpoints': True,
                'force_reset': True,
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'dropout': dropout,
                'log_tensorboard': True,
                'torch_metrics': torch_metrics,
                'lr_scheduler_cls': lr_scheduler_cls,
                'lr_scheduler_kwargs': lr_scheduler_kwargs
            }

            params["pl_trainer_kwargs"]["default_root_dir"] = str(MODEL_PATH_MAIN)
            params["pl_trainer_kwargs"]["logger"] = False
            callbacks = params["pl_trainer_kwargs"].get("callbacks", [])
            callbacks = [
                cb for cb in callbacks
                if not isinstance(cb, LearningRateMonitor)
            ]
            params["pl_trainer_kwargs"]["callbacks"] = callbacks

            model = model_class(
                **params,
                model_name=model_arch_seed,
                work_dir=MODEL_PATH_MAIN
            )

            model.fit(
                series=train_gws,
                past_covariates=train_cov,
                future_covariates=train_cov if model_arch in ['TFT', 'TiDE'] else None,
                val_series=val_gws,
                val_past_covariates=val_cov,
                val_future_covariates=val_cov if model_arch in ['TFT', 'TiDE'] else None,
                verbose=True,
                dataloader_kwargs={
                    'num_workers': num_workers
                }
            )

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
