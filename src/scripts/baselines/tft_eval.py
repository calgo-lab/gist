from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import os, glob, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from darts.models import TFTModel
from libs.utils import get_metrics
from libs.spatial_split import load_split, resolve_split_path
from libs.run_sig_tft import resolve_tft_run_sig
import torch, yaml

torch.backends.cudnn.benchmark = True
torch.set_default_dtype(torch.float32)

DATA_CFG = yaml.safe_load(Path("configs/data.yaml").read_text(encoding="utf-8"))
if not isinstance(DATA_CFG, dict):
    DATA_CFG = {}

TFT_CFG = yaml.safe_load(Path("configs/baselines/tft.yaml").read_text(encoding="utf-8"))
if not isinstance(TFT_CFG, dict):
    TFT_CFG = {}

DATASET = TFT_CFG.get("dataset", "sample")
if DATASET == "sample":
    DATA_FILE = DATA_CFG["sample_path"]
elif DATASET == "full_merged":
    DATA_FILE = DATA_CFG["full_merged_path"]
elif DATASET == "full_raw":
    DATA_FILE = DATA_CFG["full_raw_path"]
else:
    raise ValueError(f"Unknown DATASET={DATASET}")

OUTPUTS_ROOT = str(ROOT / "outputs")
os.makedirs(OUTPUTS_ROOT, exist_ok=True)
SPLITS_ROOT = ROOT / "splits"
SPLITS_ROOT.mkdir(parents=True, exist_ok=True)

STATIC_FEATURE_REGEX = (
    'eumohp_(.+)_(.+)_(.*[1])'
    '|shannongeom10kmsha|entgeom10kment|unigeom10kmuni'
    '|gwn|huek250.+_(kf).+|corine|twi'
)

training_cfg = TFT_CFG.get("training", {}) if isinstance(TFT_CFG.get("training", {}), dict) else {}
data_cfg = TFT_CFG.get("data", {}) if isinstance(TFT_CFG.get("data", {}), dict) else {}
eval_cfg = TFT_CFG.get("eval", {}) if isinstance(TFT_CFG.get("eval", {}), dict) else {}
spatial_cfg = TFT_CFG.get("spatial_split", {}) if isinstance(TFT_CFG.get("spatial_split", {}), dict) else {}
ENV_SEED = int(training_cfg.get("seed", 40))
ENV_IN_LEN = int(data_cfg.get("in_len", 52))
ENV_OUT_LEN = int(data_cfg.get("out_len", 16))
ENV_STATICS = bool(data_cfg.get("statics", False))
ENV_STRIDE = int(eval_cfg.get("stride", 1))
ENV_RETRAIN = bool(eval_cfg.get("retrain", False))
ENV_FORCE = bool(eval_cfg.get("force_reeval", False))
SPATIAL_FRACTION = float(spatial_cfg.get("train_fraction", 0.1))
SPATIAL_CLUSTERS = int(spatial_cfg.get("cluster_count", 10))
SPATIAL_SEED = int(spatial_cfg.get("split_seed", 42))
_exclude_raw = spatial_cfg.get(
    "exclude_terms",
    "geometry,x_,y_,lon,lat,koord",
)
SPATIAL_EXCLUDES = tuple(
    t.strip().lower() for t in _exclude_raw.split(",") if t.strip()
)

if __name__ == "__main__":
    gws_bb = (
        pd.read_csv(DATA_FILE, low_memory=False)
        if str(DATA_FILE).lower().endswith(".csv")
        else pq.read_table(DATA_FILE).to_pandas()
    )
    gws_bb['datum'] = pd.to_datetime(gws_bb['datum']).astype('datetime64[ns]')

    for c in ['gws','tas_5km','hurs_5km','pr_5km','tag_sin','tag_cos']:
        if c in gws_bb.columns:
            gws_bb[c] = gws_bb[c].astype('float32')
    for c in gws_bb.filter(regex=STATIC_FEATURE_REGEX).columns:
        gws_bb[c] = gws_bb[c].astype('float32')

    total_ids = gws_bb['id'].nunique()
    spatial_split_path = resolve_split_path(SPLITS_ROOT, DATASET, spatial_cfg)
    gws_bb, _spatial_info = load_split(
        gws_bb,
        static_regex=STATIC_FEATURE_REGEX,
        train_fraction=SPATIAL_FRACTION,
        n_clusters=SPATIAL_CLUSTERS,
        rng_seed=SPATIAL_SEED,
        exclude_terms=SPATIAL_EXCLUDES,
        save_path=spatial_split_path,
    )
    kept_ids = gws_bb['id'].nunique()
    share = 0.0 if total_ids == 0 else kept_ids / total_ids
    print(f"Spatial eval subset keeps {kept_ids}/{total_ids} wells ({share:.1%}).")

    static_features = None
    if ENV_STATICS:
        static_features = list(gws_bb.filter(regex=(
            'eumohp_(.+)_(.+)_(.*[1])'
            '|shannongeom10kmsha|entgeom10kment|unigeom10kmuni'
            '|gwn|huek250.+_(kf).+|corine|twi'
        )))

    gws_series = TimeSeries.from_group_dataframe(
        df=gws_bb,
        group_cols='id',
        time_col='datum',
        value_cols='gws',
        static_cols=static_features if ENV_STATICS else None,
        drop_group_cols='id'
    )
    cov_series = TimeSeries.from_group_dataframe(
        df=gws_bb,
        group_cols='id',
        time_col='datum',
        value_cols=['tas_5km','hurs_5km','pr_5km','tag_sin','tag_cos'],
        drop_group_cols='id'
    )

    train_cutoff = pd.Timestamp('20160101')
    val_cutoff = pd.Timestamp('20200101')

    train_gws, val_gws, test_gws = [], [], []
    train_cov, val_cov, test_cov = [], [], []

    for s in gws_series:
        _train, _rest = s.split_after(train_cutoff)
        _val, _test = _rest.split_after(val_cutoff)
        train_gws.append(_train)
        val_gws.append(_val)
        test_gws.append(_test)

    for s in cov_series:
        _train, _rest = s.split_after(train_cutoff)
        _val, _test = _rest.split_after(val_cutoff)
        train_cov.append(_train)
        val_cov.append(_val)
        test_cov.append(_test)

    std = StandardScaler()
    ts_scaler = Scaler(std)
    train_gws = ts_scaler.fit_transform(train_gws)
    val_gws   = ts_scaler.transform(val_gws)
    test_gws  = ts_scaler.transform(test_gws)
    gws_series= ts_scaler.transform(gws_series)

    if ENV_STATICS:
        stat_tx = StaticCovariatesTransformer(transformer_num=std)
        train_gws = stat_tx.fit_transform(train_gws)
        val_gws   = stat_tx.transform(val_gws)
        test_gws  = stat_tx.transform(test_gws)
        gws_series= stat_tx.transform(gws_series)

    cov_scaler = Scaler(StandardScaler())
    train_cov = cov_scaler.fit_transform(train_cov)
    val_cov   = cov_scaler.transform(val_cov)
    test_cov  = cov_scaler.transform(test_cov)

    train_gws_32, val_gws_32, test_gws_32, gws_series_32 = [], [], [], []
    for seq, out in [(train_gws, train_gws_32), (val_gws, val_gws_32), (test_gws, test_gws_32), (gws_series, gws_series_32)]:
        for s in seq:
            sc = s.static_covariates
            if sc is not None:
                sc = sc.astype(np.float32)
                s = s.with_static_covariates(sc)
            s = s.astype(np.float32)
            out.append(s)

    train_cov_32 = [s.astype(np.float32) for s in train_cov]
    val_cov_32   = [s.astype(np.float32) for s in val_cov]
    test_cov_32  = [s.astype(np.float32) for s in test_cov]

    train_gws = train_gws_32
    val_gws = val_gws_32
    test_gws = test_gws_32
    gws_series = gws_series_32
    train_cov = train_cov_32
    val_cov = val_cov_32
    test_cov = test_cov_32

    len_pred = (
        gws_bb[gws_bb['datum'] >= val_cutoff]
        .groupby('id')['datum'].size().unique()[0]
        - ENV_OUT_LEN - ENV_IN_LEN + 1
    )

    start_time = val_gws[1].end_time() + val_gws[1].freq + pd.Timedelta(weeks=ENV_IN_LEN)

    in_len, out_len = ENV_IN_LEN, ENV_OUT_LEN

    ids_all = pd.unique(gws_bb['id'])
    ids_keep = list(ids_all)
    lookup_ids = pd.DataFrame(ids_keep, columns=['id']).reset_index()

    print(f"Eval using date-based split; in_len={in_len}, out_len={out_len}")

    model_variants = ['dyn_stat_5km'] if ENV_STATICS else ['dyn_5km']
    seeds = [ENV_SEED]
    nn_architecture = [('TFT', TFTModel)]
    output_chunk_length = ENV_OUT_LEN

    for seed in seeds:
        for variant in model_variants:
            for model_arch, model_class in nn_architecture:
                run_sig = resolve_tft_run_sig(TFT_CFG, model="TFT")

                eval_name = f"{model_arch}_{run_sig}"
                trained_name = f"{model_arch}_{'dyn_stat_5km' if ENV_STATICS else 'dyn_5km'}_{ENV_SEED}"

                run_dir = Path(OUTPUTS_ROOT) / model_arch / eval_name
                pred_dir = run_dir / "predictions"
                run_dir.mkdir(parents=True, exist_ok=True)
                pred_dir.mkdir(parents=True, exist_ok=True)

                PREDICTION_FILE = str(pred_dir / "pred.parquet")
                METRICS_FILE    = str(run_dir / "metrics.parquet")
                SKILL_FILE      = str(run_dir / "skill_by_horizon.parquet")

                pred_mtime    = os.path.getmtime(PREDICTION_FILE) if os.path.exists(PREDICTION_FILE) else 0.0
                metrics_mtime = os.path.getmtime(METRICS_FILE) if os.path.exists(METRICS_FILE) else 0.0

                force = ENV_FORCE
                need_preds   = force or (not os.path.exists(PREDICTION_FILE))
                need_metrics = force or (not os.path.exists(METRICS_FILE)) or (pred_mtime >= metrics_mtime)

                did_preds = False

                if need_preds:
                    print(f'Calculate predictions for {eval_name}')

                    kind = None
                    art_path = None
                    trained_dir = Path(OUTPUTS_ROOT) / model_arch / eval_name
                    for name in (eval_name, trained_name):
                        cand_dir = Path(OUTPUTS_ROOT) / model_arch / name
                        ckpt_dir = cand_dir / "checkpoints"
                        for pat in (
                            str(ckpt_dir / "best-epoch*.ckpt"),
                            str(ckpt_dir / "last-epoch*.ckpt"),
                            str(ckpt_dir / "*.ckpt"),
                        ):
                            hits = sorted(glob.glob(pat), key=os.path.getmtime)
                            if hits:
                                art_path = hits[-1]
                                trained_dir = cand_dir
                                kind = "base" if art_path.endswith(".pth.tar") else "ckpt"
                                break
                        if art_path is None and (cand_dir / "_model.pth.tar").exists():
                            art_path = str(cand_dir / "_model.pth.tar")
                            trained_dir = cand_dir
                            kind = "base"
                        if art_path is not None:
                            break

                    if art_path is None:
                        raise FileNotFoundError(
                            f"No checkpoint or base model in {trained_dir}. "
                            f"Expected checkpoints/*.ckpt or _model.pth.tar"
                        )

                    if kind == "ckpt":
                        ckpt = Path(art_path)
                        work_dir = ckpt.parents[2] if ckpt.parent.name == "checkpoints" else trained_dir.parent
                        model = model_class.load_from_checkpoint(
                            work_dir=str(work_dir),
                            model_name=trained_dir.name,
                            file_name=ckpt.name,
                        )
                    else:
                        model = model_class.load(str(art_path))

                    try:
                        model.model = model.model.float()
                    except Exception:
                        pass
                    model.trainer_params["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"
                    model.trainer_params["devices"] = 1
                    model.trainer_params["precision"] = 32

                    t0 = time.perf_counter()
                    backtest_gws = model.historical_forecasts(
                        series=gws_series,
                        past_covariates=test_cov,
                        future_covariates=test_cov,
                        start=start_time,
                        forecast_horizon=output_chunk_length,
                        stride=1,
                        last_points_only=False,
                        retrain=False,
                        overlap_end=False,
                        verbose=True,
                        predict_likelihood_parameters=False,
                    )
                    t1 = time.perf_counter()
                    print(f"eval_predict_s={round(t1 - t0, 3)}")
                    backtest_gws_real = ts_scaler.inverse_transform(backtest_gws)

                    pred_final = []
                    for idx, preds in enumerate(backtest_gws_real):
                        rows = []
                        for i in range(len_pred):
                            ts = preds[i]
                            df = ts.to_dataframe().reset_index().rename(columns={'time': 'datum'})
                            df['index'] = idx
                            df['startzeitpunkt'] = df['datum'].min()
                            rows.append(df)
                        pred_final.append(pd.concat(rows, ignore_index=True))
                    pred_final = pd.concat(pred_final, ignore_index=True)
                    pq.write_table(pa.Table.from_pandas(pred_final), PREDICTION_FILE)

                    did_preds = True
                    pred_mtime = os.path.getmtime(PREDICTION_FILE) if os.path.exists(PREDICTION_FILE) else 0.0

                if did_preds or need_metrics:
                    print(f"Calculate metrics for {eval_name}")
                    pred_final = pq.read_table(PREDICTION_FILE).to_pandas()
                    pred_final = pred_final.rename(columns={'gws': 'gws_forecast'})
                    pred_final['datum'] = pd.to_datetime(pred_final['datum'])
                    pred_final['startzeitpunkt'] = pd.to_datetime(pred_final['startzeitpunkt'])

                    pred_final = pred_final.merge(lookup_ids, on='index')
                    pred_final = pred_final.merge(gws_bb[['id','datum','gws']], on=['id','datum'])

                    pred_final['horizon'] = (
                        (pred_final['datum'] - pred_final['startzeitpunkt'])/pd.Timedelta(weeks=1)
                    ) + 1

                    train_means = (
                        gws_bb[gws_bb['datum'] < train_cutoff]
                        .groupby('id')['gws'].mean()
                        .to_dict()
                    )
                    metrics = get_metrics(
                        prediction_df=pred_final,
                        real_col='gws', forecast_col='gws_forecast', id_col='id',
                        metrics_subset=['nRMSE','RMSE','NSE','KGE','rMBE','MAE'],
                        lower_quantile=None, upper_quantile=None,
                        train_means=train_means,
                    )

                    bb = gws_bb.sort_values(['id','datum']).copy()
                    bb['startzeitpunkt'] = bb.groupby('id')['datum'].shift(-1)
                    persist = (
                        bb[['id','startzeitpunkt','gws']]
                        .dropna()
                        .rename(columns={'gws': 'persist_base'})
                    )

                    dfp = pred_final.merge(persist, on=['id','startzeitpunkt'], how='left')
                    if dfp['persist_base'].isna().any():
                        dfp = dfp.dropna(subset=['persist_base'])

                    rows = []
                    for h, g in dfp.groupby(dfp['horizon'].astype(int)):
                        a = np.asarray(g['gws'], float)
                        b = np.asarray(g['gws_forecast'], float)
                        m = np.isfinite(a) & np.isfinite(b)
                        if m.sum() == 0:
                            r_m = np.nan
                        else:
                            r_m = float(np.sqrt(np.mean((a[m]-b[m])**2)))

                        a = np.asarray(g['gws'], float)
                        b = np.asarray(g['persist_base'], float)
                        m = np.isfinite(a) & np.isfinite(b)
                        if m.sum() == 0:
                            r_p = np.nan
                        else:
                            r_p = float(np.sqrt(np.mean((a[m]-b[m])**2)))

                        if (not np.isfinite(r_p)) or r_p == 0.0:
                            skill = np.nan
                        else:
                            skill = 1.0 - (r_m / r_p)
                        rows.append({'horizon': int(h), 'skill_vs_persistence': float(skill) if np.isfinite(skill) else np.nan})

                    out_df = pd.DataFrame(rows).sort_values('horizon')
                    pq.write_table(pa.Table.from_pandas(out_df), SKILL_FILE)
                    pq.write_table(pa.Table.from_pandas(metrics), METRICS_FILE)
