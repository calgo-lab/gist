import itertools
import os
import subprocess
import sys
import time
import pathlib
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
PY = sys.executable


def _load_cfg(path):
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _cfg_get(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


TFT_CONFIG = "configs/baselines/tft.yaml"
TFT_CFG = _load_cfg(TFT_CONFIG)

DATASET = _cfg_get(TFT_CFG, "dataset", default="full_merged")
_m = _cfg_get(TFT_CFG, "model", default="TFT")
MODEL = _m if isinstance(_m, str) else "TFT"
TRAIN = ROOT / "src/scripts/baselines/tft_train.py"
EVAL  = ROOT / "src/scripts/baselines/tft_eval.py"

IN_LENS = _cfg_get(TFT_CFG, "runs", "in_lens", default=[_cfg_get(TFT_CFG, "data", "in_len", default=52)])
OUT_LENS = _cfg_get(TFT_CFG, "runs", "out_lens", default=[_cfg_get(TFT_CFG, "data", "out_len", default=16)])
STATICS = _cfg_get(TFT_CFG, "runs", "statics", default=[_cfg_get(TFT_CFG, "data", "statics", default=True)])
EPOCHS = _cfg_get(TFT_CFG, "runs", "epochs", default=[_cfg_get(TFT_CFG, "training", "epochs", default=50)])
BATCH_SIZE = int(_cfg_get(TFT_CFG, "runs", "batch_size", default=_cfg_get(TFT_CFG, "training", "batch_size", default=4096)))

SPATIAL_TRAIN_FRACTION = float(_cfg_get(TFT_CFG, "spatial_split", "train_fraction", default=0.8))
SPATIAL_CLUSTER_COUNT = int(_cfg_get(TFT_CFG, "spatial_split", "cluster_count", default=20))
SPATIAL_SPLIT_SEED = int(_cfg_get(TFT_CFG, "spatial_split", "split_seed", default=42))
SPATIAL_EXCLUDE_TERMS = _cfg_get(
    TFT_CFG, "spatial_split", "exclude_terms",
    default="geometry,x_25833,y_25833"
)

SEEDS = _cfg_get(TFT_CFG, "runs", "seeds", default=[_cfg_get(TFT_CFG, "training", "seed", default=40)])


LOG = ROOT / "reports/metrics/tft/train_log.csv"
LOG.parent.mkdir(parents=True, exist_ok=True)

def run_sig(in_len, out_len, epochs, statics, seed, dataset, bs):
    spf = str(SPATIAL_TRAIN_FRACTION).replace(".", "p")
    return (
        f"in{in_len}_out{out_len}_ep{epochs}_bs{bs}_stat{int(statics)}_seed{seed}_{dataset}"
        f"_spf{spf}_sc{SPATIAL_CLUSTER_COUNT}_ss{SPATIAL_SPLIT_SEED}"
    )

def write_header_if_needed(path):
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(
            "in_len,out_len,statics,epochs,seed,train_s,eval_s,ok,"
            "rmse_h4,skill_h1,skill_h2,skill_h3,skill_h4\n"
        )

def append_row(path, row):
    with path.open("a") as f:
        f.write(row + "\n")

def _write_tft_cfg(cfg):
    pathlib.Path(TFT_CONFIG).write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )


def _with_run_params(base_cfg, in_len, out_len, epochs, statics, seed, dataset, bs):
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    cfg["dataset"] = dataset
    cfg.setdefault("data", {})
    cfg["data"]["in_len"] = in_len
    cfg["data"]["out_len"] = out_len
    cfg["data"]["statics"] = bool(statics)
    cfg.setdefault("training", {})
    cfg["training"]["epochs"] = epochs
    cfg["training"]["batch_size"] = bs
    cfg["training"]["seed"] = seed
    cfg.setdefault("spatial_split", {})
    cfg["spatial_split"]["train_fraction"] = SPATIAL_TRAIN_FRACTION
    cfg["spatial_split"]["cluster_count"] = SPATIAL_CLUSTER_COUNT
    cfg["spatial_split"]["split_seed"] = SPATIAL_SPLIT_SEED
    cfg["spatial_split"]["exclude_terms"] = SPATIAL_EXCLUDE_TERMS
    cfg["run_sig"] = run_sig(in_len, out_len, epochs, statics, seed, dataset, bs)
    return cfg


def run_one(in_len, out_len, epochs, statics, seed, dataset, bs, base_env, base_cfg):
    cfg = _with_run_params(base_cfg, in_len, out_len, epochs, statics, seed, dataset, bs)
    _write_tft_cfg(cfg)
    env = base_env.copy()

    t0 = time.perf_counter()
    subprocess.run([PY, str(TRAIN)], check=True, env=env)
    t1 = time.perf_counter()

    subprocess.run([PY, str(EVAL)], check=True, env=env)
    t2 = time.perf_counter()

    return round(t1 - t0, 3), round(t2 - t1, 3)

def read_metrics_and_skill(in_len, out_len, statics, epochs, seed, dataset, bs):
    sig = run_sig(in_len, out_len, epochs, statics, seed, dataset, bs)
    model_name = f"{MODEL}_{sig}"
    run_dir = ROOT / "outputs" / MODEL / model_name
    metrics_path = run_dir / "metrics.parquet"
    skill_path   = run_dir / "skill_by_horizon.parquet"

    rmse_h4 = s1 = s2 = s3 = s4 = ""

    if metrics_path.exists():
        m = pd.read_parquet(metrics_path)
        m4 = m[(m["metric"] == "RMSE") & (m["horizon"] == 4)]
        if not m4.empty:
            rmse_h4 = float(m4["value"].median())



    if skill_path.exists():
        s = pd.read_parquet(skill_path).set_index("horizon")["skill_vs_persistence"]
        s1 = "" if 1 not in s.index else float(s.loc[1])
        s2 = "" if 2 not in s.index else float(s.loc[2])
        s3 = "" if 3 not in s.index else float(s.loc[3])
        s4 = "" if 4 not in s.index else float(s.loc[4])

    return rmse_h4, s1, s2, s3, s4

def main():
    base_env = os.environ.copy()
    write_header_if_needed(LOG)
    base_cfg = _load_cfg(TFT_CONFIG)

    try:
        for in_len, out_len, statics, epochs in itertools.product(IN_LENS, OUT_LENS, STATICS, EPOCHS):
            for seed in SEEDS:
                ok = False
                tr_s = ev_s = ""
                try:
                    tr_s, ev_s = run_one(
                        in_len, out_len, epochs, statics, seed, DATASET, BATCH_SIZE, base_env, base_cfg
                    )
                    ok = True
                except subprocess.CalledProcessError:
                    ok = False

                rmse_h4, s1, s2, s3, s4 = read_metrics_and_skill(in_len, out_len, statics, epochs, seed, DATASET, BATCH_SIZE)
                row = f"{in_len},{out_len},{statics},{epochs},{seed},{tr_s},{ev_s},{ok},{rmse_h4},{s1},{s2},{s3},{s4}"
                append_row(LOG, row)
    finally:
        _write_tft_cfg(base_cfg)

if __name__ == "__main__":
    main()
