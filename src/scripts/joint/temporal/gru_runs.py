from __future__ import annotations

import argparse
import copy
import itertools
import os
import pathlib
import subprocess
import sys
import time

import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[4]
PY = sys.executable
TFT_CONFIG = ROOT / "configs" / "tft.yaml"
TRAIN = ROOT / "src/scripts/joint/temporal/gru_train.py"
EVAL = ROOT / "src/scripts/joint/temporal/gru_eval.py"
LOG = ROOT / "reports/gru/metrics/metrics.csv"
LOG.parent.mkdir(parents=True, exist_ok=True)


def _load_cfg(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_cfg(cfg: dict) -> None:
    TFT_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _run_sig_from_cfg(cfg: dict) -> str:
    run_sig_cfg = str(cfg.get("run_sig", "")).strip()
    if run_sig_cfg and run_sig_cfg.lower() != "auto":
        return run_sig_cfg
    data_cfg = cfg.get("data", {})
    tr_cfg = cfg.get("training", {})
    sp_cfg = cfg.get("spatial_split", {})
    dataset = cfg.get("dataset", "full_merged")
    in_len = int(data_cfg.get("in_len", 52))
    out_len = int(data_cfg.get("out_len", 16))
    epochs = int(tr_cfg.get("epochs", 20))
    bs = int(tr_cfg.get("batch_size", 1024))
    seed = int(tr_cfg.get("seed", 40))
    spf = str(float(sp_cfg.get("train_fraction", 0.8))).replace(".", "p")
    sc = int(sp_cfg.get("cluster_count", 20))
    ss = int(sp_cfg.get("split_seed", 42))
    return f"in{in_len}_out{out_len}_ep{epochs}_bs{bs}_seed{seed}_{dataset}_spf{spf}_sc{sc}_ss{ss}"


def _with_params(base_cfg: dict, in_len: int, out_len: int, epochs: int, seed: int, dataset: str, bs: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["dataset"] = dataset
    cfg.setdefault("data", {})
    cfg["data"]["in_len"] = in_len
    cfg["data"]["out_len"] = out_len
    cfg.setdefault("training", {})
    cfg["training"]["epochs"] = epochs
    cfg["training"]["batch_size"] = bs
    cfg["training"]["seed"] = seed
    cfg["run_sig"] = "auto"
    return cfg


def _read_gru_metrics(cfg: dict) -> tuple[str, float | str, float | str]:
    sig = _run_sig_from_cfg(cfg)
    metrics_path = ROOT / "outputs" / "GRU_FCOV" / f"GRU_FCOV_{sig}" / "metrics.parquet"
    if not metrics_path.exists():
        return sig, "", ""
    metrics = pd.read_parquet(metrics_path)
    out_len = int(cfg.get("data", {}).get("out_len", 16))
    row = metrics[metrics["horizon"] == out_len]
    if row.empty:
        return sig, "", ""
    rmse = float(row["RMSE"].iloc[0]) if "RMSE" in row.columns else ""
    mae = float(row["MAE"].iloc[0]) if "MAE" in row.columns else ""
    return sig, rmse, mae


def _write_header_if_needed(path: pathlib.Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(
            "dataset,in_len,out_len,epochs,batch_size,seed,train_s,eval_s,ok,run_sig,rmse_h,mae_h\n",
            encoding="utf-8",
        )


def _append_row(path: pathlib.Path, row: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(row + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRU train+eval with tft config.")
    parser.add_argument("--sweep", action="store_true", help="Use runs.* grid from tft.yaml.")
    args = parser.parse_args()

    base_cfg = _load_cfg(TFT_CONFIG)
    if not base_cfg:
        raise ValueError("Could not read configs/tft.yaml")

    dataset = str(base_cfg.get("dataset", "full_merged"))
    data_cfg = base_cfg.get("data", {})
    training_cfg = base_cfg.get("training", {})
    runs_cfg = base_cfg.get("runs", {})

    if args.sweep:
        in_lens = runs_cfg.get("in_lens", [int(data_cfg.get("in_len", 52))])
        out_lens = runs_cfg.get("out_lens", [int(data_cfg.get("out_len", 16))])
        epochs_list = runs_cfg.get("epochs", [int(training_cfg.get("epochs", 50))])
        seeds = runs_cfg.get("seeds", [int(training_cfg.get("seed", 40))])
        batch_size = int(runs_cfg.get("batch_size", int(training_cfg.get("batch_size", 4096))))
    else:
        in_lens = [int(data_cfg.get("in_len", 52))]
        out_lens = [int(data_cfg.get("out_len", 16))]
        epochs_list = [int(training_cfg.get("epochs", 50))]
        seeds = [int(training_cfg.get("seed", 40))]
        batch_size = int(training_cfg.get("batch_size", 4096))

    _write_header_if_needed(LOG)
    env = os.environ.copy()

    try:
        for in_len, out_len, epochs, seed in itertools.product(in_lens, out_lens, epochs_list, seeds):
            cfg = _with_params(
                base_cfg=base_cfg,
                in_len=int(in_len),
                out_len=int(out_len),
                epochs=int(epochs),
                seed=int(seed),
                dataset=dataset,
                bs=batch_size,
            )
            _write_cfg(cfg)

            ok = False
            train_s = ""
            eval_s = ""
            try:
                t0 = time.perf_counter()
                subprocess.run([PY, str(TRAIN)], check=True, env=env)
                t1 = time.perf_counter()
                subprocess.run([PY, str(EVAL)], check=True, env=env)
                t2 = time.perf_counter()
                train_s = round(t1 - t0, 3)
                eval_s = round(t2 - t1, 3)
                ok = True
            except subprocess.CalledProcessError:
                ok = False

            run_sig, rmse_h, mae_h = _read_gru_metrics(cfg)
            _append_row(
                LOG,
                f"{dataset},{in_len},{out_len},{epochs},{batch_size},{seed},{train_s},{eval_s},{ok},{run_sig},{rmse_h},{mae_h}",
            )
    finally:
        _write_cfg(base_cfg)


if __name__ == "__main__":
    main()
