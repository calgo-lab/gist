from __future__ import annotations

import argparse
import copy
import itertools
import os
import pathlib
import random
import subprocess
import sys
import time

import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[4]
PY = sys.executable
GRU_CONFIG = ROOT / "configs" / "gru.yaml"
HPO_CONFIG_DEFAULT = ROOT / "configs" / "hpo_gru.yaml"
TRAIN = ROOT / "src/scripts/joint/temporal/gru_train.py"
EVAL = ROOT / "src/scripts/joint/temporal/gru_eval.py"
HPO_RESULTS_DIR = ROOT / "reports/gru/hpo"


def _load_cfg(path):
    data = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    return data or {}


def _write_cfg(cfg):
    GRU_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _set_nested(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    target = cfg
    for key in keys[:-1]:
        node = target.get(key)
        if not isinstance(node, dict):
            node = {}
            target[key] = node
        target = node
    target[keys[-1]] = value


def _sample_trials(search_space, max_trials, seed):
    keys = sorted(search_space.keys())
    values = [search_space[k] for k in keys]
    full = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    rng = random.Random(seed)
    rng.shuffle(full)
    return full[: min(max_trials, len(full))]


def _run_sig_from_cfg(cfg):
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


def _read_objective(run_dir, objective):
    metrics_file = str(objective.get("file", "metrics.parquet"))
    metric_name = str(objective.get("metric", "RMSE"))
    horizon = objective.get("horizon")
    path = run_dir / metrics_file
    if not path.exists():
        raise FileNotFoundError(f"Objective file not found: {path}")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if horizon is not None and "horizon" in df.columns:
        df = df[df["horizon"] == int(horizon)]
        if df.empty:
            raise ValueError(f"No row with horizon={horizon} in {path}")
    if metric_name not in df.columns:
        raise ValueError(f"Metric {metric_name} missing in {path}")
    return float(df[metric_name].iloc[0])


def _run_train_eval(env):
    t0 = time.perf_counter()
    subprocess.run([PY, str(TRAIN)], check=True, env=env)
    t1 = time.perf_counter()
    subprocess.run([PY, str(EVAL)], check=True, env=env)
    t2 = time.perf_counter()
    return round(t1 - t0, 3), round(t2 - t1, 3)


def _run_hpo(base_cfg, hpo_config_path):
    hpo_cfg = _load_cfg(hpo_config_path)
    objective = hpo_cfg.get("objective", {})
    mode = str(objective.get("mode", "min")).lower()
    search_space = hpo_cfg.get("search_space", {})
    max_trials = int(hpo_cfg.get("max_trials", 12))
    random_seed = int(hpo_cfg.get("random_seed", 42))
    hpo_name = str(hpo_cfg.get("name", "gru_hpo"))

    trials = _sample_trials(search_space, max_trials=max_trials, seed=random_seed)
    HPO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = HPO_RESULTS_DIR / f"{hpo_name}_results.csv"

    rows = []
    best_value = None
    best_row = None
    env = os.environ.copy()

    for i, trial_params in enumerate(trials, start=1):
        cfg = copy.deepcopy(base_cfg)
        for key, value in trial_params.items():
            _set_nested(cfg, key, value)

        dataset = str(cfg.get("dataset", "full_merged"))
        seed = int(cfg.get("training", {}).get("seed", 40))
        run_sig = f"{hpo_name}_t{i:03d}_seed{seed}_{dataset}"
        cfg["run_sig"] = run_sig
        _write_cfg(cfg)

        start = time.time()
        train_s, eval_s = _run_train_eval(env=env)
        run_dir = ROOT / "outputs" / "GRU_FCOV" / f"GRU_FCOV_{run_sig}"
        objective_value = _read_objective(run_dir, objective)
        elapsed_s = round(time.time() - start, 3)

        row = {
            "trial": i,
            "run_sig": run_sig,
            "status": "ok",
            "objective": objective_value,
            "train_s": train_s,
            "eval_s": eval_s,
            "elapsed_s": elapsed_s,
            "error": "",
        }
        row.update({f"hp.{k}": v for k, v in trial_params.items()})
        rows.append(row)

        if objective_value is not None:
            is_better = best_value is None or (
                objective_value < best_value if mode == "min" else objective_value > best_value
            )
            if is_better:
                best_value = objective_value
                best_row = row

        pd.DataFrame(rows).to_csv(results_path, index=False)
        print(f"[{i}/{len(trials)}] ok run_sig={run_sig} objective={objective_value}")

    if best_row is not None:
        best_path = HPO_RESULTS_DIR / f"{hpo_name}_best.yaml"
        best_payload = {
            "hpo_name": hpo_name,
            "objective_mode": mode,
            "best_objective": best_value,
            "best_trial": best_row,
        }
        best_path.write_text(yaml.safe_dump(best_payload, sort_keys=False), encoding="utf-8")
        print(f"Best trial: run_sig={best_row['run_sig']} objective={best_value}")
        print(f"Saved: {results_path}")
        print(f"Saved: {best_path}")
    else:
        print("No successful trial completed.")
        print(f"Saved failures: {results_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo-config", default="")
    args = parser.parse_args()

    base_cfg = _load_cfg(GRU_CONFIG)
    original_cfg = copy.deepcopy(base_cfg)
    try:
        if args.hpo_config:
            _run_hpo(base_cfg=base_cfg, hpo_config_path=pathlib.Path(args.hpo_config))
        else:
            print("No --hpo-config provided. Nothing to do.")
    finally:
        _write_cfg(original_cfg)


if __name__ == "__main__":
    main()
