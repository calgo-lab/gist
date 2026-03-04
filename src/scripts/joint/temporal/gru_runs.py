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
SWEEP_LOG = ROOT / "reports/gru/metrics/metrics.csv"
HPO_RESULTS_DIR = ROOT / "reports/gru/hpo"
SWEEP_LOG.parent.mkdir(parents=True, exist_ok=True)


def _load_cfg(path: pathlib.Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _write_cfg(cfg: dict) -> None:
    GRU_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _set_nested(cfg: dict, dotted_key: str, value: object) -> None:
    keys = dotted_key.split(".")
    target = cfg
    for key in keys[:-1]:
        node = target.get(key)
        if not isinstance(node, dict):
            node = {}
            target[key] = node
        target = node
    target[keys[-1]] = value


def _sample_trials(search_space: dict, max_trials: int, seed: int) -> list[dict]:
    keys = sorted(search_space.keys())
    values = [search_space[k] for k in keys]
    full = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    rng = random.Random(seed)
    rng.shuffle(full)
    return full[: min(max_trials, len(full))]


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


def _read_objective(run_dir: pathlib.Path, objective: dict) -> float:
    metrics_file = str(objective.get("file", "metrics.parquet"))
    metric_name = str(objective.get("metric", "RMSE"))
    horizon = objective.get("horizon")

    path = run_dir / metrics_file
    if not path.exists():
        raise FileNotFoundError(f"Objective file not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if horizon is not None and "horizon" in df.columns:
        df = df[df["horizon"] == int(horizon)]
        if df.empty:
            raise ValueError(f"No row with horizon={horizon} in {path}")

    if metric_name not in df.columns:
        raise ValueError(f"Metric {metric_name} missing in {path}")

    return float(df[metric_name].iloc[0])


def _run_train_eval(env: dict) -> tuple[float, float]:
    t0 = time.perf_counter()
    subprocess.run([PY, str(TRAIN)], check=True, env=env)
    t1 = time.perf_counter()
    subprocess.run([PY, str(EVAL)], check=True, env=env)
    t2 = time.perf_counter()
    return round(t1 - t0, 3), round(t2 - t1, 3)


def _write_header_if_needed(path: pathlib.Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(
            "dataset,in_len,out_len,epochs,batch_size,seed,train_s,eval_s,ok,run_sig,rmse_h,mae_h\n",
            encoding="utf-8",
        )


def _append_row(path: pathlib.Path, row: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(row + "\n")


def _run_sweep(base_cfg: dict) -> None:
    dataset = str(base_cfg.get("dataset", "full_merged"))
    data_cfg = base_cfg.get("data", {})
    training_cfg = base_cfg.get("training", {})
    runs_cfg = base_cfg.get("runs", {})

    in_lens = runs_cfg.get("in_lens", [int(data_cfg.get("in_len", 52))])
    out_lens = runs_cfg.get("out_lens", [int(data_cfg.get("out_len", 16))])
    epochs_list = runs_cfg.get("epochs", [int(training_cfg.get("epochs", 50))])
    seeds = runs_cfg.get("seeds", [int(training_cfg.get("seed", 40))])
    batch_size = int(runs_cfg.get("batch_size", int(training_cfg.get("batch_size", 4096))))

    _write_header_if_needed(SWEEP_LOG)
    env = os.environ.copy()

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

        train_s, eval_s = _run_train_eval(env=env)
        ok = True

        run_sig, rmse_h, mae_h = _read_gru_metrics(cfg)
        _append_row(
            SWEEP_LOG,
            f"{dataset},{in_len},{out_len},{epochs},{batch_size},{seed},{train_s},{eval_s},{ok},{run_sig},{rmse_h},{mae_h}",
        )
        print(f"sweep run: ok={ok} run_sig={run_sig} rmse_h={rmse_h} mae_h={mae_h}")


def _run_hpo(base_cfg: dict, hpo_config_path: pathlib.Path) -> None:
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

    rows: list[dict] = []
    best_value: float | None = None
    best_row: dict | None = None
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
        status = "ok"
        objective_value: float | None = None
        error = ""
        train_s = ""
        eval_s = ""

        train_s, eval_s = _run_train_eval(env=env)
        run_dir = ROOT / "outputs" / "GRU_FCOV" / f"GRU_FCOV_{run_sig}"
        objective_value = _read_objective(run_dir, objective)

        elapsed_s = round(time.time() - start, 3)
        row = {
            "trial": i,
            "run_sig": run_sig,
            "status": status,
            "objective": objective_value,
            "train_s": train_s,
            "eval_s": eval_s,
            "elapsed_s": elapsed_s,
            "error": error,
        }
        row.update({f"hp.{k}": v for k, v in trial_params.items()})
        rows.append(row)

        if status == "ok" and objective_value is not None:
            is_better = best_value is None or (
                objective_value < best_value if mode == "min" else objective_value > best_value
            )
            if is_better:
                best_value = objective_value
                best_row = row

        pd.DataFrame(rows).to_csv(results_path, index=False)
        print(f"[{i}/{len(trials)}] {status} run_sig={run_sig} objective={objective_value}")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRU train/eval sweep or HPO using gru config.")
    parser.add_argument("--sweep", action="store_true", help="Use runs.* grid from gru.yaml.")
    parser.add_argument("--hpo-config", default="", help="Run HPO mode with this yaml config.")
    args = parser.parse_args()

    base_cfg = _load_cfg(GRU_CONFIG)
    original_cfg = copy.deepcopy(base_cfg)
    try:
        if args.hpo_config:
            _run_hpo(base_cfg=base_cfg, hpo_config_path=pathlib.Path(args.hpo_config))
        elif args.sweep:
            _run_sweep(base_cfg=base_cfg)
        else:
            # Single run mode: use current config values exactly once.
            _run_sweep(base_cfg=base_cfg | {"runs": {}})
    finally:
        _write_cfg(original_cfg)


if __name__ == "__main__":
    main()
