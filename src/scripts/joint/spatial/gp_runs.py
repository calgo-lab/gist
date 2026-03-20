import argparse
import copy
import itertools
import math
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
GP_CONFIG = ROOT / "configs" / "gp.yaml"
GRU_CONFIG = ROOT / "configs" / "gru.yaml"
GP_EVAL = ROOT / "src/scripts/joint/spatial/gp_eval.py"
HPO_DIR = ROOT / "reports/gp/hpo"


def load_yaml(path):
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data or {}


def default_gru_sig(gru_cfg):
    run_sig = str(gru_cfg.get("run_sig", "")).strip()
    if run_sig and run_sig.lower() != "auto":
        return run_sig
    dataset = str(gru_cfg.get("dataset", "full_merged"))
    tr = gru_cfg.get("training", {})
    dc = gru_cfg.get("data", {})
    mc = gru_cfg.get("model", {})
    sc = gru_cfg.get("spatial_split", {})
    in_len = int(dc.get("in_len", 52))
    out_len = int(dc.get("out_len", 16))
    seed = int(tr.get("seed", 40))
    epochs = int(tr.get("epochs", 50))
    bs = int(tr.get("batch_size", 4096))
    spf = str(float(sc.get("train_fraction", 0.8))).replace(".", "p")
    sc_cnt = int(sc.get("cluster_count", 20))
    ss = int(sc.get("split_seed", 42))
    return f"in{in_len}_out{out_len}_ep{epochs}_bs{bs}_seed{seed}_{dataset}_spf{spf}_sc{sc_cnt}_ss{ss}"


def run_gp(gru_run_sig, gp_run_tag, pretrain_steps, pretrain_lr, max_pretrain_pts,
           date_freq, jitter,
           backend="gpytorch", n_inducing=64, variational_lr=1e-2, use_float64=True,
           model_prefix="GRU_FCOV"):
    cmd = [
        PY, str(GP_EVAL),
        "--gru-run-sig", str(gru_run_sig),
        "--gp-run-tag", str(gp_run_tag),
        "--pretrain-steps", str(int(pretrain_steps)),
        "--pretrain-lr", str(float(pretrain_lr)),
        "--max-pretrain-pts", str(int(max_pretrain_pts)),
        "--date-freq", str(date_freq),
        "--jitter", str(float(jitter)),
        "--backend", str(backend),
        "--n-inducing", str(int(n_inducing)),
        "--variational-lr", str(float(variational_lr)),
        "--use-float64", str(use_float64).lower(),
        "--model-prefix", str(model_prefix),
    ]
    subprocess.run(cmd, check=True, env=os.environ.copy())
    run_tag = f"{model_prefix}_{gru_run_sig}__{gp_run_tag}__predobstrain"
    return ROOT / "outputs" / "gp" / run_tag


def read_metric(run_dir, metric_name):
    p = run_dir / "gp_metrics.parquet"
    df = pd.read_parquet(p)
    if "metric" in df.columns and "value" in df.columns:
        return float(df.loc[df["metric"] == metric_name, "value"].iloc[0])
    return float(df[metric_name].iloc[0])


def set_nested(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo-config", default="")
    args = parser.parse_args()

    gp_cfg = load_yaml(GP_CONFIG)
    gru_cfg = load_yaml(GRU_CONFIG)
    base_gru_sig = default_gru_sig(gru_cfg)

    if not args.hpo_config:
        print("No --hpo-config provided. Nothing to do.")
        return

    hpo_cfg = load_yaml(args.hpo_config)
    search_space = hpo_cfg.get("search_space", {})
    metric_name = str(hpo_cfg.get("objective", {}).get("metric", "NSE_id_median"))
    mode = str(hpo_cfg.get("objective", {}).get("mode", "max")).lower()
    max_trials = int(hpo_cfg.get("max_trials", 12))
    random_seed = int(hpo_cfg.get("random_seed", 42))
    hpo_name = str(hpo_cfg.get("name", "gp_hpo"))

    keys = sorted(search_space.keys())
    vals = [search_space[k] for k in keys]
    trials = [dict(zip(keys, c)) for c in itertools.product(*vals)]
    random.Random(random_seed).shuffle(trials)
    trials = trials[: min(max_trials, len(trials))]

    HPO_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = HPO_DIR / f"{hpo_name}_results.csv"
    best = None
    best_val = None
    rows = []

    for i, params in enumerate(trials, start=1):
        cfg = copy.deepcopy(gp_cfg)
        for k, v in params.items():
            set_nested(cfg, k, v)

        gru_run_sig = str(cfg.get("gru_run_sig", base_gru_sig))
        gp_run_tag = f"{hpo_name}_t{i:03d}"
        pretrain_steps = int(cfg.get("pretrain_steps", 200))
        pretrain_lr = float(cfg.get("pretrain_lr", 1e-2))
        max_pretrain_pts = int(cfg.get("max_pretrain_pts", 2000))
        date_freq = str(cfg.get("date_freq", "ME"))
        jitter = float(cfg.get("jitter", 1e-5))
        backend = str(cfg.get("backend", "gpytorch"))
        n_inducing = int(cfg.get("n_inducing", 64))
        variational_lr = float(cfg.get("variational_lr", 1e-2))
        use_float64 = cfg.get("use_float64", True)
        model_prefix = str(cfg.get("model_prefix", "GRU_FCOV"))

        t0 = time.perf_counter()
        status = "ok"
        obj = None
        error = ""
        try:
            run_dir = run_gp(
                gru_run_sig, gp_run_tag, pretrain_steps, pretrain_lr, max_pretrain_pts,
                date_freq, jitter,
                backend=backend, n_inducing=n_inducing, variational_lr=variational_lr,
                use_float64=use_float64, model_prefix=model_prefix,
            )
            obj = read_metric(run_dir, metric_name)
        except Exception as e:
            status = "failed"
            error = str(e)
        elapsed = round(time.perf_counter() - t0, 3)

        row = {
            "trial": i, "gru_run_sig": gru_run_sig, "gp_run_tag": gp_run_tag,
            "status": status, "objective": obj, "elapsed_s": elapsed, "error": error,
        }
        row.update({f"hp.{k}": v for k, v in params.items()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        if status == "ok" and (obj is None or not math.isfinite(float(obj))):
            status = "failed"
            error = "objective is non-finite (nan/inf)"
            rows[-1].update({"status": status, "error": error, "objective": obj})
            pd.DataFrame(rows).to_csv(out_csv, index=False)

        if status == "ok" and obj is not None:
            better = best_val is None or (obj > best_val if mode == "max" else obj < best_val)
            if better:
                best = row
                best_val = obj
        print(f"[{i}/{len(trials)}] {gp_run_tag} status={status} objective={obj}")

    if best is not None:
        best_path = HPO_DIR / f"{hpo_name}_best.yaml"
        best_path.write_text(yaml.safe_dump({"best_objective": best_val, "best_trial": best}, sort_keys=False), encoding="utf-8")
        print(f"Best: {best['gp_run_tag']} objective={best_val}")
        print(f"Saved: {out_csv}")
        print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
