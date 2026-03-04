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
GP_CONFIG = ROOT / "configs" / "gp.yaml"
GRU_CONFIG = ROOT / "configs" / "gru.yaml"
GP_EVAL = ROOT / "src/scripts/joint/spatial/gp_eval.py"
LOG = ROOT / "reports/gp/metrics/metrics.csv"
HPO_DIR = ROOT / "reports/gp/hpo"
LOG.parent.mkdir(parents=True, exist_ok=True)


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
    revin_tag = "r1" if bool(mc.get("use_revin", False)) else "r0"
    spf = str(float(sc.get("train_fraction", 0.8))).replace(".", "p")
    sc_cnt = int(sc.get("cluster_count", 20))
    ss = int(sc.get("split_seed", 42))
    return f"in{in_len}_out{out_len}_ep{epochs}_bs{bs}_seed{seed}_{dataset}_{revin_tag}_spf{spf}_sc{sc_cnt}_ss{ss}"


def run_gp(gru_run_sig, gp_run_tag, pretrain_steps, pretrain_lr, max_pretrain_pts, date_freq, jitter, kernel_type, isotropic):
    cmd = [
        PY,
        str(GP_EVAL),
        "--gru-run-sig", str(gru_run_sig),
        "--gp-run-tag", str(gp_run_tag),
        "--pretrain-steps", str(int(pretrain_steps)),
        "--pretrain-lr", str(float(pretrain_lr)),
        "--max-pretrain-pts", str(int(max_pretrain_pts)),
        "--date-freq", str(date_freq),
        "--jitter", str(float(jitter)),
        "--kernel-type", str(kernel_type),
        "--isotropic", str(isotropic),
    ]
    subprocess.run(cmd, check=True, env=os.environ.copy())
    run_tag = f"GRU_FCOV_{gru_run_sig}__{gp_run_tag}__predobstrain"
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
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--hpo-config", default="")
    args = parser.parse_args()

    gp_cfg = load_yaml(GP_CONFIG)
    gru_cfg = load_yaml(GRU_CONFIG)
    base_gru_sig = default_gru_sig(gru_cfg)

    if args.hpo_config:
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
            kernel_type = str(cfg.get("kernel_type", "matern32"))
            isotropic = cfg.get("isotropic", True)

            t0 = time.perf_counter()
            status = "ok"
            obj = None
            error = ""
            try:
                run_dir = run_gp(
                    gru_run_sig,
                    gp_run_tag,
                    pretrain_steps,
                    pretrain_lr,
                    max_pretrain_pts,
                    date_freq,
                    jitter,
                    kernel_type,
                    isotropic,
                )
                obj = read_metric(run_dir, metric_name)
            except Exception as e:
                status = "failed"
                error = str(e)
            elapsed = round(time.perf_counter() - t0, 3)

            row = {
                "trial": i,
                "gru_run_sig": gru_run_sig,
                "gp_run_tag": gp_run_tag,
                "status": status,
                "objective": obj,
                "elapsed_s": elapsed,
                "error": error,
            }
            row.update({f"hp.{k}": v for k, v in params.items()})
            rows.append(row)
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
        return

    runs_cfg = gp_cfg.get("runs", {}) if args.sweep else {}
    gru_run_sigs = runs_cfg.get("gru_run_sigs", [str(gp_cfg.get("gru_run_sig", base_gru_sig))])
    gp_run_tags = runs_cfg.get("gp_run_tags", [str(gp_cfg.get("gp_run_tag", "gp_pytorch"))])
    pretrain_steps_list = runs_cfg.get("pretrain_steps", [int(gp_cfg.get("pretrain_steps", 200))])
    pretrain_lr_list = runs_cfg.get("pretrain_lr", [float(gp_cfg.get("pretrain_lr", 1e-2))])
    max_pretrain_pts_list = runs_cfg.get("max_pretrain_pts", [int(gp_cfg.get("max_pretrain_pts", 2000))])
    date_freq_list = runs_cfg.get("date_freq", [str(gp_cfg.get("date_freq", "ME"))])
    jitter_list = runs_cfg.get("jitter", [float(gp_cfg.get("jitter", 1e-5))])
    kernel_type_list = runs_cfg.get("kernel_type", [str(gp_cfg.get("kernel_type", "matern32"))])
    isotropic_list = runs_cfg.get("isotropic", [gp_cfg.get("isotropic", True)])

    if not LOG.exists() or LOG.stat().st_size == 0:
        LOG.write_text(
            "gru_run_sig,gp_run_tag,pretrain_steps,pretrain_lr,max_pretrain_pts,date_freq,jitter,kernel_type,isotropic,run_s,objective\n",
            encoding="utf-8",
        )

    for values in itertools.product(
        gru_run_sigs,
        gp_run_tags,
        pretrain_steps_list,
        pretrain_lr_list,
        max_pretrain_pts_list,
        date_freq_list,
        jitter_list,
        kernel_type_list,
        isotropic_list,
    ):
        gru_run_sig, gp_run_tag, pretrain_steps, pretrain_lr, max_pretrain_pts, date_freq, jitter, kernel_type, isotropic = values
        t0 = time.perf_counter()
        run_dir = run_gp(
            gru_run_sig,
            gp_run_tag,
            pretrain_steps,
            pretrain_lr,
            max_pretrain_pts,
            date_freq,
            jitter,
            kernel_type,
            isotropic,
        )
        obj = read_metric(run_dir, "NSE_id_median")
        run_s = round(time.perf_counter() - t0, 3)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(
                f"{gru_run_sig},{gp_run_tag},{pretrain_steps},{pretrain_lr},{max_pretrain_pts},{date_freq},{jitter},{kernel_type},{isotropic},{run_s},{obj}\n"
            )
        print(f"run: gru={gru_run_sig} tag={gp_run_tag} kernel={kernel_type} iso={isotropic} objective={obj}")


if __name__ == "__main__":
    main()
