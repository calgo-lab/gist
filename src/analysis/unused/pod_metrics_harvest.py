import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import wandb
except ImportError:
    sys.exit("wandb not installed.")

ENTITY   = "gw-pred"
PROJECTS = ["gist", "gwl-interpolation"]

RE_SS  = re.compile(r"_ss\d+")
RE_GPS = re.compile(r"_gps\d+")


RE_RMSE_PW  = re.compile(r"^\s*RMSE_pw\s*:\s*([0-9.eE+\-]+)", re.M)
RE_NRMSE_PW = re.compile(r"^\s*nRMSE_pw\s*:\s*([0-9.eE+\-]+)", re.M)
RE_DONE     = re.compile(r"=== DONE:\s*(.+?)\s*===")


def pod_pipeline(pod: str) -> str:
    if pod.startswith("gw-oracle"):
        return "ORACLE"
    if pod.startswith("gw-joint"):
        return "JOINT"
    return "UNKNOWN"


def config_key_from_done(done_sig: str, pipeline: str) -> str:
    stripped = RE_SS.sub("", done_sig)
    return f"{pipeline}_{stripped}"


def fetch_pod_log(pod: str) -> dict | None:
    try:
        result = subprocess.run(
            ["kubectl", "logs", pod],
            capture_output=True, text=True, timeout=30,
        )
        log = result.stdout
    except Exception as e:
        print(f"  WARNING: kubectl logs {pod} failed: {e}")
        return None

    m_rmse = RE_RMSE_PW.search(log)
    if not m_rmse:
        return None

    rmse    = float(m_rmse.group(1))
    m_nrmse = RE_NRMSE_PW.search(log)
    nrmse   = float(m_nrmse.group(1)) if m_nrmse else np.nan

    m_done    = RE_DONE.search(log)
    pipeline  = pod_pipeline(pod)
    if m_done:
        cfg_key = config_key_from_done(m_done.group(1), pipeline)
    else:
        cfg_key = f"{pipeline}_UNKNOWN_{pod}"

    return {
        "pod":        pod,
        "config_key": cfg_key,
        "RMSE_pw":    rmse,
        "nRMSE_pw":   nrmse,
    }


def harvest_pods() -> pd.DataFrame:
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "--sort-by=.metadata.name",
             "-o", "custom-columns=NAME:.metadata.name,STATUS:.status.phase",
             "--no-headers"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        print(f"WARNING: kubectl get pods failed: {e}")
        return pd.DataFrame()

    pods = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 1:
            continue
        name = parts[0]
        if re.match(r"gw-(oracle|joint)", name):
            pods.append(name)

    print(f"  Fetching logs for {len(pods)} oracle/joint pods (parallel)...")
    rows = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(fetch_pod_log, p): p for p in pods}
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                rows.append(row)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    print(f"  Got metrics from {len(df)} pods")
    return df


RE_SS_WB  = re.compile(r"_ss\d+")
RE_GPS_WB = re.compile(r"_gps\d+")


def wb_config_key(run_name: str) -> str:
    parts   = run_name.split("__", 1)
    gru_sig = RE_SS_WB.sub("", parts[0])
    gp_tag  = RE_GPS_WB.sub("", RE_SS_WB.sub("", parts[1])) if len(parts) > 1 else ""
    return f"DEC_{gru_sig}__{gp_tag}" if gp_tag else f"DEC_{gru_sig}"


def fetch_wandb() -> pd.DataFrame:
    api  = wandb.Api(timeout=60)
    rows = []
    for proj in PROJECTS:
        try:
            runs = api.runs(f"{ENTITY}/{proj}", per_page=500)
            for r in runs:
                rmse  = r.summary.get("RMSE_pw")
                nrmse = r.summary.get("nRMSE_pw")
                if rmse is None or nrmse is None:
                    continue
                try:
                    rmse, nrmse = float(rmse), float(nrmse)
                except (TypeError, ValueError):
                    continue
                if not (np.isfinite(rmse) and np.isfinite(nrmse)):
                    continue
                rows.append({
                    "pod":        None,
                    "config_key": wb_config_key(r.name or ""),
                    "RMSE_pw":    rmse,
                    "nRMSE_pw":   nrmse,
                })
        except Exception as e:
            print(f"  WARNING: W&B {ENTITY}/{proj}: {e}")

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    print(f"  W&B: {len(df)} GP-eval runs with RMSE_pw + nRMSE_pw")
    return df


def short_label(key: str) -> str:
    if key.startswith("ORACLE_"):
        pipe = "oracle"
        rest = key[len("ORACLE_"):]
    elif key.startswith("JOINT_"):
        pipe = "joint"
        rest = key[len("JOINT_"):]
    elif key.startswith("DEC_"):
        pipe = "decoupled"
        rest = key[len("DEC_"):]
    else:
        pipe = "?"
        rest = key

    parts = rest.split("__", 1)
    gru   = parts[0]
    gp    = parts[1].strip("_") if len(parts) > 1 else ""

    m = re.search(r"(rand\d*|km|md\d+)_f(\d+)", gru)
    split_str = f"{m.group(1)} f{m.group(2)}" if m else gru[-25:]

    gp_short = (
        gp.replace("dec_prod_", "")
          .replace("oracle_prod_hr", "hr")
          .replace("joint_eval", "")
          .strip("_")
    )

    label = f"{pipe:9s}  {split_str}"
    if gp_short:
        label += f"  [{gp_short}]"
    return label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--min-seeds", type=int, default=1)
    args = parser.parse_args()

    print("Fetching W&B runs...")
    wb  = fetch_wandb()
    print("Fetching pod logs...")
    pod = harvest_pods()

    if wb.empty and pod.empty:
        print("No data found.")
        return

    df = pd.concat([wb, pod], ignore_index=True)

    grp = (
        df.groupby("config_key")
        .agg(
            label      =("config_key", lambda x: short_label(x.iloc[0])),
            n          =("RMSE_pw",    "count"),
            RMSE_mean  =("RMSE_pw",    "mean"),
            RMSE_std   =("RMSE_pw",    "std"),
            nRMSE_mean =("nRMSE_pw",   lambda x: x.dropna().mean() if x.notna().any() else np.nan),
            nRMSE_std  =("nRMSE_pw",   lambda x: x.dropna().std()  if x.notna().sum() > 1 else np.nan),
        )
        .reset_index()
    )

    if args.min_seeds > 1:
        grp = grp[grp["n"] >= args.min_seeds]

    grp = grp.sort_values("RMSE_mean").reset_index(drop=True)

    print()
    hdr = f"{'#':>3}  {'n':>4}  {'RMSE_pw':>9}  {'±':>7}  {'nRMSE_pw':>9}  {'±':>7}  label"
    print(hdr)
    print("-" * (len(hdr) + 20))
    for rank, row in grp.iterrows():
        r_std  = f"{row['RMSE_std']:.4f}"  if pd.notna(row["RMSE_std"])  else "      —"
        n_std  = f"{row['nRMSE_std']:.4f}" if pd.notna(row["nRMSE_std"]) else "      —"
        nrmse_val = f"{row['nRMSE_mean']:9.4f}" if pd.notna(row["nRMSE_mean"]) else "        —"
        print(
            f"{rank+1:>3}  "
            f"{int(row['n']):>4}  "
            f"{row['RMSE_mean']:>9.4f}  "
            f"{r_std:>7}  "
            f"{nrmse_val}  "
            f"{n_std:>7}  "
            f"{row['label']}"
        )

    print(f"\n{len(grp)} configurations, {len(df)} total seed runs")
    print("(nRMSE_pw — for joint runs: not printed by gru_gp_eval.py, shown as —)")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        grp.to_csv(out, index=False)
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
