import argparse
import subprocess
import tempfile
from pathlib import Path

SPLIT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

AFFINITY = """\
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: NotIn
                values:
              - key: gpu
                operator: NotIn
                values:
                - k80
                - p100
                - b200
                - "false"
"""


def _job_yaml(ss: int) -> str:
    job_name = f"gw-oracle-ms-{ss}"
    run_sig = f"oracle_dedup_rand_f95_ss{ss}"
    split_csv = f"splits/spatial_split_full_merged_dedup_random_f0p95_ss{ss}.csv"
    oracle_pred = f"/tmp/oracle_pred_ss{ss}.parquet"
    gp_cfg = f"/tmp/gp_oracle_ss{ss}.yaml"
    log = f"reports/gru_gp/logs/oracle_ms_ss{ss}.log"

    return f"""\
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  labels:
    app: gw-oracle-ms
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: gw-oracle-ms
    spec:
      restartPolicy: Never
{AFFINITY}\
      volumes:
      - name: gw-pred-rwx
        persistentVolumeClaim:
          claimName: gw-pred-rwx
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: "16Gi"
      containers:
      - name: gp-oracle
        image: row56/gw-pred:py312-cu124
        imagePullPolicy: Always
        resources:
          requests:
            cpu: "8"
            memory: 80Gi
            nvidia.com/gpu: 1
          limits:
            memory: 80Gi
            nvidia.com/gpu: 1
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: WANDB_MODE
          value: "disabled"
        volumeMounts:
        - name: gw-pred-rwx
          mountPath: /storage
        - name: dshm
          mountPath: /dev/shm
        command: ["bash", "-c"]
        args:
        - |
          set -euo pipefail
          source /storage/venv/bin/activate
          cd /storage/gwl-interpolation
          mkdir -p reports/gru_gp/logs

          echo "=== Building oracle pred and GP config for ss={ss} ==="
          python3 -c '
          import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, yaml
          from pathlib import Path

          IN_LEN, OUT_LEN = 52, 16
          VAL_CUTOFF = pd.Timestamp("20200101")
          SS = {ss}
          split_path = "splits/spatial_split_full_merged_dedup_random_f0p95_ss{ss}.csv"
          oracle_path = "/tmp/oracle_pred_ss{ss}.parquet"
          cfg_path    = "/tmp/gp_oracle_ss{ss}.yaml"

          gws = pq.read_table("/storage/data/merged.parquet", columns=["datum","id","gws"]).to_pandas()
          gws["datum"] = pd.to_datetime(gws["datum"])

          split = pd.read_csv(split_path)
          split_ids = set(split["id"])
          gws_split = gws[gws["id"].isin(split_ids)].sort_values(["id","datum"])
          _nw = len(split_ids)
          print(f"  ss={ss}: {{_nw}} wells in split")

          rows = []
          for gid, g in gws_split.groupby("id"):
              g = g.sort_values("datum").reset_index(drop=True)
              times = g["datum"].values
              vals  = g["gws"].values
              for i in range(IN_LEN, len(g) - OUT_LEN + 1):
                  end_time = times[i + OUT_LEN - 1]
                  if end_time <= np.datetime64(VAL_CUTOFF):
                      continue
                  for h in range(OUT_LEN):
                      v = float(vals[i + h])
                      rows.append({{
                          "id":      gid,
                          "datum":   pd.Timestamp(times[i + h]),
                          "horizon": h + 1,
                          "gws":     v if np.isfinite(v) else float("nan"),
                      }})

          df = pd.DataFrame(rows)
          _nd = df["datum"].nunique()
          _nr = len(df)
          print(f"  {{_nd}} dates, {{_nr:,}} rows")
          pq.write_table(pa.Table.from_pandas(df), oracle_path)

          cfg = yaml.safe_load(open("configs/gp/gp.yaml"))
          cfg["gp_features"] = ["siwa_verweilzeit_j"]
          cfg["gp_features_onehot"] = ["hydroraum", "gw_gespannt"]
          cfg["spatial_split"] = {{
              "file": split_path,
              "train_fraction": 0.95,
              "split_seed": SS,
              "cluster_count": 10,
              "max_dist_k": 3,
              "max_dist_percentile": 25,
              "exclude_terms": "geometry,x_25833,y_25833",
          }}
          yaml.dump(cfg, open(cfg_path, "w"), default_flow_style=False)
          print("Oracle pred and GP config ready.")
          '

          echo "=== Running GP eval (oracle) ss={ss} ==="
          {{
            time python src/scripts/joint/spatial/gp_eval.py \\
              --config "{gp_cfg}" \\
              --gru-run-sig "{run_sig}" \\
              --pred-path "{oracle_pred}" \\
              --gp-run-tag ms \\
              --pretrain-steps 300 \\
              --pretrain-lr 0.01 \\
              --max-pretrain-pts 2000 \\
              --date-freq D \\
              --jitter 0.00001 \\
              --use-float64 true \\
              --model-prefix ORACLE
          }} 2>&1 | tee "{log}"

          echo "=== DONE ss={ss} ==="
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    existing = set()
    if args.skip_existing:
        out = subprocess.run(
            ["kubectl", "get", "jobs", "--no-headers", "-o",
             "custom-columns=NAME:.metadata.name"],
            capture_output=True, text=True,
        )
        existing = set(out.stdout.split())

    submitted, skipped, failed = 0, 0, 0
    for ss in SPLIT_SEEDS:
        job_name = f"gw-oracle-ms-{ss}"
        if job_name in existing:
            print(f"  skip (exists)  {job_name}")
            skipped += 1
            continue

        yaml_str = _job_yaml(ss)
        if args.dry_run:
            print(f"  dry-run  {job_name}")
            continue

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_str)
            tmp = f.name
        try:
            result = subprocess.run(["kubectl", "apply", "-f", tmp],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  submitted  {job_name}")
                submitted += 1
            else:
                print(f"  FAILED     {job_name}: {result.stderr.strip()}")
                failed += 1
        finally:
            Path(tmp).unlink(missing_ok=True)

    print(f"\nDone: {submitted} submitted, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
