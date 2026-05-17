import argparse
import subprocess
import tempfile
from pathlib import Path

SPLIT_SEEDS = list(range(42, 52))

AFFINITY = """\
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: gpu
                operator: NotIn
                values:
                - k80
                - p100
                - b200
                - "false"
"""


def _job_yaml(ss: int) -> str:
    job_name  = f"gw-dkl-oracle-rand90-ss{ss}"
    split_csv = f"splits/spatial_split_full_merged_coloc_rand90_f90_ss{ss}.csv"
    run_tag   = f"dkl_oracle_rand90_ss{ss}"
    log       = f"reports/gru_gp/logs/{job_name}.log"

    cmd = f"""\
set -euo pipefail
source /storage/venv/bin/activate
cd /storage/gwl-interpolation
mkdir -p reports/gru_gp/logs

echo "=== DKL oracle rand90 ss={ss} ==="
{{
  time python src/scripts/joint/spatial/gp_dkl_oracle.py \\
    --config configs/gp/gp_gfa_hydroraum.yaml \\
    --split-file {split_csv} \\
    --run-tag {run_tag} \\
    --train-steps 500
}} 2>&1 | tee {log}

echo "=== DONE ss={ss} ===" """

    indented = "\n".join("          " + line for line in cmd.splitlines())

    return f"""\
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  labels:
    app: gw-dkl-oracle-rand90
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: gw-dkl-oracle-rand90
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
      - name: dkl-oracle
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
{indented}
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
        job_name = f"gw-dkl-oracle-rand90-ss{ss}"
        if job_name in existing:
            print(f"  skip (exists)  {job_name}")
            skipped += 1
            continue

        yaml_str = _job_yaml(ss)
        if args.dry_run:
            print(f"  dry-run  {job_name}")
            print(yaml_str)
            submitted += 1
            continue

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_str)
            tmp = f.name
        try:
            result = subprocess.run(
                ["kubectl", "apply", "-f", tmp],
                capture_output=True, text=True,
            )
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
