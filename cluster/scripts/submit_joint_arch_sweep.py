import subprocess
import tempfile
from pathlib import Path

SS          = 42
MODEL_SEED  = 40
LAMBDA_SP   = 0.034
SPLIT_FILE  = f"splits/spatial_split_full_merged_coloc_rand_f95_joint_ss{SS}.csv"

ARCHS = [
    {"tag": "h128l5", "hidden": 128, "layers": 5, "dropout": 0.3},
    {"tag": "h256l4", "hidden": 256, "layers": 4, "dropout": 0.2},
]

AFFINITY = """
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: gpu
                operator: In
                values:
                - a100
                - h100
"""


def _split_code():
    return f"""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path

SPLIT_FILE = Path("{SPLIT_FILE}")
if SPLIT_FILE.exists():
    split = pd.read_csv(SPLIT_FILE)
    if "spatial_val" in split["spatial_split"].values:
        print(f"Split exists: {{(split.spatial_split=='spatial_train').sum()}} train / {{(split.spatial_split=='spatial_val').sum()}} val / {{(split.spatial_split=='spatial_holdout').sum()}} holdout — skipping.")
        exit(0)
    print("Split exists but missing spatial_val — recreating.")

gws  = pd.read_parquet("/storage/data/merged.parquet")
meta = gws.groupby("id").agg(x=("x_25833","first"), y=("y_25833","first")).reset_index()
tree = cKDTree(meta[["x","y"]].values)
co_located = set()
for i, j in tree.query_pairs(r=8.0):
    co_located.add(meta.iloc[i]["id"])
    co_located.add(meta.iloc[j]["id"])
print(f"Co-located forced to train: {{len(co_located)}}")
eligible = meta[~meta["id"].isin(co_located)]["id"].values
print(f"Eligible for holdout/val: {{len(eligible)}}")

rng = np.random.default_rng({SS})
shuffled = rng.permutation(eligible)
holdout_ids = set(shuffled[:52].tolist())
val_ids     = set(shuffled[52:78].tolist())

split = pd.DataFrame({{"id": meta["id"].tolist()}})
split["spatial_split"] = "spatial_train"
split.loc[split["id"].isin(holdout_ids), "spatial_split"] = "spatial_holdout"
split.loc[split["id"].isin(val_ids),     "spatial_split"] = "spatial_val"
print(f"Split: {{(split.spatial_split=='spatial_train').sum()}} train / {{(split.spatial_split=='spatial_val').sum()}} val / {{(split.spatial_split=='spatial_holdout').sum()}} holdout")
SPLIT_FILE.parent.mkdir(exist_ok=True)
split.to_csv(SPLIT_FILE, index=False)
print(f"Saved: {{SPLIT_FILE}}")
"""


def make_yaml(arch):
    tag     = arch["tag"]
    hidden  = arch["hidden"]
    layers  = arch["layers"]
    dropout = arch["dropout"]

    job_name = f"gw-joint-arch-{tag}"
    run_sig  = (
        f"joint_in52_out16_ep50_bs8192_seed{MODEL_SEED}"
        f"_coloc_rand_f95_joint_ss{SS}_{tag}"
    )
    cfg_path = f"/tmp/joint_arch_{tag}.yaml"
    log_file = f"reports/gru_gp/logs/{job_name}.log"

    split_code = _split_code()

    cmd = f"""\
set -euo pipefail
source /storage/venv/bin/activate
cd /storage/gwl-interpolation
mkdir -p reports/gru_gp/logs

echo "=== Split: coloc rand f95 ss{SS} ==="
python3 << 'PYEOF'
{split_code}
PYEOF

echo "=== Build config ({tag}, lsp={LAMBDA_SP}) ==="
python3 -c "
import yaml
cfg = {{
  'dataset': 'full_merged',
  'run_sig': '{run_sig}',
  'data': {{'in_len': 52, 'out_len': 16}},
  'model': {{'gru_hidden': {hidden}, 'gru_layers': {layers}, 'gru_dropout': {dropout}, 'dropout': {dropout}}},
  'training': {{'epochs': 50, 'batch_size': 8192, 'seed': {MODEL_SEED},
               'num_workers': 10, 'grad_clip': 1.0,
               'early_stopping': {{'monitor': 'val_loss', 'patience': 15, 'min_delta': 0.0001, 'mode': 'min'}}}},
  'spatial_split': {{'file': '{SPLIT_FILE}'}},
  'spatial_gp': {{
    'backend': 'gpytorch', 'n_inducing': 128, 'jitter': 1e-5,
    'use_float64': True, 'lambda_spatial': {LAMBDA_SP},
    'gp_pretrain_steps': 300, 'variational_lr': 0.01,
    'max_train_points_per_horizon': 2000,
    'gp_features': [],
    'gp_features_onehot': ['hydroraum'],
  }},
  'joint': {{
    'n_gp_dates': 8, 'gp_weight': 0.5,
    'min_train_wells': 5, 'min_val_wells': 2,
    'val_gp_stride': 4, 'gru_pretrained_run_sig': '',
    'gru_lr': 1.07e-4, 'gp_lr': 1e-3,
  }},
}}
yaml.dump(cfg, open('{cfg_path}', 'w'))
print('Config written.')
"

{{
  echo "=== JOINT TRAIN {tag} ==="
  time python src/scripts/joint/gru_gp_train.py --config {cfg_path}
  echo "=== JOINT EVAL ==="
  time python src/scripts/joint/gru_gp_eval.py --config {cfg_path}
}} 2>&1 | tee {log_file}

echo "=== DONE: {run_sig} ==="
"""

    indented = "\n".join("          " + line for line in cmd.splitlines())

    return f"""\
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  labels:
    app: gw-joint-arch-sweep
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: gw-joint-arch-sweep
    spec:
      restartPolicy: Never
{AFFINITY}
      volumes:
      - name: gw-pred-rwx
        persistentVolumeClaim:
          claimName: gw-pred-rwx
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: "16Gi"
      containers:
      - name: joint-arch-sweep
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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Submitting {len(ARCHS)} arch-sweep jobs...")
    submitted, failed = 0, 0
    for arch in ARCHS:
        job_name = f"gw-joint-arch-{arch['tag']}"
        yaml_str = make_yaml(arch)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_str)
            tmp = f.name
        if args.dry_run:
            print(f"  [dry] {job_name}  ({tmp})")
            Path(tmp).unlink(missing_ok=True)
            submitted += 1
            continue
        result = subprocess.run(["kubectl", "apply", "-f", tmp], capture_output=True, text=True)
        Path(tmp).unlink(missing_ok=True)
        if result.returncode == 0:
            print(f"  submitted  {job_name}")
            submitted += 1
        else:
            print(f"  FAILED     {job_name}: {result.stderr.strip()}")
            failed += 1
    print(f"\nDone: {submitted} submitted, {failed} failed.")


if __name__ == "__main__":
    main()
