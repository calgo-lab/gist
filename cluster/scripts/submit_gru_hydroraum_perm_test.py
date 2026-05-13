
import subprocess
import tempfile
from pathlib import Path

SS         = 42
MODEL_SEED = 40
RUN_SIG    = f"in52_out16_ep50_bs4096_seed{MODEL_SEED}_gru_hr_test_coloc_rand_f95_ss{SS}"
SPLIT_FILE = f"splits/spatial_split_full_merged_coloc_rand_f95_ss{SS}.csv"

AFFINITY = """
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: NotIn
                values:
                - cl-worker24
                - cl-worker27
              - key: gpu
                operator: In
                values:
                - a100
                - h100
"""

SPLIT_CODE = f"""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path

SPLIT_FILE = Path("{SPLIT_FILE}")
if SPLIT_FILE.exists():
    split = pd.read_csv(SPLIT_FILE)
    print(f"Split exists: {{(split.spatial_split=='spatial_train').sum()}} train / {{(split.spatial_split=='spatial_holdout').sum()}} holdout — skipping.")
    exit(0)

gws  = pd.read_parquet("/storage/data/merged.parquet")
meta = gws.groupby("id").agg(x=("x_25833","first"), y=("y_25833","first")).reset_index()
tree = cKDTree(meta[["x","y"]].values)
co_located = set()
for i, j in tree.query_pairs(r=8.0):
    co_located.add(meta.iloc[i]["id"])
    co_located.add(meta.iloc[j]["id"])
print(f"Co-located forced to train: {{len(co_located)}}")
eligible = meta[~meta["id"].isin(co_located)]["id"].values
print(f"Eligible for holdout: {{len(eligible)}}")

rng = np.random.default_rng({SS})
holdout_ids = set(rng.choice(eligible, size=52, replace=False).tolist())

split = pd.DataFrame({{"id": meta["id"].tolist()}})
split["spatial_split"] = "spatial_train"
split.loc[split["id"].isin(holdout_ids), "spatial_split"] = "spatial_holdout"
print(f"Split: {{(split.spatial_split=='spatial_train').sum()}} train / {{(split.spatial_split=='spatial_holdout').sum()}} holdout")
SPLIT_FILE.parent.mkdir(exist_ok=True)
split.to_csv(SPLIT_FILE, index=False)
print(f"Saved: {{SPLIT_FILE}}")
"""

PATCH_HYDRORAUM_CODE = r"""
src_path = "src/scripts/joint/temporal/gru_train.py"
src = open(src_path).read()
marker = '    "|^gw_gespannt_bin$"'
hr_lines = (
    '\n    "|^hydroraum_Entlastungsgebiete$"'
    '\n    "|^hydroraum_Transitgebiete$"'
    '\n    "|^hydroraum_Speisungsgebiete$"'
)
if "hydroraum_Entlastungsgebiete" not in src:
    assert marker in src, f"Marker not found in {src_path}"
    src = src.replace(marker, marker + hr_lines, 1)
    open(src_path, "w").write(src)
    print("hydroraum added to STATIC_FEATURE_REGEX in gru_train.py")
else:
    print("hydroraum already present in gru_train.py — no patch needed")
"""


def make_yaml():
    job_name = "gw-gru-hr-perm-test"
    cfg_path = f"/tmp/{job_name}.yaml"
    log_file = f"reports/gru_gp/logs/{job_name}.log"

    cmd = f"""\
set -euo pipefail
source /storage/venv/bin/activate
cd /storage/gwl-interpolation
mkdir -p reports/gru_gp/logs reports/feature_importance

echo "=== Patch gru_train.py (restore hydroraum lines if needed) ==="
python3 << 'PYEOF'
{PATCH_HYDRORAUM_CODE}
PYEOF

echo "=== Split: coloc rand f95 ss{SS} ==="
python3 << 'PYEOF'
{SPLIT_CODE}
PYEOF

echo "=== Build config (add_hydroraum_onehot=true) ==="
python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/gru/gru.yaml'))
cfg['run_sig']              = '{RUN_SIG}'
cfg['dataset']              = 'full_merged'
cfg['add_hydroraum_onehot'] = True
cfg['training']['seed']     = {MODEL_SEED}
cfg['spatial_split']        = {{'file': '{SPLIT_FILE}'}}
yaml.dump(cfg, open('{cfg_path}', 'w'), default_flow_style=False)
print('Config written.')
"

echo "=== GRU train (with hydroraum) ==="
{{
  time python src/scripts/joint/temporal/gru_train.py --config {cfg_path}
}} 2>&1 | tee {log_file}

echo "=== Look up run_id by run_sig ==="
RUN_ID=$(python3 -c "
import sys; sys.path.insert(0, 'src')
from libs.run_registry import lookup_run_id
from pathlib import Path
rid = lookup_run_id(Path('outputs'), 'GRU_FCOV', '{RUN_SIG}')
print(rid)
")
echo "Run ID: $RUN_ID"

echo "=== Permutation importance ==="
time python src/analysis/gru_permutation_importance.py \\
  --run-id "$RUN_ID" \\
  --n-repeats 5 \\
  --seed 0 \\
  2>&1 | tee -a {log_file}

echo "=== DONE: hydroraum perm test ==="
echo "Results: reports/feature_importance/gru_perm_importance_${{RUN_ID}}.md"
"""

    indented = "\n".join("          " + line for line in cmd.splitlines())

    return f"""\
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  labels:
    app: gw-gru-hr-perm-test
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: gw-gru-hr-perm-test
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
      - name: gru-hr-perm-test
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

    yaml_str = make_yaml()
    if args.dry_run:
        print(yaml_str)
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_str)
        tmp = f.name

    result = subprocess.run(["kubectl", "apply", "-f", tmp], capture_output=True, text=True)
    Path(tmp).unlink(missing_ok=True)
    if result.returncode == 0:
        print("Submitted: gw-gru-hr-perm-test")
        print("Watch: kubectl logs -f job/gw-gru-hr-perm-test")
        print("Results will be at: reports/feature_importance/gru_perm_importance_<run_id>.md")
    else:
        print(f"FAILED: {result.stderr.strip()}")


if __name__ == "__main__":
    main()
