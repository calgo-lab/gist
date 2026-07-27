import subprocess
import tempfile
from pathlib import Path

SS         = 42
MODEL_SEED = 40
RUN_SIG    = f"in52_out16_ep50_bs4096_seed{MODEL_SEED}_gru_39feat_coloc_rand_f95_ss{SS}"
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

PATCH_CODE = '''
import sys

MARKER_REGEX = '    "|^hydroraum_Speisungsgebiete$"'
NEW_REGEX_LINES = (
    \'\\n    "|^fok$"\'
    \'\\n    "|^fuk$"\'
    \'\\n    "|^gwlk_[123]$"\'
    \'\\n    "|^acf_class_ord$"\'
)

for sp in [
    "src/scripts/joint/temporal/gru_train.py",
    "src/analysis/gru_permutation_importance.py",
]:
    src = open(sp).read()
    if "^acf_class_ord$" not in src:
        assert MARKER_REGEX in src, f"MARKER_REGEX not found in {sp}"
        src = src.replace(MARKER_REGEX, MARKER_REGEX + NEW_REGEX_LINES, 1)
        open(sp, "w").write(src)
        print(f"  39-feat regex added to {sp}")
    else:
        print(f"  39-feat regex already present in {sp}")

# ── gru_train.py encoding (insert before split_path line) ───────────────────
sp = "src/scripts/joint/temporal/gru_train.py"
src = open(sp).read()
SENTINEL = \'    split_path = resolve_split_path(ROOT / "splits", dataset, spatial_cfg)\'
ENCODE = (
    "    if \'gwlk\' in gws_full.columns:\\n"
    "        gws_full[\'gwlk_parsed\'] = gws_full[\'gwlk\'].str.extract(r\'(\\d+)\').astype(float)\\n"
    "        for k in [1, 2, 3]:\\n"
    "            gws_full[f\'gwlk_{k}\'] = (gws_full[\'gwlk_parsed\'] == k).astype(\'float32\')\\n"
    "    _ACF_ORDER = {\'<6\': 1, \'<9\': 2, \'<12\': 3, \'<20\': 4, \'<53\': 5, \'>53\': 6}\\n"
    "    if \'acf_class\' in gws_full.columns:\\n"
    "        gws_full[\'acf_class_ord\'] = gws_full[\'acf_class\'].map(_ACF_ORDER).astype(\'float32\')\\n"
)
if "gwlk_parsed" not in src:
    assert SENTINEL in src, f"SENTINEL not found in {sp}"
    src = src.replace(SENTINEL, ENCODE + SENTINEL, 1)
    open(sp, "w").write(src)
    print(f"  gwlk/acf encoding added to {sp}")
else:
    print(f"  gwlk/acf encoding already present in {sp}")

# ── gru_permutation_importance.py encoding (insert before return df) ─────────
sp = "src/analysis/gru_permutation_importance.py"
src = open(sp).read()
OLD_RETURN = (
    \'    if "gw_gespannt" in df.columns:\\n\'
    \'        df["gw_gespannt_bin"] = (df["gw_gespannt"] == "gespannt").astype("float32")\\n\'
    \'    return df\'
)
ENCODE_PERM = (
    "    if \'gwlk\' in df.columns:\\n"
    "        df[\'gwlk_parsed\'] = df[\'gwlk\'].str.extract(r\'(\\d+)\').astype(float)\\n"
    "        for k in [1, 2, 3]:\\n"
    "            df[f\'gwlk_{k}\'] = (df[\'gwlk_parsed\'] == k).astype(\'float32\')\\n"
    "    _ACF_ORDER = {\'<6\': 1, \'<9\': 2, \'<12\': 3, \'<20\': 4, \'<53\': 5, \'>53\': 6}\\n"
    "    if \'acf_class\' in df.columns:\\n"
    "        df[\'acf_class_ord\'] = df[\'acf_class\'].map(_ACF_ORDER).astype(\'float32\')\\n"
    "    return df"
)
NEW_RETURN = OLD_RETURN.replace("    return df", ENCODE_PERM)
if "gwlk_parsed" not in src:
    assert OLD_RETURN in src, f"OLD_RETURN pattern not found in {sp}"
    src = src.replace(OLD_RETURN, NEW_RETURN, 1)
    open(sp, "w").write(src)
    print(f"  gwlk/acf encoding added to {sp}")
else:
    print(f"  gwlk/acf encoding already present in {sp}")

print("All patches done.")
'''


def make_yaml():
    job_name = "gw-gru-39feat-perm"
    cfg_path = f"/tmp/{job_name}.yaml"
    log_file = f"reports/gru_gp/logs/{job_name}.log"

    cmd = f"""\
set -euo pipefail
source /storage/venv/bin/activate
cd /storage/gwl-interpolation
mkdir -p reports/gru_gp/logs reports/feature_importance

echo "=== Patch STATIC_FEATURE_REGEX + encoding (39 features) ==="
python3 << 'PYEOF'
{PATCH_CODE}
PYEOF

echo "=== Build config ==="
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

echo "=== GRU train (39 static features) ==="
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
  2>&1 | tee reports/feature_importance/gru_perm_importance_39feat.log

echo "=== DONE: 39-feature GRU perm importance ==="
echo "Results: reports/feature_importance/gru_perm_importance_${{RUN_ID}}.md"
"""

    indented = "\n".join("          " + line for line in cmd.splitlines())

    return f"""\
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  labels:
    app: {job_name}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: {job_name}
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
      - name: {job_name}
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
        print("Submitted: gw-gru-39feat-perm")
        print("Watch: kubectl logs -f job/gw-gru-39feat-perm")
        print(f"Results will be at: reports/feature_importance/gru_perm_importance_<run_id>.md")
    else:
        print(f"FAILED: {result.stderr.strip()}")


if __name__ == "__main__":
    main()
