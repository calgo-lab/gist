import subprocess
import tempfile
from pathlib import Path

MD_PCTS  = [10, 20, 30, 40, 50, 60, 70, 80, 90]
SS       = 42
GPS      = 1

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


def _split_code(pct):
    split_file = f"splits/spatial_split_full_merged_coloc_md{pct}_f95_ss{SS}.csv"
    return f"""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path

SPLIT_FILE = Path("{split_file}")
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

eligible = meta[~meta["id"].isin(co_located)].reset_index(drop=True)
print(f"Eligible for holdout: {{len(eligible)}}")

elig_coords = eligible[["x","y"]].values
dists, _ = cKDTree(elig_coords).query(elig_coords, k=4)
mean_dist = dists[:, 1:].mean(axis=1)
d_thresh  = np.percentile(mean_dist, {pct})
candidates = eligible.loc[mean_dist <= d_thresh, "id"].values
print(f"md{pct} candidates: {{len(candidates)}}")

n_holdout = 52
rng = np.random.default_rng({SS})
if len(candidates) < n_holdout:
    raise ValueError(f"Not enough candidates ({{len(candidates)}}) for {{n_holdout}} holdout wells.")
holdout_ids = set(rng.choice(candidates, size=n_holdout, replace=False).tolist())

split = pd.DataFrame({{"id": meta["id"].tolist()}})
split["spatial_split"] = "spatial_train"
split.loc[split["id"].isin(holdout_ids), "spatial_split"] = "spatial_holdout"
print(f"Split: {{(split.spatial_split=='spatial_train').sum()}} train / {{(split.spatial_split=='spatial_holdout').sum()}} holdout")
SPLIT_FILE.parent.mkdir(exist_ok=True)
split.to_csv(SPLIT_FILE, index=False)
print(f"Saved: {{SPLIT_FILE}}")
"""


def _oracle_pred_code(pct):
    split_file  = f"splits/spatial_split_full_merged_coloc_md{pct}_f95_ss{SS}.csv"
    oracle_path = f"/tmp/oracle_pred_md{pct}_f95_ss{SS}.parquet"
    cfg_path    = f"/tmp/gp_oracle_md{pct}_f95_ss{SS}.yaml"
    return f"""
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, yaml

IN_LEN, OUT_LEN = 52, 16
VAL_CUTOFF  = pd.Timestamp("20200101")
SPLIT_FILE  = "{split_file}"
ORACLE_PATH = "{oracle_path}"
CFG_PATH    = "{cfg_path}"

split = pd.read_csv(SPLIT_FILE)
train_ids = set(split.loc[split["spatial_split"] == "spatial_train", "id"])
print(f"  {{len(train_ids)}} spatial_train wells")

gws = pq.read_table("/storage/data/merged.parquet", columns=["datum","id","gws"]).to_pandas()
gws["datum"] = pd.to_datetime(gws["datum"])
gws = gws[gws["id"].isin(train_ids)].sort_values(["id","datum"])

rows = []
for gid, g in gws.groupby("id"):
    g = g.sort_values("datum").reset_index(drop=True)
    times, vals = g["datum"].values, g["gws"].values
    for i in range(IN_LEN, len(g) - OUT_LEN + 1):
        if times[i + OUT_LEN - 1] <= np.datetime64(VAL_CUTOFF):
            continue
        for h in range(OUT_LEN):
            v = float(vals[i + h])
            rows.append({{"id": gid, "datum": pd.Timestamp(times[i+h]),
                          "horizon": h+1, "gws": v if np.isfinite(v) else float("nan")}})
df = pd.DataFrame(rows)
print(f"  {{df['id'].nunique()}} wells, {{df['datum'].nunique()}} dates, {{len(df):,}} rows")
pq.write_table(pa.Table.from_pandas(df), ORACLE_PATH)

cfg = yaml.safe_load(open("configs/gp/gp.yaml"))
cfg["gp_features"]        = []
cfg["gp_features_onehot"] = ["hydroraum"]
cfg["spatial_split"]      = {{"file": SPLIT_FILE}}
yaml.dump(cfg, open(CFG_PATH, "w"), default_flow_style=False)
print("  oracle pred + config ready")
"""


def make_yaml(pct):
    job_name    = f"gw-oracle-md-sweep-{pct}"
    split_label = f"coloc_md{pct}_f95_ss{SS}"
    run_sig     = f"oracle_{split_label}"
    oracle_path = f"/tmp/oracle_pred_md{pct}_f95_ss{SS}.parquet"
    cfg_path    = f"/tmp/gp_oracle_md{pct}_f95_ss{SS}.yaml"
    log_file    = f"reports/gru_gp/logs/{job_name}.log"

    split_code  = _split_code(pct)
    oracle_code = _oracle_pred_code(pct)

    gp_eval = f"""\
  echo "=== GP EVAL md{pct} gps={GPS} ==="
  time python src/scripts/joint/spatial/gp_eval.py \\
    --config {cfg_path} \\
    --pred-path {oracle_path} \\
    --gru-run-sig {run_sig} \\
    --gp-run-tag md_sweep_pct{pct} \\
    --pretrain-steps 300 \\
    --pretrain-lr 0.01 \\
    --max-pretrain-pts 2000 \\
    --date-freq D \\
    --jitter 0.00001 \\
    --use-float64 true \\
    --model-prefix ORACLE \\
    --gp-seed {GPS}
"""

    cmd = f"""\
set -euo pipefail
source /storage/venv/bin/activate
cd /storage/gwl-interpolation
mkdir -p reports/gru_gp/logs

echo "=== Split: {split_label} ==="
python3 << 'PYEOF'
{split_code}
PYEOF

echo "=== Oracle pred + GP config ==="
python3 << 'PYEOF'
{oracle_code}
PYEOF

{{
{gp_eval}}} 2>&1 | tee {log_file}

echo "=== DONE: {run_sig} ==="
"""

    indented = "\n".join("          " + line for line in cmd.splitlines())

    return f"""\
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  labels:
    app: gw-oracle-md-sweep
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: gw-oracle-md-sweep
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
      - name: oracle-md-sweep
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

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Submitting {len(MD_PCTS)} oracle md-sweep jobs...")
    submitted, failed = 0, 0
    for pct in MD_PCTS:
        job_name = f"gw-oracle-md-sweep-{pct}"
        yaml_str = make_yaml(pct)
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
