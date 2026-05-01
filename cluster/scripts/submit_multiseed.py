
import argparse
import os
import subprocess
import textwrap
import tempfile
from itertools import product
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Matrix — must match cluster/scripts/generate_splits.py
# (split_type, d_percentile_or_None, train_frac, split_seeds)
# ---------------------------------------------------------------------------

MODEL_SEEDS = [40, 41, 42, 43, 44]

DECOUPLED_GROUPS = [
    # random — 0.95×10 seeds + 0.90×5 + 0.80×5 = 20 splits → ×5 model seeds = 100
    ("random", None, 0.95, [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]),
    ("random", None, 0.90, [42, 43, 44, 45, 46]),
    ("random", None, 0.80, [42, 43, 44, 45, 46]),
    # max_dist p10 ceiling — 1 split × 5 model seeds = 5
    ("max_dist", 10,  0.95, [42]),
    # max_dist p90 — 5 splits × 5 model seeds = 25
    ("max_dist", 90,  0.95, [42, 43, 44, 45, 46]),
    # max_dist p100 — 5 splits × 5 model seeds = 25
    ("max_dist", 100, 0.95, [42, 43, 44, 45, 46]),
    # kmeans — 5 splits × 5 model seeds = 25
    ("kmeans",  None, 0.95, [42, 43, 44, 45, 46]),
]
# Total decoupled: 100 + 5 + 25 + 25 + 25 = 180

JOINT_GROUPS = [
    # max_dist p90 — 2 splits × 5 model seeds = 10
    ("max_dist", 90, 0.95, [42, 43]),
]
# Grand total: 190

DATASET = "full_merged_dedup"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frac_str(f: float) -> str:
    return str(f).replace(".", "p")

def _type_abbrev(split_type: str, pct: Optional[int]) -> str:
    if split_type == "random": return "rand"
    if split_type == "kmeans": return "km"
    return f"md{pct}"

def _frac_abbrev(f: float) -> str:
    return f"f{int(round(f * 100))}"

def _split_csv(split_type, pct, frac, ss) -> str:
    fs = _frac_str(frac)
    if split_type == "random":
        return f"splits/spatial_split_{DATASET}_random_f{fs}_ss{ss}.csv"
    if split_type == "kmeans":
        return f"splits/spatial_split_{DATASET}_kmeans_f{fs}_sc10_ss{ss}.csv"
    return f"splits/spatial_split_{DATASET}_max_dist_f{fs}_k3_p{pct}_ss{ss}.csv"

def _run_sig(split_type, pct, frac, ss, ms) -> str:
    return (f"in52_out16_ep50_bs4096_seed{ms}_dedup"
            f"_{_type_abbrev(split_type, pct)}_{_frac_abbrev(frac)}_ss{ss}")

def _job_name(prefix, split_type, pct, frac, ss, ms) -> str:
    return f"gw-ms-{prefix}-{_type_abbrev(split_type, pct)}-{_frac_abbrev(frac)}-{ss}-{ms}"

# ---------------------------------------------------------------------------
# Node affinity — no nodeSelector, excludes cl-worker24 / k80 / p100 / false
# ---------------------------------------------------------------------------

AFFINITY = """\
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: NotIn
                values:
                - cl-worker24
              - key: gpu
                operator: NotIn
                values:
                - k80
                - p100
                - "false"
"""

# ---------------------------------------------------------------------------
# YAML builders
# ---------------------------------------------------------------------------

def _env_block(wandb_api_key: str) -> str:
    lines = [
        "        env:",
        "        - name: PYTHONUNBUFFERED",
        '          value: "1"',
        "        - name: WANDB_PROJECT",
        '          value: "gwl-interpolation"',
        "        - name: WANDB_ENTITY",
        '          value: "row56"',
    ]
    if wandb_api_key:
        lines += ["        - name: WANDB_API_KEY", f'          value: "{wandb_api_key}"']
    else:
        lines += ["        - name: WANDB_MODE", '          value: "disabled"']
    return "\n".join(lines)


def _spatial_cfg_literal(split_type, pct, frac, ss) -> str:
    """Python dict literal (single-quoted) with explicit file: path."""
    csv = _split_csv(split_type, pct, frac, ss)
    pct_val = pct if pct is not None else 25
    return (
        f"{{'file': '{csv}', 'train_fraction': {frac}, 'split_seed': {ss}, "
        f"'cluster_count': 10, 'max_dist_k': 3, 'max_dist_percentile': {pct_val}, "
        f"'exclude_terms': 'geometry,x_25833,y_25833'}}"
    )


def _decoupled_yaml(job_name, run_sig, split_type, pct, frac, ss, ms, wandb_key):
    spatial = _spatial_cfg_literal(split_type, pct, frac, ss)
    env = _env_block(wandb_key)
    cfg_script = (
        "import yaml; "
        "cfg = yaml.safe_load(open('configs/gru/gru.yaml')); "
        f"cfg['run_sig'] = '{run_sig}'; "
        f"cfg['training']['seed'] = {ms}; "
        "cfg['exclude_static_features'] = ['parde_seasonality', 'GW_recharge_r1000m', 'gw_gespannt_bin', 'siwa_verweilzeit_j']; "
        "cfg['gp_features'] = ['siwa_verweilzeit_j']; "
        "cfg['gp_features_onehot'] = ['hydroraum', 'gw_gespannt']; "
        f"cfg['spatial_split'] = {spatial}; "
        "yaml.dump(cfg, open('$CFG', 'w'), default_flow_style=False)"
    )

    cmd = textwrap.dedent(f"""\
        set -euo pipefail
        source /storage/venv/bin/activate
        cd /storage/gwl-interpolation

        CFG=/tmp/{job_name}.yaml
        python3 -c "{cfg_script}"

        mkdir -p reports/gru_gp/logs
        {{
          echo '=== TRAIN ==='
          time python src/scripts/joint/temporal/gru_train.py --config $CFG
          echo '=== EVAL ==='
          time python src/scripts/joint/temporal/gru_eval.py --config $CFG
          echo '=== GP ==='
          time python src/scripts/joint/spatial/gp_eval.py \\
            --config $CFG \\
            --gru-run-sig {run_sig} \\
            --gp-run-tag ms \\
            --backend gpytorch \\
            --n-inducing 64 \\
            --pretrain-steps 300 \\
            --pretrain-lr 0.01 \\
            --max-pretrain-pts 2000 \\
            --date-freq ME \\
            --jitter 0.00001 \\
            --use-float64 true \\
            --model-prefix GRU_FCOV
        }} 2>&1 | tee reports/gru_gp/logs/{job_name}.log
    """)

    return _k8s_job(job_name, "gw-multiseed", "gru-decoupled", env, cmd)


def _joint_yaml(job_name, run_sig, split_type, pct, frac, ss, ms, wandb_key):
    spatial = _spatial_cfg_literal(split_type, pct, frac, ss)
    env = _env_block(wandb_key)
    cfg_script = (
        "import yaml; "
        "cfg = yaml.safe_load(open('configs/joint/gru_gp.yaml')); "
        f"cfg['run_sig'] = '{run_sig}'; "
        f"cfg['training']['seed'] = {ms}; "
        "cfg['model']['gru_hidden'] = 128; "
        "cfg['model']['gru_layers'] = 5; "
        "cfg['model']['gru_dropout'] = 0.3; "
        "cfg['exclude_static_features'] = ['parde_seasonality', 'GW_recharge_r1000m', 'gw_gespannt_bin', 'siwa_verweilzeit_j']; "
        "cfg.setdefault('spatial_gp', {}); "
        "cfg['spatial_gp']['gp_features'] = ['siwa_verweilzeit_j']; "
        "cfg['spatial_gp']['gp_features_onehot'] = ['hydroraum', 'gw_gespannt']; "
        "cfg['spatial_gp']['lambda_spatial'] = 0.034; "
        "cfg['spatial_gp']['jitter'] = 0.001; "
        "cfg['spatial_gp']['use_float64'] = True; "
        "cfg.setdefault('joint', {}); "
        "cfg['joint']['gru_lr'] = 0.000107; "
        "cfg['joint']['val_from_train_fraction'] = 0.05; "
        f"cfg['spatial_split'] = {spatial}; "
        "yaml.dump(cfg, open('$CFG', 'w'), default_flow_style=False)"
    )

    cmd = textwrap.dedent(f"""\
        set -euo pipefail
        source /storage/venv/bin/activate
        cd /storage/gwl-interpolation

        CFG=/tmp/{job_name}.yaml
        python3 -c "{cfg_script}"

        mkdir -p reports/gru_gp/logs
        {{
          echo '=== JOINT TRAIN ==='
          time python src/scripts/joint/gru_gp_train.py --config $CFG
          echo '=== JOINT EVAL ==='
          time python src/scripts/joint/gru_gp_eval.py --config $CFG
        }} 2>&1 | tee reports/gru_gp/logs/{job_name}.log
    """)

    return _k8s_job(job_name, "gw-multiseed-joint", "gru-joint", env, cmd)


def _k8s_job(job_name, app_label, container_name, env_block, cmd):
    return f"""\
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  labels:
    app: {app_label}
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: {app_label}
    spec:
      restartPolicy: OnFailure
{AFFINITY}\
      volumes:
      - name: gw-pred-rwx
        persistentVolumeClaim:
          claimName: gw-pred-rwx
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: "8Gi"
      containers:
      - name: {container_name}
        image: row56/gw-pred:py312-cu124
        imagePullPolicy: Always
        resources:
          requests:
            cpu: "8"
            memory: 48Gi
            nvidia.com/gpu: 1
          limits:
            memory: 48Gi
            nvidia.com/gpu: 1
{env_block}
        volumeMounts:
        - name: gw-pred-rwx
          mountPath: /storage
        - name: dshm
          mountPath: /dev/shm
        command: ["bash", "-lc"]
        args:
        - |
{textwrap.indent(cmd, "          ")}"""

# ---------------------------------------------------------------------------
# Job enumeration
# ---------------------------------------------------------------------------

class JobSpec:
    def __init__(self, job_type, split_type, pct, frac, ss, ms):
        self.job_type = job_type
        self.split_type = split_type
        self.pct  = pct
        self.frac = frac
        self.ss   = ss
        self.ms   = ms

    @property
    def name(self):
        prefix = "j" if self.job_type == "joint" else "d"
        return _job_name(prefix, self.split_type, self.pct, self.frac, self.ss, self.ms)

    @property
    def run_sig(self):
        return _run_sig(self.split_type, self.pct, self.frac, self.ss, self.ms)

    def to_yaml(self, wandb_key):
        if self.job_type == "joint":
            return _joint_yaml(self.name, self.run_sig, self.split_type,
                               self.pct, self.frac, self.ss, self.ms, wandb_key)
        return _decoupled_yaml(self.name, self.run_sig, self.split_type,
                               self.pct, self.frac, self.ss, self.ms, wandb_key)


def build_job_list(include_decoupled=True, include_joint=True):
    jobs = []
    if include_decoupled:
        for split_type, pct, frac, seeds in DECOUPLED_GROUPS:
            for ss, ms in product(seeds, MODEL_SEEDS):
                jobs.append(JobSpec("decoupled", split_type, pct, frac, ss, ms))
    if include_joint:
        for split_type, pct, frac, seeds in JOINT_GROUPS:
            for ss, ms in product(seeds, MODEL_SEEDS):
                jobs.append(JobSpec("joint", split_type, pct, frac, ss, ms))
    return jobs

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--joint-only",     action="store_true")
    p.add_argument("--decoupled-only", action="store_true")
    p.add_argument("--no-wandb",       action="store_true")
    p.add_argument("--type",  help="Filter by type abbrev (rand, km, md90, ...)")
    p.add_argument("--frac",  type=int, help="Filter by frac as int (95, 90, 80)")
    p.add_argument("--ss",    type=int, help="Filter by split seed")
    p.add_argument("--ms",    type=int, help="Filter by model seed")
    args = p.parse_args()

    wandb_key = "" if args.no_wandb else os.environ.get("WANDB_API_KEY", "")
    if not wandb_key and not args.no_wandb and not args.dry_run:
        print("WARNING: WANDB_API_KEY not set — WandB will be disabled.")

    jobs = build_job_list(
        include_decoupled=not args.joint_only,
        include_joint=not args.decoupled_only,
    )

    if args.type:
        jobs = [j for j in jobs if _type_abbrev(j.split_type, j.pct) == args.type]
    if args.frac is not None:
        jobs = [j for j in jobs if int(round(j.frac * 100)) == args.frac]
    if args.ss is not None:
        jobs = [j for j in jobs if j.ss == args.ss]
    if args.ms is not None:
        jobs = [j for j in jobs if j.ms == args.ms]

    print(f"Jobs to submit: {len(jobs)}")

    if args.dry_run:
        for j in jobs:
            print(f"  {j.job_type:10s}  {j.name}  {j.run_sig}")
        return

    submitted, failed = 0, 0
    for j in jobs:
        yaml_str = j.to_yaml(wandb_key)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_str)
            tmp = f.name
        try:
            result = subprocess.run(["kubectl", "apply", "-f", tmp],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  submitted  {j.name}")
                submitted += 1
            else:
                print(f"  FAILED     {j.name}: {result.stderr.strip()}")
                failed += 1
        finally:
            Path(tmp).unlink(missing_ok=True)

    print(f"\nDone: {submitted} submitted, {failed} failed.")


if __name__ == "__main__":
    main()
