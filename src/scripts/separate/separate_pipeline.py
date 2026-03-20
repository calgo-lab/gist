import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable

USE_BASELINE = True

BASELINE_SCRIPTS = {
    "train": "src/scripts/separate/temporal/kunz_darts/tft_train.py",
    "eval": "src/scripts/separate/temporal/kunz_darts/tft_eval.py",
    "runs": "src/scripts/separate/temporal/kunz_darts/tft_runs.py",
    "spatial": "src/scripts/separate/spatial/kriging.py",
}

CUSTOM_SCRIPTS = {
    "train": "src/scripts/joint/temporal/gru_train.py",
    "eval": "src/scripts/joint/temporal/gru_eval.py",
    "runs": "src/scripts/joint/temporal/gru_runs.py",
    "spatial": "src/scripts/joint/spatial/gp_eval.py",
}

SCRIPTS = BASELINE_SCRIPTS if USE_BASELINE else CUSTOM_SCRIPTS


def main():
    stages = sys.argv[1:] if len(sys.argv) > 1 else ["train", "eval", "spatial"]
    env = os.environ.copy()
    for stage in stages:
        script_path = ROOT / SCRIPTS[stage]
        print(f"==> {stage}: {script_path}")
        subprocess.run([PYTHON, str(script_path)], check=True, env=env)


if __name__ == "__main__":
    main()
