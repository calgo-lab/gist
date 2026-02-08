import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

SCRIPTS = {
    "train": "src/scripts/separate/temporal/kunz_darts/tft_train.py",
    "eval": "src/scripts/separate/temporal/kunz_darts/tft_eval.py",
    "runs": "src/scripts/separate/temporal/kunz_darts/tft_runs.py",
    "kriging": "src/scripts/separate/spatial/kriging.py",
}


def main():
    stages = sys.argv[1:] if len(sys.argv) > 1 else ["train", "eval", "kriging"]

    base_env = os.environ.copy()

    for stage in stages:
        script_path = ROOT / SCRIPTS[stage]
        env = base_env.copy()
        print(f"==> {stage}: {script_path}")
        subprocess.run([PYTHON, str(script_path)], check=True, env=env)


if __name__ == "__main__":
    main()
