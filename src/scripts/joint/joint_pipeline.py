import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable

SCRIPTS = {
    "train": "src/scripts/joint/gru_gp_train.py",
    "eval": "src/scripts/joint/gru_gp_eval.py",
}


def main():
    stages = sys.argv[1:] if len(sys.argv) > 1 else ["train", "eval"]
    env = os.environ.copy()
    for stage in stages:
        script_path = ROOT / SCRIPTS[stage]
        print(f"==> {stage}: {script_path}")
        subprocess.run([PYTHON, str(script_path)], check=True, env=env)


if __name__ == "__main__":
    main()
