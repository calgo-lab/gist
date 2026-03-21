"""
Sequential run ID assignment and lookup.

Registry file: outputs/run_registry.csv
Columns: run_id, created_at, model_type, run_sig, <config fields...>

Usage
-----
from libs.run_registry import assign_run_id, lookup_run_id

# In train script:
run_id = assign_run_id(ROOT / "outputs", "GRU_FCOV", run_sig, meta={...})
run_dir = ROOT / "outputs" / "GRU_FCOV" / f"GRU_FCOV_{run_id}"

# In eval script:
run_id = lookup_run_id(ROOT / "outputs", "GRU_FCOV", run_sig)
run_dir = ROOT / "outputs" / "GRU_FCOV" / f"GRU_FCOV_{run_id}"
"""
from __future__ import annotations

import fcntl
from datetime import datetime
from pathlib import Path

import pandas as pd

_REGISTRY_FILE = "run_registry.csv"
_LOCK_FILE = "run_registry.lock"


def assign_run_id(
    outputs_root: Path | str,
    model_type: str,
    run_sig: str,
    meta: dict | None = None,
) -> str:
    """Register a new run and return its zero-padded 4-digit ID string."""
    outputs_root = Path(outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)
    registry_path = outputs_root / _REGISTRY_FILE
    lock_path = outputs_root / _LOCK_FILE

    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if registry_path.exists():
                df = pd.read_csv(registry_path)
                next_id = int(df["run_id"].max()) + 1 if len(df) > 0 else 1
            else:
                df = pd.DataFrame()
                next_id = 1

            row: dict = {
                "run_id": next_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "model_type": model_type,
                "run_sig": run_sig,
            }
            if meta:
                row.update(meta)

            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(registry_path, index=False)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

    return f"{next_id:04d}"


def lookup_run_id(
    outputs_root: Path | str,
    model_type: str,
    run_sig: str,
) -> str:
    """Return the most recent run ID for the given model_type + run_sig."""
    registry_path = Path(outputs_root) / _REGISTRY_FILE
    if not registry_path.exists():
        raise FileNotFoundError(f"Run registry not found: {registry_path}")

    df = pd.read_csv(registry_path)
    matches = df[(df["model_type"] == model_type) & (df["run_sig"] == run_sig)]
    if matches.empty:
        raise ValueError(
            f"No run in registry for model_type={model_type!r}, run_sig={run_sig!r}"
        )
    return f"{int(matches.iloc[-1]['run_id']):04d}"


def read_registry(outputs_root: Path | str) -> pd.DataFrame:
    """Return the full registry, or empty DataFrame if not found."""
    registry_path = Path(outputs_root) / _REGISTRY_FILE
    if not registry_path.exists():
        return pd.DataFrame()
    return pd.read_csv(registry_path)
