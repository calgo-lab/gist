import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "configs" / "experiments.yaml"


def load_registry(path=None):
    p = Path(path) if path is not None else REGISTRY_PATH
    if not p.exists():
        raise FileNotFoundError(f"experiment registry not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _merge_value(node, key, value):
    existing = node.get(key)
    if isinstance(value, dict) and isinstance(existing, dict):
        for k, v in value.items():
            _merge_value(existing, k, v)
    else:
        node[key] = value


def _apply_override(cfg, dotted_key, value):
    parts = str(dotted_key).split(".")
    node = cfg
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    _merge_value(node, parts[-1], value)


def apply_overrides(cfg, overrides):
    for key, value in (overrides or {}).items():
        _apply_override(cfg, key, value)
    return cfg


def parse_set_overrides(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        try:
            val = yaml.safe_load(raw)
        except Exception:
            val = raw
        out[key] = val
    return out


def resolve_experiment(name, extra_overrides=None, registry=None):
    registry = registry if registry is not None else load_registry()
    if name not in registry:
        raise KeyError(f"experiment '{name}' not in registry ({REGISTRY_PATH}). "
                       f"known: {sorted(registry)}")
    entry = registry[name]
    if not isinstance(entry, dict) or "base" not in entry:
        raise ValueError(f"registry entry '{name}' needs a 'base' key")

    base_path = Path(entry["base"])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"base config {base_path} did not parse to a dict")

    apply_overrides(cfg, entry.get("overrides") or {})
    apply_overrides(cfg, extra_overrides or {})
    return cfg, entry


def _git(args_list):
    try:
        out = subprocess.check_output(["git", *args_list], cwd=str(ROOT),
                                      stderr=subprocess.DEVNULL)
        return out.decode("utf-8", "replace").strip()
    except Exception:
        return None


def snapshot_run(resolved_cfg, entry, out_dir, experiment_name=None,
                 extra_overrides=None, timestamp=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8")

    entry = entry or {}
    porcelain = _git(["status", "--porcelain"])
    meta = {
        "experiment_name": experiment_name,
        "base": entry.get("base"),
        "script": entry.get("script"),
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_dirty": (bool(porcelain) if porcelain is not None else None),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "registry_overrides": entry.get("overrides") or {},
    }
    if extra_overrides:
        meta["cli_overrides"] = extra_overrides

    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
