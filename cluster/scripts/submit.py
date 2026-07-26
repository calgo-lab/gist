import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO     = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "configs" / "experiments.yaml"
JOBS     = REPO / "cluster" / "jobs"

EVAL_SIBLING = {
    "src/scripts/pipelines/temporal/gru_train.py": "src/scripts/pipelines/temporal/gru_eval.py",
    "src/scripts/pipelines/joint/gru_gp_train.py":       "src/scripts/pipelines/joint/gru_gp_eval.py",
}


def k8s_name(prefix, stem):
    name = f"{prefix}-{stem}".lower().replace("_", "-").replace(".", "-")
    return name[:63].rstrip("-")


def expand_seeds(spec):
    spec = str(spec).strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def chain_scripts(entry):
    if entry.get("scripts"):
        return entry["scripts"]
    script = entry["script"]
    out = [script]
    if script in EVAL_SIBLING:
        out.append(EVAL_SIBLING[script])
    return out


def _sub(value, ss):
    if ss is not None and isinstance(value, str):
        return value.replace("{ss}", str(ss))
    return value


def prebuild_cmds(entry, ss=None):
    cmds = []
    for step in entry.get("prebuild", []) or []:
        builder = step["builder"]
        if "/" not in builder:
            builder = f"src/prep/{builder}"
        args = " ".join(f"--{k} {_sub(str(v), ss)}" for k, v in (step.get("args") or {}).items())
        cmds.append(f"python {builder} {args}".rstrip())
    return cmds


def run_cmds(name, entry, ss=None, extra_set=None):
    if entry.get("command"):
        return [_sub(entry["command"], ss)]
    pairs = [f"{k}={_sub(v, ss)}" for k, v in (entry.get("overrides") or {}).items()
             if ss is not None and isinstance(v, str) and "{ss}" in v]
    pairs += [_sub(p, ss) for p in (extra_set or [])]
    suffix = " --set " + " ".join(pairs) if pairs else ""
    return [f'python {s} --experiment "{name}"{suffix}' for s in chain_scripts(entry)]


def build_args_block(lines):
    body = "\n".join([
        "set -euo pipefail",
        "source /storage/venv/bin/activate",
        "cd /storage/gwl-interpolation",
        *lines,
    ])
    return body + "\n"


def load_template(path):
    return yaml.safe_load(path.read_text())


def stamp(template, name, args_block, experiment):
    template["metadata"]["name"] = name
    container = template["spec"]["template"]["spec"]["containers"][0]
    container["args"] = [args_block]
    if experiment is not None:
        for env in container.get("env", []):
            if env.get("name") == "EXPERIMENT":
                env["value"] = experiment
    return template


def _literal_block(dumper, data):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


yaml.add_representer(str, _literal_block)


def apply(job, dry_run):
    text = yaml.dump(job, default_flow_style=False, sort_keys=False)
    if dry_run:
        print(text)
        return
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(text)
        tmp = fh.name
    subprocess.run(["kubectl", "apply", "-f", tmp], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="?", help="registry entry name")
    ap.add_argument("--analysis", metavar="SCRIPT", help="run an analysis script under gw-analysis.yaml")
    ap.add_argument("--seeds", help="seed set for a multiseed entry, e.g. '42-81' or '42,44'")
    ap.add_argument("--set", dest="set_overrides", nargs="*", default=[],
                    help="extra key=value overrides passed through to the script ({ss} allowed)")
    ap.add_argument("--dry-run", action="store_true", help="print the job YAML instead of applying")
    ap.add_argument("--list", action="store_true", help="list registry entries and exit")
    args = ap.parse_args()

    registry = yaml.safe_load(REGISTRY.read_text())

    if args.list:
        for name, entry in registry.items():
            print(f"{name:40s} {entry.get('script', entry.get('note',''))}")
        return

    if args.analysis:
        stem = Path(args.analysis).stem
        job = stamp(load_template(JOBS / "gw-analysis.yaml"),
                    k8s_name("gw", stem),
                    build_args_block([f"python {args.analysis}"]),
                    experiment=None)
        apply(job, args.dry_run)
        return

    if not args.experiment:
        ap.error("give an experiment name, or --analysis <script>, or --list")

    if args.experiment not in registry:
        sys.exit(f"unknown experiment '{args.experiment}'. Try --list.")
    entry = registry[args.experiment]
    name = args.experiment

    seed_spec = args.seeds or entry.get("seeds")
    seeds = expand_seeds(seed_spec) if seed_spec is not None else [None]

    for i, ss in enumerate(seeds):
        job_stem = name if ss is None else f"{name}-ss{ss}"
        lines = prebuild_cmds(entry, ss) + run_cmds(name, entry, ss, args.set_overrides)
        job = stamp(load_template(JOBS / "gw-run.yaml"),
                    k8s_name("gw", job_stem), build_args_block(lines), experiment=name)
        if args.dry_run and len(seeds) > 1 and i == 1:
            print(f"... (+{len(seeds) - 1} more seed jobs: ss{seeds[1]}..ss{seeds[-1]})\n")
        if not args.dry_run or i == 0:
            apply(job, args.dry_run)
    if not args.dry_run and len(seeds) > 1:
        print(f"Submitted {len(seeds)} seed jobs for {name} (ss{seeds[0]}..ss{seeds[-1]}).")


if __name__ == "__main__":
    main()
