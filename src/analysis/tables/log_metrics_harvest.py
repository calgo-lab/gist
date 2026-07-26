import argparse
import re
from pathlib import Path

import pandas as pd

RE_SAVED  = re.compile(r"Saved GP outputs to\s+(\S+)")
RE_PAIRS  = re.compile(r"Valid pairs:\s*(\d+)\s*\|\s*Skipped:\s*(\d+)")
RE_METRIC = re.compile(r"^\s{2,}([A-Za-z0-9_]+)\s*:\s*([0-9.eE+\-]+)\s*$")


def parse_log(path: Path) -> list[dict]:
    lines = path.read_text(errors="replace").splitlines()
    rows  = []
    run   = None
    pairs = skipped = None

    i = 0
    while i < len(lines):
        line = lines[i]

        m_saved = RE_SAVED.search(line)
        if m_saved:
            run = m_saved.group(1).rstrip("/").split("/")[-1]

        m_pairs = RE_PAIRS.search(line)
        if m_pairs:
            pairs, skipped = int(m_pairs.group(1)), int(m_pairs.group(2))

        if line.strip().startswith("Overall metrics"):
            row = {
                "log":          path.name,
                "run":          run or path.stem,
                "valid_pairs":  pairs,
                "skipped":      skipped,
            }
            j = i + 1
            while j < len(lines):
                m = RE_METRIC.match(lines[j])
                if not m:
                    break
                row[m.group(1)] = float(m.group(2))
                j += 1
            rows.append(row)
            run = pairs = skipped = None
            i = j
            continue

        i += 1

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="reports")
    parser.add_argument("--csv",  default="reports/tables/run_metrics_from_logs.csv")
    args = parser.parse_args()

    log_dir = Path(args.logs)
    logs = sorted(log_dir.rglob("*.log"))
    print(f"Scanning {len(logs)} logs under {log_dir}")

    rows = []
    for p in logs:
        rows.extend(parse_log(p))

    if not rows:
        print("No 'Overall metrics' blocks found.")
        return

    df = pd.DataFrame(rows)
    front = ["log", "run", "valid_pairs", "skipped"]
    cols  = front + [c for c in df.columns if c not in front]
    df = df[cols].sort_values("run").reset_index(drop=True)

    print(f"Extracted {len(df)} metric blocks from {df['log'].nunique()} logs")

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
