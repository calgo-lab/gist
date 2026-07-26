from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = ROOT / "configs" / "gru"
HPO_RESULTS_DIR = ROOT / "reports" / "metrics" / "gru" / "hpo"
OUT_DIR = ROOT / "reports" / "metrics" / "gru" / "hpo_tables"
OUT_CSV = OUT_DIR / "gru_hpo_search_spaces.csv"
OUT_MD = OUT_DIR / "gru_hpo_search_spaces.md"

HPO_CONFIGS = [
    CONFIG_DIR / "hpo_gru_ablation.yaml",
    CONFIG_DIR / "hpo_gru.yaml",
]

OPTIONAL_CONFIG_NAMES = [
    "hpo_gru_small_grid_v1.yaml",
    "hpo_gru_small_grid_v2.yaml",
    "hpo_gru_small_grid_v3.yaml",
    "hpo_gru_refine_v4.yaml",
]


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _format_values(values: object) -> str:
    if isinstance(values, list):
        return ", ".join(str(v) for v in values)
    return str(values)


def _discover_configs() -> list[Path]:
    files: list[Path] = []
    for path in HPO_CONFIGS:
        if path.exists():
            files.append(path)
    for name in OPTIONAL_CONFIG_NAMES:
        path = CONFIG_DIR / name
        if path.exists():
            files.append(path)
    if not files:
        files = sorted(CONFIG_DIR.glob("hpo_gru*.yaml"))
    return files


def _format_series_values(series: pd.Series) -> str:
    values = []
    for value in series.dropna().unique().tolist():
        values.append(value)
    values = sorted(values, key=lambda v: str(v))
    return ", ".join(str(v) for v in values)


def main() -> None:
    rows: list[dict[str, str]] = []
    all_hp_keys: set[str] = set()
    seen_names: set[str] = set()

    files = _discover_configs()
    for path in files:
        cfg = _load_yaml(path)
        name = str(cfg.get("name", path.stem))
        search_space = cfg.get("search_space", {})
        if not isinstance(search_space, dict):
            search_space = {}

        row: dict[str, str] = {"hpo_run": name}
        for hp_key, values in search_space.items():
            clean_key = str(hp_key).replace("model.", "").replace("training.", "")
            row[clean_key] = _format_values(values)
            all_hp_keys.add(clean_key)
        rows.append(row)
        seen_names.add(name)

    if HPO_RESULTS_DIR.exists():
        for path in sorted(HPO_RESULTS_DIR.glob("*_results.csv")):
            name = path.name.replace("_results.csv", "")
            if name in seen_names:
                continue
            df_raw = pd.read_csv(path)
            hp_cols = [c for c in df_raw.columns if c.startswith("hp.")]
            if not hp_cols:
                continue
            row = {"hpo_run": name}
            for hp_col in hp_cols:
                clean_key = hp_col.replace("hp.model.", "").replace("hp.training.", "").replace("hp.", "")
                row[clean_key] = _format_series_values(df_raw[hp_col])
                all_hp_keys.add(clean_key)
            rows.append(row)
            seen_names.add(name)

    if not rows:
        raise FileNotFoundError(
            f"No GRU HPO configs in {CONFIG_DIR} and no GRU HPO result CSVs in {HPO_RESULTS_DIR}"
        )

    ordered_cols = ["hpo_run"] + sorted(all_hp_keys)
    df = pd.DataFrame(rows)
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[ordered_cols].fillna("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")

    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
