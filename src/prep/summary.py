from pathlib import Path
import pandas as pd
import yaml


def write_spatiotemporal_range_summary(df, time_col, id_col, value_col, out_csv):
    work = df[[id_col, time_col, value_col]].dropna(subset=[id_col, time_col, value_col]).copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])

    per_well_range = work.groupby(id_col)[value_col].agg(lambda s: s.max() - s.min())
    per_timestep_range = work.groupby(time_col)[value_col].agg(lambda s: s.max() - s.min())

    rows = [
        {"scope": "temporal_per_well", "stat": "mean", "value_m": float(per_well_range.mean())},
        {"scope": "temporal_per_well", "stat": "median", "value_m": float(per_well_range.median())},
        {"scope": "temporal_per_well", "stat": "max", "value_m": float(per_well_range.max())},
        {"scope": "spatial_per_timestep", "stat": "mean", "value_m": float(per_timestep_range.mean())},
        {"scope": "spatial_per_timestep", "stat": "median", "value_m": float(per_timestep_range.median())},
        {"scope": "spatial_per_timestep", "stat": "max", "value_m": float(per_timestep_range.max())},
    ]
    out = pd.DataFrame(rows)
    out["value_m"] = out["value_m"].round(6)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


def run(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    time_col = "datum"
    id_col = "id"
    sep = ","

    data_path = Path(cfg.get("data_path", cfg.get("full_merged_path", "data/merged.parquet")))
    if not data_path.exists():
        data_path = Path("data/merged.parquet")

    if data_path.suffix.lower() in (".parq", ".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, sep=sep)

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    reports_dir = Path("reports"); reports_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = reports_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    overall_na_pct = (df.isna().values.mean() * 100)
    g = df.groupby(id_col)[time_col]
    cov = pd.DataFrame({
        "id": g.min().index,
        "start_date": g.min().values,
        "end_date": g.max().values,
        "n_obs": g.size().values,
    })
    cov["span_days"] = (cov["end_date"] - cov["start_date"]).dt.days
    cov["first_year"] = cov["start_date"].dt.year
    cov["last_year"]  = cov["end_date"].dt.year
    cov.to_csv(dataset_dir / "coverage_by_well.csv", index=False)

    range_summary_path = dataset_dir / "spatiotemporal_range_summary.csv"
    write_spatiotemporal_range_summary(df, time_col=time_col, id_col=id_col, value_col="gws", out_csv=range_summary_path)

    lines = [
        "# Data Summary",
        f"- Rows: {len(df):,}",
        f"- Wells: {df[id_col].nunique():,}",
        f"- Time span: {df[time_col].min().date()} → {df[time_col].max().date()}",
        f"- Overall NA across table: {overall_na_pct:.1f}%",
        f"- Spatiotemporal range summary: `{range_summary_path.as_posix()}`",
    ]
    (dataset_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    cfg_path = Path("configs/data.yaml")
    run(cfg_path)
