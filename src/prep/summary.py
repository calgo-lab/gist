from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import yaml


def bar_years_full_range(counts, title, outpath, tick_every):
    counts = counts.copy()
    counts.index = counts.index.astype(int)
    y0, y1 = int(counts.index.min()), int(counts.index.max())
    counts = counts.reindex(range(y0, y1 + 1), fill_value=0).astype(int)
    plt.figure(figsize=(10, 5), dpi=200)
    ax = counts.plot(kind="bar")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    years = list(counts.index)
    pos = [i for i, y in enumerate(years) if (y - years[0]) % tick_every == 0]
    ax.set_xticks(pos)
    ax.set_xticklabels([years[i] for i in pos], rotation=45, ha="right")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()


def plot_start_end_active(df, time_col, id_col, figures_dir, tick_every):
    figures_dir.mkdir(parents=True, exist_ok=True)
    start_years = df.groupby(id_col)[time_col].min().dt.year.value_counts().sort_index()
    bar_years_full_range(start_years, "Number of Wells by Start Year",
                         figures_dir / "count_wells_by_start_year.png", tick_every)
    end_years = df.groupby(id_col)[time_col].max().dt.year.value_counts().sort_index()
    bar_years_full_range(end_years, "Number of Wells by End Year",
                         figures_dir / "count_wells_by_end_year.png", tick_every)
    weeks = df[time_col].dt.to_period("W-MON").dt.start_time
    active = df.groupby(weeks)[id_col].nunique().sort_index()
    plt.figure(figsize=(10, 5), dpi=200)
    ax = active.plot()
    ax.set_title("Active Wells per Week")
    ax.set_xlabel("Week (Mon-start)")
    ax.set_ylabel("Number of Active Wells")
    plt.tight_layout()
    plt.savefig(figures_dir / "active_wells_per_week.png", dpi=200)
    plt.close()


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
    figures_dir = reports_dir / "overall" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir_tft = reports_dir / "tft" / "metrics"
    metrics_dir_tft.mkdir(parents=True, exist_ok=True)
    metrics_dir_overall = reports_dir / "overall" / "metrics"
    metrics_dir_overall.mkdir(parents=True, exist_ok=True)

    plot_start_end_active(df, time_col, id_col, figures_dir, 5)

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
    cov.to_csv(metrics_dir_overall / "coverage_by_well.csv", index=False)

    range_summary_path = metrics_dir_overall / "spatiotemporal_range_summary.csv"
    write_spatiotemporal_range_summary(df, time_col=time_col, id_col=id_col, value_col="gws", out_csv=range_summary_path)

    lines = [
        "# Data Summary",
        f"- Rows: {len(df):,}",
        f"- Wells: {df[id_col].nunique():,}",
        f"- Time span: {df[time_col].min().date()} → {df[time_col].max().date()}",
        f"- Overall NA across table: {overall_na_pct:.1f}%",
        f"- Spatiotemporal range summary: `{range_summary_path.as_posix()}`",
    ]
    (reports_dir / "overall" / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    cfg_path = Path("configs/data.yaml")
    run(cfg_path)
