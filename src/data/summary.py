from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import yaml


# Plots

def bar_years_full_range(counts, title, outpath, tick_every):
    counts = counts.copy()
    counts.index = counts.index.astype(int)
    y0, y1 = int(counts.index.min()), int(counts.index.max())
    counts = counts.reindex(range(y0, y1 + 1), fill_value=0).astype(int)

    plt.figure(figsize=(10, 5), dpi=200)
    ax = counts.plot(kind="bar")
    ax.set_title(title); ax.set_xlabel("Year"); ax.set_ylabel("Count")

    years = list(counts.index)
    pos = [i for i, y in enumerate(years) if (y - years[0]) % tick_every == 0]
    ax.set_xticks(pos)
    ax.set_xticklabels([years[i] for i in pos], rotation=45, ha="right")

    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()


def plot_start_end_active(df, time_col, id_col, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)

    start_years = df.groupby(id_col)[time_col].min().dt.year.value_counts().sort_index()
    bar_years_full_range(start_years, "Number of Wells by Start Year",
                         figures_dir / "count_wells_by_start_year.png", tick_every=5)

    end_years = df.groupby(id_col)[time_col].max().dt.year.value_counts().sort_index()
    bar_years_full_range(end_years, "Number of Wells by End Year",
                         figures_dir / "count_wells_by_end_year.png", tick_every=5)

    weeks = df[time_col].dt.to_period("W-MON").dt.start_time
    active = df.groupby(weeks)[id_col].nunique().sort_index()
    plt.figure(figsize=(10, 5), dpi=200)
    ax = active.plot()
    ax.set_title("Active Wells per Week"); ax.set_xlabel("Week (Mon-start)"); ax.set_ylabel("Number of Active Wells")
    plt.tight_layout(); plt.savefig(figures_dir / "active_wells_per_week.png", dpi=200); plt.close()


# Main

def run(cfg_path):
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_path = Path(cfg["data_path"])
    time_col = cfg.get("time_col", "datum")
    id_col = cfg.get("id_col", "id")

    reports_dir = Path("reports"); reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"; figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = reports_dir / "metrics"; metrics_dir.mkdir(parents=True, exist_ok=True)

    # read data
    if str(data_path).lower().endswith((".parq", ".parquet")):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, sep=sep)

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    # figures
    plot_start_end_active(df, time_col, id_col, figures_dir)

    # metrics
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
    cov.to_csv(metrics_dir / "coverage_by_well.csv", index=False)

    # summary.md
    lines = [
        "# Data Summary\n",
        f"- Rows: {len(df):,}",
        f"- Wells: {df[id_col].nunique():,}",
        f"- Time span: {df[time_col].min().date()} → {df[time_col].max().date()}",
        f"- Overall NA across table: {overall_na_pct:.1f}%",
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    cfg_path = Path("configs/data.yaml")
    run(cfg_path)