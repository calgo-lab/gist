from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "reports" / "gru" / "metrics"
FIG_DIR = ROOT / "reports" / "gru" / "figures"

# Seed-40 runs come from HPO so have different run_sig naming.
# Group definition: list of run_sigs belonging to each config.
CONFIGS = {
    "h192, l4, lr=4e-4": [
        "gru_small_grid_v2_t012_seed40_full_merged",
        "gru_l4_h192_bs4096_seed41_ep50",
        "gru_l4_h192_bs4096_seed42_ep50",
    ],
    "h256, l4, lr=2e-4": [
        "gru_small_grid_v2_t005_seed40_full_merged",
        "gru_l4_h256_bs4096_seed41_ep50",
        "gru_l4_h256_bs4096_seed42_ep50",
    ],
}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS_DIR / "gru_all_runs_plotstyle_metrics.csv")

    fig, ax = plt.subplots(figsize=(6, 5))

    for label, run_sigs in CONFIGS.items():
        rows = df[df["run_sig"].isin(run_sigs)]
        if rows.empty:
            print(f"No rows found for config: {label}")
            continue

        nse = rows["NSE_mean_h1_16"].dropna()
        runtime = rows["train_s"].dropna()

        mean_nse = nse.mean()
        min_nse = nse.min()
        max_nse = nse.max()
        mean_runtime = runtime.mean()

        ax.errorbar(
            mean_runtime,
            mean_nse,
            yerr=[[mean_nse - min_nse], [max_nse - mean_nse]],
            fmt="o",
            capsize=6,
            markersize=7,
            label=label,
        )

    ax.set_xlabel("Mean training time (s)")
    ax.set_ylabel("NSE mean h1–16 (median across wells)")
    ax.set_title("Config comparison: performance vs runtime\n(error bars show seed range)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()

    out = FIG_DIR / "config_seed_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
