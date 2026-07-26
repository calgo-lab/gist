from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[4]

data = {
    "random 90/10":     (1.027, 0.115, 1.121, 0.135),
    "max-distance 50%": (0.681, 0.109, 0.943, 0.239),
    "far-80%":          (5.409, 0.803, 3.614, 0.431),
    "k-means":          (1.164, 0.220, 1.339, 0.157),
}

splits = ["random 90/10", "k-means", "max-distance 50%", "far-80%"]
ts_means  = [data[s][0] for s in splits]
ts_stds   = [data[s][1] for s in splits]
jt_means  = [data[s][2] for s in splits]
jt_stds   = [data[s][3] for s in splits]

BLUE   = "#0072B2"
ORANGE = "#D55E00"

bar_h   = 0.32
y       = np.arange(len(splits))
y_ts    = y - bar_h / 2
y_jt    = y + bar_h / 2

fig, ax = plt.subplots(figsize=(6, 4))

ax.barh(y_ts, ts_means, height=bar_h, xerr=ts_stds, color=ORANGE, label="two-stage",      capsize=3, error_kw={"elinewidth": 0.9, "ecolor": "black"})
ax.barh(y_jt, jt_means, height=bar_h, xerr=jt_stds, color=BLUE,  label="jointly trained", capsize=3, error_kw={"elinewidth": 0.9, "ecolor": "black"})

ax.set_yticks(y)
ax.set_yticklabels(splits)
ax.set_xlabel("h=16 RMSE (m)")

ax.xaxis.grid(True, color="0.85", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.yaxis.grid(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(loc="lower right", frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(
    ROOT / "reports" / "figures" / "splits" / "split_strategy_joint_comparison.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved.")
