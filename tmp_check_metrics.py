import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIG_DIR = Path("reports/gru/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

rows = [
    ("Joint A / B / C",       "A",      "B",     "C",  0.1929, 0.0000),
    ("Separate A / B+C",        "A",      "—",    "B+C", 0.2093, 0.0164),
    ("Separate A+B / C",        "A+B",    "—",     "C",  0.2146, 0.0217),
]

col_labels = ["Configuration", "Train", "Val", "Test", "NRMSE", "Diff. to joint"]
cell_text  = [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows]

fig, ax = plt.subplots(figsize=(9, 2.2))
ax.axis("off")

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

# Header styling
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Highlight best NRMSE row (joint)
for j in range(len(col_labels)):
    table[1, j].set_facecolor("#d5f5e3")

# Alternating row shading for others
for j in range(len(col_labels)):
    table[2, j].set_facecolor("#f8f9fa")
    table[3, j].set_facecolor("#ffffff")

fig.suptitle("nRMSE comparison: joint vs. separate GRU+GP training", fontsize=12, y=1.02)
fig.tight_layout()
out = FIG_DIR / "joint_vs_separate_nrmse_table.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
