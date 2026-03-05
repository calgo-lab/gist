#!/usr/bin/env bash
set -euo pipefail

# Run from repo root on cluster, e.g.:
#   cd /storage/gwl-interpolation
#   bash cluster/jobs/run_clean_gp_compare.sh

ROOT="${ROOT:-/storage/gwl-interpolation}"
cd "$ROOT"

GRU_SIG="${GRU_SIG:-gru_l2_seed40_ep50_full_merged_spf0p8_sc20_ss42}"
TFT_SIG="${TFT_SIG:-in52_out16_ep50_bs4096_stat1_seed40_full_merged}"

echo "[1/7] Build monthly date list from GRU predictions"
GRU_DATES=$(python3 - <<'PY'
import pandas as pd
p="outputs/GRU_FCOV/GRU_FCOV_gru_l2_seed40_ep50_full_merged_spf0p8_sc20_ss42/predictions/pred.parquet"
df=pd.read_parquet(p, columns=["datum"])
d=pd.to_datetime(df["datum"]).drop_duplicates().sort_values()
m=pd.Series(d.values, index=d).resample("ME").first().dropna()
print(" ".join(x.strftime("%Y-%m-%d") for x in m))
PY
)

echo "[2/7] Build monthly date list from TFT predictions"
TFT_DATES=$(python3 - <<'PY'
import pandas as pd
p="outputs/TFT/TFT_in52_out16_ep50_bs4096_stat1_seed40_full_merged/predictions/pred.parquet"
df=pd.read_parquet(p, columns=["datum"])
d=pd.to_datetime(df["datum"]).drop_duplicates().sort_values()
m=pd.Series(d.values, index=d).resample("ME").first().dropna()
print(" ".join(x.strftime("%Y-%m-%d") for x in m))
PY
)

echo "[3/7] Custom GP on GRU predictions"
python3 src/scripts/joint/spatial/gp_eval.py \
  --model-prefix GRU_FCOV \
  --gru-run-sig "$GRU_SIG" \
  --gp-run-tag cmp_gru_custom \
  --pretrain-steps 0 \
  --jitter 0.001 \
  --kernel-type matern32 \
  --isotropic true \
  --date-freq ME

echo "[4/7] Sklearn kriging on GRU predictions"
python3 src/scripts/separate/spatial/kriging.py \
  --model GRU_FCOV \
  --run-sig "$GRU_SIG" \
  --dataset full_merged \
  --kriging-source pred \
  --split-tag cmp_gru_skl \
  --dates $GRU_DATES \
  --horizons 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

echo "[5/7] Custom GP on TFT predictions"
python3 src/scripts/joint/spatial/gp_eval.py \
  --model-prefix TFT \
  --gru-run-sig "$TFT_SIG" \
  --gp-run-tag cmp_tft_custom \
  --pretrain-steps 0 \
  --jitter 0.001 \
  --kernel-type matern32 \
  --isotropic true \
  --date-freq ME

echo "[6/7] Sklearn kriging on TFT predictions"
python3 src/scripts/separate/spatial/kriging.py \
  --model TFT \
  --run-sig "$TFT_SIG" \
  --dataset full_merged \
  --kriging-source pred \
  --split-tag cmp_tft_skl \
  --dates $TFT_DATES \
  --horizons 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

echo "[7/7] Build one comparison CSV"
python3 - <<'PY'
from pathlib import Path
import pandas as pd

root=Path("/storage/gwl-interpolation")
rows=[]

runs=[
  ("GRU_custom","outputs/gp/GRU_FCOV_gru_l2_seed40_ep50_full_merged_spf0p8_sc20_ss42__cmp_gru_custom__predobstrain/gp_metrics.parquet"),
  ("GRU_sklearn","outputs/gp/gru_l2_seed40_ep50_full_merged_spf0p8_sc20_ss42__cmp_gru_skl__predobstrain/gp_metrics.parquet"),
  ("TFT_custom","outputs/gp/TFT_in52_out16_ep50_bs4096_stat1_seed40_full_merged__cmp_tft_custom__predobstrain/gp_metrics.parquet"),
  ("TFT_sklearn","outputs/gp/in52_out16_ep50_bs4096_stat1_seed40_full_merged__cmp_tft_skl__predobstrain/gp_metrics.parquet"),
]

for label, rel in runs:
    p=root/rel
    if not p.exists():
        rows.append({"run":label,"metrics_found":False})
        continue
    m=pd.read_parquet(p)
    d=dict(zip(m["metric"], m["value"]))
    rows.append({
        "run":label,
        "metrics_found":True,
        "RMSE":d.get("RMSE"),
        "MAE":d.get("MAE"),
        "nRMSE":d.get("nRMSE"),
        "NSE_pooled":d.get("NSE_pooled"),
        "NSE_id_median":d.get("NSE_id_median"),
        "metrics_path":str(p),
    })

out=root/"reports/gp/metrics/clean_compare_gru_tft_custom_vs_sklearn.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(out)
PY

echo "Done."
echo "Comparison file: reports/gp/metrics/clean_compare_gru_tft_custom_vs_sklearn.csv"
