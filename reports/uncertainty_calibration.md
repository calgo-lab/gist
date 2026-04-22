# Uncertainty Calibration Report

## Metadata: Grundwasserstandsklassen

### `dss_standsklassen` (2 unique values)

```
dss_standsklassen
J    1079
N    1052
```

---

## Calibration Analysis

**Method**: leave-one-out conformal. For each test well, compute the 95th-percentile nonconformity score `q` from the other 51 wells, then check what fraction of that well's observations fall within `ŷ ± q·σ`. The calibration factor α = q replaces the naive z=1.96.

### oracle (hpo_t020)

| Metric | Value |
|--------|-------|
| Calibration factor α (median LOO) | **12.678** (vs naive z=1.96) |
| Naive PICP @95% | 54.9% |
| Calibrated PICP (α=12.678) | 95.1% |
| LOO mean per-well coverage | 92.5% |
| Naive median 95%-PI width | 1.033 m |
| Calibrated median PI width | 6.682 m |

### decoupled (hpo_sep_t118)

| Metric | Value |
|--------|-------|
| Calibration factor α (median LOO) | **12.758** (vs naive z=1.96) |
| Naive PICP @95% | 54.8% |
| Calibrated PICP (α=12.758) | 95.1% |
| LOO mean per-well coverage | 92.5% |
| Naive median 95%-PI width | 1.023 m |
| Calibrated median PI width | 6.659 m |

---

## Grundwasserstandsklassen Classification Accuracy

Using column: `dss_standsklassen`

### oracle (hpo_t020)

5-class quintile classification (boundaries per well from observed GWL):

| Metric | Value |
|--------|-------|
| Median per-well accuracy | 21.1% |
| Mean per-well accuracy | 30.0% |
| Random baseline | 20.0% |

### decoupled (hpo_sep_t118)

5-class quintile classification (boundaries per well from observed GWL):

| Metric | Value |
|--------|-------|
| Median per-well accuracy | 21.2% |
| Mean per-well accuracy | 30.0% |
| Random baseline | 20.0% |

