# Grundwasserstandsklassen Classification

**Reference**: 1991–2020 per well | **Classes**: 1=sehr niedrig (<P10), 2=niedrig (P10–P25), 3=normal (P25–P75), 4=hoch (P75–P90), 5=sehr hoch (>P90) | **Random baseline**: 20%

---

## GP — best decoupled (hpo_sep_t118, nRMSE_pw=1.1680)

Median per-well: **20.7%** | Mean: 29.3%

| Class | Name | N obs | Accuracy |
|-------|------|------:|----------:|
| 1 | sehr niedrig | 49,568 | 47.4% |
| 2 | niedrig | 58,992 | 11.2% |
| 3 | normal | 198,256 | 20.9% |
| 4 | hoch | 65,584 | 10.5% |
| 5 | sehr hoch | 72,592 | 66.3% |

## GRU-1040 (trained on all 1040 wells incl. test wells)

Median per-well: **78.3%** | Mean: 76.8%

| Class | Name | N obs | Accuracy |
|-------|------|------:|----------:|
| 1 | sehr niedrig | 68,502 | 86.3% |
| 2 | niedrig | 34,250 | 63.6% |
| 3 | normal | 64,699 | 81.8% |
| 4 | hoch | 15,835 | 54.1% |
| 5 | sehr hoch | 8,906 | 58.8% |

---

## Summary

| Model | Median GWK acc | Mean GWK acc |
|-------|---------------|-------------|
| GP (hpo_sep_t118) | 20.7% | 29.3% |
| GRU-1040 | 78.3% | 76.8% |
| Random baseline | 20.0% | 20.0% |
