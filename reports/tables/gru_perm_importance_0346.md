# GRU Feature Permutation Importance (Run 0346)

**Model**: GRU_FCOV_0346
**Split**: all 1040 wells, temporal test period (post-2020-01-01)
**Metric**: median nRMSE_pw (range-normalised per-well RMSE)
**Baseline nRMSE_pw**: 0.0467
**All-static-zeroed nRMSE_pw**: 0.0771 (delta=+0.0303)
**n_repeats**: 5 (results averaged over permutation draws)

| Rank | Type | Feature | Delta nRMSE_pw | % Change |
|------|------|---------|----------------|----------|
| 1 | dynamic | tas_5km | +0.0428 | +91.5% |
| 2 | dynamic | pr_5km | +0.0395 | +84.5% |
| 3 | dynamic | tag_cos | +0.0187 | +40.0% |
| 4 | dynamic | hurs_5km | +0.0099 | +21.2% |
| 5 | dynamic | tag_sin | +0.0098 | +20.9% |

## Summary
- Delta > 0: feature helps (shuffling it hurts performance)
- Delta < 0: feature hurts (model uses it wrong or adds noise)
- Dynamic features are permuted across windows (shuffling weather sequences across wells).
