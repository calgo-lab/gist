# GP Spatial Feature Ablation

**Pipeline**: Oracle (GP observations = true GWL values at train wells — upper bound, no GRU error)
**Split**: rmd90_test52, evaluated on 52 spatial test wells
**Metric**: nRMSE_pw = median over wells of (RMSE_well / IQR_well), lower = better
**Baseline (coords only)**: 1.7831 — GP with x/y coordinates only, no extra features
**Best**: 1.1760 — siwa + hydroraum + gw_gespannt

All runs use the same GP architecture (gpytorch, Matern-3/2, pretrain_steps=200).

---

## Core Feature Ablation (single features and combinations)

| GP Features | nRMSE_pw | vs coords-only | Notes |
|---|---|---|---|
| siwa + hydroraum + gw_gespannt | **1.1760** | −0.607 | **BEST — use this** |
| siwa + hydroraum | 1.2790 | −0.504 | |
| siwa + gw_gespannt | 1.4483 | −0.335 | |
| hydroraum + gw_gespannt | 1.5232 | −0.260 | |
| hydroraum only | 1.5776 | −0.206 | |
| gw_gespannt only | 1.6548 | −0.128 | |
| siwa only | 1.7313 | −0.052 | |
| coords only (no extra features) | 1.7831 | — | baseline |

**Key finding**: siwa_verweilzeit_j (groundwater residence time) alone contributes ~0.31 nRMSE_pw improvement in most pipelines. It is the single most important GP feature.

---

## Autocorrelation Class (acf_class = Trägheit proxy)

| GP Features | nRMSE_pw | vs coords-only | Verdict |
|---|---|---|---|
| siwa + acf_class | 1.8954 | +0.112 | **WORSE than coords-only** |
| siwa + hydroraum + gw_gespannt + acf_class | 1.9478 | +0.165 | **WORSE** |
| hydroraum + gw_gespannt + acf_class | 1.9967 | +0.214 | **WORSE** |
| acf_class only | 2.1919 | +0.409 | **MUCH WORSE** |

**Conclusion**: acf_class (autocorrelation-derived sluggishness class) fails completely as a GP feature. Sluggishness is apparently already captured by siwa_verweilzeit_j (groundwater residence time) better than any autocorrelation-derived class.

---

## Well Screen Depth (fok / fuk)

| GP Features | nRMSE_pw | vs coords-only | Verdict |
|---|---|---|---|
| siwa + hydroraum + gw_gespannt + fok | 3.6766 | +1.894 | **CATASTROPHIC** |
| siwa + hydroraum + gw_gespannt + fuk | 3.7234 | +1.940 | **CATASTROPHIC** |
| siwa + hydroraum + gw_gespannt + fok + fuk | 3.7874 | +2.004 | **CATASTROPHIC** |
| siwa + fok | 4.6402 | +2.857 | **CATASTROPHIC** |
| siwa + fuk | 4.7604 | +2.977 | **CATASTROPHIC** |
| siwa + fok + fuk | 4.8015 | +3.018 | **CATASTROPHIC** |
| fok only | 4.5601 | +2.777 | **CATASTROPHIC** |
| fok + fuk | 4.5692 | +2.786 | **CATASTROPHIC** |
| fuk only | 4.5950 | +2.812 | **CATASTROPHIC** |

**Conclusion**: Well screen top/bottom depths (fok/fuk) completely destroy GP performance when used as spatial covariance features. Well screen depth is not a valid proxy for spatial similarity of groundwater dynamics.

---

## Precipitation-Based Features

| GP Features | nRMSE_pw | vs coords-only | Verdict |
|---|---|---|---|
| pr1lag (1-week precipitation lag) | 2.0930 | +0.310 | **HURTS** |
| siwa + hydroraum + gw_gespannt + pr52lag | 2.1553 | +0.372 | **HURTS** |
| pr52lag (52-week precipitation lag) | 2.4784 | +0.695 | **HURTS** |
| siwa + hydroraum + gw_gespannt + pr52xcorr | 3.3005 | +1.517 | **CATASTROPHIC** |
| siwa + hydroraum + gw_gespannt + pr52lag + pr52xcorr | 3.2745 | +1.491 | **CATASTROPHIC** |
| pr52xcorr (52-week precipitation cross-correlation) | 4.8043 | +3.021 | **CATASTROPHIC** |

**Conclusion**: All precipitation-based GP features hurt. Precipitation variability is a temporal feature; spatial correlation of precipitation is too coarse-grained to be useful for GP covariance between wells.

---

## Other Single Features

| GP Features | nRMSE_pw | vs coords-only | Verdict |
|---|---|---|---|
| gwlk (groundwater level class) | 2.4942 | +0.711 | **HURTS** |
| siwa + hydroraum + gw_gespannt + gwlk | 3.1484 | +1.365 | **CATASTROPHIC** |

---

## Summary: Feature Value Ranking

| Feature | Role | Effect |
|---|---|---|
| siwa_verweilzeit_j | Groundwater residence time (years) | **Essential** (+0.31–0.61 improvement) |
| hydroraum | Hydrogeological zone (3-class one-hot) | **Important** (+0.13–0.26 improvement) |
| gw_gespannt | Confined vs unconfined (binary one-hot) | **Useful** (+0.09–0.13 improvement) |
| acf_class | Autocorrelation sluggishness class | **Never use** (always worse than coords-only) |
| gwlk | Groundwater level class | **Never use** (hurts) |
| pr1lag, pr52lag | Precipitation lags | **Never use** (hurts) |
| pr52xcorr | Precipitation cross-correlation | **Never use** (catastrophic) |
| fok | Well screen top depth | **Never use** (catastrophic) |
| fuk | Well screen bottom depth | **Never use** (catastrophic) |
