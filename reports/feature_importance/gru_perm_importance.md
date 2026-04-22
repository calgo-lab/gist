# GRU Static Feature Permutation Importance

**Model**: GRU_FCOV_0007 (h128, l5, do0.3, best decoupled config)
**Split**: rmd90_test52 (988 train wells)
**Metric**: nRMSE_pw on spatial_val (train-side held-out wells)
**Baseline nRMSE_pw**: 0.5117 (lower = better)
**Method**: Shuffle one feature at a time, measure increase in nRMSE_pw.
- Delta > 0 → feature HELPS (shuffling it hurts performance)
- Delta < 0 → feature HURTS (shuffling it improves performance, i.e. model uses it wrong or it adds noise)
**All features zeroed**: nRMSE_pw = 0.4340 (delta = -0.077) → static features contribute ~15% improvement overall

**Note**: Baseline is on train-side val wells, not the 52 spatial test wells. Scale is different from GP nRMSE_pw.

## Results (sorted by delta, descending)

| Rank | Feature | Short Description | Delta nRMSE_pw | % Change |
|------|---------|-------------------|----------------|----------|
| 1 | corine_1000_fractions_corineclccode_511 | CORINE: water transport network | +0.0218 | +4.3% |
| 2 | corine_1000_fractions_corineclccode_311 | CORINE: broad-leaved forest | +0.0169 | +3.3% |
| 3 | TWI_dgm50_r1000m | Topographic Wetness Index | +0.0162 | +3.2% |
| 4 | gok | Ground surface elevation (GOK) | +0.0157 | +3.1% |
| 5 | eumohp_1000_mean_sd1 | Slope (DEM std dev, 1km) | +0.0111 | +2.2% |
| 6 | corine_1000_fractions_corineclccode_121 | CORINE: industrial/commercial areas | +0.0094 | +1.8% |
| 7 | amatulli_1000_mean_shannongeom10kmsha | Shannon entropy of land use | +0.0090 | +1.8% |
| 8 | corine_1000_fractions_corineclccode_231 | CORINE: pastures | +0.0076 | +1.5% |
| 9 | corine_1000_fractions_corineclccode_141 | CORINE: sport/leisure facilities | +0.0065 | +1.3% |
| 10 | corine_1000_fractions_corineclccode_411 | CORINE: inland marshes | +0.0049 | +0.9% |
| 11 | hydroraum_Speisungsgebiete | Hydroraum: recharge zones | +0.0046 | +0.9% |
| 12 | amatulli_1000_mean_entgeom10kment | Entropy geometry of land use | +0.0041 | +0.8% |
| 13 | corine_1000_fractions_corineclccode_111 | CORINE: continuous urban fabric | +0.0025 | +0.5% |
| 14 | gwn1000_1000_mean_recharge | Groundwater recharge rate | +0.0016 | +0.3% |
| 15 | eumohp_1000_mean_dsd1 | DEM standard deviation | +0.0014 | +0.3% |
| 16 | corine_1000_fractions_corineclccode_322 | CORINE: moors and heathland | +0.0011 | +0.2% |
| 17 | eumohp_1000_mean_lp1 | Landscape position index | +0.0011 | +0.2% |
| 18 | amatulli_1000_mean_unigeom10kmuni | Geometry uniformity of land use | +0.0009 | +0.2% |
| 19 | huek250_1000_fractions_kf_9 | Hydraulic conductivity class 9 | +0.0001 | +0.0% |
| 20 | hydroraum_Transitgebiete | Hydroraum: transit zones | +0.0000 | 0.0% |
| 21 | corine_1000_fractions_corineclccode_242 | CORINE: complex cultivation patterns | -0.0004 | -0.1% |
| 22 | huek250_1000_fractions_kf_10 | Hydraulic conductivity class 10 | -0.0004 | -0.1% |
| 23 | corine_1000_fractions_corineclccode_131 | CORINE: mineral extraction sites | -0.0007 | -0.1% |
| 24 | huek250_1000_fractions_kf_11 | Hydraulic conductivity class 11 | -0.0009 | -0.2% |
| 25 | corine_1000_fractions_corineclccode_321 | CORINE: natural grassland | -0.0010 | -0.2% |
| 26 | corine_1000_fractions_corineclccode_512 | CORINE: water bodies (lakes etc.) | -0.0016 | -0.3% |
| 27 | corine_1000_fractions_corineclccode_211 | CORINE: non-irrigated arable land | -0.0050 | -1.0% |
| 28 | corine_1000_fractions_corineclccode_312 | CORINE: coniferous forest | -0.0069 | -1.3% |
| 29 | corine_1000_fractions_corineclccode_112 | CORINE: discontinuous urban fabric | -0.0105 | -2.1% |
| 30 | hydroraum_Entlastungsgebiete | Hydroraum: discharge zones | -0.0108 | -2.1% |
| 31 | corine_1000_fractions_corineclccode_313 | CORINE: mixed forest | -0.0141 | -2.8% |

## Interpretation

**Clearly helpful (delta ≥ +0.01)**: corine_511, corine_311, TWI, gok, eumohp_sd1 — top 5 features explain ~15% of performance each in relative terms.

**Marginally helpful (+0.001 to +0.009)**: corine_121, shannon_entropy, corine_231, corine_141, corine_411, hydroraum_Speisungsgebiete, entropy, corine_111, gwn_recharge, eumohp_dsd1, corine_322, eumohp_lp1, uniformity, huek250_kf9, hydroraum_Transitgebiete.

**Neutral/noise (|delta| < 0.002)**: huek250_kf9, hydroraum_Transitgebiete, corine_242, huek250_kf10, corine_131, huek250_kf11, corine_321.

**Actively harmful (delta ≤ -0.005)**: corine_512, corine_211, corine_312, corine_112, hydroraum_Entlastungsgebiete, corine_313. These 6 features hurt the model — removing them could improve performance.

**Note on hydroraum**: Only Speisungsgebiete (recharge, +0.9%) is helpful. Transitgebiete is neutral (0.0%). Entlastungsgebiete (discharge, -2.1%) actively hurts. These are one-hot encoded from the same categorical — removing Entlastungsgebiete alone isn't straightforward since it's part of the one-hot.
