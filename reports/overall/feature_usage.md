# Feature Usage (model pipeline)

This table reflects **current model inputs** (TFT/GRU + GP/kriging), not every column referenced anywhere in the repo.

| Feature | Used in models? | Usage in current models | Meaning |
|---|---|---|---|
| `datum` | yes | time index | timestamp/date of observation |
| `woche` | no | unused in current models | calendar time field |
| `monat` | no | unused in current models | calendar time field |
| `jahr` | no | unused in current models | calendar time field |
| `zeit_diff` | no | unused in current models | time difference between observations |
| `id` | yes | group id | well identifier |
| `gws` | yes | target | groundwater level (target) |
| `z_gws` | no | unused in current models | dataset field (not used in current models) |
| `zeitreihenlänge` | no | unused in current models | dataset field (not used in current models) |
| `custom_1000_mean_twi` | yes | static covariate | dataset field (not used in current models) |
| `eumohp_1000_mean_dsd1` | yes | static covariate | EUMOHP terrain/hydromorphology index |
| `eumohp_1000_mean_lp1` | yes | static covariate | EUMOHP terrain/hydromorphology index |
| `eumohp_1000_mean_sd1` | yes | static covariate | EUMOHP terrain/hydromorphology index |
| `gwn1000_1000_mean_recharge` | yes | static covariate | groundwater recharge indicator (GWN) |
| `corine_1000_fractions_corineclccode_211` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_231` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_311` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_512` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_112` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_313` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_131` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_312` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_121` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_322` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_111` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_411` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_321` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_511` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_141` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `corine_1000_fractions_corineclccode_242` | yes | static covariate | land‑use/land‑cover fractions (CORINE) |
| `amatulli_1000_mean_entgeom10kment` | yes | static covariate | geomorphological diversity index |
| `amatulli_1000_mean_shannongeom10kmsha` | yes | static covariate | geomorphological diversity index |
| `amatulli_1000_mean_unigeom10kmuni` | yes | static covariate | geomorphological diversity index |
| `huek250_1000_fractions_kf_11` | yes | static covariate | hydrogeological permeability class fraction (HUEK) |
| `huek250_1000_fractions_kf_9` | yes | static covariate | hydrogeological permeability class fraction (HUEK) |
| `huek250_1000_fractions_kf_10` | yes | static covariate | hydrogeological permeability class fraction (HUEK) |
| `tas_1km` | no | unused in current models | air temperature (1 km grid) |
| `hurs_1km` | no | unused in current models | relative humidity (1 km grid) |
| `pr_1km` | no | unused in current models | precipitation (1 km grid) |
| `tag_sin` | yes | dynamic covariate | annual seasonality (sine) |
| `tag_cos` | yes | dynamic covariate | annual seasonality (cosine) |
| `tas_5km` | yes | dynamic covariate | air temperature (5 km grid) |
| `hurs_5km` | yes | dynamic covariate | relative humidity (5 km grid) |
| `pr_5km` | yes | dynamic covariate | precipitation (5 km grid) |
| `id_original` | no | unused in current models | dataset field (not used in current models) |
| `name` | no | unused in current models | dataset field (not used in current models) |
| `typ` | no | unused in current models | dataset field (not used in current models) |
| `parameter` | no | unused in current models | dataset field (not used in current models) |
| `x_25833` | yes | spatial coordinate | easting coordinate (EPSG:25833) |
| `y_25833` | yes | spatial coordinate | northing coordinate (EPSG:25833) |
| `gok` | no | unused in current models | dataset field (not used in current models) |
| `ausbausohle` | no | unused in current models | dataset field (not used in current models) |
| `fok` | no | unused in current models | dataset field (not used in current models) |
| `fuk` | no | unused in current models | dataset field (not used in current models) |
| `logger_beginn_lfu` | no | unused in current models | dataset field (not used in current models) |
| `logger_beginn` | no | unused in current models | dataset field (not used in current models) |
| `messintervall` | no | unused in current models | dataset field (not used in current models) |
| `gwlk` | no | unused in current models | dataset field (not used in current models) |
| `sensibel` | no | unused in current models | dataset field (not used in current models) |
| `messnetz` | no | unused in current models | dataset field (not used in current models) |
| `datenquelle` | no | unused in current models | dataset field (not used in current models) |
| `gws_logger` | no | unused in current models | dataset field (not used in current models) |
| `outlier_count` | no | unused in current models | dataset field (not used in current models) |
| `acf_2018-2023_preproc` | no | unused in current models | autocorrelation feature |
| `gws_beginn_ml_prognose` | no | unused in current models | dataset field (not used in current models) |
| `ml_prognose` | no | unused in current models | dataset field (not used in current models) |
| `dss_standsklassen` | no | unused in current models | dataset field (not used in current models) |
| `gws_beginn` | no | unused in current models | dataset field (not used in current models) |
| `imputed_proportion` | no | unused in current models | dataset field (not used in current models) |
| `interpolated_proportion` | no | unused in current models | dataset field (not used in current models) |
| `imputed_proportion_ml_prognose` | no | unused in current models | dataset field (not used in current models) |
| `interpolated_proportion_ml_prognose` | no | unused in current models | dataset field (not used in current models) |
| `gaps_count` | no | unused in current models | dataset field (not used in current models) |
| `min_gap_w` | no | unused in current models | dataset field (not used in current models) |
| `max_gap_w` | no | unused in current models | dataset field (not used in current models) |
| `konsistenz` | no | unused in current models | dataset field (not used in current models) |
| `konsistenz_1991-2020` | no | unused in current models | dataset field (not used in current models) |
| `median_1991-2020` | no | unused in current models | dataset field (not used in current models) |
| `median_ugok_1991-2020` | no | unused in current models | dataset field (not used in current models) |
| `mean_range_1991-2020` | no | unused in current models | dataset field (not used in current models) |
| `mean_range_hyj_1991-2020` | no | unused in current models | dataset field (not used in current models) |
| `acf_1991-2020` | no | unused in current models | autocorrelation feature |
| `geometry` | no | unused in current models | dataset field (not used in current models) |
| `gw_gespannt` | no | unused in current models | dataset field (not used in current models) |
| `siwa_verweilzeit_j` | no | unused in current models | dataset field (not used in current models) |
| `oezg_moor_habitat` | no | unused in current models | dataset field (not used in current models) |
| `distance_wsg_z1` | no | unused in current models | dataset field (not used in current models) |
| `TWI_dgm50_r1000m` | no | unused in current models | terrain/soil/recharge indicator at 1000 m radius |
| `Slope_dgm50_r1000m` | no | unused in current models | terrain/soil/recharge indicator at 1000 m radius |
| `GW_recharge_r1000m` | no | unused in current models | terrain/soil/recharge indicator at 1000 m radius |
| `Percolation_r1000m` | no | unused in current models | terrain/soil/recharge indicator at 1000 m radius |
| `DSD_order1_r1000m` | no | unused in current models | terrain/soil/recharge indicator at 1000 m radius |
| `LP_order1_r1000m` | no | unused in current models | terrain/soil/recharge indicator at 1000 m radius |
| `SD_order1_r1000m` | no | unused in current models | terrain/soil/recharge indicator at 1000 m radius |
| `DSD_order5_r1000m` | no | unused in current models | terrain/soil/recharge indicator at 1000 m radius |
| `LP_order5_r1000m` | no | unused in current models | terrain/soil/recharge indicator at 1000 m radius |
| `SD_order5_r1000m` | no | unused in current models | terrain/soil/recharge indicator at 1000 m radius |
| `range_ratio_1991-2020` | no | unused in current models | dataset field (not used in current models) |
| `skew` | no | unused in current models | dataset field (not used in current models) |
| `min_month` | no | unused in current models | dataset field (not used in current models) |
| `max_month` | no | unused in current models | dataset field (not used in current models) |
| `longest_recession` | no | unused in current models | dataset field (not used in current models) |
| `parde_seasonality` | no | unused in current models | dataset field (not used in current models) |
| `interannual_variation` | no | unused in current models | dataset field (not used in current models) |
| `low_pulse_count` | no | unused in current models | dataset field (not used in current models) |
| `high_pulse_count` | no | unused in current models | dataset field (not used in current models) |
| `low_pulse_duration` | no | unused in current models | dataset field (not used in current models) |
| `high_pulse_duration` | no | unused in current models | dataset field (not used in current models) |
| `baseflow_index` | no | unused in current models | dataset field (not used in current models) |
| `uEZG_Zuordnung` | no | unused in current models | dataset field (not used in current models) |
| `hydroraum` | no | unused in current models | dataset field (not used in current models) |
| `HGN_anthro` | no | unused in current models | dataset field (not used in current models) |
| `DFU` | no | unused in current models | dataset field (not used in current models) |
| `KIT_rank_old` | no | unused in current models | dataset field (not used in current models) |
| `MST_pro_uEZG` | no | unused in current models | dataset field (not used in current models) |
| `MeanDist5clostestMST` | no | unused in current models | dataset field (not used in current models) |
| `hydroraum_1991-2020` | no | unused in current models | dataset field (not used in current models) |
| `dropped_lfu` | no | unused in current models | dataset field (not used in current models) |
| `PCA_R2` | no | unused in current models | dataset field (not used in current models) |
| `trend_slope_GS` | no | unused in current models | dataset field (not used in current models) |
| `trend_bewertung_GS` | no | unused in current models | dataset field (not used in current models) |
| `KIT_rank` | no | unused in current models | dataset field (not used in current models) |
| `KIT_rank_RMSE` | no | unused in current models | dataset field (not used in current models) |
| `LSTM_16w_NSE` | no | unused in current models | dataset field (not used in current models) |
| `TFT_16w_NSE` | no | unused in current models | dataset field (not used in current models) |
| `LSTM_16w_RMSE` | no | unused in current models | dataset field (not used in current models) |
| `TFT_16w_RMSE` | no | unused in current models | dataset field (not used in current models) |
| `Cluster_1980` | no | unused in current models | dataset field (not used in current models) |
| `Silhouette_score_1980` | no | unused in current models | dataset field (not used in current models) |
| `Cluster_1990` | no | unused in current models | dataset field (not used in current models) |
| `Silhouette_score_1990` | no | unused in current models | dataset field (not used in current models) |
| `pr_1W_lag` | no | unused in current models | precipitation lag/cross‑correlation feature |
| `pr_1W_xcorr` | no | unused in current models | precipitation lag/cross‑correlation feature |
| `pr_52W_lag` | no | unused in current models | precipitation lag/cross‑correlation feature |
| `pr_52W_xcorr` | no | unused in current models | precipitation lag/cross‑correlation feature |
| `acf_class` | no | unused in current models | autocorrelation feature |
