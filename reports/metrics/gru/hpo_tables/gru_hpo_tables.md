# GRU HPO Tables

## gru_ablation_revin_scheduler

|   trial | status   |   objective | run_sig                                              | use_revin   | lr_scheduler.enabled   |
|--------:|:---------|------------:|:-----------------------------------------------------|:------------|:-----------------------|
|       1 | ok       |    0.161223 | gru_ablation_revin_scheduler_t001_seed40_full_merged | True        | False                  |
|       2 | ok       |    0.211171 | gru_ablation_revin_scheduler_t002_seed40_full_merged | False       | True                   |
|       3 | ok       |    0.161223 | gru_ablation_revin_scheduler_t003_seed40_full_merged | True        | True                   |
|       4 | ok       |    0.211171 | gru_ablation_revin_scheduler_t004_seed40_full_merged | False       | False                  |

## gru_small_grid_v1

|   trial | status   |   objective | run_sig                                   |   gru_dropout |   gru_hidden |   gru_layers |     lr |
|--------:|:---------|------------:|:------------------------------------------|--------------:|-------------:|-------------:|-------:|
|       1 | ok       |    0.251387 | gru_small_grid_v1_t001_seed40_full_merged |           0.3 |           96 |            2 | 0.0001 |
|       2 | ok       |    0.252191 | gru_small_grid_v1_t002_seed40_full_merged |           0.1 |          128 |            3 | 0.0001 |
|       3 | ok       |    0.221616 | gru_small_grid_v1_t003_seed40_full_merged |           0.2 |          192 |            2 | 0.0003 |
|       4 | ok       |    0.228087 | gru_small_grid_v1_t004_seed40_full_merged |           0.2 |          192 |            3 | 0.0003 |
|       5 | ok       |    0.258357 | gru_small_grid_v1_t005_seed40_full_merged |           0.2 |           96 |            2 | 0.001  |
|       6 | ok       |    0.24608  | gru_small_grid_v1_t006_seed40_full_merged |           0.1 |          128 |            2 | 0.0001 |
|       7 | ok       |    0.26284  | gru_small_grid_v1_t007_seed40_full_merged |           0.2 |          192 |            3 | 0.001  |
|       8 | ok       |    0.246734 | gru_small_grid_v1_t008_seed40_full_merged |           0.1 |           96 |            3 | 0.001  |
|       9 | ok       |    0.26104  | gru_small_grid_v1_t009_seed40_full_merged |           0.3 |          128 |            1 | 0.0001 |
|      10 | ok       |    0.223917 | gru_small_grid_v1_t010_seed40_full_merged |           0.2 |           96 |            2 | 0.0001 |
|      11 | ok       |    0.218888 | gru_small_grid_v1_t011_seed40_full_merged |           0.3 |          192 |            2 | 0.001  |
|      12 | ok       |    0.234292 | gru_small_grid_v1_t012_seed40_full_merged |           0.2 |          192 |            2 | 0.001  |

## gru_small_grid_v2

|   trial | status   |   objective | run_sig                                   |   gru_dropout |   gru_hidden |   gru_layers |     lr |
|--------:|:---------|------------:|:------------------------------------------|--------------:|-------------:|-------------:|-------:|
|       1 | ok       |    0.230395 | gru_small_grid_v2_t001_seed40_full_merged |           0.2 |          192 |            4 | 0.0003 |
|       2 | ok       |    0.26514  | gru_small_grid_v2_t002_seed40_full_merged |           0.2 |          224 |            2 | 0.0002 |
|       3 | ok       |    0.228087 | gru_small_grid_v2_t003_seed40_full_merged |           0.2 |          192 |            3 | 0.0003 |
|       4 | ok       |    0.23114  | gru_small_grid_v2_t004_seed40_full_merged |           0.2 |          256 |            2 | 0.0005 |
|       5 | ok       |    0.229092 | gru_small_grid_v2_t005_seed40_full_merged |           0.2 |          256 |            4 | 0.0002 |
|       6 | ok       |    0.233102 | gru_small_grid_v2_t006_seed40_full_merged |           0.2 |          224 |            3 | 0.0005 |
|       7 | ok       |    0.231678 | gru_small_grid_v2_t007_seed40_full_merged |           0.2 |          256 |            3 | 0.0004 |
|       8 | ok       |    0.261097 | gru_small_grid_v2_t008_seed40_full_merged |           0.2 |          256 |            2 | 0.0003 |
|       9 | ok       |    0.257551 | gru_small_grid_v2_t009_seed40_full_merged |           0.2 |          224 |            4 | 0.0004 |
|      10 | ok       |    0.268461 | gru_small_grid_v2_t010_seed40_full_merged |           0.2 |          256 |            2 | 0.0004 |
|      11 | ok       |    0.236454 | gru_small_grid_v2_t011_seed40_full_merged |           0.2 |          256 |            3 | 0.0005 |
|      12 | ok       |    0.242845 | gru_small_grid_v2_t012_seed40_full_merged |           0.2 |          192 |            4 | 0.0004 |

## gru_small_grid_v3

|   trial | status   |   objective | run_sig                                   |   gru_dropout |   gru_hidden |   gru_layers |      lr |
|--------:|:---------|------------:|:------------------------------------------|--------------:|-------------:|-------------:|--------:|
|       1 | ok       |    0.240127 | gru_small_grid_v3_t001_seed40_full_merged |           0.2 |          192 |            4 | 0.00035 |
|       2 | ok       |    0.230295 | gru_small_grid_v3_t002_seed40_full_merged |           0.2 |          160 |            5 | 0.0004  |
|       3 | ok       |    0.237071 | gru_small_grid_v3_t003_seed40_full_merged |           0.2 |          160 |            4 | 0.0004  |
|       4 | ok       |    0.242845 | gru_small_grid_v3_t004_seed40_full_merged |           0.2 |          192 |            4 | 0.0004  |
|       5 | ok       |    0.24315  | gru_small_grid_v3_t005_seed40_full_merged |           0.2 |          192 |            5 | 0.0003  |
|       6 | ok       |    0.230395 | gru_small_grid_v3_t006_seed40_full_merged |           0.2 |          192 |            4 | 0.0003  |
|       7 | ok       |    0.22073  | gru_small_grid_v3_t007_seed40_full_merged |           0.2 |          192 |            5 | 0.0004  |
|       8 | ok       |    0.228713 | gru_small_grid_v3_t008_seed40_full_merged |           0.2 |          160 |            5 | 0.0003  |
|       9 | ok       |    0.225535 | gru_small_grid_v3_t009_seed40_full_merged |           0.2 |          160 |            5 | 0.00035 |
|      10 | ok       |    0.237523 | gru_small_grid_v3_t010_seed40_full_merged |           0.2 |          160 |            4 | 0.0003  |
|      11 | ok       |    0.23936  | gru_small_grid_v3_t011_seed40_full_merged |           0.2 |          160 |            4 | 0.00035 |
|      12 | ok       |    0.246917 | gru_small_grid_v3_t012_seed40_full_merged |           0.2 |          192 |            5 | 0.00035 |

## gru_small_refine_v4

|   trial | status   |   objective | run_sig                                     |   gru_dropout |   gru_hidden |   gru_layers |       lr |
|--------:|:---------|------------:|:--------------------------------------------|--------------:|-------------:|-------------:|---------:|
|       1 | ok       |    0.265741 | gru_small_refine_v4_t001_seed40_full_merged |           0.2 |          176 |            4 | 0.00035  |
|       2 | ok       |    0.240127 | gru_small_refine_v4_t002_seed40_full_merged |           0.2 |          192 |            4 | 0.00035  |
|       3 | ok       |    0.245305 | gru_small_refine_v4_t003_seed40_full_merged |           0.2 |          192 |            4 | 0.000375 |
|       4 | ok       |    0.23814  | gru_small_refine_v4_t004_seed40_full_merged |           0.2 |          176 |            4 | 0.000375 |
|       5 | ok       |    0.242845 | gru_small_refine_v4_t005_seed40_full_merged |           0.2 |          192 |            4 | 0.0004   |
|       6 | ok       |    0.237071 | gru_small_refine_v4_t006_seed40_full_merged |           0.2 |          160 |            4 | 0.0004   |
|       7 | ok       |    0.223943 | gru_small_refine_v4_t007_seed40_full_merged |           0.2 |          176 |            4 | 0.0004   |
|       8 | ok       |    0.23936  | gru_small_refine_v4_t008_seed40_full_merged |           0.2 |          160 |            4 | 0.00035  |
|       9 | ok       |    0.228455 | gru_small_refine_v4_t009_seed40_full_merged |           0.2 |          160 |            4 | 0.000375 |

