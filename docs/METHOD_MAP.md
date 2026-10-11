# Method-to-code map

| Manuscript component | Implementation under `legacy/` | Interpretation |
|---|---|---|
| Observed raster channels | `ALS-DiffBridge/scripts/preprocess/05_build_25d_maps_from_full_blocks.py` | Physical raster statistics from observed points |
| Shared normalization | `ALS-DiffBridge/scripts/preprocess/14_apply_shared_25d_normalization_v3.py` | Training-fit statistics; normalization files remain byte-preserved |
| Label-agnostic synthetic degradation | `ALS-DiffBridge/scripts/preprocess/12_make_label_agnostic_degraded_dales_blocks_v3.py` | Original degradation implementation |
| Point baseline and fusion topology | `ALS-DiffBridge/scripts/train/39_train_randlanet_reliability_ablation.py` | RandLA backbone, hierarchy and fusion modules |
| O-full | `ALS-DiffBridge/scripts/uq/train_restoration_control.py` | Observed raster inputs, without restoration inference |
| O-sampled | `cg_upgrade_20260914_run02/controls.py:sampled_prior` | Rasterize unique sampled observations; duplicates do not inflate density |
| O-point | `cg_upgrade_20260914_run02/controls.py:point_prior` | Concatenate the four point features twice and a zero channel; nine-channel capacity control |
| F-pre / F-rnd | `ALS-DiffBridge/scripts/uq/frozen_unet_features.py`, `train_feature_control.py` | Frozen pretrained/random U-Net feature branches; checkpoint and cache provenance matters |
| External ALS datasets | `whu25d/completion_v2/train_complete.py`, `h3d25d/run_v2/train_complete.py`, `external_controls.py` | Dataset-specific data and precision settings |
| PointNeXt | `cg_upgrade_20260914_run02/pointnext_runner.py` | Local point-only adaptation of the pinned upstream architecture |
| Fusion position | `cg_upgrade_20260914_run03/d1_controls.py` | Input-only and stage-1-only controls |
| PointNeXt plus observation | `cg_upgrade_20260914_run03/d3_runner.py` | Input-fusion adaptation; not an identical internal-stage injection |
| Additional DALES seeds | `f16_run/run_dales.py`, `f16_run/f16_evaluate.py` | Seeds 45/46 added to O-point and O-sampled; original controls use 43/44 |
| Fixed-sampled scoring | `cg_evaluation_20260914_01/evaluation.py` | Fixed-size sampled evaluation |
| Full-retained scoring | `cg_evaluation_20260914_01/full_evaluation.py` | Disjoint retained-point groups, without full-cloud interpolation |
| Scoring-population diagnostic | `f21_run/f21_coverage_isolation.py` | Rescore fixed predictions; inspect the stored confusion/order self-checks |
| Observation support size | `f44/f44_observation_scale.py` | Retained-point counts and sampled-input ratios |
| Terrain product diagnostic | `cg_analysis_20260914_01/terrain_core.py`, `terrain_evaluation.py` | Median ground grids, MAE/RMSE and coverage |

The WHU directory name is historical. The manuscript concerns the ALS subset; the corrected module description uses ALS. No dataset content or label mapping was changed during this preparation.

