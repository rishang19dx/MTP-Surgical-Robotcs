# Vector-env evaluation: nav (20261127_120058)

commit `3255749` · NVIDIA RTX A6000 · patients `../patients_ts2/ts2_s0620,../patients_ts2/ts2_s1283,../patients_ts2/ts2_s0683,../patients_ts2/ts2_s1415,../patients_ts2/ts2_s0923,../patients_ts2/ts2_s1152,../patients_ts2/ts2_s0983,../patients_ts2/ts2_s1267,../patients_ts2/ts2_s0260,../patients_ts2/ts2_s1386,../patients_ts2/ts2_s0009,../patients_ts2/ts2_s0477,../patients_ts2/ts2_s0476,../patients_ts2/ts2_s0797,../patients_ts2/ts2_s0248,../patients_ts2/ts2_s1369` · 16 episodes per patient · eval deformation `aware` · B-mode `depth` · seed 1000

| Policy | Mean success | Mean force (N) | Mean return | Final pos. error (mm) | Final yaw error (deg) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 3.98 | 35.86 | 1.4 | 2.2 |

| Patient | Policy | success | force (N) | return |
| :--- | :--- | ---: | ---: | ---: |
| ts2_s0620 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 4.00 | 34.97 |
| ts2_s1283 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 4.02 | 36.66 |
| ts2_s0683 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 4.00 | 40.88 |
| ts2_s1415 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 3.96 | 34.65 |
| ts2_s0923 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 4.03 | 36.41 |
| ts2_s1152 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 3.99 | 36.82 |
| ts2_s0983 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 3.97 | 37.58 |
| ts2_s1267 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 3.99 | 30.71 |
| ts2_s0260 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 3.98 | 35.25 |
| ts2_s1386 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 3.96 | 33.23 |
| ts2_s0009 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 3.98 | 36.84 |
| ts2_s0477 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 3.90 | 32.61 |
| ts2_s0476 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 3.98 | 37.91 |
| ts2_s0797 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 4.03 | 35.67 |
| ts2_s0248 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 4.01 | 34.97 |
| ts2_s1369 | ppo:nav_aware_depth_s0_s31_N16_1999872_steps.zip | 1.000 | 3.92 | 38.63 |
