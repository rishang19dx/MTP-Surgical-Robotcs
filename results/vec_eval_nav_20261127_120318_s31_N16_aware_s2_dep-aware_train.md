# Vector-env evaluation: nav (20261127_120318)

commit `3255749` · NVIDIA RTX A6000 · patients `../patients_ts2/ts2_s0620,../patients_ts2/ts2_s1283,../patients_ts2/ts2_s0683,../patients_ts2/ts2_s1415,../patients_ts2/ts2_s0923,../patients_ts2/ts2_s1152,../patients_ts2/ts2_s0983,../patients_ts2/ts2_s1267,../patients_ts2/ts2_s0260,../patients_ts2/ts2_s1386,../patients_ts2/ts2_s0009,../patients_ts2/ts2_s0477,../patients_ts2/ts2_s0476,../patients_ts2/ts2_s0797,../patients_ts2/ts2_s0248,../patients_ts2/ts2_s1369` · 16 episodes per patient · eval deformation `aware` · B-mode `depth` · seed 1000

| Policy | Mean success | Mean force (N) | Mean return | Final pos. error (mm) | Final yaw error (deg) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 0.996 | 3.98 | 35.72 | 1.5 | 2.2 |

| Patient | Policy | success | force (N) | return |
| :--- | :--- | ---: | ---: | ---: |
| ts2_s0620 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 0.938 | 4.00 | 32.80 |
| ts2_s1283 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 4.02 | 36.80 |
| ts2_s0683 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 3.99 | 40.73 |
| ts2_s1415 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 3.92 | 34.92 |
| ts2_s0923 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 4.02 | 36.74 |
| ts2_s1152 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 3.98 | 36.20 |
| ts2_s0983 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 3.97 | 37.55 |
| ts2_s1267 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 3.99 | 30.60 |
| ts2_s0260 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 3.97 | 35.92 |
| ts2_s1386 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 3.96 | 33.68 |
| ts2_s0009 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 4.00 | 37.52 |
| ts2_s0477 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 3.93 | 32.41 |
| ts2_s0476 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 4.01 | 37.90 |
| ts2_s0797 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 4.02 | 35.08 |
| ts2_s0248 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 3.99 | 34.01 |
| ts2_s1369 | ppo:nav_aware_depth_s2_s31_N16_1999872_steps.zip | 1.000 | 3.89 | 38.62 |
