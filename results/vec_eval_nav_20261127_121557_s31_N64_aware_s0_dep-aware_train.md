# Vector-env evaluation: nav (20261127_121557)

commit `3255749-dirty` · NVIDIA RTX A6000 · patients `../patients_ts2/ts2_s0620,../patients_ts2/ts2_s1283,../patients_ts2/ts2_s0683,../patients_ts2/ts2_s1415,../patients_ts2/ts2_s0923,../patients_ts2/ts2_s1152,../patients_ts2/ts2_s0983,../patients_ts2/ts2_s1267,../patients_ts2/ts2_s0260,../patients_ts2/ts2_s1386,../patients_ts2/ts2_s0009,../patients_ts2/ts2_s0477,../patients_ts2/ts2_s0476,../patients_ts2/ts2_s0797,../patients_ts2/ts2_s0248,../patients_ts2/ts2_s1369,../patients_ts2/ts2_s0842,../patients_ts2/ts2_s0059,../patients_ts2/ts2_s1307,../patients_ts2/ts2_s0119,../patients_ts2/ts2_s1350,../patients_ts2/ts2_s0626,../patients_ts2/ts2_s0542,../patients_ts2/ts2_s1366,../patients_ts2/ts2_s0686,../patients_ts2/ts2_s1394,../patients_ts2/ts2_s1187,../patients_ts2/ts2_s1364,../patients_ts2/ts2_s0158,../patients_ts2/ts2_s1247,../patients_ts2/ts2_s0341,../patients_ts2/ts2_s0218,../patients_ts2/ts2_s0468,../patients_ts2/ts2_s0484,../patients_ts2/ts2_s1395,../patients_ts2/ts2_s0680,../patients_ts2/ts2_s0052,../patients_ts2/ts2_s0450,../patients_ts2/ts2_s0687,../patients_ts2/ts2_s0880,../patients_ts2/ts2_s1384,../patients_ts2/ts2_s0733,../patients_ts2/ts2_s0636,../patients_ts2/ts2_s0708,../patients_ts2/ts2_s1287,../patients_ts2/ts2_s0924,../patients_ts2/ts2_s1414,../patients_ts2/ts2_s0089,../patients_ts2/ts2_s0553,../patients_ts2/ts2_s1368,../patients_ts2/ts2_s0728,../patients_ts2/ts2_s0577,../patients_ts2/ts2_s0494,../patients_ts2/ts2_s0959,../patients_ts2/ts2_s0334,../patients_ts2/ts2_s1322,../patients_ts2/ts2_s0912,../patients_ts2/ts2_s0586,../patients_ts2/ts2_s1061,../patients_ts2/ts2_s1082,../patients_ts2/ts2_s0836,../patients_ts2/ts2_s1249,../patients_ts2/ts2_s0406,../patients_ts2/ts2_s0541` · 16 episodes per patient · eval deformation `aware` · B-mode `depth` · seed 1000

| Policy | Mean success | Mean force (N) | Mean return | Final pos. error (mm) | Final yaw error (deg) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.968 | 3.99 | 34.30 | 2.0 | 2.7 |

| Patient | Policy | success | force (N) | return |
| :--- | :--- | ---: | ---: | ---: |
| ts2_s0620 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 35.14 |
| ts2_s1283 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.750 | 4.02 | 32.16 |
| ts2_s0683 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 36.53 |
| ts2_s1415 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.93 | 31.31 |
| ts2_s0923 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 4.01 | 35.34 |
| ts2_s1152 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.875 | 3.97 | 31.94 |
| ts2_s0983 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 32.38 |
| ts2_s1267 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 31.66 |
| ts2_s0260 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.938 | 3.98 | 34.32 |
| ts2_s1386 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.97 | 31.37 |
| ts2_s0009 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.938 | 3.96 | 35.32 |
| ts2_s0477 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.938 | 3.87 | 33.79 |
| ts2_s0476 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.938 | 3.98 | 34.96 |
| ts2_s0797 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 4.02 | 34.21 |
| ts2_s0248 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.95 | 33.59 |
| ts2_s1369 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.812 | 4.00 | 33.03 |
| ts2_s0842 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 4.01 | 36.57 |
| ts2_s0059 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 4.15 | 36.66 |
| ts2_s1307 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.938 | 3.99 | 36.31 |
| ts2_s0119 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 36.64 |
| ts2_s1350 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 33.03 |
| ts2_s0626 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 33.53 |
| ts2_s0542 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 40.51 |
| ts2_s1366 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.82 | 30.12 |
| ts2_s0686 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.812 | 4.02 | 31.74 |
| ts2_s1394 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 31.41 |
| ts2_s1187 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.938 | 4.00 | 32.50 |
| ts2_s1364 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 33.76 |
| ts2_s0158 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 4.01 | 35.36 |
| ts2_s1247 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.938 | 4.00 | 34.24 |
| ts2_s0341 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.96 | 37.45 |
| ts2_s0218 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.88 | 35.95 |
| ts2_s0468 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 36.02 |
| ts2_s0484 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 35.57 |
| ts2_s1395 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.750 | 3.96 | 32.53 |
| ts2_s0680 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 31.32 |
| ts2_s0052 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 4.01 | 35.53 |
| ts2_s0450 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 33.63 |
| ts2_s0687 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.97 | 32.72 |
| ts2_s0880 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.875 | 3.99 | 29.63 |
| ts2_s1384 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 4.03 | 35.43 |
| ts2_s0733 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.875 | 4.00 | 30.23 |
| ts2_s0636 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.875 | 4.02 | 34.54 |
| ts2_s0708 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 33.54 |
| ts2_s1287 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 4.28 | 36.17 |
| ts2_s0924 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 34.21 |
| ts2_s1414 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.97 | 33.43 |
| ts2_s0089 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.89 | 36.24 |
| ts2_s0553 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.96 | 36.79 |
| ts2_s1368 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 37.07 |
| ts2_s0728 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.93 | 38.55 |
| ts2_s0577 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.938 | 3.98 | 35.38 |
| ts2_s0494 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.938 | 3.94 | 31.21 |
| ts2_s0959 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 33.38 |
| ts2_s0334 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 0.875 | 3.93 | 37.72 |
| ts2_s1322 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 29.94 |
| ts2_s0912 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 34.85 |
| ts2_s0586 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 31.47 |
| ts2_s1061 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.92 | 36.35 |
| ts2_s1082 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.95 | 33.16 |
| ts2_s0836 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 36.48 |
| ts2_s1249 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 4.03 | 34.59 |
| ts2_s0406 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 37.98 |
| ts2_s0541 | ppo:nav_aware_depth_s0_s31_N64_1999872_steps.zip | 1.000 | 4.02 | 36.66 |
