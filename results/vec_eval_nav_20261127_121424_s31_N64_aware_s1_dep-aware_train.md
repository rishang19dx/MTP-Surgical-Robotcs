# Vector-env evaluation: nav (20261127_121424)

commit `3255749-dirty` · NVIDIA RTX A6000 · patients `../patients_ts2/ts2_s0620,../patients_ts2/ts2_s1283,../patients_ts2/ts2_s0683,../patients_ts2/ts2_s1415,../patients_ts2/ts2_s0923,../patients_ts2/ts2_s1152,../patients_ts2/ts2_s0983,../patients_ts2/ts2_s1267,../patients_ts2/ts2_s0260,../patients_ts2/ts2_s1386,../patients_ts2/ts2_s0009,../patients_ts2/ts2_s0477,../patients_ts2/ts2_s0476,../patients_ts2/ts2_s0797,../patients_ts2/ts2_s0248,../patients_ts2/ts2_s1369,../patients_ts2/ts2_s0842,../patients_ts2/ts2_s0059,../patients_ts2/ts2_s1307,../patients_ts2/ts2_s0119,../patients_ts2/ts2_s1350,../patients_ts2/ts2_s0626,../patients_ts2/ts2_s0542,../patients_ts2/ts2_s1366,../patients_ts2/ts2_s0686,../patients_ts2/ts2_s1394,../patients_ts2/ts2_s1187,../patients_ts2/ts2_s1364,../patients_ts2/ts2_s0158,../patients_ts2/ts2_s1247,../patients_ts2/ts2_s0341,../patients_ts2/ts2_s0218,../patients_ts2/ts2_s0468,../patients_ts2/ts2_s0484,../patients_ts2/ts2_s1395,../patients_ts2/ts2_s0680,../patients_ts2/ts2_s0052,../patients_ts2/ts2_s0450,../patients_ts2/ts2_s0687,../patients_ts2/ts2_s0880,../patients_ts2/ts2_s1384,../patients_ts2/ts2_s0733,../patients_ts2/ts2_s0636,../patients_ts2/ts2_s0708,../patients_ts2/ts2_s1287,../patients_ts2/ts2_s0924,../patients_ts2/ts2_s1414,../patients_ts2/ts2_s0089,../patients_ts2/ts2_s0553,../patients_ts2/ts2_s1368,../patients_ts2/ts2_s0728,../patients_ts2/ts2_s0577,../patients_ts2/ts2_s0494,../patients_ts2/ts2_s0959,../patients_ts2/ts2_s0334,../patients_ts2/ts2_s1322,../patients_ts2/ts2_s0912,../patients_ts2/ts2_s0586,../patients_ts2/ts2_s1061,../patients_ts2/ts2_s1082,../patients_ts2/ts2_s0836,../patients_ts2/ts2_s1249,../patients_ts2/ts2_s0406,../patients_ts2/ts2_s0541` · 16 episodes per patient · eval deformation `aware` · B-mode `depth` · seed 1000

| Policy | Mean success | Mean force (N) | Mean return | Final pos. error (mm) | Final yaw error (deg) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.949 | 3.99 | 34.09 | 1.9 | 2.8 |

| Patient | Policy | success | force (N) | return |
| :--- | :--- | ---: | ---: | ---: |
| ts2_s0620 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 35.21 |
| ts2_s1283 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.438 | 4.02 | 28.45 |
| ts2_s0683 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 36.90 |
| ts2_s1415 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.97 | 31.26 |
| ts2_s0923 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.01 | 35.50 |
| ts2_s1152 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.938 | 3.97 | 32.67 |
| ts2_s0983 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.95 | 32.74 |
| ts2_s1267 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 31.73 |
| ts2_s0260 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.625 | 4.00 | 30.21 |
| ts2_s1386 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.938 | 3.97 | 30.85 |
| ts2_s0009 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 36.46 |
| ts2_s0477 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.86 | 35.09 |
| ts2_s0476 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 35.67 |
| ts2_s0797 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.938 | 4.01 | 32.60 |
| ts2_s0248 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 33.67 |
| ts2_s1369 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 36.09 |
| ts2_s0842 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.01 | 36.70 |
| ts2_s0059 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.20 | 37.05 |
| ts2_s1307 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.812 | 3.98 | 36.24 |
| ts2_s0119 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.875 | 3.99 | 33.37 |
| ts2_s1350 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 32.87 |
| ts2_s0626 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.938 | 3.99 | 32.73 |
| ts2_s0542 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.875 | 4.00 | 37.94 |
| ts2_s1366 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.92 | 30.08 |
| ts2_s0686 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.875 | 4.03 | 32.82 |
| ts2_s1394 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 31.80 |
| ts2_s1187 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 35.25 |
| ts2_s1364 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 33.75 |
| ts2_s0158 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 35.10 |
| ts2_s1247 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.812 | 4.01 | 32.03 |
| ts2_s0341 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.97 | 36.89 |
| ts2_s0218 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.92 | 35.75 |
| ts2_s0468 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 36.24 |
| ts2_s0484 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 35.26 |
| ts2_s1395 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.688 | 3.96 | 31.27 |
| ts2_s0680 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 31.48 |
| ts2_s0052 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.02 | 35.30 |
| ts2_s0450 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 33.57 |
| ts2_s0687 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.938 | 3.97 | 32.13 |
| ts2_s0880 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 31.50 |
| ts2_s1384 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.09 | 36.02 |
| ts2_s0733 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.97 | 31.50 |
| ts2_s0636 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.500 | 4.01 | 30.03 |
| ts2_s0708 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 32.96 |
| ts2_s1287 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.25 | 36.15 |
| ts2_s0924 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 34.45 |
| ts2_s1414 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 33.76 |
| ts2_s0089 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.750 | 3.89 | 31.57 |
| ts2_s0553 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 36.91 |
| ts2_s1368 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 36.98 |
| ts2_s0728 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.94 | 38.62 |
| ts2_s0577 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 36.83 |
| ts2_s0494 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.938 | 3.95 | 31.62 |
| ts2_s0959 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 33.07 |
| ts2_s0334 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.938 | 3.92 | 38.50 |
| ts2_s1322 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 30.09 |
| ts2_s0912 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 35.01 |
| ts2_s0586 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 32.15 |
| ts2_s1061 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.91 | 36.39 |
| ts2_s1082 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.94 | 33.01 |
| ts2_s0836 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 0.938 | 4.00 | 35.50 |
| ts2_s1249 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.04 | 34.19 |
| ts2_s0406 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 37.74 |
| ts2_s0541 | ppo:nav_aware_depth_s1_s31_N64_1999872_steps.zip | 1.000 | 4.02 | 36.45 |
