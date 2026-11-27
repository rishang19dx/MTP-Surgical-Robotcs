# Vector-env evaluation: nav (20261127_121559)

commit `3255749-dirty` · NVIDIA RTX A6000 · patients `../patients_ts2/ts2_s0620,../patients_ts2/ts2_s1283,../patients_ts2/ts2_s0683,../patients_ts2/ts2_s1415,../patients_ts2/ts2_s0923,../patients_ts2/ts2_s1152,../patients_ts2/ts2_s0983,../patients_ts2/ts2_s1267,../patients_ts2/ts2_s0260,../patients_ts2/ts2_s1386,../patients_ts2/ts2_s0009,../patients_ts2/ts2_s0477,../patients_ts2/ts2_s0476,../patients_ts2/ts2_s0797,../patients_ts2/ts2_s0248,../patients_ts2/ts2_s1369,../patients_ts2/ts2_s0842,../patients_ts2/ts2_s0059,../patients_ts2/ts2_s1307,../patients_ts2/ts2_s0119,../patients_ts2/ts2_s1350,../patients_ts2/ts2_s0626,../patients_ts2/ts2_s0542,../patients_ts2/ts2_s1366,../patients_ts2/ts2_s0686,../patients_ts2/ts2_s1394,../patients_ts2/ts2_s1187,../patients_ts2/ts2_s1364,../patients_ts2/ts2_s0158,../patients_ts2/ts2_s1247,../patients_ts2/ts2_s0341,../patients_ts2/ts2_s0218,../patients_ts2/ts2_s0468,../patients_ts2/ts2_s0484,../patients_ts2/ts2_s1395,../patients_ts2/ts2_s0680,../patients_ts2/ts2_s0052,../patients_ts2/ts2_s0450,../patients_ts2/ts2_s0687,../patients_ts2/ts2_s0880,../patients_ts2/ts2_s1384,../patients_ts2/ts2_s0733,../patients_ts2/ts2_s0636,../patients_ts2/ts2_s0708,../patients_ts2/ts2_s1287,../patients_ts2/ts2_s0924,../patients_ts2/ts2_s1414,../patients_ts2/ts2_s0089,../patients_ts2/ts2_s0553,../patients_ts2/ts2_s1368,../patients_ts2/ts2_s0728,../patients_ts2/ts2_s0577,../patients_ts2/ts2_s0494,../patients_ts2/ts2_s0959,../patients_ts2/ts2_s0334,../patients_ts2/ts2_s1322,../patients_ts2/ts2_s0912,../patients_ts2/ts2_s0586,../patients_ts2/ts2_s1061,../patients_ts2/ts2_s1082,../patients_ts2/ts2_s0836,../patients_ts2/ts2_s1249,../patients_ts2/ts2_s0406,../patients_ts2/ts2_s0541` · 16 episodes per patient · eval deformation `aware` · B-mode `depth` · seed 1000

| Policy | Mean success | Mean force (N) | Mean return | Final pos. error (mm) | Final yaw error (deg) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 0.976 | 3.99 | 34.41 | 1.9 | 2.7 |

| Patient | Policy | success | force (N) | return |
| :--- | :--- | ---: | ---: | ---: |
| ts2_s0620 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 34.82 |
| ts2_s1283 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 0.562 | 4.02 | 27.63 |
| ts2_s0683 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 37.07 |
| ts2_s1415 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.96 | 31.39 |
| ts2_s0923 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.01 | 35.43 |
| ts2_s1152 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 0.938 | 3.98 | 32.75 |
| ts2_s0983 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.97 | 32.46 |
| ts2_s1267 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 31.85 |
| ts2_s0260 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 35.45 |
| ts2_s1386 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.96 | 31.63 |
| ts2_s0009 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.96 | 36.65 |
| ts2_s0477 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.90 | 34.67 |
| ts2_s0476 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 35.32 |
| ts2_s0797 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.02 | 33.48 |
| ts2_s0248 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 33.64 |
| ts2_s1369 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 0.938 | 3.97 | 35.04 |
| ts2_s0842 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.01 | 36.53 |
| ts2_s0059 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 0.812 | 4.13 | 31.52 |
| ts2_s1307 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 0.875 | 3.97 | 36.88 |
| ts2_s0119 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 36.40 |
| ts2_s1350 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 33.21 |
| ts2_s0626 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 33.73 |
| ts2_s0542 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 39.95 |
| ts2_s1366 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.81 | 30.53 |
| ts2_s0686 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.02 | 34.30 |
| ts2_s1394 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 32.04 |
| ts2_s1187 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 35.87 |
| ts2_s1364 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 33.57 |
| ts2_s0158 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 0.938 | 4.01 | 34.39 |
| ts2_s1247 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 0.938 | 4.00 | 33.84 |
| ts2_s0341 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.96 | 36.95 |
| ts2_s0218 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.96 | 35.65 |
| ts2_s0468 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 35.89 |
| ts2_s0484 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.97 | 35.86 |
| ts2_s1395 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 0.875 | 3.96 | 34.34 |
| ts2_s0680 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 31.63 |
| ts2_s0052 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.02 | 35.46 |
| ts2_s0450 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 33.20 |
| ts2_s0687 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.97 | 33.37 |
| ts2_s0880 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 31.60 |
| ts2_s1384 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.04 | 35.30 |
| ts2_s0733 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.01 | 31.68 |
| ts2_s0636 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 0.875 | 4.02 | 34.86 |
| ts2_s0708 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 32.97 |
| ts2_s1287 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.29 | 36.50 |
| ts2_s0924 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 0.938 | 3.99 | 33.38 |
| ts2_s1414 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 33.91 |
| ts2_s0089 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 0.875 | 3.91 | 34.25 |
| ts2_s0553 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 37.05 |
| ts2_s1368 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.00 | 36.44 |
| ts2_s0728 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.94 | 39.00 |
| ts2_s0577 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 35.96 |
| ts2_s0494 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 0.938 | 3.93 | 31.38 |
| ts2_s0959 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 33.37 |
| ts2_s0334 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.91 | 39.50 |
| ts2_s1322 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 30.10 |
| ts2_s0912 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 35.00 |
| ts2_s0586 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.99 | 31.36 |
| ts2_s1061 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.91 | 36.65 |
| ts2_s1082 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.94 | 33.14 |
| ts2_s0836 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 0.938 | 4.00 | 35.58 |
| ts2_s1249 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.05 | 34.84 |
| ts2_s0406 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 3.98 | 38.08 |
| ts2_s0541 | ppo:nav_aware_depth_s2_s31_N64_1999872_steps.zip | 1.000 | 4.02 | 36.16 |
