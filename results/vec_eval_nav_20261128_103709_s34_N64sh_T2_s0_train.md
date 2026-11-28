# Vector-env evaluation: nav (20261128_103709)

commit `146663b` · NVIDIA RTX A6000 · patients `../patients_ts2/ts2_s0620,../patients_ts2/ts2_s1283,../patients_ts2/ts2_s0683,../patients_ts2/ts2_s1415,../patients_ts2/ts2_s0923,../patients_ts2/ts2_s1152,../patients_ts2/ts2_s0983,../patients_ts2/ts2_s1267,../patients_ts2/ts2_s0260,../patients_ts2/ts2_s1386,../patients_ts2/ts2_s0009,../patients_ts2/ts2_s0477,../patients_ts2/ts2_s0476,../patients_ts2/ts2_s0797,../patients_ts2/ts2_s0248,../patients_ts2/ts2_s1369,../patients_ts2/ts2_s0842,../patients_ts2/ts2_s0059,../patients_ts2/ts2_s1307,../patients_ts2/ts2_s0119,../patients_ts2/ts2_s1350,../patients_ts2/ts2_s0626,../patients_ts2/ts2_s0542,../patients_ts2/ts2_s1366,../patients_ts2/ts2_s0686,../patients_ts2/ts2_s1394,../patients_ts2/ts2_s1187,../patients_ts2/ts2_s1364,../patients_ts2/ts2_s0158,../patients_ts2/ts2_s1247,../patients_ts2/ts2_s0341,../patients_ts2/ts2_s0218,../patients_ts2/ts2_s0468,../patients_ts2/ts2_s0484,../patients_ts2/ts2_s1395,../patients_ts2/ts2_s0680,../patients_ts2/ts2_s0052,../patients_ts2/ts2_s0450,../patients_ts2/ts2_s0687,../patients_ts2/ts2_s0880,../patients_ts2/ts2_s1384,../patients_ts2/ts2_s0733,../patients_ts2/ts2_s0636,../patients_ts2/ts2_s0708,../patients_ts2/ts2_s1287,../patients_ts2/ts2_s0924,../patients_ts2/ts2_s1414,../patients_ts2/ts2_s0089,../patients_ts2/ts2_s0553,../patients_ts2/ts2_s1368,../patients_ts2/ts2_s0728,../patients_ts2/ts2_s0577,../patients_ts2/ts2_s0494,../patients_ts2/ts2_s0959,../patients_ts2/ts2_s0334,../patients_ts2/ts2_s1322,../patients_ts2/ts2_s0912,../patients_ts2/ts2_s0586,../patients_ts2/ts2_s1061,../patients_ts2/ts2_s1082,../patients_ts2/ts2_s0836,../patients_ts2/ts2_s1249,../patients_ts2/ts2_s0406,../patients_ts2/ts2_s0541` · 16 episodes per patient · eval deformation `aware` · B-mode `depth` · seed 1000

| Policy | Mean success | Mean force (N) | Mean return | Final pos. error (mm) | Final yaw error (deg) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.950 | 3.99 | 34.04 | 2.0 | 2.8 |

| Patient | Policy | success | force (N) | return |
| :--- | :--- | ---: | ---: | ---: |
| ts2_s0620 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 4.00 | 34.72 |
| ts2_s1283 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.750 | 4.02 | 31.92 |
| ts2_s0683 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.875 | 3.99 | 35.15 |
| ts2_s1415 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.938 | 3.96 | 30.50 |
| ts2_s0923 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 4.01 | 35.41 |
| ts2_s1152 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.812 | 3.95 | 30.76 |
| ts2_s0983 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.938 | 3.95 | 31.46 |
| ts2_s1267 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 32.05 |
| ts2_s0260 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.938 | 3.98 | 34.88 |
| ts2_s1386 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.97 | 31.98 |
| ts2_s0009 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.97 | 36.01 |
| ts2_s0477 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.875 | 3.90 | 32.84 |
| ts2_s0476 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.875 | 4.00 | 34.28 |
| ts2_s0797 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.938 | 4.01 | 33.09 |
| ts2_s0248 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.938 | 3.99 | 31.61 |
| ts2_s1369 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.812 | 4.00 | 33.14 |
| ts2_s0842 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 4.01 | 36.67 |
| ts2_s0059 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 4.17 | 36.61 |
| ts2_s1307 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.875 | 4.01 | 36.32 |
| ts2_s0119 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.938 | 3.99 | 35.30 |
| ts2_s1350 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.97 | 32.80 |
| ts2_s0626 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 33.47 |
| ts2_s0542 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 39.90 |
| ts2_s1366 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.86 | 29.82 |
| ts2_s0686 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 4.03 | 34.74 |
| ts2_s1394 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 31.57 |
| ts2_s1187 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 4.00 | 35.18 |
| ts2_s1364 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.875 | 3.99 | 32.61 |
| ts2_s0158 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.938 | 4.01 | 33.62 |
| ts2_s1247 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.812 | 4.00 | 31.34 |
| ts2_s0341 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.97 | 37.09 |
| ts2_s0218 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.80 | 35.87 |
| ts2_s0468 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.98 | 36.14 |
| ts2_s0484 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.98 | 35.77 |
| ts2_s1395 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.812 | 3.95 | 33.05 |
| ts2_s0680 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 4.00 | 31.66 |
| ts2_s0052 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 4.03 | 35.50 |
| ts2_s0450 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.98 | 33.94 |
| ts2_s0687 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.97 | 32.44 |
| ts2_s0880 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 31.64 |
| ts2_s1384 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 35.27 |
| ts2_s0733 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.98 | 32.46 |
| ts2_s0636 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.625 | 4.03 | 32.03 |
| ts2_s0708 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.938 | 4.00 | 32.39 |
| ts2_s1287 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 4.40 | 35.98 |
| ts2_s0924 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.938 | 3.98 | 32.24 |
| ts2_s1414 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.875 | 3.97 | 31.78 |
| ts2_s0089 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.90 | 36.21 |
| ts2_s0553 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 36.81 |
| ts2_s1368 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 36.40 |
| ts2_s0728 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.96 | 38.54 |
| ts2_s0577 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.938 | 3.98 | 35.76 |
| ts2_s0494 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.875 | 3.93 | 31.48 |
| ts2_s0959 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.98 | 33.44 |
| ts2_s0334 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.94 | 39.38 |
| ts2_s1322 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 30.09 |
| ts2_s0912 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 34.43 |
| ts2_s0586 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 31.25 |
| ts2_s1061 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.85 | 37.32 |
| ts2_s1082 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.938 | 3.94 | 31.85 |
| ts2_s0836 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 36.02 |
| ts2_s1249 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.938 | 4.03 | 33.36 |
| ts2_s0406 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.875 | 3.97 | 35.42 |
| ts2_s0541 | ppo:nav_aware_depth_s0_s34_N64sh_1999872_steps.zip | 0.938 | 4.02 | 35.81 |
