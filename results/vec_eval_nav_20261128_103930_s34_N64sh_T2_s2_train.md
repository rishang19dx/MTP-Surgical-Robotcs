# Vector-env evaluation: nav (20261128_103930)

commit `146663b` · NVIDIA RTX A6000 · patients `../patients_ts2/ts2_s0620,../patients_ts2/ts2_s1283,../patients_ts2/ts2_s0683,../patients_ts2/ts2_s1415,../patients_ts2/ts2_s0923,../patients_ts2/ts2_s1152,../patients_ts2/ts2_s0983,../patients_ts2/ts2_s1267,../patients_ts2/ts2_s0260,../patients_ts2/ts2_s1386,../patients_ts2/ts2_s0009,../patients_ts2/ts2_s0477,../patients_ts2/ts2_s0476,../patients_ts2/ts2_s0797,../patients_ts2/ts2_s0248,../patients_ts2/ts2_s1369,../patients_ts2/ts2_s0842,../patients_ts2/ts2_s0059,../patients_ts2/ts2_s1307,../patients_ts2/ts2_s0119,../patients_ts2/ts2_s1350,../patients_ts2/ts2_s0626,../patients_ts2/ts2_s0542,../patients_ts2/ts2_s1366,../patients_ts2/ts2_s0686,../patients_ts2/ts2_s1394,../patients_ts2/ts2_s1187,../patients_ts2/ts2_s1364,../patients_ts2/ts2_s0158,../patients_ts2/ts2_s1247,../patients_ts2/ts2_s0341,../patients_ts2/ts2_s0218,../patients_ts2/ts2_s0468,../patients_ts2/ts2_s0484,../patients_ts2/ts2_s1395,../patients_ts2/ts2_s0680,../patients_ts2/ts2_s0052,../patients_ts2/ts2_s0450,../patients_ts2/ts2_s0687,../patients_ts2/ts2_s0880,../patients_ts2/ts2_s1384,../patients_ts2/ts2_s0733,../patients_ts2/ts2_s0636,../patients_ts2/ts2_s0708,../patients_ts2/ts2_s1287,../patients_ts2/ts2_s0924,../patients_ts2/ts2_s1414,../patients_ts2/ts2_s0089,../patients_ts2/ts2_s0553,../patients_ts2/ts2_s1368,../patients_ts2/ts2_s0728,../patients_ts2/ts2_s0577,../patients_ts2/ts2_s0494,../patients_ts2/ts2_s0959,../patients_ts2/ts2_s0334,../patients_ts2/ts2_s1322,../patients_ts2/ts2_s0912,../patients_ts2/ts2_s0586,../patients_ts2/ts2_s1061,../patients_ts2/ts2_s1082,../patients_ts2/ts2_s0836,../patients_ts2/ts2_s1249,../patients_ts2/ts2_s0406,../patients_ts2/ts2_s0541` · 16 episodes per patient · eval deformation `aware` · B-mode `depth` · seed 1000

| Policy | Mean success | Mean force (N) | Mean return | Final pos. error (mm) | Final yaw error (deg) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.965 | 4.00 | 34.04 | 2.2 | 2.7 |

| Patient | Policy | success | force (N) | return |
| :--- | :--- | ---: | ---: | ---: |
| ts2_s0620 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 4.00 | 33.52 |
| ts2_s1283 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.875 | 4.02 | 33.61 |
| ts2_s0683 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 37.02 |
| ts2_s1415 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 3.97 | 29.81 |
| ts2_s0923 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.01 | 35.59 |
| ts2_s1152 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.812 | 3.98 | 30.29 |
| ts2_s0983 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.98 | 32.26 |
| ts2_s1267 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.00 | 31.74 |
| ts2_s0260 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.875 | 3.98 | 33.49 |
| ts2_s1386 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.98 | 31.89 |
| ts2_s0009 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.98 | 36.96 |
| ts2_s0477 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.90 | 34.27 |
| ts2_s0476 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.875 | 3.99 | 33.89 |
| ts2_s0797 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 4.02 | 33.26 |
| ts2_s0248 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.98 | 33.78 |
| ts2_s1369 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 4.00 | 34.69 |
| ts2_s0842 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.01 | 36.79 |
| ts2_s0059 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.13 | 36.99 |
| ts2_s1307 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 3.96 | 37.35 |
| ts2_s0119 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 3.98 | 35.02 |
| ts2_s1350 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 3.98 | 31.98 |
| ts2_s0626 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.00 | 33.28 |
| ts2_s0542 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 39.71 |
| ts2_s1366 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 3.91 | 27.98 |
| ts2_s0686 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 4.02 | 31.33 |
| ts2_s1394 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.00 | 31.91 |
| ts2_s1187 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 35.21 |
| ts2_s1364 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 33.40 |
| ts2_s0158 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.688 | 3.99 | 28.02 |
| ts2_s1247 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.00 | 34.65 |
| ts2_s0341 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.97 | 36.90 |
| ts2_s0218 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.92 | 35.96 |
| ts2_s0468 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.98 | 36.23 |
| ts2_s0484 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.98 | 35.65 |
| ts2_s1395 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.688 | 3.96 | 29.33 |
| ts2_s0680 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 3.99 | 29.53 |
| ts2_s0052 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.01 | 35.78 |
| ts2_s0450 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 33.42 |
| ts2_s0687 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.97 | 32.76 |
| ts2_s0880 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 31.28 |
| ts2_s1384 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.06 | 35.35 |
| ts2_s0733 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.875 | 4.00 | 29.06 |
| ts2_s0636 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.02 | 36.30 |
| ts2_s0708 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 4.00 | 32.31 |
| ts2_s1287 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.64 | 36.38 |
| ts2_s0924 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 3.99 | 32.65 |
| ts2_s1414 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.00 | 33.38 |
| ts2_s0089 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.89 | 36.10 |
| ts2_s0553 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 4.03 | 34.76 |
| ts2_s1368 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.00 | 37.31 |
| ts2_s0728 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.95 | 38.68 |
| ts2_s0577 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 3.96 | 33.54 |
| ts2_s0494 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.94 | 32.87 |
| ts2_s0959 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 33.54 |
| ts2_s0334 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.93 | 39.69 |
| ts2_s1322 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 30.20 |
| ts2_s0912 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 0.938 | 3.99 | 33.14 |
| ts2_s0586 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.99 | 31.58 |
| ts2_s1061 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.89 | 36.74 |
| ts2_s1082 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.90 | 33.24 |
| ts2_s0836 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.00 | 36.45 |
| ts2_s1249 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.03 | 34.51 |
| ts2_s0406 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 3.98 | 37.74 |
| ts2_s0541 | ppo:nav_aware_depth_s2_s34_N64sh_1999872_steps.zip | 1.000 | 4.01 | 36.61 |
