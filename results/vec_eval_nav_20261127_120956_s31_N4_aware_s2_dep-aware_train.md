# Vector-env evaluation: nav (20261127_120956)

commit `3255749-dirty` · NVIDIA RTX A6000 · patients `../patients_ts2/ts2_s0620,../patients_ts2/ts2_s1283,../patients_ts2/ts2_s0683,../patients_ts2/ts2_s1415` · 16 episodes per patient · eval deformation `aware` · B-mode `depth` · seed 1000

| Policy | Mean success | Mean force (N) | Mean return | Final pos. error (mm) | Final yaw error (deg) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ppo:nav_aware_depth_s2_s31_N4_1999872_steps.zip | 1.000 | 3.98 | 34.98 | 0.8 | 1.4 |

| Patient | Policy | success | force (N) | return |
| :--- | :--- | ---: | ---: | ---: |
| ts2_s0620 | ppo:nav_aware_depth_s2_s31_N4_1999872_steps.zip | 1.000 | 4.00 | 34.02 |
| ts2_s1283 | ppo:nav_aware_depth_s2_s31_N4_1999872_steps.zip | 1.000 | 4.04 | 34.82 |
| ts2_s0683 | ppo:nav_aware_depth_s2_s31_N4_1999872_steps.zip | 1.000 | 3.99 | 37.77 |
| ts2_s1415 | ppo:nav_aware_depth_s2_s31_N4_1999872_steps.zip | 1.000 | 3.92 | 33.29 |
