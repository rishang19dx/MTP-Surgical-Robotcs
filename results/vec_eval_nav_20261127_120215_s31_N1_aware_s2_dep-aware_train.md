# Vector-env evaluation: nav (20261127_120215)

commit `3255749` · NVIDIA RTX A6000 · patients `../patients_ts2/ts2_s0620` · 16 episodes per patient · eval deformation `aware` · B-mode `depth` · seed 1000

| Policy | Mean success | Mean force (N) | Mean return | Final pos. error (mm) | Final yaw error (deg) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ppo:nav_aware_depth_s2_s31_N1_1999872_steps.zip | 1.000 | 4.00 | 37.47 | 0.4 | 0.7 |

| Patient | Policy | success | force (N) | return |
| :--- | :--- | ---: | ---: | ---: |
| ts2_s0620 | ppo:nav_aware_depth_s2_s31_N1_1999872_steps.zip | 1.000 | 4.00 | 37.47 |
