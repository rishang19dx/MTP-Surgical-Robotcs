# Vector-env evaluation: nav (20261126_214039)

commit `3255749` · NVIDIA RTX A6000 · patients `dev` · 16 episodes per patient · eval deformation `aware` · B-mode `depth` · seed 1000

| Policy | Mean success | Mean force (N) | Mean return | Final pos. error (mm) | Final yaw error (deg) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| random | 0.042 | 3.98 | -4.84 | 21.7 | 23.0 |
| blind | 0.088 | 4.00 | 15.38 | 11.1 | 4.7 |
| oracle | 1.000 | 3.98 | 37.14 | 0.8 | 0.1 |
| ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.465 | 3.98 | 21.23 | 8.6 | 5.6 |

| Patient | Policy | success | force (N) | return |
| :--- | :--- | ---: | ---: | ---: |
| ts2_s0416 | random | 0.062 | 4.00 | -9.27 |
| ts2_s1334 | random | 0.062 | 3.99 | -6.43 |
| ts2_s1105 | random | 0.000 | 3.98 | -2.91 |
| ts2_s1235 | random | 0.062 | 3.98 | -1.66 |
| ts2_s0768 | random | 0.000 | 3.96 | -9.92 |
| ts2_s0550 | random | 0.000 | 3.99 | -0.44 |
| ts2_s0250 | random | 0.062 | 3.90 | -0.96 |
| ts2_s0860 | random | 0.062 | 3.99 | -3.97 |
| ts2_s1206 | random | 0.062 | 3.98 | -5.25 |
| ts2_s0896 | random | 0.125 | 3.99 | -5.75 |
| ts2_s1344 | random | 0.062 | 3.99 | -4.02 |
| ts2_s0369 | random | 0.000 | 3.99 | -9.65 |
| ts2_s0864 | random | 0.125 | 3.99 | -4.56 |
| ts2_s0939 | random | 0.062 | 3.98 | -1.96 |
| ts2_s0965 | random | 0.000 | 3.98 | -9.59 |
| ts2_s0940 | random | 0.000 | 3.98 | -4.08 |
| ts2_s0300 | random | 0.000 | 3.99 | -7.78 |
| ts2_s1008 | random | 0.000 | 3.99 | -4.08 |
| ts2_s1340 | random | 0.000 | 3.99 | -8.32 |
| ts2_s0592 | random | 0.000 | 3.99 | -2.26 |
| ts2_s0390 | random | 0.000 | 3.97 | -1.96 |
| ts2_s0109 | random | 0.062 | 3.98 | -0.06 |
| ts2_s0495 | random | 0.000 | 3.99 | -1.02 |
| ts2_s0650 | random | 0.188 | 4.00 | -7.85 |
| ts2_s1207 | random | 0.062 | 4.00 | 0.93 |
| ts2_s1212 | random | 0.000 | 3.98 | -14.56 |
| ts2_s0054 | random | 0.062 | 4.00 | -3.30 |
| ts2_s0416 | blind | 0.062 | 4.00 | 13.89 |
| ts2_s1334 | blind | 0.188 | 3.99 | 18.65 |
| ts2_s1105 | blind | 0.188 | 3.99 | 18.83 |
| ts2_s1235 | blind | 0.125 | 4.01 | 15.16 |
| ts2_s0768 | blind | 0.062 | 3.99 | 14.74 |
| ts2_s0550 | blind | 0.062 | 3.99 | 16.11 |
| ts2_s0250 | blind | 0.062 | 3.99 | 12.06 |
| ts2_s0860 | blind | 0.188 | 4.00 | 16.46 |
| ts2_s1206 | blind | 0.062 | 3.98 | 13.36 |
| ts2_s0896 | blind | 0.000 | 4.00 | 12.32 |
| ts2_s1344 | blind | 0.062 | 3.99 | 16.94 |
| ts2_s0369 | blind | 0.188 | 4.00 | 15.96 |
| ts2_s0864 | blind | 0.062 | 4.00 | 12.12 |
| ts2_s0939 | blind | 0.062 | 4.00 | 16.05 |
| ts2_s0965 | blind | 0.062 | 3.98 | 14.21 |
| ts2_s0940 | blind | 0.188 | 4.00 | 17.16 |
| ts2_s0300 | blind | 0.125 | 3.99 | 13.42 |
| ts2_s1008 | blind | 0.125 | 4.00 | 17.58 |
| ts2_s1340 | blind | 0.000 | 4.00 | 18.73 |
| ts2_s0592 | blind | 0.062 | 4.00 | 19.26 |
| ts2_s0390 | blind | 0.062 | 3.99 | 17.49 |
| ts2_s0109 | blind | 0.000 | 4.00 | 16.77 |
| ts2_s0495 | blind | 0.000 | 4.00 | 15.77 |
| ts2_s0650 | blind | 0.062 | 4.00 | 7.06 |
| ts2_s1207 | blind | 0.062 | 4.00 | 14.59 |
| ts2_s1212 | blind | 0.125 | 4.00 | 13.69 |
| ts2_s0054 | blind | 0.125 | 4.00 | 16.86 |
| ts2_s0416 | oracle | 1.000 | 4.00 | 34.65 |
| ts2_s1334 | oracle | 1.000 | 3.97 | 38.65 |
| ts2_s1105 | oracle | 1.000 | 3.94 | 40.28 |
| ts2_s1235 | oracle | 1.000 | 4.03 | 34.61 |
| ts2_s0768 | oracle | 1.000 | 4.07 | 35.74 |
| ts2_s0550 | oracle | 1.000 | 3.95 | 38.96 |
| ts2_s0250 | oracle | 1.000 | 3.92 | 34.58 |
| ts2_s0860 | oracle | 1.000 | 3.97 | 37.53 |
| ts2_s1206 | oracle | 1.000 | 3.88 | 34.49 |
| ts2_s0896 | oracle | 1.000 | 4.01 | 36.02 |
| ts2_s1344 | oracle | 1.000 | 3.95 | 37.92 |
| ts2_s0369 | oracle | 1.000 | 4.00 | 35.61 |
| ts2_s0864 | oracle | 1.000 | 3.97 | 35.34 |
| ts2_s0939 | oracle | 1.000 | 3.95 | 37.85 |
| ts2_s0965 | oracle | 1.000 | 3.86 | 38.18 |
| ts2_s0940 | oracle | 1.000 | 4.01 | 37.14 |
| ts2_s0300 | oracle | 1.000 | 3.95 | 34.86 |
| ts2_s1008 | oracle | 1.000 | 4.00 | 38.54 |
| ts2_s1340 | oracle | 1.000 | 3.99 | 41.57 |
| ts2_s0592 | oracle | 1.000 | 3.97 | 39.97 |
| ts2_s0390 | oracle | 1.000 | 3.92 | 39.40 |
| ts2_s0109 | oracle | 1.000 | 4.03 | 39.58 |
| ts2_s0495 | oracle | 1.000 | 4.01 | 40.97 |
| ts2_s0650 | oracle | 1.000 | 4.01 | 31.84 |
| ts2_s1207 | oracle | 1.000 | 3.99 | 36.64 |
| ts2_s1212 | oracle | 1.000 | 4.01 | 34.62 |
| ts2_s0054 | oracle | 1.000 | 4.01 | 37.19 |
| ts2_s0416 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.875 | 4.00 | 28.77 |
| ts2_s1334 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 1.000 | 3.98 | 36.53 |
| ts2_s1105 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.062 | 4.00 | 12.95 |
| ts2_s1235 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.000 | 3.98 | 12.91 |
| ts2_s0768 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.562 | 3.97 | 24.20 |
| ts2_s0550 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.938 | 3.98 | 34.61 |
| ts2_s0250 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.188 | 3.84 | 9.89 |
| ts2_s0860 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.688 | 3.97 | 24.10 |
| ts2_s1206 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 1.000 | 3.93 | 31.86 |
| ts2_s0896 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.375 | 4.00 | 14.33 |
| ts2_s1344 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.062 | 3.96 | 11.76 |
| ts2_s0369 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.688 | 4.00 | 27.05 |
| ts2_s0864 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.812 | 3.99 | 26.78 |
| ts2_s0939 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.250 | 3.98 | 19.58 |
| ts2_s0965 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.438 | 3.95 | 22.38 |
| ts2_s0940 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.875 | 3.99 | 32.41 |
| ts2_s0300 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.312 | 3.99 | 16.22 |
| ts2_s1008 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.625 | 4.01 | 23.21 |
| ts2_s1340 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.625 | 3.99 | 31.66 |
| ts2_s0592 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.250 | 3.98 | 19.44 |
| ts2_s0390 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.062 | 3.96 | 15.83 |
| ts2_s0109 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.750 | 4.00 | 29.25 |
| ts2_s0495 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.375 | 3.99 | 23.67 |
| ts2_s0650 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.250 | 4.00 | 12.52 |
| ts2_s1207 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.250 | 3.99 | 14.20 |
| ts2_s1212 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.188 | 3.95 | 14.79 |
| ts2_s0054 | ppo:nav_aware_depth_s0_budget32_1999872_steps.zip | 0.062 | 4.00 | 2.34 |
