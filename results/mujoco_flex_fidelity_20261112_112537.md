# MuJoCo flex fidelity vs Neo-Hookean reference (20261112_112537)

commit `3409010` · MuJoCo 4.0.0 · gate DEC-053 · reference `results/fem_fidelity_20261107_234638.json` · identical nodal contact forces (linear contact solution), as DEC-047

| Patient | Site | Force (N) | Material | Comp. linear (mm) | Comp. Neo-Hookean (mm) | Comp. flex (mm) | Rel. err flex | Rel. err linear | Settled (steps) | Inverted | Scored |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| sample_totalsegmentator_patient | paraspinal (-20, -45) | 2 | svk | 1.66 | 1.61 | 3.97 | +145.9 % | +2.7 % | yes (110) | 30 | yes |
| sample_totalsegmentator_patient | paraspinal (-20, -45) | 2 | snh | 1.66 | 1.61 | 1.63 | +1.2 % | +2.7 % | yes (67) | 0 | yes |
| sample_totalsegmentator_patient | paraspinal (-20, -45) | 5 | svk | 3.42 | 3.22 | 15.42 | +379.5 % | +6.3 % | yes (544) | 729 | yes |
| sample_totalsegmentator_patient | paraspinal (-20, -45) | 5 | snh | 3.42 | 3.22 | 3.31 | +3.0 % | +6.3 % | yes (68) | 0 | yes |
| sample_totalsegmentator_patient | paraspinal (-20, -45) | 10 | svk | 6.12 | 5.47 | 24.21 | +342.8 % | +11.9 % | yes (490) | 2180 | yes |
| sample_totalsegmentator_patient | paraspinal (-20, -45) | 10 | snh | 6.12 | 5.47 | 5.88 | +7.6 % | +11.9 % | yes (76) | 0 | yes |
| sample_totalsegmentator_patient | paraspinal (-20, -45) | 15 | svk | 8.92 | 7.73 | 30.29 | +291.7 % | +15.3 % | yes (532) | 3664 | yes |
| sample_totalsegmentator_patient | paraspinal (-20, -45) | 15 | snh | 8.92 | 7.73 | 14.80 | +91.5 % | +15.3 % | yes (194) | 11 | yes |
| sample_totalsegmentator_patient | midline (0, -32.5) | 2 | svk | 3.53 | 3.55 | 11.33 | +219.2 % | -0.5 % | yes (200) | 179 | yes |
| sample_totalsegmentator_patient | midline (0, -32.5) | 2 | snh | 3.53 | 3.55 | 3.71 | +4.5 % | -0.5 % | yes (89) | 0 | yes |
| sample_totalsegmentator_patient | midline (0, -32.5) | 5 | svk | 4.84 | 5.85 | 15.20 | +159.8 % | -17.3 % | yes (574) | 635 | yes |
| sample_totalsegmentator_patient | midline (0, -32.5) | 5 | snh | 4.84 | 5.85 | 6.73 | +15.1 % | -17.3 % | yes (250) | 0 | yes |
| sample_totalsegmentator_patient | midline (0, -32.5) | 10 | svk | 6.93 | 8.86 | 27.64 | +211.9 % | -21.9 % | yes (1128) | 3515 | yes |
| sample_totalsegmentator_patient | midline (0, -32.5) | 10 | snh | 6.93 | 8.86 | 9.97 | +12.5 % | -21.9 % | yes (87) | 0 | yes |
| sample_totalsegmentator_patient | midline (0, -32.5) | 15 | svk | 9.56 | 12.53 | 29.84 | +138.2 % | -23.7 % | yes (803) | 4068 | yes |
| sample_totalsegmentator_patient | midline (0, -32.5) | 15 | snh | 9.56 | 12.53 | 15.68 | +25.2 % | -23.7 % | yes (362) | 7 | yes |
| sonogym_s0015 | paraspinal (-20, -45) | 2 | svk | 1.94 | 1.85 | 8.77 | +374.9 % | +5.1 % | yes (198) | 196 | yes |
| sonogym_s0015 | paraspinal (-20, -45) | 2 | snh | 1.94 | 1.85 | 1.87 | +1.2 % | +5.1 % | yes (85) | 0 | yes |
| sonogym_s0015 | paraspinal (-20, -45) | 5 | svk | 4.27 | 3.85 | 18.32 | +375.4 % | +10.8 % | yes (265) | 1017 | yes |
| sonogym_s0015 | paraspinal (-20, -45) | 5 | snh | 4.27 | 3.85 | 3.97 | +3.1 % | +10.8 % | yes (88) | 0 | yes |
| sonogym_s0015 | paraspinal (-20, -45) | 10 | svk | 8.29 | 7.04 | 32.40 | +360.1 % | +17.8 % | yes (926) | 3982 | yes |
| sonogym_s0015 | paraspinal (-20, -45) | 10 | snh | 8.29 | 7.04 | 23.70 | +236.6 % | +17.8 % | yes (465) | 18 | yes |
| sonogym_s0015 | paraspinal (-20, -45) | 15 | svk | 12.36 | 6.66 | 35.09 | +426.8 % | +85.6 % | yes (505) | 4357 | no: Neo-Hookean not converged in DEC-047 |
| sonogym_s0015 | paraspinal (-20, -45) | 15 | snh | 12.36 | 6.66 | 35.78 | +437.2 % | +85.6 % | yes (152) | 39 | no: Neo-Hookean not converged in DEC-047 |
| sonogym_s0015 | midline (0, -32.5) | 2 | svk | 1.66 | 1.59 | 8.25 | +419.9 % | +4.4 % | yes (351) | 292 | yes |
| sonogym_s0015 | midline (0, -32.5) | 2 | snh | 1.66 | 1.59 | 1.61 | +1.5 % | +4.4 % | yes (74) | 0 | yes |
| sonogym_s0015 | midline (0, -32.5) | 5 | svk | 3.33 | 3.08 | 10.96 | +255.5 % | +8.1 % | yes (185) | 442 | yes |
| sonogym_s0015 | midline (0, -32.5) | 5 | snh | 3.33 | 3.08 | 3.18 | +3.2 % | +8.1 % | yes (75) | 0 | yes |
| sonogym_s0015 | midline (0, -32.5) | 10 | svk | 6.32 | 5.61 | 26.29 | +368.9 % | +12.8 % | yes (709) | 3039 | yes |
| sonogym_s0015 | midline (0, -32.5) | 10 | snh | 6.32 | 5.61 | 6.15 | +9.8 % | +12.8 % | yes (83) | 0 | yes |
| sonogym_s0015 | midline (0, -32.5) | 15 | svk | 9.34 | 8.41 | 34.39 | +309.2 % | +11.1 % | yes (1284) | 6409 | yes |
| sonogym_s0015 | midline (0, -32.5) | 15 | snh | 9.34 | 8.41 | 26.42 | +214.3 % | +11.1 % | yes (367) | 27 | yes |
| sonogym_s0014 | paraspinal (-20, -45) | 2 | svk | 3.56 | 3.62 | 8.84 | +143.9 % | -1.8 % | yes (203) | 96 | yes |
| sonogym_s0014 | paraspinal (-20, -45) | 2 | snh | 3.56 | 3.62 | 3.70 | +2.0 % | -1.8 % | yes (182) | 0 | yes |
| sonogym_s0014 | paraspinal (-20, -45) | 5 | svk | 7.21 | 9.16 | 21.76 | +137.6 % | -21.2 % | yes (833) | 1717 | yes |
| sonogym_s0014 | paraspinal (-20, -45) | 5 | snh | 7.21 | 9.16 | 9.39 | +2.5 % | -21.2 % | yes (184) | 0 | yes |
| sonogym_s0014 | paraspinal (-20, -45) | 10 | svk | 13.67 | 16.00 | 32.24 | +101.5 % | -14.6 % | yes (909) | 3933 | yes |
| sonogym_s0014 | paraspinal (-20, -45) | 10 | snh | 13.67 | 16.00 | 17.00 | +6.3 % | -14.6 % | yes (186) | 0 | yes |
| sonogym_s0014 | paraspinal (-20, -45) | 15 | svk | 20.28 | 24.33 | 35.67 | +46.6 % | -16.7 % | yes (659) | 4507 | yes |
| sonogym_s0014 | paraspinal (-20, -45) | 15 | snh | 20.28 | 24.33 | 39.47 | +62.2 % | -16.7 % | yes (284) | 25 | yes |
| sonogym_s0014 | midline (0, -32.5) | 2 | svk | 3.42 | 3.41 | 6.47 | +89.7 % | +0.1 % | yes (204) | 45 | yes |
| sonogym_s0014 | midline (0, -32.5) | 2 | snh | 3.42 | 3.41 | 3.47 | +1.8 % | +0.1 % | yes (167) | 0 | yes |
| sonogym_s0014 | midline (0, -32.5) | 5 | svk | 7.65 | 8.42 | 21.22 | +152.0 % | -9.2 % | yes (740) | 1201 | yes |
| sonogym_s0014 | midline (0, -32.5) | 5 | snh | 7.65 | 8.42 | 8.62 | +2.3 % | -9.2 % | yes (170) | 0 | yes |
| sonogym_s0014 | midline (0, -32.5) | 10 | svk | 14.43 | 14.95 | 33.26 | +122.4 % | -3.5 % | yes (1182) | 3595 | yes |
| sonogym_s0014 | midline (0, -32.5) | 10 | snh | 14.43 | 14.95 | 16.08 | +7.6 % | -3.5 % | yes (182) | 0 | yes |
| sonogym_s0014 | midline (0, -32.5) | 15 | svk | 21.14 | 20.00 | 39.65 | +98.3 % | +5.7 % | yes (875) | 5409 | no: Neo-Hookean not converged in DEC-047 |
| sonogym_s0014 | midline (0, -32.5) | 15 | snh | 21.14 | 20.00 | 42.31 | +111.6 % | +5.7 % | yes (222) | 30 | no: Neo-Hookean not converged in DEC-047 |

## Gate (DEC-053)

| Material | G1 (≤5 N, all within 10 %) | G2 (10–15 N median vs linear) | G3 (finite, no inversion) | G4 (warm re-settle) | G5 | Passes |
| :--- | :---: | :--- | :---: | :--- | :--- | :---: |
| svk | FAIL | FAIL (251.8 % vs 14.9 %) | FAIL | fail (50467 ms) | see ReconstructionPipeline/tests/test_mujoco_flex.py::test_per_tet_scaling_equals_changing_young | **no** |
| snh | FAIL | FAIL (18.8 % vs 14.9 %) | FAIL | fail (12511 ms) | see ReconstructionPipeline/tests/test_mujoco_flex.py::test_per_tet_scaling_equals_changing_young | **no** |
