# Paraspinal muscle thickness under compression (20261108_021028)

commit `6420d21` · pre-registration DEC-050 · forces 1, 2, 3, 4, 5, 8, 10 N (above 5 N: linear-FEM caveat, DEC-047)

**Gate:** measurable (detect ≥ 0.5 and median |err| < 3 mm at ≤ 2 N): FAIL (detect 0.38, |err| 7.302135927593293); compression 1→5 N > 0.377 mm: PASS (0.75 mm).

**H4** (image thickness falls with force, 1–5 N): supported, median slope -0.35804020100502354 mm/N.

**H5** (force-linear correction fitted on train, test): {"k_mm_per_N": -1.8426121095827954, "test_median_abs_err_uncorrected_mm": 7.53768849971906, "test_median_abs_err_corrected_mm": 2.5333687264892, "supported": true}

| Patient | Split | Reference thickness (mm) | GT compression 1→5 N (mm) | Detect ≤ 2 N | Median abs err ≤ 2 N (mm) | Image slope (mm/N) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| sonogym_s0015 | train | 33.2 | 1.32 | 0.75 | 4.899497966670516 | -0.3391959798995001 |
| sonogym_s0010 | train | 34.9 | 0.75 | 0.50 | 6.407035482588729 | -0.4522613065326671 |
| sonogym_s0012 | train | 36.9 | 1.13 | 0.58 | 10.175879607847591 | -0.489949748743719 |
| sonogym_s0038 | train | 29.4 | 0.75 | 0.75 | 5.276382273765066 | -0.11306532663316476 |
| sonogym_s0024 | train | 42.2 | -0.38 | 0.00 | None | 0.28266331658292704 |
| sonogym_s0004 | train | 31.8 | 0.75 | 0.42 | 13.944723828953123 | -0.8668341708542682 |
| sonogym_s0028 | train | 32.4 | 0.57 | 0.33 | 6.218593424888116 | -0.18844221105527875 |
| sonogym_s0014 | train | 31.7 | -2.83 | 0.00 | None | None |
| sonogym_s0029 | test | 36.4 | 1.13 | 0.67 | 9.987437435130982 | -0.45226130653266117 |
| sonogym_s0006 | test | 40.3 | 1.13 | 0.17 | 8.197236372597857 | -0.37688442211054696 |
| sonogym_s0034 | test | 36.0 | 0.57 | 0.00 | None | 6.674703196033882e-15 |
| sonogym_s0030 | test | 27.1 | -2.83 | 0.00 | None | None |
