# Thread B v2: muscle thickness under probe force (_20261112_151349)

commit `563d540` · pre-registration DEC-056 · imaging depth + fascia (DEC-055) · forces [1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0] N (primary 1–5 N; 8–10 N reported, DEC-047 caveat)

## Criteria (test patients)

| Criterion | Value | Pass |
| :--- | :--- | :---: |
| G: detection ≤ 2 N ≥ 0.5 | 1.00 | |
| G: median abs error ≤ 2 N < 3 mm | 1.51 mm | **pass** |
| H4a: deformable image slope < 0 in every measurable patient (2) | see table | pass |
| H4b: rigid image slope within ±0.15 mm/N | see table | pass |
| H4c: median tracking gap ≤ 0.1 mm/N | 0.075 mm/N | pass |
| **H4** (a and b and c) | | **pass** |
| **H5**: force-aware beats intercept-only by ≥ 0.377 mm | 1.31 vs 1.32 mm (uncorrected 0.75); k = -0.087 mm/N | **FAIL** |

## Per patient (scored sites; slopes over 1–5 N)

| Patient | Split | Scored sites | Detect ≤ 2 N | Median abs err ≤ 2 N (mm) | Bias (mm) | Image slope, deformable | GT slope, deformable | Image slope, rigid (mm/N) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sonogym_s0015 | train | 6 | 1.00 | 0.94 | +0.75 | -0.339 | -0.339 | -0.019 |
| sonogym_s0010 | train | 6 | 1.00 | 1.13 | +1.13 | -0.245 | -0.226 | -0.038 |
| sonogym_s0012 | train | 6 | 0.92 | 1.32 | -0.57 | -0.245 | -0.283 | +0.094 |
| sonogym_s0038 | train | 5 | 1.00 | 0.38 | +0.38 | -0.188 | -0.113 | +0.075 |
| sonogym_s0024 | train | 2 | 0.00 | — | — | +0.188 | -0.085 | +0.000 |
| sonogym_s0004 | train | 3 | 1.00 | 1.13 | +1.13 | -0.226 | -0.188 | +0.038 |
| sonogym_s0028 | train | 6 | 1.00 | 1.32 | +0.85 | -0.113 | -0.160 | -0.038 |
| sonogym_s0014 | train | 0 | — | — | — | — | — | — |
| sonogym_s0029 | test | 6 | 1.00 | 0.75 | +0.75 | -0.302 | -0.339 | +0.019 |
| sonogym_s0006 | test | 6 | 1.00 | 2.26 | +2.26 | -0.245 | -0.226 | +0.038 |
| sonogym_s0034 | test | 2 | 0.00 | — | — | — | -0.160 | — |
| sonogym_s0030 | test | 0 | — | — | — | — | — | — |

Train (development set, not scored): G pass (detect 1.00, err 1.13 mm), H4 pass (a True, b True, c True).
