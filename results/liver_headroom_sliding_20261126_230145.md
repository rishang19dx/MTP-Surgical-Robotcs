# Liver target-shift headroom — SLIDING liver (DEC-060) (20261126_230145)

pre-registration DEC-060 · commit `3255749` · deformation-only shift |u| of liver targets (deformable − kinematic) · gate: median at 5 N ≥ 2.0 mm in ≥ 2/3 of measurable patients at every liver modulus

## Gate

| Liver E (kPa) | Measurable patients | ≥ 2 mm | Pass |
| ---: | ---: | ---: | :---: |
| 2.3 | 10 | 0 | **FAIL** |
| 4.1 | 10 | 0 | **FAIL** |
| 5.9 | 10 | 0 | **FAIL** |

**Overall: FAIL**

## Per patient (median over sites × targets)

| Patient | Split | Sites | |u| 5 N, E 2.3 | |u| 5 N, E 4.1 | |u| 5 N, E 5.9 | in-plane / out-of-plane 5 N, E 4.1 | |u| by force 1–10 N, E 4.1 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| sonogym_s0004 | train | 6 | 1.00 | 0.82 | 0.74 | 0.82 / 0.08 | 0.24 0.42 0.58 0.70 0.82 1.13 1.31 |
| sonogym_s0006 | test | — | — | — | — | — | not measurable: liver touches the CT field of view within 10 mm of a site |
| sonogym_s0010 | train | 6 | 1.09 | 0.96 | 0.82 | 0.95 / 0.09 | 0.26 0.49 0.67 0.82 0.96 1.30 1.49 |
| sonogym_s0012 | train | 6 | 1.23 | 1.05 | 0.95 | 1.04 / 0.15 | 0.29 0.53 0.73 0.90 1.05 1.43 1.63 |
| sonogym_s0014 | train | 6 | 0.64 | 0.52 | 0.48 | 0.51 / 0.10 | 0.13 0.25 0.35 0.44 0.52 0.74 0.87 |
| sonogym_s0015 | train | 6 | 0.34 | 0.31 | 0.30 | 0.31 / 0.05 | 0.07 0.13 0.20 0.26 0.31 0.46 0.54 |
| sonogym_s0024 | train | 6 | 0.86 | 0.73 | 0.66 | 0.72 / 0.13 | 0.22 0.37 0.51 0.62 0.73 1.02 1.18 |
| sonogym_s0028 | train | 6 | 0.43 | 0.40 | 0.37 | 0.38 / 0.06 | 0.11 0.20 0.27 0.34 0.40 0.56 0.64 |
| sonogym_s0029 | test | 6 | 1.64 | 1.45 | 1.33 | 1.45 / 0.09 | 0.41 0.75 1.02 1.25 1.45 1.95 2.21 |
| sonogym_s0030 | test | 6 | 0.60 | 0.54 | 0.49 | 0.54 / 0.03 | 0.14 0.26 0.37 0.46 0.54 0.75 0.86 |
| sonogym_s0034 | test | — | — | — | — | — | not measurable: liver touches the CT field of view within 10 mm of a site |
| sonogym_s0038 | train | 6 | 1.57 | 1.35 | 1.23 | 1.32 / 0.23 | 0.39 0.69 0.94 1.15 1.35 1.82 2.06 |

Reference (not decisive): one axial pixel 0.377 mm; lumbar muscle force effect ≈ 1.0–1.3 mm over 1–5 N (DEC-056).
