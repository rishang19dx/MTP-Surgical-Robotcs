# Liver target-shift headroom (20261122_102208)

pre-registration DEC-062 · commit `f84441f` · deformation-only shift |u| of liver targets (deformable − kinematic) · gate: median at 5 N ≥ 2.0 mm in ≥ 2/3 of measurable patients at every liver modulus

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
| sonogym_s0004 | train | 6 | 1.10 | 0.90 | 0.80 | 0.88 / 0.10 | 0.26 0.46 0.63 0.77 0.90 1.21 1.38 |
| sonogym_s0006 | test | — | — | — | — | — | not measurable: liver touches the CT field of view within 10 mm of a site |
| sonogym_s0010 | train | 6 | 1.33 | 1.04 | 0.87 | 1.03 / 0.05 | 0.34 0.58 0.76 0.91 1.04 1.34 1.49 |
| sonogym_s0012 | train | 6 | 1.37 | 1.11 | 1.01 | 1.09 / 0.13 | 0.35 0.60 0.80 0.97 1.11 1.46 1.65 |
| sonogym_s0014 | train | 6 | 0.71 | 0.59 | 0.50 | 0.58 / 0.08 | 0.18 0.32 0.43 0.51 0.59 0.76 0.84 |
| sonogym_s0015 | train | 6 | 0.34 | 0.30 | 0.28 | 0.30 / 0.05 | 0.08 0.15 0.21 0.26 0.30 0.42 0.49 |
| sonogym_s0024 | train | 6 | 1.15 | 0.93 | 0.79 | 0.93 / 0.08 | 0.31 0.51 0.68 0.82 0.93 1.20 1.35 |
| sonogym_s0028 | train | 6 | 0.48 | 0.40 | 0.36 | 0.39 / 0.07 | 0.15 0.24 0.30 0.35 0.40 0.51 0.57 |
| sonogym_s0029 | test | 6 | 1.86 | 1.53 | 1.32 | 1.51 / 0.11 | 0.49 0.85 1.13 1.35 1.53 1.93 2.13 |
| sonogym_s0030 | test | 6 | 0.74 | 0.61 | 0.53 | 0.61 / 0.03 | 0.20 0.34 0.45 0.54 0.61 0.79 0.88 |
| sonogym_s0034 | test | — | — | — | — | — | not measurable: liver touches the CT field of view within 10 mm of a site |
| sonogym_s0038 | train | 6 | 1.58 | 1.30 | 1.13 | 1.28 / 0.21 | 0.45 0.74 0.96 1.15 1.30 1.65 1.83 |

Reference (not decisive): one axial pixel 0.377 mm; lumbar muscle force effect ≈ 1.0–1.3 mm over 1–5 N (DEC-056).
