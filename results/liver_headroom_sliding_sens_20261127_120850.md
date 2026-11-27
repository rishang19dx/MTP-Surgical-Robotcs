# Liver target-shift headroom — SLIDING liver (DEC-060) — penalty sensitivity 30.0 (20261127_120850)

pre-registration DEC-060 · commit `3255749-dirty` · deformation-only shift |u| of liver targets (deformable − kinematic) · gate: median at 5 N ≥ 2.0 mm in ≥ 2/3 of measurable patients at every liver modulus

## Gate

| Liver E (kPa) | Measurable patients | ≥ 2 mm | Pass |
| ---: | ---: | ---: | :---: |
| 4.1 | 3 | 0 | **FAIL** |

**Overall: FAIL**

## Per patient (median over sites × targets)

| Patient | Split | Sites | |u| 5 N, E 4.1 | in-plane / out-of-plane 5 N, E 4.1 | |u| by force 1–10 N, E 4.1 |
| :--- | :--- | ---: | ---: | ---: | :--- |
| sonogym_s0004 | train | 6 | 0.82 | 0.81 / 0.08 | 0.24 0.42 0.58 0.70 0.82 1.13 1.31 |
| sonogym_s0010 | train | 6 | 0.96 | 0.95 / 0.09 | 0.26 0.49 0.67 0.82 0.96 1.30 1.49 |
| sonogym_s0015 | train | 6 | 0.31 | 0.31 / 0.05 | 0.07 0.13 0.20 0.26 0.31 0.46 0.54 |

Reference (not decisive): one axial pixel 0.377 mm; lumbar muscle force effect ≈ 1.0–1.3 mm over 1–5 N (DEC-056).
