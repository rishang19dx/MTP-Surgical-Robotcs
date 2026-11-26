# Liver clamped-wall margin sensitivity (20261126_205228)

pre-registration DEC-063 · commit `3255749` · tied liver 4.1 kPa · fitted fat 3.38 kPa · one site per subject · outcome: median |u| of liver targets at 5 N

**Verdict: INCONCLUSIVE** (artefact if R = |u|(90)/|u|(30) ≥ 1.5 in ≥ 2/3; null stands if R < 1.2 in ≥ 2/3)

| Subject | Site (x, z) | |u| m=30 (mm) | |u| m=45 (mm) | |u| m=60 (mm) | |u| m=90 (mm) | R | |u|(last)/|u|(prev) | ≥ 2 mm at any m |
| :--- | :--- |  ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| sonogym_s0004 | (0, -10) | 0.97 | 1.18 | 1.25 | 1.40 | 1.44 | 1.12 | no |
| sonogym_s0010 | (0, -10) | 1.23 | 1.48 | 1.63 | 1.77 | 1.44 | 1.09 | no |
| sonogym_s0012 | — | — | — | — | — | — | — | not eligible: largest box reaches the CT field-of-view edge (17385 body voxels) |
| sonogym_s0038 | (0, -10) | 1.25 | 1.49 | 1.66 | 1.89 | 1.52 | 1.14 | no |

Box size (x × z mm) and build time per margin:

- sonogym_s0004: m=30: [105, 70], 9 s; m=45: [135, 100], 16 s; m=60: [165, 130], 82 s; m=90: [225, 190], 337 s
- sonogym_s0010: m=30: [105, 70], 8 s; m=45: [135, 100], 30 s; m=60: [165, 130], 85 s; m=90: [225, 190], 309 s
- sonogym_s0038: m=30: [105, 70], 8 s; m=45: [135, 100], 32 s; m=60: [165, 130], 91 s; m=90: [225, 190], 371 s
