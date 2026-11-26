# DEC-067: muscle thickness under force at scale (muscle_thickness_dec067_s11.json)

commit `d15f5fa-dirty` · pre-registration DEC-067 · split `data/splits_v2.json` · stiffness draw none (shared moduli) · forces [1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0] N

Test patients scored 127, measurable 112 (detection ≤ 2 N ≥ 0.5). Bootstrap 10000 resamples of patients, 95 % percentile intervals.

| Criterion | Value | Verdict |
| :--- | :--- | :---: |
| G: median detection ≤ 2 N ≥ 0.5 and median abs error < 3 mm | 1.00, 1.51 mm | **pass** |
| H4a: deformable image slope median < 0 (CI upper < 0), ≥ 75 % negative | -0.264 [-0.283, -0.236] mm/N, 95% negative (GT slope median -0.264) | **pass** |
| H4b: median abs rigid slope, CI upper ≤ 0.15 mm/N | 0.042 (upper 0.066) | **pass** |
| H4c: median tracking gap, CI upper ≤ 0.1 mm/N | 0.113 (upper 0.132) | **FAIL** |
| **H4** | | **FAIL** |
| **H5**: force-aware beats intercept-only by ≥ 0.377 mm | 2.49 vs 2.65 mm (uncorrected 1.13); k = 0.426 mm/N | **FAIL** |
| **S1.3 P1 gate**: median (headroom − noise floor) ≥ 1.0 mm and ≥ 50 % of patients | 0.00 [-0.19, 0.14] mm, 8% ≥ 1 mm (n = 112) | **OUT** |

## Reported, not decisive

- DEC-056 original rules on these test patients: G True, H4a (every patient) False, H4b (every patient) False, H4c False.
- Without flagged patients: measurable 110, H4 False, P1 gate OUT.
- New test patients only (the 4 old test patients were analysed in DEC-056): measurable 110, H4 False, P1 gate OUT.
- Train (development set): measurable 209, H4 False.
- Spearman ρ of the deformable image slope vs midline soft tissue +0.34, vs paraspinal fat +0.43 (n = 112).
