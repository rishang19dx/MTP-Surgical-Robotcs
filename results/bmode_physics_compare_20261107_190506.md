# Legacy vs depth B-mode physics on the benchmark raster (20261107_190506)

commit `7818454` · NVIDIA GeForce RTX 2050 · 4 envs per patient · 72-frame raster · aware mode · shared moduli

Depth physics: DEC-042 (fat 0.48 / muscle 1.09 dB/cm/MHz, receiver noise before TGC) + DEC-043 (fixed TGC 10.9 dB/cm, cap 81.75 dB).

| Patient | Cov@3 legacy | Cov@3 depth | Prec@2 legacy | Prec@2 depth | Points/frame legacy | Points/frame depth |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| sample_totalsegmentator_patient | 0.895 | 0.873 | 0.999 | 0.999 | 99.2 | 81.0 |
| sonogym_s0015 | 0.917 | 0.890 | 1.000 | 1.000 | 92.7 | 77.5 |
| sonogym_s0010 | 0.910 | 0.873 | 0.999 | 1.000 | 85.4 | 55.0 |
| sonogym_s0012 | 0.811 | 0.436 | 0.939 | 0.977 | 65.9 | 13.3 |
| sonogym_s0038 | 0.767 | 0.067 | 0.994 | 1.000 | 86.9 | 8.5 |
| sonogym_s0024 | 0.163 | 0.000 | 1.000 | nan | 27.2 | 0.0 |
| sonogym_s0004 | 0.799 | 0.421 | 0.994 | 0.998 | 80.9 | 13.2 |
| sonogym_s0028 | 0.966 | 0.006 | 1.000 | 1.000 | 83.7 | 2.6 |
| sonogym_s0014 | 0.241 | 0.036 | 1.000 | 1.000 | 24.6 | 1.7 |
| sonogym_s0029 | 0.839 | 0.689 | 1.000 | 1.000 | 85.9 | 44.3 |
| sonogym_s0006 | 0.807 | 0.532 | 0.999 | 1.000 | 71.6 | 32.4 |
| sonogym_s0034 | 0.375 | 0.000 | 0.993 | nan | 15.5 | 0.0 |
| sonogym_s0030 | 0.083 | 0.059 | 1.000 | 1.000 | 11.7 | 2.5 |
| **mean legacy** | 0.659 | | 0.994 | | | |
| **mean depth** | 0.376 | | 0.998 | | | |
