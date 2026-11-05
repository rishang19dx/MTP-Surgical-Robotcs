# Compression sweep (20261105_235841)

commit `50f4919` · NVIDIA GeForce RTX 2050 · forces [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0] N · shared moduli · 3x3 untilted poses over the L4 window · fixed speckle seed

Echo shift: per pose, median over scanlines with bone detected at both the lowest and the highest force; then the median over poses.

| Patient | Thickness (mm) | Bone visible, lowest F | highest F | Paired poses | Median echo shift (mm) | Paired poses shift > 1 px | Indent past contact at max F (mm) | B-mode MAD vs lowest F | Gate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| sample_totalsegmentator_patient | 36.0 | 1.00 | 1.00 | 9 | 7.54 | 1.00 | 9.6 | 0.086 | pass |
| sonogym_s0015 | 46.0 | 0.78 | 1.00 | 7 | 6.41 | 1.00 | 8.1 | 0.082 | pass |
| sonogym_s0010 | 48.0 | 0.78 | 1.00 | 6 | 9.61 | 1.00 | 13.7 | 0.103 | pass |
| sonogym_s0012 | 56.0 | 1.00 | 1.00 | 9 | 7.54 | 1.00 | 9.5 | 0.077 | pass |
| sonogym_s0038 | 55.0 | 1.00 | 1.00 | 9 | 9.05 | 1.00 | 12.4 | 0.099 | pass |
| sonogym_s0024 | 77.5 | 0.33 | 1.00 | 3 | 8.67 | 1.00 | 11.3 | 0.075 | pass |
| sonogym_s0004 | 58.0 | 1.00 | 1.00 | 9 | 8.67 | 1.00 | 11.2 | 0.083 | pass |
| sonogym_s0028 | 65.0 | 0.67 | 1.00 | 6 | 8.67 | 1.00 | 11.5 | 0.085 | pass |
| sonogym_s0014 | 76.0 | 0.00 | 0.89 | 0 | nan | nan | 12.9 | 0.081 | pass |
| sonogym_s0029 | 52.0 | 0.89 | 1.00 | 8 | 7.91 | 1.00 | 11.0 | 0.087 | pass |
| sonogym_s0006 | 55.0 | 0.44 | 1.00 | 4 | 8.48 | 1.00 | 12.9 | 0.097 | pass |
| sonogym_s0034 | 73.0 | 0.44 | 1.00 | 3 | 9.05 | 1.00 | 12.2 | 0.070 | pass |
| sonogym_s0030 | 85.0 | 0.67 | 1.00 | 6 | 8.29 | 1.00 | 10.5 | 0.057 | pass |
