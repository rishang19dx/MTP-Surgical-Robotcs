# Muscle ground truth vs the 75 mm image window (20261112_142624)

commit `f97a27b` · forces [1.0, 2.0, 3.0, 4.0, 5.0] N · a (pose, force) is truncated when ≥ 50% of the central columns have muscle in the last 3 rows

| Patient | Split | Muscle depth at 1 N (mm) | Truncated (pose, force) | Poses clean at all forces | GT compression 1→5 N, all poses (mm) | … clean poses only (mm) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| sonogym_s0015 | train | 9.6–42.0 | 0% | 6/6 | +1.32 | +1.32 |
| sonogym_s0010 | train | 16.4–50.3 | 0% | 6/6 | +0.75 | +0.75 |
| sonogym_s0012 | train | 13.8–50.1 | 0% | 6/6 | +1.13 | +1.13 |
| sonogym_s0038 | train | 26.9–57.3 | 0% | 6/6 | +0.75 | +0.75 |
| sonogym_s0024 | train | 30.5–74.4 | 43% | 3/6 | -0.38 | +0.75 |
| sonogym_s0004 | train | 22.4–52.6 | 0% | 6/6 | +0.75 | +0.75 |
| sonogym_s0028 | train | 27.3–59.0 | 0% | 6/6 | +0.57 | +0.57 |
| sonogym_s0014 | train | 40.3–75.0 | 60% | 2/6 | -2.83 | +0.19 |
| sonogym_s0029 | test | 14.9–51.8 | 0% | 6/6 | +1.13 | +1.13 |
| sonogym_s0006 | test | 15.6–56.3 | 0% | 6/6 | +1.13 | +1.13 |
| sonogym_s0034 | test | 33.2–68.6 | 0% | 6/6 | +0.57 | +0.57 |
| sonogym_s0030 | test | 46.5–75.0 | 87% | 0/6 | -2.83 | — |
