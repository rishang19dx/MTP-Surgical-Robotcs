# Perception re-tuning for the depth B-mode (20261107_190845)

commit `9a5b1d0` · NVIDIA GeForce RTX 2050 · protocol DEC-044 · 54 configurations scored on 9 train patients; test patients scored only for the defaults and the selected configuration

Selected: `{"shadow_threshold": 0.9, "attenuation_decay_rate": 0.4, "intensity_threshold": 0.75, "min_confidence": 0.05}`

| Split | Configuration | Mean Cov@3 | Mean Prec@2 |
| :--- | :--- | ---: | ---: |
| train | defaults | 0.403 | 0.998 |
| train | selected | 0.678 | 0.981 |
| **test** | defaults | 0.318 | 1.000 |
| **test** | selected | 0.428 | 0.979 |

| Test patient | Cov@3 defaults | Cov@3 selected | Prec@2 defaults | Prec@2 selected |
| :--- | ---: | ---: | ---: | ---: |
| sonogym_s0029 | 0.684 | 0.833 | 1.000 | 1.000 |
| sonogym_s0006 | 0.528 | 0.796 | 1.000 | 0.999 |
| sonogym_s0034 | 0.000 | 0.010 | nan | 0.917 |
| sonogym_s0030 | 0.059 | 0.072 | 1.000 | 1.000 |
