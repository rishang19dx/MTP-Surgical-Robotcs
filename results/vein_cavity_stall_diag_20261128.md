# DEC-075 post-hoc diagnosis: where the patient sweeps stall (28 Nov 2026)

Script `ReconstructionPipeline/evaluation/vein_stall_diag.py`; 4 TRAINING veins already in the DEC-075 gate (test veins untouched); fat 4.886 kPa, h = 2 mm, cavity lumen. Post-hoc, not a pre-registered result.

| Vein | vein depth / diameter (mm) | stall indent (mm) / force (N) / AP ratio | min J (tissue) | min-J tet: dist. to face edge / skin (mm) | 20 lowest-J tets: fat / under face / ≤ 3 mm from edge / ≤ 3 mm from vein | wall pairs active | failing Newton step |
| :--- | :--- | :--- | :--- | :--- | :--- | ---: | :--- |
| ts2_s0000_left | 22 / 16 | 10.6 / 2.06 / 0.953 | 0.19 (fat) | 0.0 / 1.8 | 20 / 19 / 13 / 0 | 0 | rel. residual 4.6e-03, line search α 9.5e-07, rejections 'energy_up', full step inverts 0 tets |
| ts2_s0004_left | 38 / 15 | 8.9 / 1.12 / 0.972 | 0.36 (fat) | 0.0 / 1.5 | 20 / 20 / 15 / 0 | 0 | rel. residual 2.9e-04, line search α 9.5e-07, rejections 'energy_up', full step inverts 0 tets |
| ts2_s0028_right | 37 / 14 | 5.1 / 0.27 / 0.995 | 0.57 (fat) | 0.0 / 1.5 | 20 / 14 / 17 / 0 | 0 | rel. residual 2.6e-04, line search α 9.5e-07, rejections 'energy_up', full step inverts 0 tets |
| ts2_s0095_left | 29 / 18 | 11.4 / 2.69 / 0.886 | 0.32 (fat) | 1.0 / 1.1 | 20 / 20 / 15 / 0 | 0 | rel. residual 7.7e-05, line search α 1.5e-05, rejections 'energy_up', full step inverts 0 tets |

Reading: in every vein the solve fails at the lateral edge of the rigid flat probe face (±22.5 mm), in superficial fat 1–2 mm under the skin, with no inverted tet; the line search rejects every step because the energy does not decrease (not because tets invert), at a small residual. The vein walls never touch, and the lowest-J tets are 35–51 mm from the cavity.
