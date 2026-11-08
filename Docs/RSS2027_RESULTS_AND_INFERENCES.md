# RSS 2027 work: results so far and what they support

Status on 7 November 2026, branch `audit-fixes`. This page collects every result produced for the RSS 2027 direction
(DEC-041 to DEC-044), names its source file, and says what can and cannot be inferred from it. It is the input for
the Stage 1 extended abstract (due 4 December 2026). Every number below comes from `results/`, a test, or a
DECISIONS entry that names its source; nothing is copied from scratch runs.

All runs are on a laptop RTX 2050 (4 GB). The workstation GPU has not been reachable yet, so none of the numbers
are from it.

---

## 1. What exists

`mtp_gym` is a GPU-batched Gymnasium `VectorEnv` for robotic lumbar ultrasound. It has:

- patient-specific soft tissue from the precomputed linear FEM;
- force from the same tissue model, under admittance control;
- B-mode images rendered from the deformed CT.

There are 13 patients: the 12 TotalSegmentator subjects SonoGym uses, plus the original sample. The split is 9 train
and 4 held out, stratified by soft-tissue thickness (`data/splits.json`).

| Environment | Purpose | Action |
| :--- | :--- | :--- |
| `MTP/FreeScan-v0` | Simulation cost only, no reward | 4-DoF pose |
| `MTP/Recon-v0` | Bone-surface reconstruction (SonoGym task 2 analogue) | 4-DoF pose |
| `MTP/Nav-v0` | Navigation to the transverse L4 plane (SonoGym task 1 analogue) | 3-DoF pose |
| `MTP/ForceScan-v0` | Reconstruction with commanded contact force (new; only meaningful with soft tissue) | 4-DoF pose + force 1–15 N |

The batched path is checked against the existing single environment by tests:

- FEM force, torque, contact stiffness, field and controller queries: `test_fem_batched.py`.
- Deformed slices: `test_batched_render.py`.
- Force, indentation and coupling after reset and step: `test_gym_api.py`.
- Batched B-mode bone segmentation, pixel for pixel: `test_vec_reconstruction.py`.

The suite has 139 tests plus 1 strict expected failure (the H2 result, section 6).

---

## 2. Simulation cost

Source: `results/throughput_20261105_235332.md`. FreeScan task, sample patient, legacy B-mode (before DEC-042). The
file names commit `8f89f63`, which is `bbea34f` after the author rewrite (DECISIONS erratum).

| Mode | 1 env | 8 envs | 32 envs | 128 envs |
| :--- | ---: | ---: | ---: | ---: |
| `aware` (FEM-deformed image), env-steps/s | 44.8 | 280.4 | 529.5 | 732.4 |
| `rigid` (undeformed image, FEM force control kept), env-steps/s | 51.8 | 338.8 | 664.8 | 944.1 |

**Inference.** Deforming the image costs about 22 % of throughput at 128 environments (732 vs 944). Batching gives
about 16× over one environment. Both modes include the FEM force controller, so this is the price of the image
warp and field, not of having soft tissue at all.

**Not supported yet.** Any comparison with SonoGym's speed (different GPU, different task); the cost on the
workstation GPU; the cost of the depth B-mode (not re-measured).

---

## 3. Patients and tissue

Sources: `results/patients_fem_20261105_225601.json` and `data/patients/*/meta.json`.

**Import.** The SonoGym CTs are true RAS. They are rotated into frame P (LPS), anchored at L4, and the CT table is
removed (DEC-041 T4). Midline soft tissue at L4 ranges from 15 to 78 mm across patients.

**FEM build.** For every patient the Maxwell–Betti asymmetry is ≤ 7.4e-6 and the PCG residual < 1e-5. The build
takes 28–72 s per patient.

**Shared moduli (user decision).** Per-patient calibration had spread the modulus scale from 0.62 to 2.86,
equalising force across anatomy. All patients now share E_fat / E_muscle = 10.96 / 21.92 kPa (the sample patient's
calibration). With the same moduli, at 6 mm past first contact on the midline, the thin back s0015 gives 9.43 N
and the thick back s0014 gives 3.48 N (`test_multi_patient.py`).

**Inference.** Force now depends on anatomy, which is the precondition for any patient-dependent force behaviour.

**Not supported yet.** That the absolute stiffness is right. It rests on one calibration to a literature force
curve and has not been compared with a nonlinear FEM or with published moduli (task T14).

---

## 4. Does probe force change the image? (T9)

Source: `results/compression_sweep_20261105_235841.md`. 13 patients, 9 poses each, 1–15 N, legacy B-mode without
the coupling loss, measured from the image only.

- **Echo shift.** In the 12 patients with paired scanlines, the bone echo moves 6.4–9.6 mm toward the probe
  between 1 and 15 N (median per patient). It moves at 100 % of the paired poses.
- **Visibility.** In thick backs, force decides whether bone is visible at all. s0014 shows bone at 0 of 9 poses
  at 1 N and 8 of 9 at 15 N; s0024 goes from 3 to 9 of 9.
- **The gate passes for all 13.**

**Inference: corrected on 8 Nov (DEC-051).** The original inference said this shift was something a rigid simulator
cannot produce. That is wrong. `results/kinematic_vs_deformation_20261108_020955.md` renders the same indented face
poses through rigid tissue and gets the same shift: the deformation part is 0.00 mm in 11 of 12 patients. The probe
presses 8–14 mm deeper and the bone is fixed, so the shift is kinematic. The H3 inference ("reconstruction that
ignores compression will be biased") is also withdrawn: that bias only appears if the indentation is ignored, which
is true by construction. What remains is that force (through indentation) decides whether bone in thick backs
enters the field of view, and the FEM supplies the anatomy-dependent force-to-indentation relation.

**Statistic history.** Two statistics were replaced during development because they mixed structures or poses
(DEC-041 progress T9); neither produced a reported number.

---

## 5. B-mode realism: legacy vs depth physics (DEC-042, DEC-043)

Source: `results/bmode_physics_compare_20261107_190506.md`, commit `7818454`. Benchmark raster, 4 environments per
patient.

**What changed (literature-anchored, opt-in, default in the gym):**

- per-tissue attenuation, fat 0.48 and muscle 1.09 dB/cm/MHz;
- receiver noise before the TGC, anchored at 500-wavelength penetration;
- a fixed muscle-appropriate TGC of 10.9 dB/cm.

| | Legacy | Depth physics |
| :--- | ---: | ---: |
| Mean Cov@3, 13 patients | 0.659 | 0.376 |
| Backs with ≤ 33 mm paraspinal fat | 0.81–0.92 | 0.44–0.89 |
| Backs with ≥ 42 mm paraspinal fat | 0.08–0.97 | 0.00–0.07 |

Across the 12 SonoGym subjects, coverage under the depth physics correlates with paraspinal fat at Spearman
ρ = −0.79 (p = 0.002, DEC-043 outcome; computed from the results JSON and the meta files).

**Inference.** With realistic depth physics, image quality, and therefore reconstruction, falls with subcutaneous
fat. That matches clinical experience that ultrasound quality declines with BMI. The legacy model was "too easy"
partly because depth cost nothing.

**Caveat.** Part of the collapse is perception, not physics. The over-gained fat lowers the image-only confidence
below the segmenter's gate (inspected on images; DEC-043).

---

## 6. Hypothesis H2 (best force rises with tissue thickness): pre-registered, failed

Sources:

- DEC-042: the predictions were committed in `3280829` before the code.
- `test_vec_force_scan.py`.
- DEC-044 outcome.

Protocol: thin s0015 vs thick s0014, 9 held poses each, `aware` mode, reward = 2 × scanline bone visibility
− 0.05 × F.

| Prediction | Result |
| :--- | :--- |
| P1: best force higher for the thick back | Failed: 2 N for both (4 N for both with the legacy model) |
| P2: thick-back visibility gain > 0.10 from 4 to 12 N | Failed: 0.128 → 0.149 (with the DEC-044 perception: 0.074 → 0.112) |
| P3: rigid control unchanged by force (< 0.03) | Held: identical |

**Inference.**

- In this simulator, beyond the acoustic-coupling knee (1 → 2 N), pressing harder does not buy image quality,
  even though it moves bone about 8 mm closer (section 4). The gain from shorter depth is small compared with the
  force cost in the reward.
- H2 is reframed, as pre-registered: policies should learn the minimal force for acoustic coupling and field of
  view.
- This is a reportable negative result. It also says the force-scan task's optimum is close to "just enough
  force", which a learned policy can be tested against.

**Not supported.** Any claim that force helps imaging in obese patients. Clinical sources report that sonographers
press harder (e.g. AuntMinnie on probes for obese patients), but we found no study showing that it improves image
quality.

---

## 7. Perception re-tuning on the training patients (DEC-044)

Source: `results/perception_tuning_20261107_190845.md`, commit `9a5b1d0`. The grid and selection rule were
committed in `f487492` before the run.

| | Cov@3 defaults → selected | Prec@2 |
| :--- | :--- | :--- |
| Train (9) | 0.403 → 0.678 | 0.998 → 0.981 |
| Held out (4) | 0.318 → 0.428 | 1.000 → 0.979 |

Held-out patients: s0029 0.68 → 0.83, s0006 0.53 → 0.80, s0034 (52 mm fat) 0.00 → 0.01, s0030 (58 mm fat)
0.06 → 0.07.

**Inference.**

- Thresholds matched to the image model recover moderately fatty backs on unseen patients at a small precision
  cost.
- The fattest backs are not a threshold problem. They are a hard case for any fixed preset and stay open for
  learned policies or for gain control.
- Precision is now 0.98, closer to, but still above, the 0.8–0.9 reported for real spine ultrasound (README
  Limitations).

**Caveat.** Three of the four selected values sit on the edge of the grid; the grid was not extended.

---

## 8. Navigation and force-scan tasks: logic checks only

- **Navigation** (`test_vec_navigation.py`, sample patient, 64 episodes). A privileged oracle succeeds in 100 %
  of episodes (3.9 steps on average); random actions in 0 %. This shows the task logic, not that the plane can be
  found from images.
- **Force scan** (`test_vec_force_scan.py`). The commanded force is tracked exactly by the controller (sensed =
  commanded in the T7 measurement, DEC-041 progress).

---

## 9. What the evidence supports for the Stage 1 abstract

**Claims supported now:**

1. A GPU-batched robotic-ultrasound gym with patient-specific FEM soft tissue, force and deformed B-mode, on the
   same 12 patients as SonoGym, checked component by component against a single-environment reference.
2. ~~Compression moves anatomy in the image by 6–10 mm; a rigid simulator cannot represent this (T9).~~ Withdrawn
   (DEC-051): the shift is probe kinematics. Supported instead: an anatomy-dependent force-to-indentation relation
   (thin vs thick back 9.4 vs 3.5 N at the same indentation), with FEM fidelity within 5 % at 2 N (DEC-047).
3. With literature depth physics, reconstruction difficulty grows with subcutaneous fat (ρ = −0.79), giving a
   patient-generalisation benchmark with a realistic hard tail.
4. A pre-registered negative result: in simulation, extra force beyond coupling does not buy image quality (H2
   reframed).

**Since added** (DEC-045 to DEC-047, see DECISIONS.md):

- **FEM vs Neo-Hookean reference** (`results/fem_fidelity_20261107_234638.md`): within 5 % at 2 N and up to about
  30 % at 5–15 N. Claim fidelity only at low force.
- **Navigation** (`results/vec_eval_nav_20261107_223741.md`): after the hidden-offset fix, PPO succeeds in 100 % on
  its 2 training patients (blind baseline 6 %) and 7.8 % on held-out patients. It learns from the image but does not
  generalise from 2 patients.
- **Reconstruction reward.** The area reward rewarded wandering; it is now coverage gain (DEC-046), and the earlier
  reconstruction and force checkpoints are void.

**Still to produce** (tasks/todo.md):
- Learned baselines on the three tasks (T10).
- H1: rigid-trained vs deformable-trained policies.
- H3: reconstruction bias against force, with and without correction.
- Throughput with the depth B-mode, and on the workstation GPU.

---

## 10. Overnight run, 8 November (DEC-049 to DEC-051)

All runs on the laptop RTX 2050; results committed in `699c2ba`.

**Thread C: kinematic vs deformation** (`results/kinematic_vs_deformation_20261108_020955.md`). The bone-echo shift
is identical with and without the FEM warp at the same face poses: deformation part 0.00 mm in 11 of 12 patients.
Claim 2 above is withdrawn.

**Thread B: paraspinal muscle thickness** (`results/muscle_compression_20261108_021028.md`). The muscle does compress:
0.75 mm median from 1 to 5 N (gate ii passes). But it cannot be measured from the simulated image: detection 0.38,
median error 7.3 mm (gate i fails), because the CT-derived B-mode does not render the fascia. As pre-registered,
this thread moves to Stage 2. Three thick backs show negative compression, which is not yet explained.

**Thread A: generalisation pilot** (`results/vec_eval_nav_20261108_075931.md`, 64 held-out episodes per policy).

| Training | Held-out success |
| :--- | ---: |
| 1 patient | 0.0 % |
| 2 patients (seeds 0, 1) | 7.8 %, 4.7 % |
| 3 patients | 4.7 % |
| 2 patients + FEM augmentation (seeds 0, 1) | 14.1 %, 4.7 % |
| 2 patients + warp control | 3.1 % |
| 2 patients + imaging augmentation | 1.6 % |

Against the pre-registrations: H1 is not supported, and H2 and H3 show no detectable difference within the ±12-point
interval.

**Inference.** The generalisation gap is robust: 100 % on training patients vs ≤ 14 % held out, against a blind
baseline of 3 %. Whether more patients or physics augmentation close it cannot be decided at this scale. That test
is the server study.

**Reconstruction retrained with the coverage reward** (`results/vec_eval_recon_20261108_020410.md`). PPO does not beat
random (0.32–0.34 vs 0.34; raster 0.43). The old H1 (trained rigid vs deformable) shows no difference.

### Where the evidence stands for Stage 1 (8 November)

**Supported:**

- the gym and its component checks;
- 13 patients with a held-out split;
- an anatomy-dependent force-to-indentation relation, with FEM fidelity within 5 % at ≤ 2 N;
- the fat-dependent imaging difficulty (with the confound noted);
- the benchmark-validity checks (blind baseline, reward audit);
- the pre-registered negatives (H2 force; the pilot H1–H3);
- the generalisation gap.

**Not supported:**

- any task where soft-tissue deformation changes policy outcomes;
- any learned policy that generalises.

**Next test:** the server-scale generalisation study, under its own pre-registration.

