# Task list: RSS 2027 soft-tissue US gym (see `tasks/plan.md`, DEC-041)

Test command: `./.venv/bin/python -m pytest ReconstructionPipeline/tests/ -v`.
Every task also obeys the AGENTS.md invariants:
- numbers only from `results/`;
- the perception path is image-only;
- coordinates go through `coordinates.py` helpers;
- DECISIONS.md and FLOW.md are updated.

## Schedule (set 2026-11-05; Stage 1 due 4 Dec 2026 AoE)

| Task | Planned | Status |
| :--- | :--- | :--- |
| T1 batched FEM | 5 Nov | done 5 Nov |
| T2 batched deformed B-mode | 6 Nov | done 5 Nov |
| T3 VectorEnv core | 7–9 Nov | done 5 Nov |
| T8 throughput into `results/` | 9 Nov | done 5 Nov (`results/throughput_20261105_235332.md`, RTX 2050, clean commit; re-run on the server GPU) |
| T4 multi-patient assets (SonoGym cases + varied BMI) | 6–12 Nov | done 5 Nov: 12 SonoGym subjects + sample, split in `data/splits.json`, shared moduli; own varied-BMI picks still to add |
| T5 reconstruction task on VectorEnv | 10–12 Nov | done 5 Nov (`MTP/Recon-v0`) |
| **Checkpoint A** | 13 Nov | |
| T6 navigation task | 12–15 Nov | done 7 Nov (`MTP/Nav-v0`; oracle 100 %, random 0 %) |
| T9 compression-changes-image gate | 13–14 Nov | done 5 Nov: passes on all 13 patients (`compression_sweep.py`) |
| T14 fidelity vs nonlinear FEM | 14–19 Nov | |
| **Checkpoint B** | 20 Nov | |
| T7 force-scan task | 19–21 Nov | implemented 7 Nov (`MTP/ForceScan-v0`); pre-registered H2 prediction failed under DEC-042 depth physics; H2 reframed |
| T10 PPO baselines on the 3 tasks | 20–24 Nov | |
| T11/T12 preliminary H1/H2 runs (server GPU) | 22–26 Nov | |
| T13 H3 compression bias | 24–26 Nov | |
| T15 Stage 1 abstract | 26 Nov–3 Dec | |
| **Checkpoint C / submit** | 2–4 Dec | |

---

## Phase 1: Stage 1 sprint (due 4 Dec 2026)

### T1: Batched FEM contact and field (M) — DONE 2026-11-05 (`simulation/fem_batched.py`)
**Description:** Extend `fem_tissue` to take B probe poses for one patient. Share the compliance and `G_lat` across envs.
The contact solve runs on the GPU with a fixed iteration count, and the field is evaluated as one batched matmul.
**Acceptance:**
- [x] For B = 1 the result matches the current single-env solver (force and field, max abs error < 1e-4 relative). In
  fp64 the force matches to 1e-6 relative over 24 mixed queries.
- [x] Batches of B = 1…128 run with no Python loop over envs.
- [x] Peak VRAM and time per step are logged (test output, `-s`).
**Verify:**
- [x] New `test_fem_batched.py` passes (8 tests).
- [x] The existing `test_fem_tissue.py` stays green (full suite: 111 passed).

**Depends on:** none.
**Files:** `ReconstructionPipeline/simulation/fem_tissue.py`, `tests/test_fem_batched.py`.

### T2: Batched deformed B-mode (S) — DONE 2026-11-05 (`test_batched_render.py`)
**Description:** Batch the slice extraction (`grid_sample` backward warp through the FEM field) and the model-based B-mode
over B envs.
**Acceptance:**
- [ ] For B = 1 the output is bit-identical (or within fp tolerance) to the current path.
- [ ] Time per frame at B = 128 is measured.
- [ ] No privileged outputs (`energy`, `bone_mask`) reach the observations.
**Verify:**
- [ ] `test_ultrasound_sim.py` and `test_slice_extractor.py` stay green.
- [ ] Add a batched test.

**Depends on:** T1.
**Files:** `simulation/slice_extractor.py`, `simulation/ultrasound_model_based.py`, a test.

### T3: Gymnasium VectorEnv core (M) — DONE 2026-11-05 (`mtp_gym/`, `test_gym_api.py`)
**Description:** Add a new `mtp_gym` package: a torch-native `VectorEnv` (reset, step, auto-reset), quasi-static probe
kinematics with no PyBullet, and `gymnasium.register` IDs.
**Acceptance:**
- [ ] `gymnasium.make_vec("MTP/Recon-v0", num_envs=N)` works.
- [ ] Spaces are declared.
- [ ] `gymnasium.utils.env_checker` passes for N = 1.
- [ ] A parity test shows the same image and force as the PyBullet path for a fixed pose.
- [ ] The test that the streaming side never imports pybullet still passes.

**Verify:** new `test_gym_api.py`; the full suite.
**Depends on:** T1, T2.
**Files:** `mtp_gym/__init__.py`, `mtp_gym/vector_env.py`, `mtp_gym/kinematics.py`, a test.

### T4: Multi-patient assets (M) — DONE 2026-11-05 for the SonoGym subjects (`import_sonogym_subjects.py`, `build_patient_fem.py`, `test_multi_patient.py`)
**Description:** Generalise `prepare_patient_assets` beyond the hardcoded case-0004. Build at least 8 patients
(identify SonoGym's TotalSegmentator subjects), precompute and cache the FEM for each, and commit a fixed train/held-out
split file.
**Acceptance:**
- [ ] A loop over cases runs with no hardcoded paths.
- [ ] Each patient passes the FEM reciprocity and patch tests.
- [ ] `data/splits.json` is committed.
- [ ] The tissue thickness at the scan site for each patient is logged, for use in H2.

**Verify:** the tests; a log of precompute time and VRAM per patient.
**Depends on:** none (can run in parallel with T1–T3).
**Files:** `data/prepare_patient_assets.py`, `data/splits.json`.

### Checkpoint A (≈ 13 Nov)
- [ ] The suite is green.
- [ ] `make_vec` with N = 64 steps on one patient.
- [ ] Review with the user.

### T8: Throughput benchmark into results/ (S) — DONE 2026-11-05 (`results/throughput_20261105_223122.md`, RTX 2050)
**Description:** Add a script that measures steps/s and VRAM against N ∈ {1, 8, 32, 128} for three tissue modes: rigid,
Kelvinlet and FEM. It writes `results/throughput_<date>.json` and `.md`. It replaces the uncitable measurement in
`.agents/`.
**Acceptance:**
- [ ] The JSON includes the GPU name, the commit and the settings.
- [ ] The table is reproducible from one command.

**Depends on:** T3.
**Files:** `ReconstructionPipeline/evaluation/throughput_benchmark.py`.

### T5: Reconstruction task on the VectorEnv (M) — DONE 2026-11-05 (`mtp_gym/tasks/reconstruction.py`, `test_vec_reconstruction.py`)
**Description:** Port `env_reconstruction` semantics: 4-DoF action, submodular coverage reward, and a voxel or point-cloud
observation built from B-mode segmentation (not ground truth, unlike SonoGym). Run it per patient, with the split.
**Acceptance:**
- [ ] The metrics (Cov@3, Prec@2 vis, Chamfer) match `benchmark_runner` for the raster policy on case-0004 within
  tolerance.
- [ ] A rigid / FEM toggle flag exists.

**Depends on:** T3, T4.
**Files:** `mtp_gym/tasks/reconstruction.py`, a test.

### T6: Navigation-to-plane task (M) — DONE 2026-11-07 (`mtp_gym/tasks/navigation.py`, `test_vec_navigation.py`)
**Description:** SonoGym-style navigation: the observation is the B-mode image, the action is 3-DoF tangential, the
reward is the decrease in pose distance to a target standard plane, and success is < 3 mm and < 5°. Targets are defined
per patient from TotalSegmentator vertebra labels. Labels are used only for the reward and goal, never for observations.
**Acceptance:**
- [ ] A scripted oracle reaches success above 95 %.
- [ ] A random policy stays below 10 %.
- [ ] A rigid / FEM toggle flag exists.

**Depends on:** T3, T4.
**Files:** `mtp_gym/tasks/navigation.py`, a test.

### T9: Gate: does compression change the simulated image? (S) — DONE 2026-11-05 (passes on all 13 patients)
**Description:** Before investing in the force task, sweep the force from 1 to 15 N at fixed poses on all patients and
measure image change. Measures: echo depth of bone, image-only confidence, and SSIM to the 1 N image. This replaces the
DEC-040 single-patient check (22.2 → 16.6 mm). Write the result to `results/`.
**Acceptance:**
- [ ] There is a monotone, patient-dependent effect, **or** a documented negative result. If negative, H2 is redesigned
  before T7.

**Depends on:** T2, T4.
**Files:** `evaluation/compression_sweep.py`.

### T14: Fidelity check against a nonlinear FEM reference (M)
**Description:** On the same tet mesh and boundary conditions, run a Neo-Hookean reference offline (SOFA, FEBio or a
small PyTorch Newton solver) for probe indentation at 1–15 N. Compare force–indentation curves and the displacement
field (RMS and max mm) with the linear precomputed model. Compare the stiffness (N/mm) and moduli with published values,
with each citation verified: erector spinae and multifidus SWE, subcutaneous fat, and Remus 2024 abdominal indentation.
**Acceptance:**
- [ ] `results/fem_fidelity_<date>.json` and `.md` exist.
- [ ] The force range where linear error is below a stated threshold (e.g. 1 mm) is reported.

**Depends on:** none.
**Files:** `evaluation/fem_fidelity.py`, a test for the reference solver on a cube patch.

### Checkpoint B (≈ 20 Nov)
- [ ] T5, T6, T8, T9 and T14 have results in `results/`.
- [ ] Decide the H2 design from T9.
- [ ] Review with the user.

### T7: Force-regulated scan task (M)
**Description:** Add the force set-point to the action (bounded at 1–20 N). The reward is the image-only quality term
(confidence/visibility from the B-mode) plus a coverage term, minus a force penalty. Run it with per-patient
stiffness/thickness and optional stiffness randomisation. It only works with FEM; the rigid mode is the control.
**Acceptance:**
- [ ] The reward is computed from the B-mode only. A test proves no privileged inputs (QUIZME C1).
- [ ] A scripted "thickness-aware" heuristic beats a constant-force baseline in FEM mode.

**Depends on:** T3, T4, T9.
**Files:** `mtp_gym/tasks/force_scan.py`, a test.

### T10: Baselines wired to the VectorEnv (M)
**Description:** Wire PPO (SB3 or skrl) to all three tasks, plus raster/NBV heuristics. Add A2C and SAC as cheap extras.
The existing PPO/CFM/diffusion paths are adapted to read from the VectorEnv.
**Acceptance:**
- [ ] One command per (task, policy, seed) writes results JSON.
- [ ] PPO learns above random on each task (single seed is acceptable for Stage 1).

**Depends on:** T5, T6 (T7 for the force task).
**Files:** `planning/train_vec_ppo.py`, `evaluation/benchmark_runner.py`.

### T11 / T12: Preliminary H1 and H2 runs (S each)
- **H1:** train rigid vs FEM on the reconstruction and navigation tasks, then evaluate both in FEM.
- **H2:** train on the force task across patients, then correlate the learned force with thickness.

Single seed and training patients only for Stage 1. Label them clearly as preliminary.
**Depends on:** T10.

### T13: H3 compression bias in reconstruction (S)
**Description:** Reconstruct at fixed forces from 2 to 15 N with three correction modes: none, oracle field, and field
estimated from force plus compliance. Report Chamfer and registration error against force.
**Depends on:** T5.

### T15: Stage 1 extended abstract (M)
**Description:** 6 pages (5 + 1). Sections: hypotheses H1–H4; method (gym, FEM, B-mode, tasks); preliminary results
(T8, T9, T14, plus T11, T12 or T13); planned experiments as the Extension Charter draft. Double-blind, no links.
Update README, FLOW and DECISIONS. Every number cites a `results/` file.
**Depends on:** results from the tasks above.

### Checkpoint C (≈ 2 Dec): submission readiness
- [ ] Every number traces to `results/`.
- [ ] Positioning re-checked against arXiv (CRESSim-Neo, SonoGym follow-ups, NVIDIA I4H).
- [ ] The user has proofread it.
- [ ] Submitted by 4 Dec AoE.

---

## Phase 2: Stage 2 (to 16 Apr 2027), tracked coarsely

- [ ] T16 Release: LICENSE, hosted assets (HF), install docs, LeRobot-format demo export, checkpoints.
- [ ] T17 (optional) Isaac Lab / SonoGym interop wrapper around the torch FEM module.
- [ ] T18 Full baselines: submodular PPO, ACT, diffusion, CFM; 3–5 seeds; 100 eval episodes; held-out patients.
- [ ] T19 US realism: clutter, reverberation, domain randomisation, and a learned-US option. Fixes the open issue that
  "precision ~97 % is too easy".
- [ ] T20 Full H1–H4 experiments, ablation of the force observation, and stiffness randomisation.
- [ ] T21 Linear FEM limits: friction and nonlinear displacement, if T14 shows they are needed.
- [ ] T22 The 8-page paper and the supplementary zip.
