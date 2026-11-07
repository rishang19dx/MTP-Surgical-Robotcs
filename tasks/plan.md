# Revised direction (DEC-048, 8 Nov 2026): two threads chosen to maximise the chance of publication

The 8 Nov self-review (hard-reviewer pass) found the original central claim weak. "Deformation changes what
policies must learn" does not hold for a rigid bone target: with a tracked probe, reconstruction is correct
whether or not the tissue deforms (aware vs unaware Cov@3 88.0 vs 88.8). H2 (force) failed when pre-registered,
and H3 as drafted was true by construction. The paper is re-aimed at two gaps that the literature recognises
and our tools can close. Nothing guarantees acceptance. The design aims to be publishable whatever the outcome:
every hypothesis is pre-registered, and a negative result still informs.

## Thread A (primary): close the cross-patient generalisation gap of image-based robotic-US policies

**The gap.**
- SonoGym: side error 2.3 mm on seen patients vs 27.3 mm on a new patient (their stated main limitation).
- Jiang et al., IJRR 2024: 92 % vs 46 % on unseen patients.
- Intercostal RL (2025): generalises poorly to unseen rib cages.
- Ours: navigation 100 % on 2 training patients vs 7.8 % on 4 held-out patients (DEC-045).
- No work found that uses biomechanical soft-tissue augmentation to improve robot-policy generalisation.

**Hypotheses** (exact predictions and thresholds are pre-registered in DECISIONS before the runs):
- **H1 Scaling.** Held-out navigation success rises with the number of training patients (1 → 3 on the laptop;
  1 → 3 → 9 → 30+ on the server).
- **H2 Physics augmentation.** At a fixed patient count, FEM-based augmentation raises held-out success. The
  augmentation is a random pre-deformation (superposed FEM fields from random skin loads, within the ≤ 5 N regime
  where DEC-047 shows the linear model within 5 %) plus stiffness randomisation.
- **H3 It must be physical.** FEM augmentation beats a non-physical control of matched displacement magnitude
  (random smooth elastic warps of the CT) and plain imaging randomisation.

**Protocol.**
- Primary task: navigation. Secondary: reconstruction.
- Baselines: random, blind (pose only) and oracle on every evaluation.
- Seeds: 3 (Stage 1 laptop: 1–2), 5 for Stage 2.
- Metrics: success (< 3 mm / < 5°) plus SonoGym's position and rotation error, for comparability.
- Data: 13 patients now. The TotalSegmentator v2 dataset (1,228 CTs, 23.6 GB) is filtered on the server for a
  larger training pool and held-out set.

## Thread B: a soft target where deformation provably matters, paraspinal muscle thickness

**Why this target.**
- Ultrasound measurement of erector spinae / multifidus thickness is used clinically (rehabilitative US imaging),
  and probe pressure is a known error source.
- Unlike bone, the muscle itself is compressed, so the measured quantity changes. This is not a rigid shift that a
  tracked pose removes.
- It lies inside the existing L4 FEM region, and TotalSegmentator labels it ("autochthon").

**Hypotheses** (pre-registered):
- **H4.** Image-measured muscle thickness under-reads by an amount that grows with probe force, in the deformable
  simulator. The rigid simulator predicts no change; it is the control.
- **H5.** A force-aware correction (sensed force + patient FEM compliance → predicted compression) reduces the bias
  on held-out patients.
- Restricted to 1–5 N (DEC-047). Forces above 5 N are reported only with the model-error caveat. The magnitude is
  compared with published probe-pressure effects on muscle thickness, once found and verified.

**Gate (by 16 Nov).** Thread B continues only if both hold:
- the muscle boundaries are measurable from the simulated B-mode (image only);
- compression at ≤ 5 N exceeds the axial pixel size.

If either fails, Thread B becomes Stage 2 or future work.

## Thread C: fixes from the self-review (this week)
- Correct the overstated echo-shift claim in the README and the results doc. The shift is of the order of the
  probe indentation, i.e. largely kinematic. Quantify it with a warp vs no-warp comparison at the same face pose.
- Retire H3 "unaware back-projection bias" (true by construction).
- State the caveats:
  - the supine CT geometry;
  - the fat modulus (11 kPa vs the 1–2 kPa reported by the literature search, unverified);
  - the frictionless, quasi-static model with a kinematic robot.

## Schedule

| Dates | Work |
| :--- | :--- |
| 8–12 Nov | Thread C fixes. Implement FEM pre-deformation augmentation and the elastic-warp control. Import the autochthon labels (IDs checked as for L4). Pre-register H1–H5. Finish recording the current pipeline results. |
| 12–16 Nov | Laptop: H1 with 1 vs 2 vs 3 training patients, H2/H3 at 2 patients (navigation). Thread B gate. |
| 16–24 Nov | Laptop: Thread B measurement study (13 patients, 1–5 N, rigid control). Server runs (9 patients, seeds, TotalSegmentator import) if reachable by about 18 Nov; otherwise Stage 1 uses the laptop results. |
| 24 Nov – 4 Dec | Stage 1 abstract (6 pages): contributions = gym (system) + generalisation study (A) + soft-target study (B); preliminary results; Stage 2 charter. |
| Feb – Apr 2027 | Full study (5 seeds, 100 trials, 30+ patients, imitation baseline, release). Backup venue (RA-L/IROS or MICCAI, early-March deadlines to be confirmed) if Stage 1 is rejected on 5 Feb. |

---

# Original plan (5 Nov; partly superseded by DEC-048 above)

Written 2026-11-05. Decision record: DEC-041 in `DECISIONS.md`. Research behind this plan (gitignored scratch, not citable as a
source of numbers): `.agents/research/sonogym_gap_analysis.md`, `.agents/research/softbody_and_positioning.md`,
`.agents/research/deep_research_result.json`.

## Deadlines (confirmed on https://roboticsconference.org/information/cfp/, 2026-11-05)

| Date (AoE) | What |
| :--- | :--- |
| **4 Dec 2026** | Stage 1: 6-page extended abstract (5 pages of content + 1 of references), no supplementary material, double-blind, no external links. Reviewers judge the hypothesis and methodology; results may be preliminary. |
| 5 Feb 2027 | Stage 1 decision (reject, or invite to rebuttal) |
| 12 Feb 2027 | Rebuttal |
| 26 Feb 2027 | Stage 2 decision (invite to execute an "Extension Charter") |
| **16 Apr 2027** | Final paper, 8 pages excluding references (supplementary zip ≤100 MB by 19 Apr) |

## Overview

Build an open, GPU-batched robotic-ultrasound learning environment. Everything in it is patient-specific CT/TotalSegmentator anatomy:
- soft-tissue compression from the linear FEM, with contact force consistent with it;
- B-mode images warped by the deformation.

It should match SonoGym on two of its tasks (bone reconstruction, navigation to a standard plane) and add a third task that
only makes sense with deformation (force-regulated scanning). The paper is hypothesis-led ("Does tissue deformation
change what ultrasound policies must learn?") and treats the gym as the instrument.

## Positioning (what we may and may not claim)

- **SonoGym** (arXiv 2507.01152, NeurIPS 2025) has rigid anatomy and no contact force. Its own limitations section says so.
- **CRESSim-Neo** (arXiv 2608.25192, Aug 2026) already has a batched GPU XPBD UltrasoundScan env.
  - So "the first parallel soft-body US gym" is **not** claimable.
  - It has no CT anatomy, no FEM, no fidelity check, no force reward and no policy results.
- **Goksel & Salcudean** (TMI 2009) and **Bürger et al.** (TMI 2013) predate us on deformable US simulation.
- Claim the combination instead:
  - patient-specific CT anatomy;
  - FEM checked against a reference;
  - force-consistent compression and deformed B-mode;
  - SonoGym task parity plus a force task;
  - benchmarked RL, imitation and classical policies with held-out patients.
- Re-check arXiv for new versions of CRESSim-Neo, SonoGym follow-ups and NVIDIA Isaac-for-Healthcare deformation before
  each deadline.

## Hypotheses (Stage 1 is reviewed on these)

Each can fail, and a negative result is reportable.

- **H1 Transfer gap.** A policy trained on rigid anatomy loses performance when evaluated with deformation.
  - Metrics: Cov@3 and Prec@2 for reconstruction; success rate (<3 mm, <5°) for navigation.
  - The gap grows as tissue gets softer or thicker.
  - Control: train in the deformable sim.
  - Note: DEC-040 already showed only a small coverage gap for the aware vs unaware pipeline on one patient, so H1 may
    fail for reconstruction. That is a result.
- **H2 Force is learnable and patient-dependent.** With probe force as an action and a reward computed from the image
  only, the learned force rises with subcutaneous-tissue thickness.
  - Expected direction: the in-vivo BMI trend reported by Work 2017 (verify that citation before use).
  - In the rigid sim force cannot change the image, so there is nothing for this task to learn.
  - First check that compression changes the simulated B-mode at all (Task 9 gate).
- **H3 Compression bias in reconstruction.** Uncorrected reconstruction or registration error grows with probe force
  over 2–15 N. Correcting with the sensed force plus the precomputed compliance recovers most of it.
  - Comparison points: no correction, oracle field, estimated field.
- **H4 Fidelity and cost.**
  - The precomputed linear FEM stays within a stated error of a nonlinear (Neo-Hookean) FEM reference over the
    clinical force range.
  - Its stiffness lies inside published ranges.
  - Deformation costs a bounded share of throughput at N = 1…128 envs.

## Architecture Decisions (DEC-041)

1. **Keep the PyTorch FEM; batch it on the GPU. No Isaac Lab port before Stage 1.**
   - PhysX deformables do not report contact forces, and Newton is experimental in Isaac Lab 3.0.
   - Our precomputed compliance is shared per patient. Per env, only the contact solve, one field matmul and one
     `grid_sample` warp are needed.
   - Isaac Lab/SonoGym interop is an optional Stage 2 item (Task 17).
2. **Two execution paths share one physics core.**
   - (a) A batched, physics-light **training path**: a Gymnasium `VectorEnv` with no PyBullet and a quasi-static
     probe-pose kinematic model, all tensors on the GPU.
   - (b) The existing **1 kHz PyBullet + Slicer path** for the interactive demo and the systems validation (invariant 2
     unchanged).
   - Both call the same `fem_tissue` and `ultrasound_model_based` functions. A parity test asserts identical images and
     forces for the same pose.
3. **Action and observation spaces follow SonoGym where tasks overlap.**
   - Navigation: 3-DoF tangential.
   - Reconstruction: 4-DoF.
   - The force task adds a scalar force set-point.
   - The perception path stays image-only (invariant 5).
4. **Patients.** Use ≥8 TotalSegmentator cases (the subjects SonoGym uses where we can identify them, since its assets
   are MIT), with a fixed train/held-out split stored in the repo.
5. **Statistics.** At least 3 training seeds (5 if time allows) and 100 evaluation episodes per policy and patient split.
   Numbers only from `results/`.

## Dependency graph

```
T1 batched FEM ──┬─> T2 batched B-mode ──> T3 VectorEnv core ──┬─> T5 recon task ──┬─> T10 baselines (PPO/A2C/SAC via SB3/skrl)
                 │                                             ├─> T6 nav task ────┤
T4 multi-patient assets ───────────────────────────────────────┤                   ├─> T11 H1 experiment
                                                               └─> T7 force task ──┴─> T12 H2 experiment
T8 throughput bench (needs T3) ; T9 compression-changes-image gate (needs T2) ; T13 H3 (needs T5)
T14 fidelity vs nonlinear FEM (independent; needs only fem_tissue)
T15 Stage 1 abstract (needs preliminary T8, T9, T11 or T12, T14)
```

## Phase plan

### Phase 1 — Stage 1 sprint (5 Nov → 4 Dec 2026)

| Week | Goal | Tasks |
| :--- | :--- | :--- |
| W1 (5–11 Nov) | GPU-batched core runs N envs on one patient | T1, T2, T3, T8 |
| W2 (12–18 Nov) | Multi-patient assets; tasks registered; compression gate | T4, T5, T6, T9, T14 (start) |
| W3 (19–25 Nov) | Force task + first preliminary runs (single seed allowed) | T7, T10 (PPO only), T11 or T12 preliminary, T14 |
| W4 (26 Nov–4 Dec) | Write the 6-page abstract; figures from `results/` only | T15 (+ DEC/FLOW/README updates) |

**Stage 1 minimum** if time runs short:
- T1–T3, T5, T8, T9 and T14;
- one preliminary H1 run.

That is enough to state the hypotheses, the method, and evidence that the instrument works.

### Phase 2 — Build-out (Dec → 5 Feb), rebuttal (12 Feb), final (16 Apr)

- T6/T7 complete.
- Full baselines: PPO, A2C, SAC, submodular PPO, ACT, diffusion, CFM, raster, NBV.
- 3–5 seeds, held-out patients; H1–H3 full runs.
- US realism: clutter, reverberation, domain randomisation (the open issue that "the simulator is too easy").
- Optional Isaac Lab wrapper (T17).
- Release packaging (T16).
- 8-page paper.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| H1 shows no gap (as DEC-040's 88.0 vs 88.8 % suggests for coverage) | High | Lead with H2/H3, where deformation is necessary by construction of the task, not by the metric. Report H1 honestly; vary stiffness and thickness to find where the gap appears. |
| Batched contact solve is slow or unstable | High | Fixed-iteration projected Gauss–Seidel / active-set solve on the GPU. Fall back to the per-env path in a process pool (about 2 days) to keep training unblocked. |
| Linear FEM is wrong at 10–15 N | Med | T14 quantifies it; restrict claims to the validated range. Nonlinear displacement is a Stage 2 extension. |
| Field VRAM (233 MB fp16 per patient) × 8 patients | Med | Only patients in the current batch are resident; assign envs to patients in blocks; RTX 5000 Blackwell Pro memory is ample for 2–4 resident patients. |
| CRESSim-Neo or NVIDIA ship CT-based deformable US first | Med | Differentiate on validated FEM, force tasks and benchmarked policies; re-check arXiv weekly. |
| Agent usage limits (memory note) | Low | Run agents in small batches; keep research notes in `.agents/research/`. |
| Stage 1 judged as "just a gym" | Med | Lead with hypotheses H1–H4; gym is the method section. |

## Open Questions (need the user)

- Which TotalSegmentator subjects: the exact SonoGym cases (best for comparability) or our own selection with
  varied BMI (best for H2)? Possibly both.
- Can any real data be borrowed (a lab's phantom US frames, or a public US dataset with probe force) for a realism check?
  A simulation-only paper is acceptable, but this would strengthen H4.
- Authorship, affiliation, and whether a preprint will be posted (it affects anonymity).
