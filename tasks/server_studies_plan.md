# Server study plan (from 27 Nov 2026, after DEC-064)

Goal: results, not a paper deadline (RSS dropped 26 Nov). Every study below is pre-registered as its own DEC
(prediction, gate, patients, seeds, analysis script frozen) **before** it runs, and negatives are reported.
Compute figures marked "est." are planning guesses, not results; S0.3 replaces them with measurements.

Resources: 399 new patients (`data/patients_ts2`, DEC-064) + 13 old; shared 8 × A6000 (usually 3–4 free).

---

## Stage 0 — Foundations (blocks everything)

### S0.1 Patient split v2
- Fixed rule: sort the 399 by midline soft tissue at L4 and take every 3rd as test (as DEC-041), so thin to thick
  backs are in both sets; the 13 old patients keep their old split.
- A small dev subset of the train patients for any tuning; the test set is not touched until a study's run.
- Decide the 3 old patients that fail the DEC-064 filter (s0004, s0006, s0012): keep with a flag or drop.
- Output: `data/splits_v2.json`, DEC-065. Cost: an hour, CPU.

### S0.2 Measurability QA over all 412
- The DEC-051 window rule (muscle untruncated at all forces), the scan-window skin coverage (ts2_s0353, ts2_s0740),
  and whether the liver and femoral vein are present (TotalSegmentator labels) per patient.
- Output: a per-patient eligibility table in `results/`, used by S1–S2 as frozen inclusion lists.
- Cost: est. < 1 h, 1 GPU.

### S0.3 Training pilot (sizing only, no hypothesis)
- One nav PPO run on train patients at 512 and 1024 envs, 1 GPU, 2M steps: wall time per million steps, VRAM, and
  whether the many-patient env (patients rotated on the GPU) works.
- Output: measured cost per run, which sets the seeds/steps in S3. Cost: est. 1–2 h.

---

## Stage 1 — Measurement studies at scale (no training; cheap; days)

### S1.1 Muscle thickness under force at scale (DEC-056 replication, H4/H5)
- Question: does the image-measured muscle thickness fall with force, tracking ground truth, while the rigid
  simulator stays flat — across hundreds of held-out patients instead of 2?
- Design: the frozen DEC-056 detector v3 and analysis unchanged; held-out v2 patients that pass S0.2; 1–5 N; rigid
  vs deformable. H5 (force-aware correction) refit on the v2 train patients, tested on held out.
- Outcomes: per-patient slope (mm/N) with a CI across patients; slope vs fat and muscle thickness; H5 error vs no
  correction.
- Why first: the one positive result so far, rests on n = 2. Highest value per GPU hour.

### S1.2 Modulus sensitivity (P4)
- Repeat S1.1 with per-patient random modulus scale (`e_scale`, run-time, no recompute), range fixed from the
  literature before the run. Question: does the H4 conclusion survive unknown stiffness?

### S1.3 Spine force headroom at scale (DEC-058 at scale)
- Per-patient headroom (outcome range over the force range vs the rigid noise floor) on all eligible patients.
- Output: the distribution, and whether any pre-defined subgroup (e.g. thick backs) clears the noise floor by the
  DEC-058 margin. This is the gate for a force-choosing policy on the spine (S3.4).

### S1.4 Liver: finish the pending studies
- (a) DEC-063 at the 90 mm margin (needs RAM, now available), on the same 3 subjects — settles whether the clamped
  box causes the liver null.
- (b) DEC-060 sliding liver (pre-registered 16 Nov, never run): run as registered.
- (c) Only if (a) or (b) changes the picture: liver headroom on many TotalSegmentator v2 patients (needs the liver
  importer extended to the v2 zip), its own pre-registration.

---

## Stage 2 — Physics upgrade (only if Stage 1 shows we need it)

### S2.1 Large-strain FEM (Neo-Hookean, GPU, offline use)
- Why: DEC-062 found no vein reaches 50 % collapse by 20 N in linear FEM, and the linear model is ~30 % off at
  5–15 N (DEC-047). Vein collapse, forces above 5 N and liver at higher force all need it.
- Scope: an offline nonlinear solve for measurement studies only; the 1 kHz loop keeps the linear field
  (invariant 2 unchanged).
- Gate before any use: agreement with a reference nonlinear solve (the `fem_fidelity.py` setup) and with the
  published popliteal/femoral collapse data (Sato 2022, Eitminavičius 2026) on calibration only.
- Needs a QUIZME grilling before code (major capability).

### S2.2 Vein compressibility re-run (DEC-061 rule, with large strain)
- Re-run the DEC-061 F50 spread gate on femoral veins from the v2 patients. This re-runs a failed gate after adding
  physics: the justification (DEC-062 says large strain is needed) is recorded before the run and disclosed.

---

## Stage 3 — Policy studies (training-heavy; weeks; most of the GPU time)

### S3.1 Thread A H1: navigation generalisation vs number of training patients
- Train on 1, 4, 16, 64, 256 patients (nested subsets), 3 seeds (5 at the largest points if the pilot allows);
  evaluate on all held-out v2 patients. Prediction: held-out success rises with N (the laptop gap was 100 % train
  vs 7.8 % held out).
- This is the main candidate for a clear result.

### S3.2 Thread A H2/H3: does FEM augmentation help generalisation?
- At the N where S3.1 shows the largest remaining gap: none vs FEM pre-deformation augmentation vs non-physical
  warp control (displacement RMS matched) vs imaging randomisation. 3+ seeds.

### S3.3 Navigation deformation-only gap without a ceiling (P3, DEC-057 at scale)
- Rigid vs kinematic vs deformable training, tested on held-out patients where success is not at 0.98–1.00.

### S3.4 Force-choosing policy (P1) — only if S1.3 or S1.4/S2.2 passes its headroom gate
- Train in deformable and kinematic sims; primary metric is the deformation-only gap on the task outcome.

### S3.5 Benchmark refresh
- Retrain PPO / CFM / diffusion on the v2 train set (checkpoints predate DEC-040); 6-policy benchmark on held-out
  patients, 3 seeds.

---

## Stage 4 — Realism (parallel track, changes the simulator)
- US clutter and reverberation (precision ~97 % vs 0.8–0.9 in the literature). Changes every perception number, so it
  goes in as a versioned simulator change, followed by a re-run of the headline results (S1.1, S3.1) as a
  sensitivity check — not mixed into the runs above.

---

## Order and GPU use

| Order | Study | Needs | GPUs | Est. time |
| :--- | :--- | :--- | :--- | :--- |
| 1 | S0.1 split, S0.2 QA, S0.3 pilot | — | 1 | ~1 day incl. pre-reg |
| 2 | S1.1 muscle at scale + S1.2 sensitivity | S0.1, S0.2 | 1–2 | 1–2 days |
| 2 (parallel, CPU/RAM) | S1.4a liver 90 mm, S1.4b sliding | — | 0–1 | 1 day |
| 3 | S1.3 spine headroom at scale | S0.2 | 1 | 1 day |
| 4 (start) | S3.1 H1 scaling — long runs start while 5–6 are analysed | S0.3 | 2–4 | 1–2 weeks |
| 5 | S3.3 nav without ceiling | S3.1 runs | shared with S3.1 | — |
| 6 | Decision point: does any target pass a headroom gate? | S1.3, S1.4 | — | — |
| 7 | S2.1 large-strain FEM (grill first) → S2.2 vein | decision 6 | 1 | 1–2 weeks |
| 8 | S3.2 augmentation, S3.4 force policy (if gated in), S3.5 benchmark | S3.1, 6/7 | 2–4 | weeks |
| 9 | Stage 4 realism + sensitivity re-runs | after headline results | 1 | 1 week |
