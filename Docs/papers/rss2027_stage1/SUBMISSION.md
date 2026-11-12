# RSS 2027 Stage 1 extended abstract: submission kit

**Deadline: 4 Dec 2026, 11:59 pm AoE, on OpenReview.**

Rules (from the RSS 2027 call for papers):
- **Page limit:** 6 pages = at most 5 pages of content + 1 page of references.
- **No extras:** no appendix and no supplementary material.
- **Double-blind:** no author names, no acknowledgements, and no links to GitHub, video or lab pages.
- **Template:** unmodified.
- **Layout marks:** no page numbers and no "Draft" marks.
- **Fonts:** Type-1 (or embedded) fonts only.
- **References:** hallucinated references lead to desk rejection.

## Files

| File | What it is |
|---|---|
| `paper.tex` | The paper (RSS template, `IEEEtran` conference class, `plainnat` + `natbib`) |
| `references.bib` | 11 references. Each was checked on 13 Nov 2026 against its DOI record or arXiv page. |
| `IEEEtran.cls` | From the official RSS template (`roboticsconference.org/docs/paper-template-latex.tar.gz`) |
| `fig_*.pdf` | Made by `make_figures.py`. Data figures are read from `results/`; illustrations are rendered by the simulator. |

**To build:** `pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`, or upload the folder to Overleaf.

## Before submitting (checklist)

- [ ] **Compile and check the length.** The paper has not been compiled yet, because there is no LaTeX on the dev laptop. The text is about 2,600 words, with 1 full-width figure, 3 column figures and 2 tables, which should fit in 5 pages. Check it.
- [ ] **Paper-ID.** Replace `Paper-ID [add your ID here]` with the OpenReview ID once it is assigned.
- [ ] **Fonts.** In the PDF, open Document Properties → Fonts. Everything must be embedded, with no Type 3. The figures embed TrueType (matplotlib `pdf.fonttype 42`).
- [ ] **Anonymity.**
  - No names and no institute in the PDF metadata. `\pdfinfo` already says Anonymous; check Overleaf's own metadata too.
  - No repository links.
  - Patient IDs (s0006, s0029, ...) are public TotalSegmentator subject IDs, which is fine.
- [ ] **Supervisor sign-off** on:
  - the framing;
  - the Stage 2 plan (P1–P4), because Stage 2 is judged on carrying out the Extension Charter, so promise only what can be delivered by 16 Apr 2027.
- [ ] **Liver / abdomen.** Deliberately *not* in the Stage 2 plan; it is still open with the supervisor. If it is added, it becomes a Charter obligation.
- [ ] **Authorship.** Set the author list in OpenReview now: changes between stages need Program Chair approval.
- [ ] **Institutional email.** OpenReview profiles without one can take up to 2 weeks to be moderated, so create the profile early.

## OpenReview form fields (paste-ready)

**Title:** When Does Soft Tissue Matter for Robot Learning in Ultrasound? A Pre-Registered Study on Patient-Specific Deformable Anatomy

**Abstract** (plain text):

> Robot-learning simulators for ultrasound treat the patient as rigid, although a probe can only image by pressing on soft tissue. Whether this simplification matters has not been measured. We pose it as a falsifiable question and answer it with a simulator in which each of 13 CT patients has a GPU-batched finite-element model of the soft tissue, a force-controlled probe, and B-mode images rendered from the deformed anatomy. The key methodological device is a kinematic control mode that presses the probe into undeformed tissue; it splits any rigid-versus-deformable difference into a part due to probe indentation, which a rigid simulator can reproduce, and a part due to deformation, which it cannot. All hypotheses, thresholds and analysis scripts are fixed before each run. Preliminary results give one positive and one negative answer. (i) A soft-tissue measurement depends on deformation: paraspinal muscle thickness read from the image falls by 0.25-0.30 mm/N on held-out patients, tracking the true compression, while the rigid control stays within ±0.04 mm/N. (ii) Policy behaviour in navigation does not: policies trained with and without deformation differ by -0.8 success points, and the residual gap to rigid training is explained by indentation. We propose to test whether deformation changes behaviour once the policy controls the contact force, and to scale the measurement result to many patients.

**Keywords:** robotic ultrasound; soft-tissue simulation; finite element method; reinforcement learning; sim-to-real; pre-registration; medical robotics

**TL;DR:** Separating probe indentation from tissue deformation shows that soft tissue changes an ultrasound measurement but not navigation behaviour; Stage 2 tests force-choosing policies.

## Where each number comes from

| Number in the paper | Source |
|---|---|
| H1 slopes, errors, detection, the H5 follow-up (1.31 vs 1.32 mm) | `results/muscle_thickness_20261112_151349.md` (DEC-056) |
| Muscle thickness 38–51 mm (true thickness at 1 N, scored sites) | `results/muscle_thickness_20261112_151349.json` |
| H2 table, gaps, transfer to kinematic | `results/nav_transfer_20261112_183531.md` (DEC-057) |
| FEM vs Neo-Hookean (−4.9 to +1.8 % at 2 N, ±30 %) | `results/fem_fidelity_20261107_234638` (DEC-047) |
| 9.43 vs 3.48 N | test `test_multi_patient.py::test_anatomy_changes_force_under_shared_moduli` (DEC-041) |
| 530 / 665 env-steps/s | `results/throughput_20261105_235332.md` |
| 40–53 min per 300k-step run | `checkpoints/vec_ppo/nav_*_laptop2p_{transfer,offset}.json` (`wall_s`) |
| MuJoCo flex gate | `results/mujoco_flex_fidelity_20261112_112537.md` (DEC-053) |
| Reconstruction 88.0 vs 88.8 % | DEC-048 |
| Echo shift is kinematic (11 of 12) | `results/kinematic_vs_deformation_20261108_020955` |
| Navigation 100 % vs at most 14.1 % held out | `results/vec_eval_nav_20261107_223741`, `results/vec_eval_nav_20261108_075931` |
| Fascia 1.65 mm, 1650 m/s, 1.61 dB/(cm MHz) | DEC-055 (Pirri 2023 Table 3; Miles 1996) |
