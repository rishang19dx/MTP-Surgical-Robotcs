# Questions for the supervisor: where to take the soft-tissue work

*Prepared 13 Nov 2026. The full evidence is in `report.tex` in this folder.*

## 1. The situation in two minutes

Open the meeting with this.

**What I set out to do**
- Take SonoGym's robotic-ultrasound tasks.
- Add patient-specific soft tissue: probe force, deformation, and an image rendered from the deformed anatomy.
- Find out what the deformation changes.

**What we found** (every test was pre-registered):

| Question | Answer | Strength |
|---|---|---|
| Does deformation change a **measurement**? (paraspinal muscle thickness vs probe force) | **Yes.** About 0.25–0.30 mm thinner per newton, tracking the truth. A rigid simulator shows nothing. | 2 measurable held-out patients; effect about 1 mm over 1–5 N |
| Does deformation change **policy behaviour**? (navigation to the L4 view) | **No.** Gap −0.8 points. Indentation explains the difference from SonoGym-style training, and a rigid simulator can add indentation. | 2 patients, 2 seeds, ceiling, force fixed by the task |
| Does deformation change **bone reconstruction**? | **No** (88.0 vs 88.8 % coverage) | — |
| Do learned policies **generalise** to new patients? | **No** (≤ 14 % held out vs 100 % on training patients) | laptop scale only |
| Can **MuJoCo 4** soft bodies replace our FEM? | **No.** Either 90–420 % wrong, or about 380× too slow | pre-registered gate |

**The pattern behind all of it.** Deformation only matters when both of these hold:
1. the **target itself is soft**, and
2. the outcome depends on its change by more than the noise.

The spine's targets are mostly **bone**, which does not deform. That is why I think we are at a deadlock *for this anatomy*. It is not necessarily a deadlock for the idea.

**The time pressure**
- The RSS 2027 Stage 1 abstract is due **4 Dec 2026** (3 weeks).
- The final paper is due 16 Apr 2027.

## 2. Questions to ask, most important first

For each question: why I'm asking, what I would suggest, and what the answer changes.

### Q1. Is "where soft tissue matters and where it doesn't" a publishable contribution, and where?
- **Why.** We have one small positive result and several clean negatives, all pre-registered. Some venues value a well-done negative result ("rigid simulators are enough for navigation to bony targets"); others will see "no effect".
- **My suggestion.** Submit the Stage 1 abstract to RSS framed as a map of where soft tissue matters. Keep a backup: MICCAI (early March), or IROS/RA-L.
- **Ask specifically.**
  - Would a reviewer at RSS find this interesting enough?
  - Would MICCAI or IPCAI suit a measurement-centred paper better?
- **What it changes.** If RSS is wrong for this framing, the next 3 weeks go into building a new result instead of writing.

### Q2. Should we switch to (or add) a softer organ, such as the liver?
- **Why.** The rule above says to look for targets that **move or change size** under the probe. TotalSegmentator already labels the liver, kidneys, spleen and major vessels, so the pipeline can be reused. Candidate tasks:
  - **Liver lesion or vessel targeting** (biopsy or ablation guidance). The target shifts under the probe, so a policy trained on rigid anatomy aims at the wrong place.
  - **Vein compressibility**, as in compression ultrasound for deep vein thrombosis. Whether the vein collapses under the probe *is* the diagnosis, so deformation is the outcome.
  - **Organ size** (kidney length, liver span) under pressure. This is the abdominal version of our muscle result.
- **Honest costs.** Our FEM is linear and accurate only up to about 5 N, so it would need:
  - **organ sliding**, which it does not model;
  - **breathing motion**, which is often larger than the probe effect;
  - **rib and bowel-gas shadowing** in the image model;
  - for veins, **fluid-filled collapse**, which a linear FEM cannot do;
  - **liver stiffness values** from the elastography literature.

  That is weeks of work, so it is probably not feasible before 4 Dec. It is realistic for the April paper.
- **Ask specifically.**
  - Which abdominal task is clinically most meaningful in your view?
  - Do you know a clinician (radiologist or hepatologist) who could tell us whether probe-induced target shift is a real problem in practice?
  - Would you rather have spine + liver in one paper, or liver alone?
- **What it changes.** The whole direction for the final paper.

### Q3. Should I try one more policy-level test on the spine, where the policy chooses the force?
- **Why.** Navigation could not show a deformation effect because the task fixes the force at 4 N. In a measurement task where the policy chooses the force, a rigid-trained policy is never penalised for pressing hard, so it should press harder and measure the muscle thinner.
- **Risk.**
  - Our measurement result already predicts the size of the effect without any RL (about 0.9 mm from 1 to 4 N), so a reviewer may say the policy adds nothing.
  - Force-scan PPO did not learn at laptop scale.
- **My suggestion.** Only on the server, and only after Q1/Q2 are settled.
- **What it changes.** Whether there is any *policy* claim in the paper.

### Q4. Is the muscle-thickness result new, or is the probe-pressure effect already well documented clinically?
- **Why.** Clinicians already know that probe pressure changes muscle thickness readings. The contribution would then be that a **simulator** can reproduce and quantify it per patient, which rigid simulators cannot. I need to know how strong that is.
- **Ask specifically.**
  - Do you know studies that measured muscle thickness at controlled probe forces (in mm per N)?
  - If so, can we compare our 0.25–0.30 mm/N against them? That would be our first real-data anchor.

### Q5. Can we get any real data, even a little?
- **Why.** The biggest limitation is that every result is simulator-only. Even a small real anchor would greatly strengthen the paper:
  - one gel phantom of known stiffness scanned at several forces, or
  - a handful of volunteer scans with a force sensor.
- **Ask specifically.**
  - Is there an ultrasound machine, a phantom, a force sensor or a robot arm in the lab or a partner lab?
  - Is ethics approval needed for volunteer scans, and how long does it take?

### Q6. GPU server access
- **Why.** The workstation is only reachable from the institute network. Every scaling experiment needs it:
  - more patients (the 23.6 GB TotalSegmentator v2 download);
  - many seeds;
  - millions of training steps;
  - a liver pilot.
- **Ask specifically.**
  - Can I get VPN access, or a guaranteed time slot?
  - Can I use it for about two weeks of continuous runs?

### Q7. Scope for the next 3 weeks
- **Ask directly.** "If you had to choose, which should I do by 4 Dec?"
  - **(a)** Write up what we have.
  - **(b)** Start the liver pilot.
  - **(c)** Run the server experiments for generalisation.
- **My recommendation.** **(a)** now, with (b) or (c) as the plan the abstract promises for the full paper. RSS Stage 1 judges the hypothesis and method, not finished results.

## 3. Things worth saying up front
- **We stopped the MuJoCo soft-body line** after it failed a pre-registered gate. MuJoCo is still useful as the robot engine.
- **All results are pre-registered and reproducible from scripts.** Negative results were not re-run with relaxed thresholds.
- **One disclosed process slip.** A held-out patient (s0006) was viewed during a visual check before the muscle study. It is recorded in DEC-056.

## 4. Quick numbers, if asked
- **Muscle slope:** −0.302 and −0.245 mm/N (image) vs −0.339 and −0.226 mm/N (truth) on s0029 and s0006. Rigid: +0.019 and +0.038 mm/N. (`results/muscle_thickness_20261112_151349.md`)
- **Navigation:** deformable-trained 0.98–1.00 success and kinematic-trained 1.00 on deformable tissue; rigid-trained 0.80 and 0.95. (`results/nav_transfer_20261112_183531.md`)
- **FEM accuracy:** within 5 % of a large-strain reference at 2 N; up to ±30 % at 5–15 N. (`results/fem_fidelity_20261107_234638`)
- **MuJoCo flex:** SVK is 90–420 % off at 2 N. Stable Neo-Hookean takes 12.5 s per re-settle against a 33 ms budget. (`results/mujoco_flex_fidelity_20261112_112537.md`)
