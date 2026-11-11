# MuJoCo bridge plan (DEC-052, branch `mujoco-bridge`)

Written 2026-11-10. Decision record: DEC-052 in `DECISIONS.md`. This is a side track: PyBullet stays the
default backend and the RSS runs (`tasks/plan.md`) are not touched.

## 0. Facts checked before planning

| Claim | What was found | How it was checked |
| :--- | :--- | :--- |
| "MuJoCo 4 has been released" | True: 4.0.0, released 6 Nov 2026. It requires Python ≥ 3.11. In our Python 3.10 venv pip therefore offers 3.17.0 as the newest version (the last release for 3.10). | PyPI JSON (`requires_python`), 2026-11-10 |
| MuJoCo simulates soft bodies natively | True since 3.0, not new in 4.0 (`flexcomp dim="3"` tetrahedra, `elasticity young/poisson/damping`, `pin`). Flex elasticity now needs the `discrete` integrator (3.13+), which the docs call "under active development, and subject to change". | `mj_printSchema`, compile errors, changelog |
| Material model | Saint Venant–Kirchhoff on linear tets, one Young/Poisson pair per flex. A Neo-Hookean option exists only as an experimental C-API switch (3.15+). No per-element stiffness in MJCF. | XML reference, changelog |
| 3.17.0 vs 4.0.0 for soft bodies | 3.17.0 uses twice the first Lamé parameter in 3D (ν = 0.45 behaves like about 0.47) and distributes contact forces with weights that put a false torque on elements. Both are fixed in 4.0.0. 4.0.0 itself has a sparse-Cholesky bug in the Newton solver (issue 3767), so use CG there. | 4.0.0 changelog |
| It runs with our stack | 3.17.0 imports with numpy 1.26.4 on Python 3.10. 4.0.0 is installed in a second venv, `.venv-mj4` (Python 3.11). | both venvs |
| We can reuse the robot model | MuJoCo loads `pybullet_data/franka_panda/panda.urdf` directly. `panda_grasptarget` at the rest posture is at (0.4842, 0, 0.4110) in both engines. | scratch script (these coordinates); `test_backends_agree_on_the_probe_pose_for_the_same_target` checks agreement at a scan pose |
| Soft tissue at 1 kHz | No. A 21×21×11 tet slab at 4 mm (4,851 vertices, 24,000 tets, E = 15 kPa, ν = 0.45) costs about 70 ms per step on 3.17.0 and about 90 ms on 4.0.0; an 11×11×6 slab at 8 mm costs about 5 ms and 8 ms. Our patient mesh is larger (15,259 nodes, 71,232 tets). | scratch spike, laptop CPU, not a `results/` number |
| GPU soft bodies | MJX (JAX) does not support flex. MuJoCo Warp lists flex as experimental and publishes no tetrahedral timings. | MJX feature table, MuJoCo Warp README |
| Indentation force validated? | No published comparison of flex indentation force against FEM or an analytical solution was found. The contact is a soft constraint, so its compliance adds to the tissue's. | docs, issue tracker search |
| Existing MuJoCo–Slicer bridge | None found. | web and issue search |

The spike timings are scratch measurements. They decide the architecture, and they must be regenerated into
`results/` by the Stage 2 study before anyone cites them. The research notes with sources are scratch, not
committed.

## 1. Decisions from the grilling (QUIZME A1, A3, F2, E4; user, 2026-11-10)

1. **Role: staged and gated.** Stage 1 replaces PyBullet's job (robot, probe, fixtures, viewer) and keeps the GPU
   FEM as the tissue model. Stage 2 adds MuJoCo flex as a third tissue backend only if it passes a pre-registered
   gate.
2. **Limitations to attack:** friction and sliding, large-strain accuracy, real robot dynamics, dynamic tissue
   response. Each one gets a measurement that can fail (section 4).
3. **Clock:** flex tissue runs in its own process and reports F0, K_c and δ0, exactly as the FEM does. The 1 kHz
   loop keeps `F = F0 + K_c(δ − δ0)`. Invariant 2 is unchanged.
4. **Priority:** side track, `--physics-backend mujoco` beside PyBullet, PyBullet default until parity and rate are
   in `results/`.
5. **Versions (asked after MuJoCo 4.0.0 was confirmed):** the rigid backend stays in `.venv` (Python 3.10) on
   MuJoCo 3.17.0; the rigid-body features it uses are the same in 4.0. The flex tissue process and its study run
   in `.venv-mj4` (Python 3.11, MuJoCo 4.0.x). That is the only code allowed outside `.venv`.

## 2. Architecture

```
                         streaming side (unified_server, 30 Hz; imports no engine)
                               │ ("target", {...tissue: F0, K_c, δ0...})      ▲ contact / view snapshots
                               ▼                                              │
   physics_common.py  ── PhysicsChannels, run_physics_loop, tick(), force law, CPU pinning, create_backend()
        ├── pybullet_backend.PyBulletBackend   (default)
        └── mujoco_backend.MujocoBackend       (Stage 1)       ← 1 kHz, rigid only
   tissue model (TissueModel protocol, streaming side or its own process)
        ├── FEMTissueModel (default)   ├── KelvinletTissueModel   └── MujocoFlexTissue (Stage 2, gated)
```

- **World frame.** The MuJoCo world is the existing PB frame (Z-up, metres, same offsets), so every helper in
  `coordinates.py` and the exported `meshes/*.obj` stay valid. The only new conversion is the quaternion order:
  (x, y, z, w) at the backend boundary, (w, x, y, z) inside MuJoCo. A contract test covers it.
- **Robot.** The same Panda URDF, compiled by MuJoCo with a pedestal offset of 0.30 m. The probe is a body welded
  to `panda_grasptarget` (a child body, so no constraint drift), with the same 0.2 kg mass and cylinder.
- **Control ("real robot dynamics").** Torque level: `τ = qfrc_bias + M(q)·(Kp·(q_d − q) + Kd·(q̇_d − q̇))`, clipped
  to the URDF effort limits (87 N·m joints 1–4, 12 N·m joints 5–7). `q_d` comes from damped-least-squares IK on
  `mj_jac`, warm-started from the previous solution, a few iterations per tick. No IK offset workaround.
- **Tissue force.** `mj_applyFT` at the probe face, against the beam, every tick, with the shared force law.
- **Fixtures.** A keep-out mesh needs a model recompile in MuJoCo (`MjSpec`), with the state carried over. This
  costs one long tick when a fixture arrives. Mesh geoms collide as convex hulls, so a concave fixture is
  over-approximated (stated limitation; PyBullet uses the concave trimesh).
- **Viewer.** `mujoco.viewer.launch_passive` in the viewer process, mirroring joints and probe pose; skin patch
  through `update_mesh`, overlays through `user_scn`. Never on the physics loop.

## 3. Stage 1 (rigid backend) — done on 2026-11-11, TDD

Tests: `ReconstructionPipeline/tests/test_physics_backends.py`, written first and failing first. Every contract
test runs against both backends. Full suite: 184 passed, 2 xfailed (before: 154 passed, 1 xfailed).

| # | Task | Test | Status |
| :-- | :--- | :--- | :--- |
| 1.1 | `physics_common.py`: engine-free loop, channels, force law; `BACKENDS`, `create_backend()` | `test_physics_common_imports_no_engine`, `test_backend_registry_*` | done |
| 1.2 | `MujocoBackend`: scene from the shared URDF, IK, torque control, teleport | `test_teleport_then_hold_*`, `test_probe_orientation_*`, `test_moving_target_*`, `test_tilting_sweep_is_tracked`, `test_backends_agree_*` | done |
| 1.3 | Tissue force law on the MuJoCo probe | `test_tissue_force_*` | done |
| 1.4 | Virtual fixture via recompile | `test_virtual_fixture_blocks_the_probe`, `test_invalid_fixture_is_ignored` | done (convex hull only) |
| 1.5 | Server flag `--physics-backend`, `physics_backend=` kwarg | `test_server_rejects_unknown_*`, `test_mujoco_inline_server_*`, `test_streaming_side_never_imports_an_engine` | done |
| 1.6 | Rate | `test_median_tick_fits_the_one_millisecond_budget`; `results/physics_rate_20261111_mujoco_vs_pybullet.json` | measured on a loaded laptop; repeat on an idle one |
| 1.7 | Viewer process (GUI) | hand check on 2026-11-11 (FLOW recipe 13); no automated test | done |
| 1.8 | Docs: DECISIONS (DEC-052), FLOW, AGENTS, QUIZME | – | done |

What the tests found:
- **PyBullet's arm is rigid under the tissue force.** Under a 30 N push on the probe it moves by less than
  1 µm (asserted); the torque-controlled MuJoCo arm moves by about 1.1 mm with the chosen gains (printed by
  `test_tissue_force_acts_on_the_dynamics_as_each_backend_models_the_arm`). The deflection is not fed back to
  the tissue solve or the image, so the DEC-048 caveat stands for all reported results.
- **Tilting sweep.** Worst end-effector error 7.3 mm with PyBullet and 5.6 mm with MuJoCo (printed by
  `test_tilting_sweep_is_tracked`).
- **A data bug, not fixed here.** `meshes/patient_l4_vertebra.obj` is in frame P (mm), not in the PB frame
  (metres) like the other two PyBullet meshes. PyBullet draws it about 1000× too large; the MuJoCo viewer skips
  it with a warning.

Open after Stage 1:
- The physics process pins itself to the highest-numbered CPU, which on the i5-1240P is an efficiency core.
  Whether that hurts the MuJoCo tick more (it has more Python in it) is a conjecture. Pinning to a performance
  core is a change to the shared loop and needs its own measurement.
- The rate must be re-measured on an idle machine, with the parent commit alongside: the committed runs do not
  show 1 kHz for either backend and do not rule out a slowdown of PyBullet.
- A concave keep-out fixture is over-approximated by its convex hull.
- Review gates: one code-review agent and one audit agent (invariants 2, 4, 6; "can each new test fail?").

## 4. Stage 2 (flex tissue) — gate first, pre-registered

Stage 2 starts with a study script, not with integration. The gate and thresholds go into DECISIONS **before**
the script is run on the patient mesh.

**Study** (`ReconstructionPipeline/evaluation/mujoco_flex_fidelity.py`, writes `results/mujoco_flex_*.json/.md`):

1. Export the patient tet mesh (`FEMTissueModel.mesh`: nodes, tets, fixed set) to a `flexcomp` with `pin`
   (`dof="full"`: the reduced trilinear/quadratic modes can only pin cage nodes, not bone-surface vertices).
   Stiffness is one value per flex, so fat and muscle need either two coupled flexes or one averaged modulus;
   the study states which and measures the cost of the choice. The mesh is exported from `.venv` to a file and
   the flex runs in `.venv-mj4`.
2. Apply the same loading as DEC-047: the two sites × three patients, 2 / 5 / 10 / 15 N.
3. Compare compression under the probe and the RMS field difference against the Neo-Hookean reference
   (`fem_nonlinear.py`) and the linear FEM.
4. Measure step time, steps to settle, and whether the run stays finite at ν = 0.45. Sweep the contact
   `solref`/`solimp` to show how much of the indentation is contact penetration and not tissue strain.
5. Coarsening sweep (4, 6, 8 mm; trilinear and quadratic reduced modes) to find what resolution is affordable.

**Proposed gate (to be confirmed and committed before the run):**

| Criterion | Pass |
| :--- | :--- |
| Accuracy at ≤ 5 N | compression within 10 % of Neo-Hookean in every converged case |
| Accuracy at 10–15 N | median error no worse than the linear FEM's (DEC-047: up to 31 %) |
| Stability | finite and non-inverted at ν = 0.45 for all cases |
| Cost | a settled solution in ≤ 33 ms wall time (one streaming frame), or ≤ 100 ms with the gate recorded as "10 Hz tissue" |
| Heterogeneity | fat and muscle moduli both represented |

If the gate fails, the result is recorded as a negative finding and Stage 2 stops there.

**If it passes**, `MujocoFlexTissue` runs in its own process:
- a probe face welded to a mocap body follows the 1 kHz probe pose from the `view` snapshot (a bare mocap body
  has no contact velocity and under-reports force; flex vertices cannot attach to mocap bodies);
- it publishes F0 (contact force on the probe), K_c (finite difference over a small indentation step), δ0, the
  deformed skin vertices and the nodal displacement field, resampled to the lattice for the backward warp;
- per-limitation measurements, each able to fail:

| Limitation (DECISIONS) | Measurement | Fails if |
| :--- | :--- | :--- |
| Frictionless contact (DEC-040) | tangential force during a lateral sweep at 4 N, μ ∈ {0, 0.3} | no difference between μ = 0 and μ = 0.3 |
| Linear, small strain (DEC-047) | the study above at 10–15 N | not better than linear |
| Kinematic robot (DEC-048) | Stage 1 gives a compliant arm in the server only. To touch the caveat, the measured probe pose must feed the tissue solve and the image; then compare reconstruction with and without it | no measurable change |
| Quasi-static tissue | force overshoot and settling time after a 2 mm step indentation | overshoot is solver noise (changes sign with timestep) |

**Not in scope:** the gym (`mtp_gym`) stays on the batched PyTorch FEM. A stepped flex model cannot serve its
admittance loop, which makes many quasi-static solves per env step.

## 5. Risks

1. Flex cost: 5–90× the tick budget in the scratch spike; handled by the separate process and the gate.
2. Near-incompressibility (ν = 0.45): ill-conditioning warning seen in the spike.
3. Contact force may be governed by `solref`/`solimp` softness rather than elasticity; the study must show the
   force changes with Young's modulus as expected.
4. Fixture recompile stalls the loop once per fixture.
5. Time: RSS Stage 1 is due 4 Dec 2026. Stage 1 here is time-boxed to about a week; Stage 2 beyond the gate study
   waits if it threatens Thread A.
