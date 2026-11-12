# MuJoCo Bridge (branch `mujoco-bridge`)

This branch adds MuJoCo as an alternative physics engine to the project described in the main
[`README.md`](README.md), staged in two parts (DEC-052, DEC-053 in [`DECISIONS.md`](DECISIONS.md)):

1. **Stage 1 (done): MuJoCo as a rigid-body backend.** `--physics-backend mujoco` replaces PyBullet for the robot,
   probe, table, virtual keep-out fixtures and viewer in the 1 kHz loop. The GPU FEM stays the tissue model.
   `--physics-backend pybullet` remains the **default**; nothing on `main` changes behaviour.
2. **Stage 2 (gated): MuJoCo flex as a tissue model.** A live MuJoCo 4 soft-body (flex) simulation of the patient
   mesh, in its own process, runs as `--tissue-backend mujoco-flex`. It is **demo only** — it does not back any
   metric, benchmark or gym run — unless it passes a pre-registered accuracy/stability/cost gate against a
   Neo-Hookean reference. See [Stage 2: the gate](#stage-2-the-gate) for where that result lives.

This branch is a **side track**: it was built off `audit-fixes` to attack four tracked limitations (friction,
real robot dynamics, dynamic tissue response, large-strain accuracy) without disturbing the RSS 2027 runs on
`main`. It has not been merged. Read [Limitations](#limitations) and [Status](#status) before citing anything
from it.

---

## Why two engines

PyBullet's robot is effectively **kinematic**: its position motors are velocity-limited constraints well above
the Panda's torque limits, so a tissue-force push on the probe moves the end effector by under 1 µm (measured,
DEC-052). That is fine for the reconstruction task (a tracked probe), but it means "robot dynamics" is not
actually modelled anywhere in the project, and friction, compliance and contact response are all absent from the
rigid side of the simulation. MuJoCo was evaluated because:

- it ships a genuine torque-controlled articulated-body solver (so robot compliance under force becomes
  real, not assumed away), and
- it claims native soft-body (flex) simulation, which is the only way to test whether a *different* continuum
  solver agrees with the project's own GPU FEM, rather than only checking the FEM against itself.

Facts checked before committing to this (DEC-052 section 0, `tasks/mujoco_plan.md`):

| Claim | Finding |
| :--- | :--- |
| "MuJoCo 4 has been released" | True (4.0.0, 6 Nov 2026). It requires Python ≥ 3.11; a Python 3.10 venv silently offers 3.17.0 as "newest" instead of erroring, which is why we first thought it didn't exist. |
| MuJoCo flex is new in 4.0 | False — it exists since 3.0 (Saint Venant–Kirchhoff on linear tets, one modulus per flex). 4.0.0 fixes two 3.17.0 soft-body bugs (doubled first Lamé parameter in 3D; a contact-weighting bug that puts false torque on elements). |
| Flex can share the project's 1 ms tick | No. A 4 mm tet slab the size of our ROI costs 70–90 ms/step on CPU (scratch spike). This is why flex runs in its own process, never in the 1 kHz loop. |
| A published check of flex force against FEM/analytical solutions | None found — this project's DEC-053 study is, as far as we found, a novel comparison. |

---

## Architecture

```
 unified_server.py (--physics-backend {pybullet,mujoco}, --tissue-backend {fem,kelvinlet,mujoco-flex})
   │
   ├─ 1 kHz physics loop (own process) ── physics_common.py (engine-free: channels, tick, force law,
   │     │                                 fixture validation, CPU pinning) ─┬─ pybullet_backend.py (default)
   │     │                                                                  └─ mujoco_backend.py  (Stage 1)
   │     └─ probe force F = F0 + K_c(δ − δ0) applied every tick, from whichever tissue model is selected
   │
   ├─ tissue model (streaming side, synchronous) ─┬─ FEMTissueModel        (default; GPU, quasi-static)
   │                                              ├─ KelvinletTissueModel (analytic, comparison)
   │                                              └─ MujocoFlexTissueModel (DEMO ONLY, DEC-053) ──┐
   │                                                                                               │
   └─ viewer process (--gui): mirrors joints/probe, renders the FEM/flex-deformed skin              │
                                                                                                     │
       MujocoFlexTissueModel (runs in .venv, this process) ──IPC (multiprocessing.connection)──► mujoco_flex_worker.py
                                                                                             (subprocess, .venv-mj4,
                                                                                              Python 3.11, MuJoCo 4)
```

- **Stage 1** (`mujoco_backend.py`) replaces `pybullet_backend.py` under the same `PhysicsBackendBase` contract
  (`physics_common.py`), so every test in `test_physics_backends.py` runs against both engines.
- **Stage 2** (`mujoco_flex_tissue.py` + `mujoco_flex_worker.py`) is a *tissue* model, not a physics backend: it
  implements the same `TissueModel` protocol as `FEMTissueModel`/`KelvinletTissueModel` and is selected
  independently of `--physics-backend`. It runs in a **second Python interpreter** (`.venv-mj4`) because MuJoCo 4
  needs Python ≥ 3.11 while the rest of the project is pinned to 3.10 (AGENTS.md invariant 1, amended for this one
  case). The two processes talk over a `multiprocessing.connection` socket using plain builtins and raw bytes
  (never numpy pickles — the two venvs run different numpy versions).

### Stage 1: the rigid backend (`mujoco_backend.py`)

- **World frame**: identical to PyBullet's — the PB frame of `coordinates.py` (Z-up, metres). Quaternions cross
  the boundary as `(x, y, z, w)`; MuJoCo is `(w, x, y, z)` internally.
- **Robot**: the same Franka Panda URDF PyBullet loads, compiled by MuJoCo (gripper fingers removed, 7 DoF). The
  probe is a child body of `panda_grasptarget`, so there is no constraint to drift (unlike PyBullet, which welds
  it with a constraint).
- **Control (torque level)**, unlike PyBullet's kinematic position motors:

  ```
  τ = qfrc_bias + M(q) · (Kp (q_cmd − q) + Kd (q̇_cmd − q̇))     [clipped to the URDF effort limits]
  ```

  `q_cmd` tracks a damped-least-squares IK solution to the commanded end-effector pose, run **one iteration per
  tick until it converges** (up to `IK_MAX_TICKS = 300`) — an earlier version stopped after 3 iterations and left
  a single streamed target up to 95 mm off; see the DEC-052 addendum. The IK's convergence lead is interpolated
  into a joint-speed-limited command, and the PD error term (not the command itself) is clamped to a 5 mrad
  following-error bound, so a blocked arm pushes with bounded force without the clamp fighting the
  velocity feed-forward (this ordering bug was the cause of a torque-saturated 1 rad/s joint oscillation found in
  the same review — fixed, test-first, in `test_physics_backends.py`).
- **Fixtures** need one `MjSpec` recompile per fixture (MuJoCo cannot add a geom to a compiled model). Each
  fixture **triangle becomes its own thin convex prism** (4 mm thick, `FIXTURE_SHELL_M`), so the union of prisms
  follows a concave surface or a flat sheet — a single mesh collider in MuJoCo can only be a convex hull, which
  would fill in concave keep-out regions. Limit: 2,000 triangles per fixture (`MAX_FIXTURE_TRIANGLES`); larger
  meshes must be decimated first. PyBullet collides with the triangle mesh directly and has no such limit.
- **Viewer**: `mujoco.viewer.launch_passive` in the (optional) separate viewer process. Never touches OpenGL in
  the headless physics process.

### Stage 2: the flex tissue (`mujoco_flex_tissue.py`, `mujoco_flex_worker.py`)

- **Mesh**: the same CT-derived ROI, Dirichlet (fixed) node set and per-cell fat/muscle moduli as the GPU FEM
  (`ReconstructionPipeline/simulation/tet_mesh.py`), but a coarser element size for the live demo (6–8 mm vs the
  FEM's 4 mm) so it can run close to real time.
- **Material**: one MuJoCo flex, either Saint Venant–Kirchhoff (`svk`, MuJoCo's default) or the experimental
  Stable Neo-Hookean material (`snh`, `elastic3d=1`, MjSpec-only). **Per-tet moduli in one flex**: MuJoCo only
  exposes one Young's modulus per *flex* in MJCF, but the compiled per-tet stiffness matrix
  (`model.flex_stiffness`, 24 values/tet) turns out to scale linearly with E in 23 of its 24 slots for both
  materials — so heterogeneous (fat vs. muscle) tissue is represented by scaling each tet's compiled stiffness
  after the fact, rather than needing one flex per tissue type (which can't share vertices anyway). This is
  asserted by `test_per_tet_scaling_equals_changing_young` and is the finding that made G5 of the gate (below)
  satisfiable at all.
- **Live simulation** (`mujoco_flex_worker.py live`): a dynamic (not quasi-static) MuJoCo simulation with a probe
  mocap body that MuJoCo's own contact solver presses into the flex. It runs in `.venv-mj4` as a subprocess,
  paced to real time, and publishes state (contact force, probe depth, nodal displacement field) over the IPC
  socket at ~30 Hz. `MujocoFlexTissueModel` (in `.venv`, the main process) keeps the **latest published state**
  in a background thread and answers `solve()`/`force_at()` from it — it never blocks the 1 kHz loop or the
  streaming loop waiting on the flex, exactly like the GPU FEM. The consequence is that this tissue model lags
  the commanded probe pose by the worker's own settling dynamics; it is a genuinely dynamic tissue response
  (one of the four tracked limitations), not an approximation of one.
- **What it reports**: reaction force (MuJoCo contact force on the probe, along `-beam`), the worker's *actual*
  probe-face depth (so force and indentation sent to the 1 kHz loop belong to the same state), a secant contact
  stiffness (no tangent is available from a dynamic sim), and the nodal displacement field resampled onto the
  same lattice convention as the FEM's `G_lat`, so the Slicer skin patch and backward-warp code are unchanged.
  Geometry queries (first contact, max indentation, tissue thickness, skin mesh) are delegated to the FEM model
  built from the same CT — Stage 2 only replaces the *mechanics*, not the geometry pipeline.
- **Demo-only fencing**: `TISSUE_BACKENDS = ("fem", "kelvinlet", "mujoco-flex")` in `unified_server.py`, but
  `UltrasoundReconstructionEnv` and `benchmark_runner.py` only accept `"fem"`/`"kelvinlet"` — they raise/exit on
  `"mujoco-flex"` (tested: `test_mujoco_flex_is_not_accepted_by_the_env_or_the_benchmark`). The server prints a
  demo-only banner at startup when this backend is selected. This fence does not move unless the gate below
  passes.

---

## Quickstart

```bash
# 0. One-time: the flex worker's interpreter (MuJoCo 4 needs Python >= 3.11; nothing else may use this venv)
uv venv --python 3.11 .venv-mj4 && uv pip install --python .venv-mj4/bin/python mujoco==4.0.0 numpy

# 1. Stage 1 tests (contract tests run against both PyBullet and MuJoCo)
./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_physics_backends.py -q -s
./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_coordinates.py -q        # L4-mesh-frame regression

# 2. Stage 2 (flex worker) tests — skipped automatically if .venv-mj4 is absent
./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_mujoco_flex.py -q

# 3. Physics-rate benchmark, either backend (always re-measure after a control-loop change)
./.venv/bin/python surgical_physics_bridge/src/unified_server.py --benchmark-seconds 20 --physics-backend mujoco
./.venv/bin/python surgical_physics_bridge/src/unified_server.py --benchmark-seconds 20 --physics-backend pybullet

# 4. Live server with the MuJoCo rigid backend (Stage 1), FEM tissue (unchanged accuracy)
./.venv/bin/python surgical_physics_bridge/src/unified_server.py --mode raster_l4 --fps 30 --gui \
    --physics-backend mujoco

# 5. Live server with the MuJoCo rigid backend AND the demo-only live flex tissue (Stage 2)
./.venv/bin/python surgical_physics_bridge/src/unified_server.py --mode raster_l4 --fps 30 --gui \
    --physics-backend mujoco --tissue-backend mujoco-flex --flex-element-mm 8

# 6. In another terminal: 3D Slicer with the (unchanged) bridge script
Slicer --python-script Slicer/slicer_surgical_physics_bridge.py

# 7. One-shot recorded demonstration (server + flex + Slicer, auto-captures + assembles GIFs)
bash scripts/demo_mujoco_flex_slicer.sh --seconds 60
#   writes demonstrations/mujoco_flex/{mujoco_flex_tissue.gif, mujoco_flex_slicer.gif,
#                                       mujoco_flex_contact_sheet.png, README.md}

# 8. The pre-registered Stage 2 gate study itself (slow: tens of minutes, see Status below)
./.venv/bin/python ReconstructionPipeline/evaluation/mujoco_flex_fidelity.py
#   writes results/mujoco_flex_fidelity_<stamp>.json/.md
```

FLOW.md recipes 13 (Stage 1) and 14 (Stage 2) hold the canonical, copy-pasteable versions of the commands above,
including the Wayland-safe snapshot environment variables.

### New flags on `unified_server.py`

| Flag | Values | Notes |
| :--- | :--- | :--- |
| `--physics-backend` | `pybullet` (default) \| `mujoco` | Rigid-body engine of the 1 kHz loop (robot, probe, fixtures, viewer). |
| `--tissue-backend` | `fem` (default) \| `kelvinlet` \| `mujoco-flex` | `mujoco-flex` is demo only (DEC-053); rejected by the env, benchmark and gym. |
| `--flex-element-mm` | float, default `8.0` | Element size of the live flex mesh; 8 mm runs near real time on a laptop CPU. |

---

## The live Slicer demonstration

`scripts/demo_mujoco_flex_slicer.sh` is the single entry point for "show me soft-body deformation in MuJoCo,
bridged to Slicer":

1. Starts `unified_server.py` with `--physics-backend mujoco --tissue-backend mujoco-flex`, with opt-in offscreen
   snapshotting enabled on both the MuJoCo robot viewer (`MTP_VIEWER_SNAPSHOT_DIR`, reused from DEC-039) and the
   flex worker itself (`MTP_FLEX_SNAPSHOT_DIR`, new: the worker renders its own camera looking at the probe/tissue
   contact, with `flex_frames.csv` recording the actual contact force and probe depth per frame — these are the
   same numbers the physics loop received, not hand-authored captions).
2. Starts 3D Slicer with the existing bridge script, with Slicer-side snapshotting enabled
   (`MTP_SLICER_SNAPSHOT_DIR`, new) and an auto-exit timer (`MTP_SLICER_EXIT_AFTER_S`) so the whole thing runs
   unattended for a fixed duration and shuts itself down.
3. Calls `ReconstructionPipeline/scripts/make_mujoco_flex_demo.py`, which assembles the captured frames (plus
   their CSV telemetry) into two GIFs and a contact sheet, with every number in the generated `README.md` read
   back from the CSVs — never typed by hand (AGENTS.md invariant 6).

All of the opt-in snapshot mechanisms follow the DEC-039 pattern already used for PyBullet/MuJoCo viewer
snapshots on Wayland (where external capture tools like `import`/`xwininfo` cannot see the windows): each side
renders or grabs itself, from inside its own process.

---

## Stage 2: the gate

Because MuJoCo flex has no published validation against a reference continuum solver, DEC-053 pre-registers a
gate — written into `DECISIONS.md` **before** the study (`ReconstructionPipeline/evaluation/mujoco_flex_fidelity.py`)
was run on any patient case — so the result can't be massaged after the fact. The gate compares flex compression
against the same Neo-Hookean reference used for the GPU FEM's own fidelity check (DEC-047), under **identical
prescribed nodal forces** (isolating the material model from contact), across 3 patients × 2 sites × 4 force
levels × 2 materials (SVK, Stable Neo-Hookean):

| # | Criterion | Pass condition |
| :-- | :--- | :--- |
| G1 | Accuracy at 2/5 N | every scored case within 10 % of the Neo-Hookean reference |
| G2 | Accuracy at 10/15 N | median absolute relative error ≤ the linear FEM's own error on the same cases |
| G3 | Stability at ν = 0.45 | every case finite, no tet inverted |
| G4 | Cost | a warm-started re-settle after a 0.33 mm indentation step ≤ 33 ms wall (or ≤ 100 ms, recorded as "10 Hz tissue") |
| G5 | Heterogeneity | fat and muscle moduli both represented (satisfied structurally by the per-tet scaling above; asserted by `test_per_tet_scaling_equals_changing_young`) |

**A material passes only if G1–G4 all hold.** If neither material passes, flex remains demo-only indefinitely —
it does not get a second chance with relaxed thresholds. The full criteria, rationale and alternatives considered
are in DEC-053.

**Where the result lives**: `results/mujoco_flex_fidelity_<stamp>.json/.md`, with the gate pass/fail table at the
end of the `.md` file, and a dated follow-up entry under DEC-053 in `DECISIONS.md` once the study has completed.
Do not treat any flex force or compression number as validated until that gate entry exists and says "pass".

---

## Status

As of this branch's last commit:

- **Stage 1 is implemented, tested, and under live review.** `test_physics_backends.py` runs its full contract
  suite against both engines; see the DEC-052 entry and its two addenda in `DECISIONS.md` for the defects found
  and fixed during that review (a streamed-target tracking bug, a control-loop limit cycle, missing orientation
  reporting, unvalidated fixture input, concave-fixture over-approximation, and the long-standing L4-mesh-frame
  bug inherited from `main`).
- **The physics rate has not yet been re-measured since the most recent control-loop fix.** The only committed
  rate comparison (`results/physics_rate_20261111_mujoco_vs_pybullet.json`) predates it and was taken on a loaded
  machine; it does not show either backend reliably holding 1 kHz and does not separate the two engines. Do not
  cite it as a performance comparison. A fresh idle-machine run is owed before any MuJoCo-vs-PyBullet rate claim.
- **The Stage 2 gate study is in progress / has not yet produced a committed result.** Expect it to take on the
  order of an hour on a laptop CPU across the three patients and both materials — most of the wall time is the
  flex solver settling at large (10–15 N) loads, where SVK in particular has shown a growing number of inverted
  elements as compression increases on some cases. **Do not assume the gate passes.** Check
  `results/mujoco_flex_fidelity_*.md` and the corresponding `DECISIONS.md` entry for the actual outcome before
  using `--tissue-backend mujoco-flex` for anything beyond a visual demonstration.
- **No recorded demonstration is committed yet.** `demonstrations/mujoco_flex/` is generated by
  `scripts/demo_mujoco_flex_slicer.sh` on demand; it is not currently checked in.

---

## Limitations

- **This is a side track off `audit-fixes`, not merged to `main`.** `--physics-backend pybullet` and
  `--tissue-backend fem` remain the defaults everywhere; none of the RSS 2027 results on `main` use anything from
  this branch.
- **The MuJoCo robot's compliance is real but unused downstream.** Even with torque control, the measured
  deflection under tissue force is not fed back into the tissue solve, the B-mode simulation or the
  reconstruction — the "kinematic robot" caveat from DEC-048 stands for every reported result regardless of
  physics backend (DEC-052).
- **Fixture concavity is approximated, not exact.** The per-triangle-prism scheme (4 mm thick) follows the
  surface far better than a single convex hull, but it is still a shell of finite thickness, not the exact
  mesh PyBullet collides with. Fixtures over 2,000 triangles are refused outright.
- **Stage 2 is a dynamic, lagging tissue model, not a drop-in FEM replacement.** It reports state from a live
  subprocess that settles asynchronously; its force/indentation pairing is self-consistent (both read from the
  same worker state) but is not quasi-static like the FEM, and nothing about its accuracy is established until
  the DEC-053 gate entry says so.
- **No automated test runs the Slicer side of the flex demo inside Slicer itself** — consistent with the
  existing limitation on `main` (the Slicer bridge has no automated test inside Slicer at all). The demo script's
  correctness is checked by eye against the generated GIFs/CSVs, not by a test suite.
- **The gains and thresholds (10 Hz control bandwidth, 5 mrad following error, the flex gate's 10 % / cost
  bounds) are chosen, not identified from hardware or clinical requirements.**

---

## Where to look next

- [`DECISIONS.md`](DECISIONS.md) — `DEC-052` (Stage 1, with two review addenda) and `DEC-053` (the Stage 2 gate,
  pre-registered) are the primary source of truth for what was decided, why, and what was found. Read these
  before `tasks/mujoco_plan.md`, which is the original plan and may be stale where DECISIONS.md supersedes it.
- [`FLOW.md`](FLOW.md) — recipes 13 and 14 are the copy-pasteable commands for everything in this file, plus the
  coordinate-frame and contact-snapshot-field notes specific to the MuJoCo backends.
- [`AGENTS.md`](AGENTS.md) — the `.venv-mj4` exception to invariant 1, and the updated `CONTACT_FIELDS`/launch
  command references.
- [`ReconstructionPipeline/tests/test_physics_backends.py`](ReconstructionPipeline/tests/test_physics_backends.py),
  [`test_mujoco_flex.py`](ReconstructionPipeline/tests/test_mujoco_flex.py),
  [`test_coordinates.py`](ReconstructionPipeline/tests/test_coordinates.py) — the contract tests referenced
  throughout this file.
