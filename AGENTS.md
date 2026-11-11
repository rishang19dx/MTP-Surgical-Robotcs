# Project: MTP - Surgical Physics Bridge & Autonomous Robotic Ultrasound

## Tech Stack & Architecture
- **Language & Runtime**: Python 3.10 (`./.venv/bin/python`). There is no compiled module; the 3D Slicer side is a Python script (`Slicer/slicer_surgical_physics_bridge.py`).
- **Physics Core**: PyBullet (default) or MuJoCo 3.17 (`--physics-backend mujoco`, DEC-052); the shared loop is in `physics_common.py`. The 1 kHz loop runs in its own process (robot IK/dynamics, rigid contacts, virtual fixtures) and applies the tissue force `F = F0 + K_c(δ − δ0)`. An optional GUI viewer runs as a separate process.
- **Soft Tissue**: Patient-specific precomputed linear tetrahedral FEM on the GPU (`simulation/fem_tissue.py`, default). The regularised Kelvinlet model (`simulation/tissue_model.py`) is the alternative.
- **Clinical Frontend**: 3D Slicer 5.x via OpenIGTLink (`pyigtl` on both sides, port 18944).
- **Simulation & Perception**: PyTorch (grid_sample reslicing with backward warping, model-based B-mode, image-only confidence, B-mode bone segmentation, PointNet encoder with LayerNorm).
- **Learning & Planning**: Submodular PPO, Continuous Flow Matching (CFM), Karras EDM diffusion policy, heuristic raster/orbital, greedy NBV.
- **Geometry**: Open3D, NiBabel, Trimesh, SciPy (cKDTree metrics).
- **Dependencies**: one top-level `requirements.txt` (read by `setup.py`).

## Standard Commands
- **Run Tests**: `./.venv/bin/python -m pytest ReconstructionPipeline/tests/ -v`
- **Run 6-Policy Benchmark**: `./.venv/bin/python ReconstructionPipeline/evaluation/benchmark_runner.py --patient-dir data/sample_totalsegmentator_patient --num-trials 3 --policies raster,orbital,nbv,ppo,cfm,diffusion --soft-body`. Without `--patient-dir` the runner uses a synthetic phantom. A missing checkpoint for a requested learned policy is an error; use `--heuristics-only` if no checkpoints exist.
- **Generate Patient Assets**: `./.venv/bin/python -m ReconstructionPipeline.data.prepare_patient_assets --export-totalseg data/sample_totalsegmentator_patient`. Without arguments it only builds the synthetic phantom smoke check.
- **Launch Server**: `./.venv/bin/python surgical_physics_bridge/src/unified_server.py --mode raster_l4 --fps 30 [--gui] [--tissue-backend fem|kelvinlet] [--physics-backend pybullet|mujoco] [--urdf <robot.urdf>]`
- **Measure Physics Rate**: `./.venv/bin/python surgical_physics_bridge/src/unified_server.py --benchmark-seconds 20 [--physics-backend pybullet|mujoco]`
- **Slicer Bridge**: `Slicer --python-script Slicer/slicer_surgical_physics_bridge.py`. To start the server and Slicer together: `bash scripts/launch_dual_pybullet_slicer.sh`.
- **Train Submodular PPO**: `./.venv/bin/python ReconstructionPipeline/planning/train_submodular_ppo.py` (`--dry-run` for a smoke test)
- **Train CFM + Diffusion Policy**: `./.venv/bin/python ReconstructionPipeline/planning/train_diffusion_policy.py`

## Invariants & Critical Boundaries
1. **Python Environment**: ALWAYS use `./.venv/bin/python` (Python 3.10). Never use the global `/usr/bin/python`, which lacks PyTorch and pyigtl.
   - One exception (DEC-052): the MuJoCo flex tissue process and its study run in `./.venv-mj4` (Python 3.11, MuJoCo 4.x), because MuJoCo 4 does not support Python 3.10. Nothing else may use it.
2. **Clock Decoupling**: Physics MUST step at 1000 Hz, decoupled from streaming (30 Hz) and visualization.
   - Only the physics loop calls the rigid-body engine (PyBullet or MuJoCo). The streaming side never imports either (tests check this).
   - The physics loop never waits on the FEM solver.
   - 3D Slicer or GUI stutter must never destabilize contact physics.
   - Measure the rate with `--benchmark-seconds` after any change to the loop.
3. **In-Place Buffer Mutation**: For deformable meshes and reconstructed point clouds in Slicer, update pre-allocated VTK buffers in place: bind a numpy array once with `numpy_to_vtk(deep=False)`, write into it, then call `Modified()`. Do not create new nodes or arrays per frame.
4. **Coordinate Frames**: Frame P (simulation, mm) is **LPS** (+X left, +Y posterior, +Z superior), even though the NIfTI header and the `*_ras.obj` names say RAS.
   - P → Slicer RAS is `P_TO_RAS = diag(−1, −1, 1)`, applied exactly once by the `PatientToRAS` transform node.
   - P → PyBullet is `patient_to_pybullet_points` (m, Z-up).
   - Use only the helpers in `surgical_physics_bridge/src/coordinates.py` and keep `tests/test_coordinates.py` green.
5. **Acoustic Simulation**: B-mode generation is model-based:
   - attenuation in dB→Np over the physical pixel spacing,
   - interface transmission and specular reflection,
   - Rayleigh speckle,
   - a fixed TGC, independent of the tissue.

   Target: < 2 ms per frame on GPU. This is a design budget that no test asserts; measure it if you change the simulator. Perception must use the B-mode image only. The simulator's `energy`, `effective_energy` and `bone_mask` outputs and `segment_from_label_map` are privileged or oracle data and must not feed the reconstruction.
6. **Metrics Must Be Able to Fail; Numbers Come from `results/`**:
   - No metric may be true by construction. Examples: a constant force, hardcoded indentation, or ground-truth leakage.
   - Every number in README, FLOW, DECISIONS or Docs must come from `results/` (benchmark JSON/markdown, physics-rate JSON), a test, or a docstring that names its source. Name the results file next to each table.
   - Regenerate results with the scripts. Do not edit `results/benchmark_latest.md` by hand.
   - Use no marketing language ("SOTA", "clinical-grade", "validated") unless a result shows it. State limitations plainly.

## Mandatory Agent Operating Files
Every agent working on this codebase MUST adhere to and maintain these files:
1. **[`DECISIONS.md`](DECISIONS.md)**: chronological log of architectural, research and algorithmic decisions. Record every new architectural choice, parameter shift or major design change with context, alternatives and rationale. Mark old entries "Superseded" or add errata instead of rewriting them.
2. **[`FLOW.md`](FLOW.md)**: map of execution flows, entry points, data routes, coordinate frames and verification recipes. Consult it before modifying entry points or data routes, and update it afterwards.
3. **[`QUIZME.md`](QUIZME.md)**: pre-implementation grilling protocol. When the user requests a major architectural change, refactoring or new pipeline capability, halt and grill the user on invariants, physics and trade-offs before writing code.

## Local Scratch
`.agents/` is local agent scratch (audit notes, logs, temporary scripts and runs). It is gitignored and may contain credentials. Never commit it, and never cite it as the source of a documented number.
