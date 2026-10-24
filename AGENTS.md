# Project: MTP - Surgical Physics Bridge & Autonomous Robotic Ultrasound

## Tech Stack & Architecture
- **Language & Runtime**: Python 3.10 (`./.venv/bin/python`), C++17 (3D Slicer Loadable Module)
- **Physics Core**: PyBullet (1000 Hz decoupled contact dynamics, soft tissue, virtual fixtures)
- **Clinical Frontend**: 3D Slicer (v5.0+, OpenIGTLink TCP binary sockets via `pyigtl` and `vtkMRMLIGTLConnectorNode`)
- **Simulation & Perception**: PyTorch (GPU grid sampling, ray acoustics, confidence mapping, PointNet encoder)
- **Learning & Planning**: Submodular MDP, PPO, Continuous Flow Matching (CFM), Karras EDM 3D Diffusion Policy
- **Registration & Geometry**: Open3D (ICP registration, point cloud downsampling), NiBabel, Trimesh

## Standard Commands
- **Run Tests**: `./.venv/bin/python -m pytest ReconstructionPipeline/tests/ -v`
- **Run 6-Policy Benchmark**: `./.venv/bin/python ReconstructionPipeline/evaluation/benchmark_runner.py --num-trials 3`
- **Generate / Test Patient Assets**: `./.venv/bin/python -m ReconstructionPipeline.data.prepare_patient_assets`
- **Launch Physics Server**: `./.venv/bin/python surgical_physics_bridge/src/server.py --urdf <path_to_robot.urdf>`
- **Train Submodular PPO**: `./.venv/bin/python ReconstructionPipeline/planning/train_submodular_ppo.py`
- **Train Diffusion Policy**: `./.venv/bin/python ReconstructionPipeline/planning/train_diffusion_policy.py`

## Invariants & Critical Boundaries
1. **Python Environment**: ALWAYS use `./.venv/bin/python`. Never use global `/usr/bin/python` which lacks PyTorch and pyigtl dependencies.
2. **Clock Decoupling**: Physics MUST step at 1000 Hz decoupled from visualization (30–60 FPS). 3D Slicer UI stutter must never destabilize contact physics.
3. **In-Place Buffer Mutation**: For deformable meshes and reconstructed point clouds, update pre-allocated VTK C++ memory buffers directly to prevent memory leaks and UI lag.
4. **Coordinate Frames**: Convert between PyBullet world (meters, Z-up) and 3D Slicer (millimeters, RAS space) using validated transformation utilities (`surgical_physics_bridge/src/coordinates.py`).
5. **Acoustic Simulation**: Ultrasound B-mode generation is model-based ray tracing (attenuation, impedance reflection, Rayleigh speckle, shadow factor), executing in $< 2$ ms on GPU.
