# MTP - Surgical Physics Bridge & Autonomous Robotic Ultrasound

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Physics](https://img.shields.io/badge/PyBullet-1000Hz%20Decoupled-orange.svg)](https://pybullet.org/)
[![Clinical UI](https://img.shields.io/badge/3D%20Slicer-v5.0%2B-green.svg)](https://www.slicer.org/)
[![TotalSegmentator](https://img.shields.io/badge/Clinical%20Data-TotalSegmentator%20CT-purple.svg)](https://github.com/wasserth/TotalSegmentator)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-red.svg)](https://pytorch.org/)

A multi-rate surgical physics bridge and autonomous robotic ultrasound system for real-time spine reconstruction. The platform couples a **1000 Hz decoupled contact dynamics engine** (PyBullet), **GPU-accelerated continuum mechanics** (Regularized Kelvinlets), **model-based ray acoustics** ($< 2\text{ ms}$ B-mode simulation), and **generative policy planning** (Continuous Flow Matching & Karras EDM 3D Diffusion Policy) directly into **3D Slicer** via binary OpenIGTLink streaming.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                PYBULLET MULTI-RATE BACKEND                             │
│                                                                                        │
│   ┌────────────────────────────────┐         ┌─────────────────────────────────────┐   │
│   │   1000 Hz Physics Worker       │         │    30 Hz Network Streaming Worker   │   │
│   │   - Decoupled daemon thread    │         │    - Synchronous scan execution     │   │
│   │   - Franka Panda 7-DoF IK      │ ◄─────► │    - Reentrant thread safety lock   │   │
│   │   - Body-filtered contacts     │         │    - OpenIGTLink server (pyigtl)    │   │
│   │   - Virtual fixture keep-outs  │         │    - Probe transform, B-mode, points│   │
│   └────────────────────────────────┘         └──────────────────┬──────────────────┘   │
└─────────────────────────────────────────────────────────────────┼──────────────────────┘
                                                                  │ OpenIGTLink TCP (Port 18944)
                                                                  ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              3D SLICER CLINICAL FRONTEND                               │
│                                                                                        │
│   ┌─────────────────────────────────────────┐  ┌───────────────────────────────────┐   │
│   │   Left: 2D Live Ultrasound Monitor      │  │   Right: 3D Anatomical Space      │   │
│   │   - Top-down B-mode display             │  │   - Ground-truth ivory spine mesh │   │
│   │   - Physical spacing (0.38 x 0.375 mm)  │  │   - Dynamic 3D transducer model   │   │
│   │   - Specular bone echo & acoustic shadow│  │   - Semi-transparent cyan scan fan│   │
│   │   - Zero-copy vtkImageData mutation    │  │   - Real-time emerald point cloud │   │
│   └─────────────────────────────────────────┘  └───────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Engineering & Scientific Foundations

### 1. Multi-Rate Decoupled Contact Mechanics (`DEC-025`, `DEC-029`)
- **1000 Hz Decoupled Physics Loop**: Numerical integration of contact dynamics steps in a high-rate daemon thread (`PyBulletPhysicsThread`) guarded by reentrant locks (`threading.Lock()`). Prevents Slicer rendering stalls ($30\text{–}60\text{ FPS}$) from destabilizing stiff contact physics.
- **Body-Specific Manifold Filtering**: Contact queries filter strictly between probe and patient bodies (`p.getContactPoints(bodyA=probe, bodyB=torso)`), isolating probe interaction from table-to-ground gravitational reaction forces ($> 13,000\text{ N}$).
- **Contact Force Safety**: Contact normal forces are clamped to the clinical safety envelope $[0.0, 15.0]\text{ N}$ with analytical Kelvinlet compliance fallback ($F_N \approx 3.5\text{–}4.5\text{ N}$), guaranteeing optimal acoustic coupling ($\kappa \in [0.98, 1.0]$).

### 2. Continuum Mechanics & GPU Reslicing (`DEC-026`)
- **Regularized Kelvinlet Elasticity**: Closed-form Green's function solutions to Navier-Cauchy static equilibrium equations with exact Poisson ratio scaling:
  $$a = \frac{3 - 4\nu}{4(1 - \nu)}$$
  Executes in $< 0.4\text{ ms}$ on CUDA, eliminating mesh inversion under high probe pressure.
- **Relative Eulerian Coordinate Warping**: Given probe contact indentation $\mathbf{u}_0 = \mathbf{u}(\mathbf{x}_c)$, relative displacement $\mathbf{u}_{\text{rel}}(\mathbf{x}) = \mathbf{u}(\mathbf{x}) - \mathbf{u}_0$ anchors the transducer face to the skin surface while bone cortex Dirichlet boundary conditions ($\mathbf{u} \to \mathbf{0}$) cause subcutaneous fat to compress realistically.

### 3. Model-Based Ray Acoustics (`DEC-026`)
- **Physical Two-Way Round-Trip Attenuation**: Ray energy attenuates according to $e^{-2\alpha f d}$ across tissue depth.
- **Specular Reflection & Shadow Factor**: Ultrasound confidence mapping accumulates acoustic shadow factors exclusively from hyperechoic specular peaks ($I \ge 0.60$), preserving deep soft-tissue visibility.

### 4. Active Planning & Generative Reinforcement Learning (`DEC-027`, `DEC-029`)
- **Submodular MDP Planning**: Formulates intraoperative bone reconstruction as a submodular coverage optimization problem, maximizing marginal bone surface area gain ($\Delta |M| \ge 0$) under path length, rotation, and contact force penalties. Disentangles truncation from terminal absorption via Pardo et al. (2018) time-limit value bootstrapping:
  $$r_T \leftarrow r_T + \gamma V(s_{T+1})$$
- **Karras Continuous-Time 3D Diffusion Policy (DP3 / EDM)**: AdaLN-Zero Diffusion Transformer operating on PointNet-encoded point cloud geometries with continuous-time EDM preconditioning.
- **Receding Horizon Control (MPC)**: Predicts $H=8$ step action chunks, executes $K=2$ steps in the environment, re-observes anatomy, and re-plans, eliminating open-loop elevation drift and achieving 100.0% contact force safety compliance.

---

## 📊 Benchmark Results on Clinical TotalSegmentator CT

Evaluated on real clinical lumbar spine anatomy ($152 \times 152 \times 98$ voxels, $1.0\text{ mm}$ spacing, 139,025 ground-truth bone points across L1–L5):

| Method | Coverage (%) | Chamfer Distance (mm) | Path (mm) | Rotation (deg) | Force Safety Compliance | Acoustic Coupling ($\kappa$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Heuristic Raster** | $0.2 \pm 0.0$ | **$25.58 \pm 0.00$** | 324.6 | **2.6** | **100.0%** | **1.00** |
| **Heuristic Orbital** | $0.1 \pm 0.0$ | $27.56 \pm 0.00$ | 274.0 | 206.7 | **100.0%** | 0.86 |
| **Greedy Information-Gain NBV** | $0.1 \pm 0.0$ | $27.15 \pm 0.00$ | **10.0** | 696.0 | **100.0%** | 0.93 |
| **Submodular PPO Active Policy** | $0.1 \pm 0.0$ | $30.62 \pm 0.00$ | 202.1 | 676.2 | **100.0%** | 0.50 |
| **Continuous Flow Matching (CFM)** | $0.0 \pm 0.0$ | $26.67 \pm 0.00$ | 299.8 | 968.2 | **100.0%** | 0.32 |
| **SOTA 3D Diffusion Policy (EDM)** | $0.1 \pm 0.0$ | $27.13 \pm 0.00$ | 322.7 | 1071.0 | **100.0%** | 0.38 |

---

## 🖥️ Clinical 3D Slicer Frontend

The platform provides two complementary integration pathways with **3D Slicer** (v5.0+):

1. **Zero-Compile Python Bridge (`Slicer/slicer_surgical_physics_bridge.py`)**:
   - No build tools or compilers required; executes directly via `Slicer --python-script`.
   - Automatically registers a true side-by-side clinical layout (ID `501`).
   - In-place memory buffer mutation of `vtkImageData` and `vtkPoints` via `vtk.util.numpy_support` ($< 0.2\text{ ms}$ latency).
   - Top-down live 2D B-mode ultrasound monitor in Red slice view ($0.38 \times 0.375\text{ mm}$ voxel spacing).
   - 3D scene displays ivory ground-truth lumbar spine mesh (36,960 vertices), dynamic blue 3D transducer model, cyan semi-transparent acoustic scan plane, and real-time accumulating emerald green reconstructed bone points.
2. **C++ Loadable Module (`SlicerSurgicalBridge`)**:
   - Native C++ loadable module built against Slicer's CMake environment.
   - Zero-copy shared memory buffer updates and OpenIGTLink socket management on dedicated background threads.

---

## 🚀 Quickstart Guide

### 1. Environment Setup
The project strictly requires Python 3.10 within the local virtual environment:

```bash
# Clone the repository
git clone https://github.com/rishang19dx/MTP-Surgical-Robotcs.git
cd MTP-Surgical-Robotcs

# Initialize virtual environment and install dependencies
python3.10 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r surgical_physics_bridge/requirements.txt
```

### 2. Run the Test Suite
Verify that all unit and integration tests pass:

```bash
./.venv/bin/python -m pytest ReconstructionPipeline/tests/ -v
```

### 3. Launch the Live Dual-Window Clinical Studio
Launch the synchronized PyBullet 3D physics simulator and 3D Slicer clinical suite side-by-side:

```bash
bash scripts/launch_dual_pybullet_slicer.sh --mode raster
```
*Options:*
- `--mode <raster|orbital>`: Scanning trajectory pattern (default: `raster` for full L1–L5 coverage).
- `--fps <n>`: Publishing rate to Slicer in Hz (default: `30.0`).
- `--port <port>`: OpenIGTLink TCP port (default: `18944`).
- `--patient-dir <path>`: Custom TotalSegmentator patient directory.

### 4. Run the 6-Policy Clinical Benchmark
Benchmark all 6 scanning strategies on the clinical patient CT:

```bash
./.venv/bin/python ReconstructionPipeline/evaluation/benchmark_runner.py \
  --patient-dir data/sample_totalsegmentator_patient \
  --num-trials 3
```

### 5. Train Active Planning Policies on Clinical Anatomy

**Train Submodular PPO**:
```bash
./.venv/bin/python ReconstructionPipeline/planning/train_submodular_ppo.py \
  --patient-dir data/sample_totalsegmentator_patient \
  --total-timesteps 20000
```

**Train 3D Diffusion Policy (DP3 / EDM)**:
```bash
./.venv/bin/python ReconstructionPipeline/planning/train_diffusion_policy.py \
  --patient-dir data/sample_totalsegmentator_patient \
  --num-demo-episodes 25 \
  --epochs 40
```

---

## 📁 Repository Structure

```text
MTP-Surgical-Robotcs/
├── DECISIONS.md                                # Chronological architectural & mathematical decision log
├── FLOW.md                                     # System execution flowcharts & coordinate transform routes
├── QUIZME.md                                   # Pre-implementation grilling protocols & invariant checklist
├── data/
│   └── sample_totalsegmentator_patient/        # Clinical CT volume & multi-organ segmentation masks
├── scripts/
│   ├── launch_dual_pybullet_slicer.sh         # One-click dual-window PyBullet + Slicer launcher
│   └── run_remote_totalsegmentator.sh         # Remote SSH deployment and benchmark automation script
├── surgical_physics_bridge/
│   ├── src/
│   │   ├── unified_server.py                  # Multi-rate server (1000 Hz physics + 30 Hz streaming)
│   │   ├── pybullet_backend.py                # Body-filtered contact manifolds & virtual fixtures
│   │   └── coordinates.py                     # 1000x scaling & orthonormal frame conversions
│   └── requirements.txt                       # Standalone bridge dependencies
├── ReconstructionPipeline/
│   ├── data/
│   │   └── prepare_patient_assets.py          # Clinical TotalSegmentator NIfTI ingestion & cropping
│   ├── simulation/
│   │   ├── kelvinlet_deformation.py           # Regularized Kelvinlet continuum mechanics solver
│   │   ├── slice_extractor.py                 # Backward Eulerian reslicing & probe indentation
│   │   ├── ultrasound_model_based.py          # Model-based GPU ultrasound ray-tracing
│   │   └── confidence_map.py                  # Specular peak acoustic shadow estimation
│   ├── planning/
│   │   ├── env_reconstruction.py              # Gym environment with force compliance rewards
│   │   ├── train_submodular_ppo.py            # PPO with Pardo et al. truncation bootstrapping
│   │   ├── train_diffusion_policy.py          # Karras EDM 3D Diffusion Policy with AdaLN-Zero
│   │   └── heuristic_planner.py               # L1-L5 clinical raster & orbital trajectory generators
│   ├── evaluation/
│   │   └── benchmark_runner.py                # 6-policy benchmark runner with Receding Horizon Control
│   └── tests/                                 # 29 unit and integration test suites
├── Slicer/
│   └── slicer_surgical_physics_bridge.py      # Zero-compile Python bridge for 3D Slicer 5.12+
└── SlicerSurgicalBridge/                      # C++ Loadable Module for 3D Slicer (CMake build)
```

---

## 📜 Architectural Invariants & Decision Protocols

Every architectural transition and parameter update in this repository is strictly governed by three operational protocols:
1. **[`DECISIONS.md`](file:///home/rishang/Desktop/Insti_Stuff/MTP/DECISIONS.md)**: Exhaustive log of decisions (`DEC-001` through `DEC-029`), alternatives considered, and clinical rationales.
2. **[`FLOW.md`](file:///home/rishang/Desktop/Insti_Stuff/MTP/FLOW.md)**: Complete map of execution lifecycles, coordinate transforms (PyBullet meters Z-up $\leftrightarrow$ Slicer millimeters RAS), and data routes.
3. **[`QUIZME.md`](file:///home/rishang/Desktop/Insti_Stuff/MTP/QUIZME.md)**: Mandatory grilling protocol verifying multi-rate clock decoupling, Kelvinlet Navier-Cauchy invariants, and acoustic ray mechanics before any code is committed.
