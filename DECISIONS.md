# DECISIONS.md - Architectural & Research Decision Log

A chronological record of all architectural, research, algorithmic, and engineering decisions for the **Surgical Physics Bridge & Autonomous Robotic Ultrasound (MTP)** project.

---

## Chronological Index

| Date | ID | Title | Status |
| :--- | :--- | :--- | :--- |
| **Aug 2025** | `DEC-001` | Selection of OpenIGTLink over ROS / ROS2 Middleware | **Accepted** |
| **Aug 2025** | `DEC-002` | Selection of PyBullet for 1000 Hz Contact Physics | **Accepted** |
| **Aug 2025** | `DEC-003` | 3D Slicer C++ Loadable Module with In-Place Memory Mutation | **Accepted** |
| **Sep 2025** | `DEC-004` | Model-Based GPU Ray Acoustics over 2D GAN Texturing | **Accepted** |
| **Sep 2025** | `DEC-005` | Mathematical Coordinate Transformation (PyBullet to Slicer RAS) | **Accepted** |
| **Sep 2025** | `DEC-006` | Native Virtual Keep-Out Fixtures via PyBullet GJK/EPA | **Accepted** |
| **Oct 2025** | `DEC-007` | Submodular MDP Formulation for Active View Selection | **Accepted** |
| **Oct 2025** | `DEC-008` | PointNet 3D Geometric State Representation over 3D CNNs | **Accepted** |
| **Oct 2025** | `DEC-009` | 6-Policy Comparative Benchmarking Suite | **Accepted** |
| **Oct 2026** | `DEC-010` | Lateral Flank Scanning Approach (Patient Lying Down) | **Accepted** |
| **Oct 2026** | `DEC-011` | Pure 2D B-Mode Ultrasound Signal Surface Extraction | **Accepted** |
| **Oct 2026** | `DEC-012` | Interactive Slicer Fiducial Markups to Closed-Loop IK | **Accepted** |
| **Oct 2026** | `DEC-013` | Research Direction: SonoGym Extension with Soft Body Deformations | **Accepted** |
| **Oct 2026** | `DEC-014` | Deformable Slicing via Backward Eulerian Coordinate Warping | **Accepted** |
| **Oct 2026** | `DEC-015` | Regularized Kelvinlets as Fast Differentiable Elasticity Prior | **Accepted** |
| **Oct 2026** | `DEC-016` | Relative Eulerian Warping with Probe Indentation Compensation | **Accepted** |
| **Oct 2026** | `DEC-017` | Evaluation of SOFA Framework vs Regularized Kelvinlets for Real-Time Soft Body Ultrasound | **Accepted** |
| **Oct 2026** | `DEC-018` | Remote SSH Deployment & Ingestion Protocol for Real TotalSegmentator Clinical Data | **Accepted** |

---

## Detailed Decision Records

### `DEC-001`: Selection of OpenIGTLink over ROS / ROS2 Middleware
* **Date**: August 2025
* **Context**: The simulation platform requires high-speed bidirectional communication between the physics simulation engine and 3D Slicer.
* **Alternatives Considered**:
  1. *ROS / ROS2*: De facto robotics standard; requires multi-node daemon setup, IPC serialization, and complex environment management on clinical PCs.
  2. *gRPC / Protocol Buffers*: Fast binary streaming, but lacks native 3D Slicer viewer support.
  3. *OpenIGTLink (Selected)*: Native medical device protocol supported directly in 3D Slicer (`OpenIGTLinkIF`).
* **Decision**: Adopt OpenIGTLink via `pyigtl` on Python server and `vtkMRMLIGTLConnectorNode` in C++ Slicer module over raw TCP sockets (port 18944).
* **Rationale**: Eliminates middleware overhead; supports raw binary multidimensional arrays (`NDArrayMessage`) with zero-copy deserialization.

---

### `DEC-002`: Selection of PyBullet for 1000 Hz Contact Physics
* **Date**: August 2025
* **Context**: Need a physically grounded contact engine to simulate probe-tissue interactions and normal force compliance ($F_N$).
* **Alternatives Considered**:
  1. *NVIDIA Isaac Sim (PhysX)*: GPU parallel, but high workstation overhead and difficult headless C++ bridging to 3D Slicer.
  2. *SOFA Framework*: High clinical biomechanics, but slow real-time multi-body IK and robotics control.
  3. *PyBullet (Selected)*: 1000 Hz stable numerical integration, URDF tree parsing, built-in GJK/EPA collision hulls.
* **Decision**: Run PyBullet in a decoupled background loop at 1000 Hz ($dt = 0.001\text{ s}$), publishing to Slicer at 30–60 FPS.
* **Rationale**: Decoupled timing ensures that 3D Slicer visual rendering stalls never destabilize physical contact mechanics.

---

### `DEC-003`: 3D Slicer C++ Loadable Module with In-Place Memory Mutation
* **Date**: August 2025
* **Context**: High-frequency streaming of moving soft tissue vertices and reconstructed bone point clouds into 3D Slicer.
* **Alternatives Considered**:
  1. *Re-instantiating `vtkMRMLModelNode` every frame*: Destroys and recreates scene nodes; triggers memory allocation churn and UI freeze.
  2. *In-Place Memory Buffer Mutation (Selected)*: Reaching directly into `vtkPoints->GetData()` and calling `SetComponent()`.
* **Decision**: Implement `SlicerSurgicalBridge` as a C++ Loadable Module utilizing in-place VTK buffer mutation.
* **Rationale**: Achieves smooth 60 FPS visual rendering with zero memory leaks and zero garbage collection spikes.

---

### `DEC-004`: Model-Based GPU Ray Acoustics over 2D GAN Texturing
* **Date**: September 2025
* **Context**: Generating synthetic ultrasound images from CT volume data in real-time ($< 5\text{ ms}$).
* **Alternatives Considered**:
  1. *2D Generative GAN (SonoGym baseline pix2pix)*: Generates plausible speckle texture, but hallucinates boundaries, lacks physical ray attenuation, cannot model off-axis bone shadows, and requires retraining per patient.
  2. *Model-Based GPU Ray Acoustics (Selected)*: Physical depth attenuation $E(u) = E_0 \exp(-f \int \alpha dv)$, specular impedance reflection $R(u)$, and Rayleigh speckle $B(u)$ using PyTorch CUDA grid sampling.
* **Decision**: Implement GPU ray-acoustic simulator executing in $< 2\text{ ms}$ on GPU.
* **Rationale**: Zero-shot generalizability across any patient CT volume; deterministic acoustic shadows behind bone interfaces.

---

### `DEC-005`: Mathematical Coordinate Transformation (PyBullet to Slicer RAS)
* **Date**: September 2025
* **Context**: PyBullet uses a Z-up, X-forward, Y-left metric coordinate system (meters). 3D Slicer uses Right-Anterior-Superior (RAS) in millimeters.
* **Alternatives Considered**:
  1. *Ad-hoc axis swapping in GUI code*: Fragile, bug-prone.
  2. *Rigid Homogeneous Transformation Matrix (Selected)*: Explicit $4 \times 4$ transformation matrix $T_{\text{conv}}$:
     $$T_{\text{conv}} = \begin{bmatrix} 0 & 1000 & 0 & 0 \\ 1000 & 0 & 0 & 0 \\ 0 & 0 & 1000 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
* **Decision**: Centralize all coordinate conversions in `surgical_physics_bridge/src/coordinates.py`.
* **Rationale**: Mathematically rigorous spatial consistency between PyBullet kinematics and Slicer MRML displays.

---

### `DEC-006`: Native Virtual Keep-Out Fixtures via PyBullet GJK/EPA
* **Date**: September 2025
* **Context**: Clinical safety requires preventing robotic tools from breaching critical structures (e.g. spinal canal, nerves).
* **Alternatives Considered**:
  1. *Distance margin checking in Slicer UI*: Open-loop, cannot prevent kinematic penetration.
  2. *PyBullet GJK/EPA Collision Barrier (Selected)*: Clinician delineates keep-out mesh in Slicer; geometry is sent via JSON to PyBullet to instantiate rigid collision objects.
* **Decision**: Enforce virtual fixtures directly within the 1000 Hz physics solver.
* **Rationale**: Guarantees physical repulsion forces and prevents tool breach at the dynamics level.

---

### `DEC-007`: Submodular MDP Formulation for Active View Selection
* **Date**: October 2025
* **Context**: Designing an autonomous trajectory planner to eliminate acoustic shadows around complex lumbar vertebra anatomy.
* **Alternatives Considered**:
  1. *Open-loop Raster Sweeps*: Blind zigzags, leaves lateral facets in permanent acoustic shadow.
  2. *Greedy Next-Best-View (1-step)*: Gets trapped in flat local optima.
  3. *Submodular MDP (Selected)*: Monotone submodular reward based on greedy marginal surface gain with diminishing returns:
     $$r_t = \Delta|M_{t+1}| - w_{\text{path}} \|\Delta \mathbf{x}\| - w_{\text{rot}} |\Delta \theta|$$
* **Decision**: Train PPO on a submodular diminishing-returns reward.
* **Rationale**: Mathematically bounds coverage efficiency; prevents redundant re-scanning of already mapped voxels.

---

### `DEC-008`: PointNet 3D Geometric State Representation over 3D CNNs
* **Date**: October 2025
* **Context**: Encoding accumulated 3D bone point clouds into a state representation for reinforcement learning.
* **Alternatives Considered**:
  1. *3D Voxel Grid + 3D CNN ($40 \times 40 \times 40$)*: Memory-heavy, slow 3D convolutions on GPU.
  2. *PointNet Permutation-Invariant Encoder (Selected)*: Processes downsampled $N \times 3$ point sets via shared MLPs and max-pooling to 128-dim latent vector.
* **Decision**: Implement lightweight PointNet encoder (`ReconstructionPipeline/planning/pointnet_encoder.py`).
* **Rationale**: Sub-millisecond forward pass; eliminates voxel quantization artifacts and scales smoothly with point cloud density.

---

### `DEC-009`: 6-Policy Comparative Benchmarking Suite
* **Date**: October 2025
* **Context**: Scientifically evaluating autonomous scanning strategies.
* **Decision**: Benchmark 6 distinct techniques under identical anatomical conditions:
  1. Linear Raster Sweep (Clinical baseline)
  2. Orbital Facet Sweep (Heuristic baseline)
  3. Greedy NBV (Information-theoretic baseline)
  4. Submodular PPO (Deep RL with PointNet)
  5. Continuous Flow Matching (CFM with 8-step action chunking)
  6. SOTA 3D Diffusion Policy (DP3 / Karras EDM with AdaLN DiT)
* **Outcome**: CFM achieved highest bone coverage ($11.2 \pm 3.5\%$) and lowest Chamfer distance ($5.51\text{ mm}$); Submodular PPO achieved shortest path length ($202.7\text{ mm}$).

---

### `DEC-010`: Lateral Flank Scanning Approach (Patient Lying Down)
* **Date**: October 23, 2026
* **Context**: User noted that clinical spine ultrasound procedures (e.g., LLIF, facet injections) are performed with the patient lying down horizontally, not standing or being viewed from the dorsal top.
* **Decision**: 
  - Patient lies horizontally on the operating bed with spine aligned along Z.
  - Probe is positioned at the right lateral flank ($X_{\text{skin}} \approx 48\text{ mm}$) pointing inward along $-X$ into the body.
  - Normal vector: $\vec{n} = [-1, 0, 0]^\top$. Rotation matrix $\mathbf{R}_{W \leftarrow U} \in SO(3)$ with $\det = +1.0$.
* **Rationale**: Replicates genuine clinical operating room geometry and acoustic windows into lateral vertebra facets.

---

### `DEC-011`: Pure 2D B-Mode Ultrasound Signal Surface Extraction
* **Date**: October 23, 2026
* **Context**: Previous baseline extracted surface contours from ground-truth CT segmentation slices (`segment_from_label_map`). User required pure ultrasound-based reconstruction.
* **Decision**: 
  - CT volume is strictly used as an acoustic simulation medium for ray-tracing.
  - 3D point cloud accumulation is driven solely by 2D B-mode signal processing:
    1. Hyperechoic specular peak detection: $I(u, v) \ge I_{\text{thresh}}$ and $I(u, v) > I(u, v \pm 1)$.
    2. Distal acoustic shadow verification: $I_{\text{peak}} \ge 1.25 \cdot \bar{I}_{\text{shadow}}$.
    3. Acoustic confidence gating: $C(u, v) \ge 0.35$.
* **Rationale**: Zero ground-truth CT label leakage; realistic clinical ultrasound signal processing.

---

### `DEC-012`: Interactive Slicer Fiducial Markups to Closed-Loop IK
* **Date**: October 23, 2026
* **Context**: Clinical interaction required enabling surgeons to click target regions on the CT in Slicer to guide the robot.
* **Decision**: Added `vtkMRMLMarkupsFiducialNode` listener in `qSlicerSurgicalBridgeModuleWidget.cxx` transmitting `{"roi_target": [x, y, z]}` via `StringMessage: CMD`, resolved on server via Kinova 7-DoF Damped Least Squares IK.
* **Rationale**: Empowers surgeon-in-the-loop target initialization before autonomous active view scanning begins.

---

### `DEC-013`: Research Direction: SonoGym Extension with Soft Body Deformations
* **Date**: October 23, 2026
* **Context**: User initiated new research direction to address SonoGym's primary limitation: rigid patient models with zero tissue deformation under probe contact.
* **Decision**: Formulated the **SonoGym-SoftBody** architecture: separating high-rate GPU soft-body mechanics from deformable ultrasound ray-tracing.
* **Rationale**: Normal contact force ($F_N$) modulates both tissue deformation and ultrasound acoustic coupling; modeling this coupling is essential for sim-to-real transfer.

---

### `DEC-014`: Deformable Slicing via Backward Eulerian Coordinate Warping
* **Date**: October 23, 2026
* **Context**: Computing ultrasound slices through deforming soft tissue in real time ($< 3\text{ ms}$).
* **Alternatives Considered**:
  1. *Tetrahedral re-meshing on every time step*: Computationally prohibitive ($> 50\text{ ms}$).
  2. *Backward Eulerian Coordinate Warping (Selected)*: Given displacement field $\mathbf{u}(\mathbf{x})$, sample undeformed CT at $\mathbf{X}_{\text{ref}} = \mathbf{x} - \mathbf{u}(\mathbf{x})$ using `torch.nn.functional.grid_sample`.
* **Decision**: Adopt backward Eulerian coordinate warping for deformable ultrasound slice extraction.
* **Rationale**: Executes in $< 1.5\text{ ms}$ on GPU; maintains identical tensor interface with model-based ray acoustic shader.

---

### `DEC-015`: Regularized Kelvinlets as Fast Differentiable Elasticity Prior
* **Date**: October 23, 2026
* **Context**: Selecting the deformation engine for fast reinforcement learning training loops (512+ parallel environments).
* **Alternatives Considered**:
  1. *SOFA FEM*: High clinical accuracy, but slow for RL rollouts ($15-35\text{ ms}$ per step).
  2. *PyBullet loadSoftBody*: CPU-bound, tetrahedral inversion issues.
  3. *Regularized Kelvinlets (Selected for DRL)*: Closed-form Green's function solutions to Navier-Cauchy elasticity equations. Executes in $< 0.3\text{ ms}$ on GPU, fully differentiable via PyTorch autograd.
* **Decision**: Use Regularized Kelvinlets for fast RL training; use NVIDIA Warp FEM for high-fidelity IsaacLab validation.
* **Rationale**: Eliminates mesh inversion under high probe pressure while preserving GPU-parallel throughput.

---

### `DEC-016`: Relative Eulerian Warping with Probe Indentation Compensation
* **Date**: October 27, 2026
* **Context**: In robotic ultrasound scanning, the probe maintains continuous contact with the skin surface under varying normal contact force $F_N$. Absolute backward coordinate warping $\mathbf{X}_{\text{ref}} = \mathbf{x} - \mathbf{u}(\mathbf{x})$ causes the skin to detach from the probe face in the probe's local coordinate frame when the probe is placed kinematically at the unindented surface.
* **Alternatives Considered**:
  1. *Dynamic kinematic probe position shifting*: Shift the robot's physical IK target inward by $\delta(F_N)$. Compounded control latency and potential virtual fixture collisions.
  2. *Relative Eulerian Coordinate Warping in Probe Frame (Selected)*: Compute probe face indentation $\mathbf{u}_0 = \mathbf{u}(\mathbf{x}_c)$ and define the relative displacement field $\mathbf{u}_{\text{rel}}(\mathbf{x}) = \mathbf{u}(\mathbf{x}) - \mathbf{u}_0$. The reference coordinate is sampled via $\mathbf{X}_{\text{ref}}(\mathbf{x}) = \mathbf{x} - \mathbf{u}_{\text{rel}}(\mathbf{x}) = \mathbf{x} + (\mathbf{u}_0 - \mathbf{u}(\mathbf{x}))$.
* **Decision**: Implement relative Eulerian coordinate warping in `VolumeSliceExtractor.extract_slices`.
* **Rationale**: 
  - Transducer face ($\mathbf{x} = \mathbf{x}_c$) always samples reference skin surface ($\mathbf{X}_{\text{ref}} = \mathbf{x}_c$), preventing acoustic voids or hovering artifacts.
  - Subcutaneous tissue layers compress realistically, bringing bone reflection peaks physically shallower in B-mode echograms under higher probe pressure.
  - Preserves $< 1.8\text{ ms}$ GPU latency budget.

---

### `DEC-017`: Evaluation of SOFA Framework vs Regularized Kelvinlets for Real-Time Soft Body Ultrasound
* **Date**: October 27, 2026
* **Context**: Investigated the viability of utilizing the Python extension of SOFA (`SofaPython3`) for real-time in-the-loop soft body deformation and compared its performance against Regularized Kelvinlets.
* **Empirical Findings**:
  - *Axial Profile Congruence*: Kelvinlet depth decay matches Co-rotational FEM with $R^2 = 0.9959$ ($> 99.5\%$ correlation) on lumbar patient anatomy.
  - *Solve Latency*: SOFA Co-rotational FEM requires $15 - 40\text{ ms}$ per step on CPU; Regularized Kelvinlets execute in $0.375\text{ ms}$ on CUDA ($> 6,600\times$ faster).
  - *Multi-Instance Parallelism*: SOFA is process-bound and cannot scale to 512+ parallel GPU environments for SonoGym reinforcement learning / diffusion policy training.
  - *Mesh Stability*: High normal forces ($F_N > 8\text{ N}$) risk tetrahedral element inversion in FEM, whereas Kelvinlets use singularity-free Green's functions.
* **Decision**: 
  1. Maintain **Regularized Kelvinlets** as the primary real-time in-the-loop deformation solver for the 1000 Hz PyBullet contact loop, ultrasound reslicing ($< 1.8\text{ ms}$), and fast DRL rollouts.
  2. Adopt **SOFA / Co-Rotational FEM** as an **offline clinical validation benchmark** to evaluate policy sim-to-real generalization across nonlinear hyperelastic deformations.
  3. If true GPU tetrahedral FEM is required in-the-loop, evaluate **NVIDIA Warp (`warp-lang`)** as the GPU-parallel alternative.

---

### `DEC-018`: Remote SSH Deployment & Ingestion Protocol for Real TotalSegmentator Clinical Data
* **Date**: October 28, 2026
* **Context**: Moving from synthetic lumbar phantoms to real clinical patient data requires training and benchmarking on multi-gigabyte NIfTI CT volumes from TotalSegmentator on remote SSH GPU servers/clusters.
* **Decision**: 
  1. Built an all-in-one automation script [`scripts/run_remote_totalsegmentator.sh`](file:///home/rishang/Desktop/Insti_Stuff/MTP/scripts/run_remote_totalsegmentator.sh) supporting modular phases (`setup`, `prepare`, `train`, `benchmark`, `all`).
  2. Standardized remote data ingestion via `load_totalsegmentator_subject` in `prepare_patient_assets.py` to auto-crop spine bounding boxes and generate acoustic impedance property maps directly from clinical NIfTI masks.
  3. Formulated fast rsync transfer protocols excluding local `.venv` and video caches to enable $< 30\text{ s}$ repo deployment to any remote compute node.
* **Rationale**: Eliminates manual remote configuration errors, ensures pure ultrasound signal extraction without ground-truth label leakage on real clinical scans, and accelerates DRL/Diffusion Policy training on high-VRAM remote GPUs.
