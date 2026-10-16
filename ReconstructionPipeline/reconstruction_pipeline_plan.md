# Implementation Plan: 3D Ultrasound Reconstruction & Active View Planning Pipeline

## 1. Executive Summary & Objective

This plan details the end-to-end implementation for an **Active Robotic 3D Ultrasound Reconstruction and Next-Best-View (NBV) Planning Pipeline**, synthesizing the architectural insights from **"Advanced 3D Reconstruction and Active View Planning in Robotic Ultrasound: Optimizing the SonoGym Pipeline"** and the foundational framework from **"SonoGym: High Performance Simulation for Challenging Surgical Tasks with Robotic Ultrasound" (Ao et al., 2025)**.

The system bridges into our existing **PyBullet physics backend** (`surgical_physics_bridge`) and **3D Slicer OpenIGTLink frontend** (`SlicerSurgicalBridge`), transforming the simulator into an intelligent, autonomous robotic ultrasound scanning and intraoperative reconstruction platform.

```
                      +-------------------------------------------------+
                      |              3D Slicer (Frontend)               |
                      |  - Live 2D US Viewport & B-mode Display         |
                      |  - Reconstructed 3D Bone Point Cloud / Mesh     |
                      |  - Real-time Trajectory & Robot Model Display   |
                      +------------------------^------------------------+
                                               | OpenIGTLink (pyigtl)
                                               v
+---------------------+       +-----------------------------------------+
| Patient Anatomy &   |       |         Robotic Simulation Core         |
| Acoustic Simulation |<----->| - PyBullet: Rigid contacts & probe pose |
| - CT Volume & Label |       | - Pinocchio / IK: Analytical QP control |
| - GPU Grid Sampling |       | - End-Effector: 4-DoF [dx, dy, da, db]  |
| - MB Ray Tracing /  |       +--------------------^--------------------+
|   LB Pix2Pix Model  |                            |
+----------+----------+                            v
           |                   +----------------------------------------+
           +------------------>|  Perception & Surface Accumulator     |
           Extracted B-mode &  |  - Bone Surface Extractor (Seg/Dropout)|
           Confidence Map      |  - Local-to-Global Coordinate Mapping  |
                               |  - Point Cloud / TSDF Accumulation     |
                               +-------------------+--------------------+
                                                   |
                                                   v
                               +----------------------------------------+
                               |     Active NBV / DRL Planner           |
                               |  - Submodular MDP (Diminishing Return) |
                               |  - PointNet / 3D Grid State Encoder    |
                               |  - Policy: Submodular PPO / Heuristic  |
                               +----------------------------------------+
```

---

## 2. Core Architecture & Mathematical Foundations

### 2.1 State, Action, and Observation Space
Following SonoGym Task 2 (Bone Surface Reconstruction) and the report's active view formulation:
- **Kinematic Constraints**:
  - Probe frame $\{U\}$ maintained tangent to skin surface (z-axis along surface normal).
  - 4-DoF action space: $a_t = [\Delta x_t, \Delta y_t, \Delta \alpha_t, \Delta \beta_t]^\top$
    - $[\Delta x_t, \Delta y_t]$: Tangential translation along skin surface.
    - $\Delta \alpha_t$: Yaw (rotation about probe normal / z-axis).
    - $\Delta \beta_t$: Pitch (tilt about elevation / y-axis to maintain orthogonality to curved bone facets).
- **Observation Space**:
  - **Volumetric Representation**: $40 \times 40 \times 40$ voxel grid at $3\,\text{mm}$ resolution centered on the target vertebra $\{V\}$, transformed to probe frame ${}^U M_{0:t}$.
  - **Lightweight Alternative**: Unstructured Point Cloud representation ($N \times 3$) processed via a PointNet encoder to eliminate 3D CNN memory overhead.
  - **Ultrasound Confidence Map**: Per-pixel acoustic confidence weighting based on incidence angle $\Theta$ and attenuation depth.

### 2.2 Submodular Objective Function & Reward Formulation
Because re-scanning previously observed regions yields diminishing informational value, the trajectory objective is defined as a monotone submodular set function:

$$F(\tau_{0:t}) := |M_{0:t}| - w_2 \sum_{h=0}^t \left( |\Delta x_h| + |\Delta y_h| + w_3 |\Delta \alpha_h| + w_3 |\Delta \beta_h| \right)$$

- $|M_{0:t}|$: Total reconstructed bone surface area / number of unique voxels covered.
- $w_2 = 0.01, w_3 = 1.0$: Penalty weights penalizing physical path length and probe rotation.
- **Reward for Step $t$**: Greedy marginal gain:
  $$r(s_t, a_t) = F(\tau_{0:t+1}) - F(\tau_{0:t}) = \Delta |M_{t+1}| - w_2 \left( |\Delta x_t| + |\Delta y_t| + w_3 |\Delta \alpha_t| + w_3 |\Delta \beta_t| \right)$$

---

## 3. Phased Implementation Roadmap

### Phase 1: Patient Assets & Acoustic Ultrasound Simulator
**Goal**: Implement fast GPU-accelerated 2D ultrasound slice extraction and acoustic simulation from 3D CT volumes and TotalSegmentator label maps.

1. **Asset Management (`ReconstructionPipeline/assets/`)**:
   - Download sample TotalSegmentator lumbar spine dataset (CT volume + L1–L5 vertebra label maps).
   - Generate patient skin surface mesh (STL) and bone meshes for collision and ground-truth evaluation.
2. **GPU Slicing Engine (`ReconstructionPipeline/simulation/slicer_cuda.py`)**:
   - Given probe pose $T_{W \to U}$, compute the 2D imaging plane in CT coordinate space.
   - Use PyTorch `torch.nn.functional.grid_sample` (bilinear/trilinear interpolation) to extract 2D CT slices and label maps at $200 \times 150$ resolution ($0.5\,\text{mm/px}$).
3. **Ultrasound Physics Engine (`ReconstructionPipeline/simulation/us_simulator.py`)**:
   - **Model-Based (MB) Simulator**:
     - Attenuation map: $E(u) = E_0 \exp\left(-f \int_0^y \alpha(x, v) dv\right)$.
     - Reflection term: $R(u) = \left|E(u) \cos\Theta(u) \frac{Z(x, y+\delta y) - Z(x, y)}{Z(x, y+\delta y) + Z(x, y)}\right| \cdot P(u) \otimes G(u)$.
     - Backscattering term: $B(u) = E(u) P(u) \otimes T(u)$ with multi-scale Gaussian noise maps.
     - Total B-mode intensity: $I(u) = R(u) + B(u)$.
   - **Ultrasound Confidence Map**:
     - Compute incidence angle $\Theta(u)$ between ultrasound ray and bone surface normal.
     - Generate confidence weight $C(u) = \exp(-\Theta(u)^2 / 2\sigma_\theta^2) \cdot \text{ShadowFactor}(u)$.

---

### Phase 2: Decoupled Kinematics & Surface Reconstruction Pipeline
**Goal**: Integrate lightweight kinematics in the PyBullet server and build the real-time point cloud / TSDF accumulator.

1. **Analytical Kinematics (`surgical_physics_bridge/src/kinematics/`)**:
   - Integrate `pinocchio` or optimized PyBullet differential IK to avoid standard iterative solver latency.
   - Constrain probe contact to patient skin mesh using Signed Distance Fields (SDF) or ray-cast normal alignment.
2. **Bone Surface Segmenter & Noise Model (`ReconstructionPipeline/reconstruction/segmentation.py`)**:
   - Simulator mode: Direct boundary extraction from label map with configurable missing rate (20% dropout) and specular loss at oblique incidence angles ($> 30^\circ$).
   - Inference mode: Lightweight 2D U-Net / MobileNet segmenter for synthetic B-mode images.
3. **Point Cloud & Voxel Accumulator (`ReconstructionPipeline/reconstruction/accumulator.py`)**:
   - Back-project 2D segmented bone contour points into 3D world space using current tracked probe pose $T_{W \to U}$:
     $$P_{\text{world}} = T_{W \to U} \cdot [x_{\text{pixel}} \cdot s_x, y_{\text{pixel}} \cdot s_y, 0, 1]^\top$$
   - Filter points by ultrasound confidence score $C(u) > \tau_{\text{conf}}$.
   - Maintain:
     - Accumulated Point Cloud with radius outlier removal and voxel downsampling ($1.0\,\text{mm}$).
     - $40 \times 40 \times 40$ relative voxel grid (${}^U M_{0:t}$) centered on target vertebra for RL observation.

---

### Phase 3: Active View Planner (Submodular DRL & Baselines)
**Goal**: Implement the Next-Best-View planner and train submodular DRL policies against heuristic baselines.

1. **Gym/Gymnasium Environment (`ReconstructionPipeline/gym/us_recon_env.py`)**:
   - Compliant with Gymnasium API.
   - Random initial probe pose within $30 \times 30\,\text{mm}^2$ window over vertebra.
   - 300 steps per episode.
   - Reward calculator implementing marginal gain $r_t = F(\tau_{0:t+1}) - F(\tau_{0:t})$.
2. **Baselines & Benchmarks (`ReconstructionPipeline/planners/`)**:
   - **Open-Loop Heuristic Scanner**: Standard raster (meander/zigzag) linear sweep across $X \times Y$.
   - **Greedy Shannon Entropy / Ray-Casting NBV Planner**: Classic Information Gain baseline.
3. **Submodular DRL Agent (`ReconstructionPipeline/learning/`)**:
   - **Network Architecture**:
     - Option A: 3D CNN encoder ($40 \times 40 \times 40$) + shared MLP for actor/critic (matching SonoGym).
     - Option B: Lightweight PointNet encoder operating on downsampled point cloud $(N \times 3)$.
   - **Algorithm**: PPO with KL-Adaptive learning rate (`skrl` or `stable-baselines3`).
   - Validate emergence of adaptive circular/orbital trajectories around spinal facets.

---

### Phase 4: Integration with 3D Slicer via OpenIGTLink
**Goal**: Stream ultrasound frames, reconstructed point clouds, and planned trajectories to 3D Slicer in real time.

1. **OpenIGTLink Protocol Extensions (`surgical_physics_bridge/src/server.py`)**:
   - `IMAGE` message: 2D synthetic ultrasound B-mode slice ($200 \times 150$, 8-bit grayscale).
   - `POINT` / `POLYDATA` message: Reconstructed bone point cloud and mesh streamed at 10–20 Hz.
   - `TRANSFORM` message: Probe frame $\{U\}$ and planned next-view target pose.
2. **3D Slicer Visualization (`SlicerSurgicalBridge`)**:
   - Live Ultrasound slice viewer widget placed in 3D coordinate space attached to the probe end-effector.
   - Color-coded point cloud display (e.g., colored by confidence or reconstruction time).
   - Ground truth CT vertebra overlay to visually inspect Chamfer Distance and surface coverage.

---

### Phase 5: Quantitative Evaluation & Benchmarking Suite
**Goal**: Automated evaluation comparing reconstruction accuracy, efficiency, and robustness.

| Metric | Target | Evaluation Method |
| :--- | :--- | :--- |
| **Coverage Ratio** | $\ge 75\%$ of dorsal vertebra surface | Ratio of reconstructed points within $\epsilon = 2\,\text{mm}$ of GT CT mesh |
| **Chamfer Distance** | $\le 1.5\,\text{mm}$ | Bidirectional point-to-mesh shortest distance |
| **Trajectory Length** | $\le 450\,\text{mm}$ | Cumulative Cartesian path length of probe tip |
| **Total Rotation** | $\le 250^\circ$ | Cumulative angular displacement ($\Delta \alpha + \Delta \beta$) |
| **Inference Latency** | $\ge 30\,\text{FPS}$ ($< 33\,\text{ms}$) | End-to-end slice extraction + inference + IK |

---

## 4. File Structure & Module Organization

```
MTP/
├── ReconstructionPipeline/
│   ├── assets/                     # Downloaded CT volumes & segmentations
│   │   └── download_totalseg.py    # Automated asset downloader
│   ├── simulation/
│   │   ├── slicer_cuda.py          # PyTorch grid_sample CT/label extraction
│   │   ├── us_simulator.py         # Model-based ultrasound ray-tracer
│   │   └── confidence_map.py       # Acoustic incidence & shadow confidence
│   ├── reconstruction/
│   │   ├── segmentation.py         # 2D bone segmentation & dropout simulator
│   │   └── accumulator.py          # 3D Point Cloud & Voxel Grid aggregator
│   ├── gym/
│   │   └── us_recon_env.py         # Gymnasium Submodular MDP environment
│   ├── planners/
│   │   ├── heuristic_raster.py     # Linear sweep baseline
│   │   └── greedy_nbv.py           # Shannon entropy ray-cast baseline
│   ├── learning/
│   │   ├── models/                 # 3D CNN & PointNet feature extractors
│   │   └── train_submodular_ppo.py # PPO training script
│   └── eval/
│       └── benchmark_metrics.py    # Chamfer distance & coverage calculator
├── surgical_physics_bridge/        # Existing PyBullet server
│   └── src/
│       ├── us_igtl_bridge.py       # OpenIGTLink streamer for US image & cloud
│       └── server.py               # Integrated physics + ultrasound loop
└── SlicerSurgicalBridge/           # Existing C++ Slicer loadable plugin
```
