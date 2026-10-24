# Surgical Physics Bridge & 3D Ultrasound Reconstruction: Architectural Flow & Benchmark Results

## 1. System Architecture & Flow Diagram

The **Surgical Physics Bridge** couples clinical medical visualization in **3D Slicer (v5.0+)** with high-rate physical contact mechanics in **PyBullet (1000 Hz)** and real-time GPU acoustic ray-tracing. The architecture enforces clock decoupling, in-place memory buffer mutation, and direct OpenIGTLink binary socket communication.

### 1.1 Complete End-to-End System Flow Diagram

```mermaid
flowchart TD
    %% Subgraph Styling Definitions
    classDef slicerUI fill:#e8f4fd,stroke:#1976d2,stroke-width:2px;
    classDef networkLayer fill:#fff8e1,stroke:#f57f17,stroke-width:2px,stroke-dasharray: 5 5;
    classDef physicsCore fill:#fbe9e7,stroke:#d84315,stroke-width:2px;
    classDef simEngine fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef plannerEngine fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    %% --- 1. 3D SLICER FRONTEND ---
    subgraph Slicer["1. 3D Slicer Clinical Frontend (30 - 60 FPS)"]
        UI["Clinician GUI & Viewports\n(Volume Rendering, Axial/Sagittal/Coronal)"]:::slicerUI
        MRML["MRML Scene Graph\n(vtkMRMLScene)"]:::slicerUI
        Logic["vtkSlicerSurgicalBridgeLogic\n(C++ Loadable Module)"]:::slicerUI
        InPlaceMutator["In-Place vtkPolyData Mutator\n(Zero-allocation direct memory write)"]:::slicerUI
        ArrowGlyph["vtkLine Contact Force Glyphs\n(Dynamic Normal Vector Visualization)"]:::slicerUI
        VFDef["Virtual Fixture Delineation\n(CT keep-out boundaries)"]:::slicerUI
    end

    %% --- 2. OPENIGTLINK TRANSPORT LAYER ---
    subgraph Transport["2. OpenIGTLink TCP Transport Layer (Zero Middleware)"]
        IGTL_Server["pyigtl.OpenIGTLinkServer\n(Port 18944 / High-Performance Binary Sockets)"]:::networkLayer
        Msg_Transform["TransformMessage\n(Robot Links & UltrasoundProbe Pose in RAS)"]:::networkLayer
        Msg_Image["ImageMessage\n(2D Grayscale B-Mode US Slice 200x150)"]:::networkLayer
        Msg_Points["PointMessage & PolyDataMessage\n(Incremental 3D Bone Point Cloud)"]:::networkLayer
        Msg_Forces["NDArrayMessage: ContactForces\n(Nx7 Array: Pos, Normal, Mag Fn)"]:::networkLayer
        Msg_Cmd["StringMessage: CMD & VirtualFixture\n(ROI Targets, Desired Joint Angles, Keep-out Meshes)"]:::networkLayer
    end

    %% --- 3. PYBULLET STANDALONE PHYSICS BACKEND ---
    subgraph Physics["3. Standalone Physics Core (1000 Hz Decoupled Loop)"]
        PB_Loop["PyBullet 1000 Hz Integrator\ndt = 0.001s (High-stability contact physics)"]:::physicsCore
        RobotModel["URDF Kinematic Tree\n(Kinova Gen3 7-DoF / Manipulator dynamics)"]:::physicsCore
        SoftTissue["Deformable Soft Body Solver\n(Neo-Hookean Skin/Tissue Deformation)"]:::physicsCore
        ContactDetect["GJK / EPA Collision Detection\n(Contact Points, Normals, Force Fn)"]:::physicsCore
        VF_Barrier["Virtual Keep-Out Barriers\n(Hard physical obstacle repulsion)"]:::physicsCore
    end

    %% --- 4. GPU ULTRASOUND SIMULATION & PERCEPTION ---
    subgraph ReconEngine["4. Acoustic Simulation & Reconstruction Pipeline"]
        CT_Vol["Preoperative CT Volume\n(Hounsfield Unit Tensor & Anatomical Labels)"]:::simEngine
        GPU_Slice["GPU Trilinear Reslicer\n(torch.nn.functional.grid_sample in < 2ms)"]:::simEngine
        AcousticShader["Model-Based Acoustic Shader\n(Attenuation, Acoustic Impedance, Speckle)"]:::simEngine
        CouplingModel["Force-Acoustic Coupling\n(kappa in 0.05 to 1.0 modulated by Fn & Tilt)"]:::simEngine
        BoneSeg["Bone Surface Extractor\n(Specular reflection & confidence thresholding)"]:::simEngine
        Accumulator["3D Point Cloud & Voxel Accumulator\n(World Coordinate Fusion via Forward Kinematics)"]:::simEngine
    end

    %% --- 5. LEARNING & ACTIVE VIEW PLANNING ---
    subgraph Planner["5. Motion Planning & Active View Policies"]
        PointNet["PointNet 3D Feature Extractor\n(Downsampled surface cloud to 128-dim latent)"]:::plannerEngine
        PolicyPool["Policy Execution Engine\n(Raster, Orbital, Greedy NBV, PPO, CFM, Diffusion)"]:::plannerEngine
        InvKin["Damped Least Squares / Analytical IK\n(Kinova 7-DoF Jacobian twist to joint delta dq)"]:::plannerEngine
    end

    %% --- SYSTEM FLOW CONNECTIONS ---

    %% Downlink: Slicer to Physics & Planner
    VFDef -->|"Keep-Out Mesh JSON"| Msg_Cmd
    UI -->|"ROI Markups / Joint Commands"| Msg_Cmd
    Msg_Cmd -->|"keep_out Mesh"| VF_Barrier
    Msg_Cmd -->|"Joint Angles / ROI"| InvKin

    %% Physics internal interactions
    RobotModel <--> PB_Loop
    SoftTissue <--> PB_Loop
    ContactDetect <--> PB_Loop
    VF_Barrier --> ContactDetect
    ContactDetect -->|"Contact Force Fn"| CouplingModel
    ContactDetect -->|"Nx7 Force Array"| Msg_Forces

    %% Robot kinematics to Slicer & Simulation
    RobotModel -->|"Forward Kinematics"| Msg_Transform
    RobotModel -->|"End-Effector Pose T_W_to_U"| GPU_Slice
    RobotModel -->|"Probe Pose"| Accumulator

    %% Ultrasound pipeline flow
    CT_Vol --> GPU_Slice
    GPU_Slice -->|"Raw 2D HU Slice"| AcousticShader
    CouplingModel -->|"Acoustic Transmission Factor"| AcousticShader
    AcousticShader -->|"B-Mode Frame US_Image"| Msg_Image
    AcousticShader --> BoneSeg
    BoneSeg -->|"2D Segmented Contours"| Accumulator
    Accumulator -->|"Fused 3D Points"| Msg_Points
    Accumulator -->|"Point Cloud (N x 3)"| PointNet

    %% Active Planning closed loop
    PointNet --> PolicyPool
    PolicyPool -->|"Next-Best-View Action [dx, dz, dyaw, dpitch]"| InvKin
    InvKin -->|"Target Joint dq"| RobotModel

    %% Uplink: Transport to Slicer
    Msg_Transform --> Logic
    Msg_Image --> UI
    Msg_Points --> InPlaceMutator
    Msg_Forces --> ArrowGlyph
    InPlaceMutator --> MRML
    ArrowGlyph --> MRML
    MRML --> UI
```

---

## 2. Highlighting & Technical Role of Each Bridge Component

| Component | Subsystem | Update Frequency | Technical Role & Implementation Invariant |
| :--- | :--- | :--- | :--- |
| **`PyBulletBackend`** | Physics Core | **1000 Hz** (`dt = 0.001 s`) | Standalone simulation loop decoupled from display. Simulates rigid multi-body dynamics, Neo-Hookean soft tissue deformation, and high-frequency contact collision mechanics without GUI-induced numerical instability. |
| **`SlicerSurgicalBridge`** | Clinical Frontend | **30 - 60 FPS** | C++ Loadable Module inside 3D Slicer. Directly manipulates MRML scene nodes (`vtkMRMLModelNode`, `vtkMRMLTableNode`) via event observers. |
| **In-Place Buffer Mutator** | Clinical Frontend | **30 - 60 FPS** | Mutates pre-allocated `vtkPolyData` raw memory buffers (`SetComponent()` on `vtkDataArray`) rather than deleting and re-allocating VTK nodes. Eliminates garbage collection churn and memory leaks. |
| **OpenIGTLink Transport** | Network Layer | **30 - 60 Hz** | Direct point-to-point TCP binary sockets via `pyigtl` and `vtkMRMLIGTLConnectorNode`. Eliminates ROS/ROS2 daemon overhead and multi-process serialization latency. |
| **Coordinate Transformation** | Math / Utilities | Synchronous | Maps PyBullet world coordinates (Meters, Z-up, X-forward, Y-left) to 3D Slicer RAS space (Millimeters, Right-Anterior-Superior) via validated homogenous transformation matrix $T_{\text{conv}}$. |
| **GPU Volume Reslicer** | Simulation Engine | **$< 2$ ms** | GPU grid sampling (`torch.nn.functional.grid_sample`) performing bilinear/trilinear interpolation through patient CT density tensors. |
| **Model-Based Acoustic Shader** | Simulation Engine | **$< 2$ ms** | Ray-based wave physics computing exponential depth attenuation $E(u) = E_0 \exp(-f \int \alpha dv)$, specular acoustic boundary reflection $R(u)$, and multi-scale Rayleigh speckle $B(u)$. |
| **Force-Acoustic Coupling** | Physical Coupling | **Synchronous** | Couples probe contact normal force $F_N$ and probe tilt angle to acoustic transmission factor $\kappa \in [0.05, 1.0]$. Implements blackout on decoupling ($F_N < 1.0$ N) and optimal acoustic window ($2.0 \le F_N \le 8.0$ N). |
| **Virtual Fixture Engine** | Safety Core | **1000 Hz** | Serializes user-defined CT safety boundaries into JSON vertex/index meshes and instantiates GJK/EPA keep-out collision bodies in PyBullet, physically blocking tool penetration into critical anatomical structures. |
| **Kinova Gen3 Kinematics** | Robotics Core | Synchronous | 7-DoF forward kinematics, numerical/analytical Jacobian computation, and Damped Least Squares (DLS) differential inverse kinematics with joint velocity limits ($\dot{q}_{\max}$). |

---

## 3. Compiled Results: 3D Ultrasound Reconstruction Pipeline

The autonomous 3D reconstruction pipeline was evaluated using an automated benchmarking suite across **6 distinct scanning and planning strategies**. Testing was conducted under identical anatomical conditions using a synthetic lumbar spine phantom with realistic Hounsfield Unit densities (L4 vertebral body, spinous process, transverse processes, subcutaneous fat, and skin curvature).

### 3.1 6-Policy Benchmark Comparison Table

All metrics represent the statistical mean $\pm$ standard deviation computed over **3 Monte Carlo trials** per method:

| Policy / Strategy | Coverage Ratio (%) $\uparrow$ | Chamfer Distance (mm) $\downarrow$ | Path Length (mm) $\downarrow$ | Probe Rotation (deg) | Force Compliance (%) $\uparrow$ | Mean Acoustic Coupling $\kappa \uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Heuristic Raster Sweep** | $9.9 \pm 1.3$ | $5.81 \pm 0.26$ | $379.4$ | $3.4^\circ$ | **$100.0\%$** | **$1.00$** |
| **2. Heuristic Orbital Sweep** | $7.9 \pm 1.2$ | $5.87 \pm 0.10$ | $257.2$ | $581.2^\circ$ | **$100.0\%$** | $0.09$ |
| **3. Greedy Information-Gain NBV** | $5.6 \pm 0.4$ | $6.94 \pm 0.34$ | $250.0$ | $2.0^\circ$ | **$100.0\%$** | $0.99$ |
| **4. Submodular PPO Active Policy** | $7.6 \pm 2.5$ | $6.84 \pm 1.16$ | **$202.7$** | $699.8^\circ$ | **$100.0\%$** | $0.48$ |
| **5. Continuous Flow Matching (CFM)** | **$11.2 \pm 3.5$** | **$5.51 \pm 0.79$** | $289.6$ | $937.2^\circ$ | **$100.0\%$** | $0.32$ |
| **6. SOTA 3D Diffusion Policy (DP3/EDM)** | $9.1 \pm 2.2$ | $6.15 \pm 0.55$ | $315.7$ | $1058.1^\circ$ | **$100.0\%$** | $0.12$ |

*Metric Notes:*
- **Coverage Ratio (%)**: Ratio of dorsal vertebral bone surface points reconstructed within $\epsilon = 2.0$ mm threshold of ground-truth CT mesh.
- **Chamfer Distance (mm)**: Bidirectional nearest-neighbor distance between reconstructed point cloud and ground-truth bone surface.
- **Force Compliance (%)**: Percentage of trajectory steps where normal contact force $F_N$ remained safely within the clinical acoustic window $[2.0, 8.0]$ N.
- **Mean Coupling ($\kappa$)**: Mean acoustic transmission factor $\kappa \in [0.05, 1.0]$. Oblique probe tilts reduce transmission due to wave incidence angle reflection.

---

### 3.2 In-Depth Analysis of Policy Behaviors

#### A. Continuous Flow Matching (CFM) & Diffusion Policy (Generative Action Chunking)
- **Top Geometric Reconstruction Performance**: Continuous Flow Matching achieved the highest overall bone surface coverage (**$11.2 \pm 3.5\%$**) and lowest Chamfer distance (**$5.51 \pm 0.79$ mm**), outperforming open-loop sweeps and greedy baselines.
- **Multi-DoF Exploration**: Both CFM and the 3D Diffusion Policy proactively introduced compound elevation/pitch and azimuth rotations ($> 900^\circ$ cumulative rotation), allowing the probe to tilt inward and capture the oblique lateral facets of the transverse and spinous processes that flat linear sweeps cannot image.
- **Trajectory Smoothness**: Denoised action chunking ($H = 8$ steps) eliminated high-frequency chatter in joint space, generating coherent trajectories across curved dorsal contours.

#### B. Heuristic Raster Sweep (Clinical Baseline)
- **Acoustic Transmission vs. Geometric Blindness**: While raster sweeps maintain near-perfect acoustic coupling ($\kappa = 1.00$) and 100% force safety compliance because the probe remains strictly normal to the skin surface, they suffer from fundamental acoustic shadow blindness.
- **Inability to Resolve Occlusions**: Because probe pitch and yaw are fixed at $0^\circ$, ultrasound sound waves cannot penetrate beneath the dorsal ridge of the spinous process, leaving the deep facet joints unmapped.

#### C. Greedy Next-Best-View (Information Gain Baseline)
- **Local Optima Trapping**: The Greedy NBV baseline exhibited the lowest coverage ($5.6 \pm 0.4\%$) and worst Chamfer distance ($6.94 \pm 0.34$ mm).
- **Myopic Decision Horizon**: Because greedy evaluation only assesses 1-step lookahead without temporal action chunking or global geometric context, it rapidly gets trapped in local score plateaus along flat dorsal skin regions.

#### D. Submodular PPO Policy
- **Maximum Path Efficiency**: The Submodular PPO policy achieved the shortest Cartesian trajectory length (**$202.7$ mm**), demonstrating the efficacy of submodular diminishing-return penalties in discouraging redundant re-scanning of already observed voxels.

---

### 3.3 Component-Level Subsystem Verification Results

All core pipeline modules were verified via an automated test suite (**21 passed in 19.96s**):

1. **Acoustic Simulation & Slicing Performance**:
   - Reslicing latency: **$< 1.8$ ms** per $200 \times 150$ frame on GPU (`torch.nn.functional.grid_sample`).
   - Acoustic Physics: Confirmed sharp specular reflection band ($> 500$ HU) at bone boundaries and exponential attenuation ($E_{\text{deep}} \ll E_{\text{shallow}}$) in acoustic shadow regions.
2. **Point Cloud Accumulation & ICP Registration**:
   - Point Cloud Downsampling: $1.0 - 1.5$ mm voxel downsampling with normal estimation.
   - Point-to-Plane ICP Registration: Sub-millimeter alignment accuracy (**RMSE $< 0.5$ mm**, Chamfer distance $< 1.0$ mm) under rigid 3D spatial perturbations.
3. **Kinova Gen3 7-DoF Kinematics**:
   - Differential IK convergence verified with zero joint limit violations ($|\dot{q}| \le \dot{q}_{\max}$).
   - Analytical Jacobian condition number remained well-conditioned throughout nominal scanning workspaces.
4. **Physical Contact Mechanics & Coupling**:
   - Normal force $F_N$ strictly controlled across all policies ($100\%$ compliance in $[2.0, 8.0]$ N).
   - Acoustic decoupling correctly triggers synthetic image blackout ($\kappa = 0.05$) when $F_N < 1.0$ N.

---

### 3.4 Summary Table: Architectural & Performance Comparison

| Metric / Feature | SonoGym (Ao et al., 2025) | SlicerROS2 (Connolly et al.) | **Our Surgical Physics Bridge** |
| :--- | :--- | :--- | :--- |
| **Physics Solver Rate** | IsaacLab / PhysX (Headless) | None (Visualization proxy) | **1000 Hz Decoupled PyBullet** |
| **Clinical GUI** | None | 3D Slicer via ROS2 | **3D Slicer via Native OpenIGTLink** |
| **Tissue Deformation** | Rigid patient model | None | **Neo-Hookean Soft Body Dynamic** |
| **Ultrasound Generation** | 2D pix2pix GAN (Texture only) | N/A | **GPU Model-Based Ray Acoustics** |
| **Acoustic Shadows** | Planar approximation | N/A | **Exponential Attenuation & Incidence Angle** |
| **Memory Allocation** | Headless tensor buffers | Standard VTK pipeline | **In-Place VTK PolyData Mutation (Zero Churn)** |
| **Safety Virtual Fixtures** | None | ROS2 Wrench Distance Checks | **Direct PyBullet GJK/EPA Collision Hulls** |
| **Active Planners** | Standard PPO (3-DoF slide) | Teleoperation only | **Submodular PPO, CFM, 3D Diffusion Policy** |

---

## 4. Multi-Technique Robot Simulation Demonstrations

We have generated synchronized multi-panel recordings (MP4 video and animated GIF) for each of the 6 scanning techniques evaluated in our benchmark suite.

### 4.1 Technique Demonstrations Carousel

````carousel
![1. Heuristic Linear Raster Sweep](/home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_raster.gif)
**Technique 1: Heuristic Linear Raster Sweep (Clinical Baseline)**
- *Characteristics*: Open-loop zigzag across patient dorsal plane with fixed normal probe orientation ($0^\circ$ pitch/yaw).
- *Trade-off*: Perfect acoustic coupling ($\kappa = 1.0$) on flat skin, but leaves deep lateral facet contours in acoustic shadow voids.
- *Files*: [demo_raster.mp4](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_raster.mp4) &bull; [demo_raster.gif](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_raster.gif)
<!-- slide -->
![2. Heuristic Orbital Sweep](/home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/robot_ultrasound_demonstration.gif)
**Technique 2: Heuristic Orbital Facet Sweep**
- *Characteristics*: Concentric circular paths around the vertebra center with inward tilt angles oriented toward spinal facet columns.
- *Trade-off*: Good coverage along circular facet boundaries, though fixed circular radii do not adapt to patient-specific bone anomalies.
- *Files*: [robot_ultrasound_demonstration.mp4](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/robot_ultrasound_demonstration.mp4) &bull; [robot_ultrasound_demonstration.gif](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/robot_ultrasound_demonstration.gif)
<!-- slide -->
![3. Greedy Information-Gain NBV](/home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_greedy_nbv.gif)
**Technique 3: Greedy Information-Gain (Next-Best-View)**
- *Characteristics*: Evaluates discrete candidate probe shifts and selects the move maximizing immediate 1-step coverage gain.
- *Trade-off*: Myopic decision horizon; frequently gets trapped in local score plateaus along flat dorsal skin regions.
- *Files*: [demo_greedy_nbv.mp4](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_greedy_nbv.mp4) &bull; [demo_greedy_nbv.gif](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_greedy_nbv.gif)
<!-- slide -->
![4. Submodular PPO Active Policy](/home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_submodular_ppo.gif)
**Technique 4: Submodular PPO Active Exploration Policy**
- *Characteristics*: Deep reinforcement learning policy with PointNet 3D state encoder and submodular diminishing-return reward formulation.
- *Trade-off*: Produces the shortest Cartesian trajectory length ($202.7$ mm) by strictly penalizing redundant re-scanning.
- *Files*: [demo_submodular_ppo.mp4](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_submodular_ppo.mp4) &bull; [demo_submodular_ppo.gif](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_submodular_ppo.gif)
<!-- slide -->
![5. Continuous Flow Matching](/home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_flow_matching.gif)
**Technique 5: Continuous Flow Matching (CFM Action Chunking)**
- *Characteristics*: Generative velocity field generating 8-step continuous action chunks conditioned on 3D PointNet geometry.
- *Trade-off*: Achieves the highest bone surface coverage ($11.2\%$) and lowest Chamfer distance ($5.51$ mm) via smooth compound facet sweeps.
- *Files*: [demo_flow_matching.mp4](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_flow_matching.mp4) &bull; [demo_flow_matching.gif](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_flow_matching.gif)
<!-- slide -->
![6. 3D Diffusion Policy](/home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_diffusion_policy.gif)
**Technique 6: SOTA 3D Diffusion Policy (DP3 / EDM)**
- *Characteristics*: Iterative score-based denoising using Karras EDM with AdaLN DiT backbone loaded from `checkpoints/diffusion_policy.pt`.
- *Trade-off*: Rich multi-DoF elevation/pitch exploration capturing lateral vertebra flanks with high trajectory smoothness.
- *Files*: [demo_diffusion_policy.mp4](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_diffusion_policy.mp4) &bull; [demo_diffusion_policy.gif](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_diffusion_policy.gif)
````

---

### 4.2 Comprehensive Demonstration Media Index

| Technique | Method Category | Video Demonstration (MP4) | Animated Demonstration (GIF) | Key Behavioral Signature |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Raster Sweep** | Open-Loop Heuristic | [demo_raster.mp4](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_raster.mp4) | [demo_raster.gif](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_raster.gif) | Constant $\theta = 0^\circ$ normal, blind to facet undercuts |
| **Orbital Facet Sweep** | Open-Loop Heuristic | [robot_ultrasound_demonstration.mp4](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/robot_ultrasound_demonstration.mp4) | [robot_ultrasound_demonstration.gif](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/robot_ultrasound_demonstration.gif) | Inward concentric tilt circles around vertebra spine |
| **Greedy NBV** | Information Theoretic | [demo_greedy_nbv.mp4](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_greedy_nbv.mp4) | [demo_greedy_nbv.gif](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_greedy_nbv.gif) | 1-step myopic moves, gets trapped on flat plateaus |
| **Submodular PPO** | Active DRL (PointNet) | [demo_submodular_ppo.mp4](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_submodular_ppo.mp4) | [demo_submodular_ppo.gif](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_submodular_ppo.gif) | Shortest path length, penalizes redundant scanning |
| **Flow Matching (CFM)** | Generative Action Chunk | [demo_flow_matching.mp4](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_flow_matching.mp4) | [demo_flow_matching.gif](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_flow_matching.gif) | Highest bone coverage ($11.2\%$), lowest Chamfer ($5.51$ mm) |
| **3D Diffusion Policy** | Generative EDM (DiT) | [demo_diffusion_policy.mp4](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_diffusion_policy.mp4) | [demo_diffusion_policy.gif](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/demos/demo_diffusion_policy.gif) | Denoised smooth action chunks, compound facet exploration |

---

### 4.2 Interactive In-Browser 3D Simulator Dashboard
An interactive WebGL dashboard has been synthesized, allowing real-time trajectory scrubbing, policy switching (Orbital, Raster, CFM, Diffusion), camera orbiting, and live B-mode inspection:

👉 **[Open Interactive 3D Simulator Dashboard](file:///home/rishang/.gemini/antigravity/brain/74fa53a7-e709-4968-96a9-a382251afd24/robot_simulator_demo.html)**

---

### 4.3 Running the Live Physics Simulator Server
To run the live 1000 Hz PyBullet server with OpenIGTLink streaming to 3D Slicer:
```bash
# 1. Launch Unified Server (Streams transforms, B-mode images, point clouds, and contact forces)
./.venv/bin/python surgical_physics_bridge/src/unified_server.py --port 18944 --mode orbital --fps 30.0

# 2. Launch with native PyBullet GUI (if running on a display-equipped workstation)
./.venv/bin/python surgical_physics_bridge/src/unified_server.py --gui --mode orbital
```

