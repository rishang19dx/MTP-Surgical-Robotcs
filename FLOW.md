# FLOW.md - Codebase Execution Flow & Architecture Guide

A comprehensive architectural map detailing all entry points, execution flows, communication protocols, and runtime dependencies for the **Surgical Physics Bridge & Autonomous Robotic Ultrasound (MTP)** codebase.

---

## 1. Directory of Primary Entry Points

| Purpose / Task | Entry Point Script | Standard Invocation Command |
| :--- | :--- | :--- |
| **Unified Server (Physics + US + Slicer)** | [`surgical_physics_bridge/src/unified_server.py`](file:///home/rishang/Desktop/Insti_Stuff/MTP/surgical_physics_bridge/src/unified_server.py) | `./.venv/bin/python surgical_physics_bridge/src/unified_server.py --port 18944 --mode orbital --fps 30.0` |
| **Standalone Physics Server (PyBullet)** | [`surgical_physics_bridge/src/server.py`](file:///home/rishang/Desktop/Insti_Stuff/MTP/surgical_physics_bridge/src/server.py) | `./.venv/bin/python surgical_physics_bridge/src/server.py --port 18944 --gui` |
| **6-Policy Benchmark Runner** | [`ReconstructionPipeline/evaluation/benchmark_runner.py`](file:///home/rishang/Desktop/Insti_Stuff/MTP/ReconstructionPipeline/evaluation/benchmark_runner.py) | `./.venv/bin/python ReconstructionPipeline/evaluation/benchmark_runner.py --num-trials 3` |
| **Patient Asset Generation (CT/Spine)** | [`ReconstructionPipeline/data/prepare_patient_assets.py`](file:///home/rishang/Desktop/Insti_Stuff/MTP/ReconstructionPipeline/data/prepare_patient_assets.py) | `./.venv/bin/python -m ReconstructionPipeline.data.prepare_patient_assets` |
| **Train Submodular PPO (Active DRL)** | [`ReconstructionPipeline/planning/train_submodular_ppo.py`](file:///home/rishang/Desktop/Insti_Stuff/MTP/ReconstructionPipeline/planning/train_submodular_ppo.py) | `./.venv/bin/python ReconstructionPipeline/planning/train_submodular_ppo.py --num-updates 20` |
| **Train 3D Diffusion Policy (DP3/EDM)** | [`ReconstructionPipeline/planning/train_diffusion_policy.py`](file:///home/rishang/Desktop/Insti_Stuff/MTP/ReconstructionPipeline/planning/train_diffusion_policy.py) | `./.venv/bin/python ReconstructionPipeline/planning/train_diffusion_policy.py --epochs 50 --batch-size 32` |
| **Run Unit & Integration Test Suite** | [`ReconstructionPipeline/tests/`](file:///home/rishang/Desktop/Insti_Stuff/MTP/ReconstructionPipeline/tests) | `./.venv/bin/python -m pytest ReconstructionPipeline/tests/ -v` |
| **Remote TotalSegmentator Automation (SSH)** | [`scripts/run_remote_totalsegmentator.sh`](file:///home/rishang/Desktop/Insti_Stuff/MTP/scripts/run_remote_totalsegmentator.sh) | `bash scripts/run_remote_totalsegmentator.sh --dataset-dir /data/s0010 --mode all` |
| **3D Slicer Loadable C++ Module** | [`SlicerSurgicalBridge/`](file:///home/rishang/Desktop/Insti_Stuff/MTP/SlicerSurgicalBridge) | Built via CMake against Slicer SDK; loaded as Qt extension |

---

## 2. End-to-End System Execution Flows

### Flow A: Offline Patient Asset Generation
Generates synthetic or TotalSegmentator clinical CT volumes, anatomical label maps, dorsal/flank skin meshes, and ground-truth bone point clouds.

```mermaid
flowchart LR
    CTGen["PatientAssetGenerator\n(prepare_patient_assets.py)"] --> SynthCT["CT Volume (120x120x80)\nHounsfield Units (-1000 to +1200)"]
    CTGen --> Labels["Label Map (120x120x80)\n0=Background, 1=Bone, 2=Tissue"]
    CTGen --> SkinMesh["Skin Surface Mesh (STL)\nLateral Flank & Dorsal Curvature"]
    CTGen --> GTPoints["GT Vertebra Point Cloud\nN x 3 Bone Surface Coordinates"]
    SynthCT --> OutDir["ReconstructionPipeline/data/sample_patient/"]
    Labels --> OutDir
    SkinMesh --> OutDir
    GTPoints --> OutDir
```

---

### Flow B: Live Physics Simulation & OpenIGTLink Streaming
The primary runtime service loop executed by [`unified_server.py`](file:///home/rishang/Desktop/Insti_Stuff/MTP/surgical_physics_bridge/src/unified_server.py). Decouples 1000 Hz PyBullet contact dynamics from 30–60 Hz Slicer OpenIGTLink publishing.

```mermaid
sequenceDiagram
    autonumber
    participant PB as PyBullet Backend (1 kHz)
    participant Server as Unified Server Loop
    participant Env as US Reconstruction Env
    participant IGTL as OpenIGTLink Server (Port 18944)
    participant Slicer as 3D Slicer Client

    loop Every 1 ms (1000 Hz)
        Server->>PB: Step simulation (dt = 0.001s)
        PB->>Server: Contact forces & link poses
    end

    loop Every 33.3 ms (30 Hz)
        Server->>Env: Step active probe action [dy, dz, dyaw, dpitch]
        Env->>Server: B-mode image, point cloud, normal force Fn
        Server->>IGTL: Send TransformMessage ("UltrasoundProbe", links)
        Server->>IGTL: Send ImageMessage ("US_Image", 240x160 grayscale)
        Server->>IGTL: Send PointMessage / PolyDataMessage ("ReconstructedBone")
        Server->>IGTL: Send NDArrayMessage ("ContactForces", Nx7)
        IGTL->>Slicer: TCP Binary Stream
        Slicer->>Slicer: In-place vtkPolyData mutation & 2D slice update
    end

    opt Clinician Interaction
        Slicer->>IGTL: StringMessage ("CMD", {"roi_target": [x,y,z]})
        IGTL->>Server: Receive Slicer ROI markups
        Server->>Server: Convert RAS to PB world frame
        Server->>Server: Kinova 7-DoF DLS Inverse Kinematics
    end
```

---

### Flow C: Ultrasound Simulation & Pure B-Mode Surface Extraction
Extracts 3D bone point clouds **strictly from 2D ultrasound signals** (zero ground-truth CT label leakage), featuring real-time **GPU soft-tissue deformation** via Regularized Kelvinlets.

```mermaid
flowchart TD
    ProbePose["Tracked Ultrasound Probe Pose\nT_{W <- U} (Lateral Flank)"] --> WarpGrid["Backward Eulerian Coordinate Warping\nX_ref = x - (u(x) - u_0)"]
    Force["Probe Contact Normal Force Fn\n& Inward Incidence Angle"] --> Kelvinlet["Regularized Kelvinlet Elasticity\n(kelvinlet_deformation.py)\nmu=12 kPa, nu=0.48, epsilon=15 mm"]
    Kelvinlet --> WarpGrid
    WarpGrid --> GridSample["GPU Volume Reslicer\ntorch.nn.functional.grid_sample (< 1.8 ms)"]
    CTVol["CT Hounsfield Unit Volume\nDensity rho and Speed c"] --> GridSample
    GridSample --> HU_Slice["Deformed 2D Hounsfield Unit Slice (200x150)"]

    HU_Slice --> AcousticShader["Model-Based Acoustic Shader\n(ultrasound_model_based.py)"]
    Force --> Coupling["Force-Acoustic Coupling\nkappa in [0.05, 1.0]"]
    Coupling --> AcousticShader

    AcousticShader -->|"Attenuation + Reflection + Speckle"| BMode["2D B-Mode Ultrasound Image\n(Normalized [0, 1])"]
    AcousticShader --> ConfMap["Acoustic Confidence Map C(u, v)\nShadow Factor & Beam Incidence"]

    BMode --> Seg["BoneSurfaceSegmenter\n(bone_segmentation.py)"]
    ConfMap --> Seg

    subgraph Direct_US_Segmentation["Direct B-Mode Signal Extraction"]
        Seg --> S1["1. Specular Peak Detection:\nI(u, v) >= 0.22 and I > I(u, v +- 1)"]
        S1 --> S2["2. Acoustic Shadow Verification:\nI_peak >= 1.25 * I_shadow"]
        S2 --> S3["3. Confidence Gating:\nC(u, v) >= 0.35"]
        S3 --> Contours["2D Extracted Bone Pixels (u, v)"]
    end

    Contours --> Accum["PointCloudAccumulator\n(point_cloud_accumulator.py)"]
    ProbePose --> Accum
    Accum -->|"p_world = T_{W <- U} * [u*sx, v*sy, 0, 1]"| VoxelGrid["Incremental 3D Point Cloud\n(Voxel Downsampled at 1.5 mm)"]
```

---

### Flow D: Active View Planning & Closed-Loop Control Loop
Reinforcement Learning (Submodular PPO) and Generative Policies (Continuous Flow Matching, 3D Diffusion Policy).

```mermaid
flowchart LR
    AccumPoints["Accumulated 3D Point Cloud\n(PointCloudAccumulator)"] --> PointNet["PointNet State Encoder\n(pointnet_encoder.py)"]
    ProbePose["Current Probe Pose\n[c1, c2, yaw, pitch]"] --> PointNet
    PointNet --> State["Fused 192-dim Latent State s_t"]

    State --> PolicyPool{"Active Policy"}
    PolicyPool -->|"Submodular PPO"| PPO["PPO Actor-Critic Network\nGaussian Policy pi(a|s)"]
    PolicyPool -->|"Diffusion Policy"| DP["AdaLN DiT Denoising\nKarras EDM (8-step chunk)"]
    PolicyPool -->|"Flow Matching"| CFM["Vector Field Integration\nEuler ODE (8-step chunk)"]

    PPO --> Action["Delta Action a_t = [dy, dz, dyaw, dpitch]"]
    DP --> Action
    CFM --> Action

    Action --> DLS_IK["Kinova 7-DoF DLS Inverse Kinematics\n(kinova_kinematics.py)"]
    DLS_IK --> JointCmd["Joint Velocity / Position Target dq"]
    JointCmd --> PyBullet["PyBullet Physics Core (1 kHz)\nContact Mechanics & Deformation"]
    PyBullet --> AccumPoints
```

---

### Flow E: Clinician-in-the-Loop Slicer Fiducial Target Workflow
Enables interactive point-and-click target selection in 3D Slicer.

```mermaid
flowchart TD
    Clinician["Clinician in 3D Slicer"] -->|"Clicks Vertebra Facet"| FiducialNode["vtkMRMLMarkupsFiducialNode"]
    FiducialNode --> Widget["qSlicerSurgicalBridgeModuleWidget"]
    Widget -->|"Click 'Send ROI Target'"| Logic["vtkSlicerSurgicalBridgeLogic::SendROITarget(x, y, z)"]
    Logic --> NetText["vtkMRMLTextNode ('CMD')\n{'roi_target': [x_ras, y_ras, z_ras]}"]
    NetText --> Connector["vtkMRMLIGTLConnectorNode"]
    Connector -->|"OpenIGTLink TCP Socket"| PyServer["unified_server.py: handle_incoming_messages()"]

    subgraph Server_Target_Resolution["Server Kinematic Target Resolution"]
        PyServer --> CoordConv["convert_pose_pb_to_slicer Inverse\np_PB = R_conv^T * (p_RAS / 1000 - t_conv)"]
        CoordConv --> LateralProj["Project to Flank Skin Surface Normal\nX_skin(y, z) pointing along -X"]
        LateralProj --> IK["KinovaGen3Kinematics.inverse_kinematics()"]
        IK --> RobotPose["Robot Probe Dispatched to Lateral Target"]
    end
```

---

## 3. Coordinate System Conversions & Invariants

| Space / Frame | Origin & Axis Convention | Units | Conversion to/from |
| :--- | :--- | :--- | :--- |
| **PyBullet World** | Origin at surgical table base. **X-forward, Y-left, Z-up**. | **Meters** | Base reference frame for all rigid/soft dynamics. |
| **3D Slicer RAS** | Origin at CT scan center. **Right (+X), Anterior (+Y), Superior (+Z)**. | **Millimeters** | $\mathbf{p}_{\text{RAS}} = \mathbf{T}_{\text{conv}} \cdot (\mathbf{p}_{\text{PB}} \times 1000)$ |
| **Ultrasound Probe $\{U\}$** | Origin at center of transducer face. **Beam along $+Y_{\text{probe}}$**. | **Millimeters** | Transformed into PyBullet world via $\mathbf{T}_{W \leftarrow U}$. |
| **Lateral Flank Geometry** | Patient lying horizontally along Z. Flank skin at $X \approx +48\text{ mm}$, beam directed along **$-X$** toward spine. | **Millimeters** | Orthonormal matrix $\mathbf{R}_{W \leftarrow U}$ with columns $[[0,0,1]^\top, [-1,0,0]^\top, [0,-1,0]^\top]$. |

---

## 4. Standard Developer Verification Recipes

### 1. Full Unit Test Suite (21 Tests)
```bash
./.venv/bin/python -m pytest ReconstructionPipeline/tests/ -v
```

### 2. Run 6-Policy Benchmark (3 Trials per Method)
```bash
./.venv/bin/python ReconstructionPipeline/evaluation/benchmark_runner.py --num-trials 3
```

### 3. Launch Unified Server (Streaming to 3D Slicer)
```bash
./.venv/bin/python surgical_physics_bridge/src/unified_server.py --port 18944 --mode orbital --fps 30.0
```

### 4. Train 3D Diffusion Policy
```bash
./.venv/bin/python ReconstructionPipeline/planning/train_diffusion_policy.py --epochs 50 --batch-size 32
```

### 5. Train Submodular PPO
```bash
./.venv/bin/python ReconstructionPipeline/planning/train_submodular_ppo.py --num-updates 20 --num-steps 128
```
