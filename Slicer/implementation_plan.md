# Slicer-Physics Backend Integration Architecture

Develop a robust, decoupled architecture integrating 3D Slicer with a physics backend (PyBullet initially, migrating to MuJoCo) for surgical simulation. This architecture will prioritize a strict separation between the clinical/semantic visualization loop and the real-time control/physics loop, enabling haptic control and advanced tissue simulation without being bottlenecked by GUI rendering.

## User Review Required

> [!IMPORTANT]
> The primary design principle is that **3D Slicer / SlicerIGT is not part of the 1 kHz safety-critical loop**. Slicer will handle clinical visualization, planning, and safety-region authoring (SDFs), while the separate physics backend handles robot dynamics, soft tissue, and haptic feedback. Slicer will consume state asynchronously via a latest-state buffer.

Please review the proposed stages and abstract backend interface below to ensure they align with your project goals before we begin implementation.

## Open Questions

> [!WARNING]
> - What specific robot model (e.g., PSM/dVRK, UR5) are we targeting for Stage 1?
> - For the shared memory / latest-state buffer (Stage 2), is there a preferred IPC mechanism you'd like to use (e.g., Python `multiprocessing.shared_memory`, zeroMQ, or a stripped-down ROS 2 topic purely for local transport)?
> - Where should the `SurgicalSimulationBackend` Python codebase live relative to the Slicer module? Should they be separate packages communicating via IPC?

## Proposed Changes

### 1. Abstract Physics Backend Interface

We will define an engine-neutral Python interface to prevent the Slicer module from tightly coupling to PyBullet or MuJoCo.

#### [NEW] `simulation_backend.py`
A base class defining the contract for the physics engine:
```python
class SurgicalSimulationBackend:
    def load_robot(self, robot_description): ...
    def load_anatomy(self, collision_mesh, visual_mesh, material_params): ...
    def set_joint_command(self, q_des, dq_des=None, tau=None): ...
    def step(self, dt): ...
    def get_link_poses(self) -> dict[str, "SE3"]: ...
    def get_tool_state(self) -> dict: ...
    def get_contacts(self) -> list: ...
    def get_deformed_vertices(self) -> "Nx3 array | None": ...
    def set_virtual_fixture(self, signed_distance_field, policy): ...
```

### 2. Implementation Roadmap

The development will be structured into the following verifiable stages:

#### Stage 0: Coordinate Contract
- **Goal**: Establish a single, documented conversion between Slicer RAS coordinates and the simulator world.
- **Action**: Define units, transform directions, quaternion conventions, and fixed patient/robot/tool frames.

#### Stage 1: Rigid PyBullet Bridge
- **Goal**: Implement the `SurgicalSimulationBackend` using PyBullet.
- **Action**: Load a URDF (e.g., dVRK/PSM) in PyBullet. Represent it visually in Slicer via MRML models and a transform hierarchy.

#### Stage 2: Timing and Reliability
- **Goal**: Implement asynchronous state transfer.
- **Action**: Create a lock-free/seqlock-style latest-state buffer. Slicer will consume this state at its render rate (e.g., 50 Hz). The simulator process will log physics rate, missed snapshots, and latency metrics.

#### Stage 3: Image-Guided Safety Scene
- **Goal**: Implement virtual fixtures authored in Slicer.
- **Action**: Convert a Slicer segmentation into a mesh and precompute a Signed Distance Field (SDF). Pass this to the simulator to compute a repulsive guidance field or velocity projection for the tool tip.

#### Stage 4: MuJoCo / Soft-Tissue Prototype
- **Goal**: Introduce deformable physics for tasks where rigid collision is inadequate.
- **Action**: Implement a MuJoCo backend for a calibrated benchmark tissue scene. Send deformed vertex data back to Slicer for visualization.

#### Stage 5: Control + Haptics
- **Goal**: Close the high-frequency loop.
- **Action**: Implement a separate servo process with an explicit watchdog, decoupled entirely from the Slicer GUI.

#### Stage 6: Surgical Task Benchmark
- **Goal**: Validate the system on a relevant surgical task.
- **Action**: Implement a task (e.g., tumor-margin avoidance or needle insertion) and compare guidance policies (no guidance vs. virtual fixture vs. learned policy).

## Verification Plan

### Automated Tests
- **Coordinate Transformations**: Random-pose round trips between Slicer RAS and the simulator frame must yield negligible pose error.
- **Forward Kinematics**: Verify that Slicer link poses match the simulator's forward kinematics over a full workspace sweep.
- **Safety Field**: Automated tests driving a tool path into forbidden regions must correctly trigger virtual fixture responses.

### Manual / System Verification
- **Performance Profiling**: Measure and report 95th/99th percentile latency, dropped frames, and state age at display.
- **Decoupling Validation**: Demonstrate bounded, stable haptic/physics behavior even if the Slicer visualization thread is intentionally stalled or delayed.
