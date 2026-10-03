# MTP - Surgical Physics Bridge

This repository contains the ongoing work for bridging **3D Slicer** with a **PyBullet** physics backend to create a surgical simulation environment. The goal is to provide a robust framework where rigid robots and deformable anatomical models simulated in PyBullet are mirrored in real-time within 3D Slicer's visualization ecosystem.

## 🏗️ What is Implemented

- **Client-Server Architecture**:
  - A decoupled ZeroMQ (ZMQ) network bridge separating the physics engine (Python) and the 3D Slicer UI.
  - The physics server publishes simulation state at ~50 Hz, and subscribes to incoming control commands.
- **Slicer Plugin (`SlicerSurgicalBridge`)**:
  - Contains a UI widget to connect/disconnect to the server.
  - Subscribes to the physics server to receive robotic link poses.
  - Dynamically creates and updates `vtkMRMLLinearTransformNode`s based on incoming link poses in real-time.
  - Provides a UI to send desired joint configurations (`q_des`) back to the simulation.
- **PyBullet Physics Backend (`pybullet_backend.py`)**:
  - **Robot Loading**: Ability to load URDF robots.
  - **Soft Body Anatomy**: Experimental support for Neo-Hookean deformable soft bodies using `p.loadSoftBody()`.
  - **Rigid Anatomy**: Support for standard rigid collision and visual meshes.
  - **Coordinate Conversion**: Integrates a translation layer (`convert_pose_pb_to_slicer`) to map between PyBullet's and Slicer's coordinate systems.
  - **Joint Control**: Standard positional joint motor control for robot manipulation.

## 🚧 What is Left / Critical Evaluation

The current state is a functional prototype for rigid body tracking, but is missing critical features to be considered a complete surgical simulator:

### 1. Deformable Mesh Synchronization
- **Current State**: While PyBullet is generating soft body simulations and calculating deformed vertices (`get_deformed_vertices()`), this data is **not** being transmitted over ZMQ, nor is 3D Slicer consuming it.
- **Action Required**: The ZMQ state message needs to include vertex arrays. The Slicer plugin must be updated to retrieve these arrays and dynamically update the `vtkPolyData` of a `vtkMRMLModelNode` in real-time. 

### 2. Network Optimization (JSON vs Binary)
- **Current State**: The ZMQ bridge transmits data using JSON strings. 
- **Action Required**: JSON is too slow and bloated for sending high-density mesh vertex data at 50Hz. The pipeline must be upgraded to use a binary serialization protocol like **MessagePack** or **Protocol Buffers (Protobuf)** to maintain real-time performance when transmitting soft-body deformations.

### 3. Haptics and Contact Forces
- **Current State**: `pybullet_backend.py` extracts contact points and normal forces (`get_contacts()`), but this data is not exposed to the server loop or sent to Slicer.
- **Action Required**: Force and contact data needs to be published back to the client. This is essential if the project intends to support haptic feedback devices (like Geomagic Touch) or visual force-feedback indicators in Slicer.

### 4. Virtual Fixtures
- **Current State**: The `set_virtual_fixture` method exists as a stub. It stores a signed distance field (SDF) and a policy, but the physics `step()` function entirely ignores it.
- **Action Required**: Virtual fixture constraints (e.g., active safety boundaries, forbidden regions) need to be mathematically enforced during the physics step, modifying the robot's control commands or applying penalty forces before stepping the simulation.

### 5. Control Interface
- **Current State**: Joint commands are currently sent by manually typing a comma-separated string into a Slicer text box (`0.0, 0.0, 0.0...`).
- **Action Required**: This is a placeholder. Real interaction requires integrating an external tracker (via OpenIGTLink), a virtual joystick widget, or inverse kinematics (IK) dragging directly within the Slicer 3D view.
