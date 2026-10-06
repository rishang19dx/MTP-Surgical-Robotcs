# MTP - Surgical Physics Bridge

This repository contains the ongoing work for bridging **3D Slicer** with a **PyBullet** physics backend to create a surgical simulation environment. The goal is to provide a robust framework where rigid robots and deformable anatomical models simulated in PyBullet are mirrored in real-time within 3D Slicer's visualization ecosystem.

## 🏗️ What is Implemented

- **Client-Server Architecture**:
  - A decoupled OpenIGTLink network bridge separating the physics engine (Python) and the 3D Slicer UI.
  - The physics server publishes simulation state (robot link transforms and deformable mesh vertices) at ~50 Hz, and receives incoming control commands.
- **Slicer Plugin (`SlicerSurgicalBridge`)**:
  - Developed as a high-performance **C++ Loadable Slicer Module**.
  - Uses `vtkMRMLIGTLConnectorNode` to handle OpenIGTLink sockets natively in a background thread without freezing the UI.
  - Exposes Python APIs for UI logic while relying on C++ for heavy binary data processing.
- **PyBullet Physics Backend (`pybullet_backend.py`)**:
  - **Robot Loading**: Ability to load URDF robots.
  - **Soft Body Anatomy**: Experimental support for Neo-Hookean deformable soft bodies using `p.loadSoftBody()`.
  - **Rigid Anatomy**: Support for standard rigid collision and visual meshes.
  - **Coordinate Conversion**: Integrates a translation layer (`convert_pose_pb_to_slicer`) to map between PyBullet's and Slicer's coordinate systems.
  - **Joint Control**: Standard positional joint motor control for robot manipulation.

## 🎯 Completed Phases

The following features have been successfully implemented:

### Phase 1: Architectural Overhaul & Data Transport
- **Network Optimization (JSON vs Binary)**: Transitioned from ZMQ/JSON to the **OpenIGTLink** protocol (via `pyigtl` on the server and `SlicerOpenIGTLinkIF` on the client) for high-performance binary serialization of transforms and high-density mesh vertex data.

### Phase 2: Feature Completeness for Surgical Simulation
- **Deformable Mesh Synchronization (Client Side)**: The Slicer C++ plugin now processes incoming OpenIGTLink arrays (`DeformedAnatomy`) and dynamically updates the `vtkPolyData` of a `vtkMRMLModelNode` in real-time without reallocating memory.
- **Haptics and Contact Forces**: `pybullet_backend.py` extracts contact points and normal forces (`get_contacts()`), packaging them into an OpenIGTLink `ContactForces` NDArrayMessage. The Slicer C++ plugin reconstructs these forces into interactive vector arrows via `vtkPolyData` lines dynamically.
- **Virtual Fixtures**: The Slicer C++ plugin provides a "Virtual Fixtures" UI to select a `vtkMRMLModelNode`. Clicking the export button serializes the full mesh (vertices and indices) into a JSON `StringMessage` via OpenIGTLink. The PyBullet server natively decodes this and spawns a static collision body (`keep_out` policy) that physically blocks the robot from entering the forbidden zone.

## 🚧 What is Left / Critical Evaluation

The simulator now handles soft bodies, haptics, and virtual fixtures, but one major component remains for real-time human interaction:

### 1. Control Interface
- **Current State**: Joint commands are currently sent by manually typing a comma-separated string into a Slicer text box (`0.0, 0.0, 0.0...`).
- **Action Required**: This is a placeholder. Real interaction requires integrating an external tracker (via OpenIGTLink), a virtual joystick widget, or inverse kinematics (IK) dragging directly within the Slicer 3D view.

## 🛠️ Prerequisites

### Python Dependencies (Physics Server)
The standalone physics server requires a standard Python environment (Python 3.8+ recommended). Install the required dependencies:
```bash
pip install -r surgical_physics_bridge/requirements.txt
```
*(Dependencies include: `pybullet`, `numpy`, `scipy`, `pyigtl`)*

### 3D Slicer (Client UI)
You will need to install **3D Slicer** (version 5.0 or later is recommended). 
Since the `SlicerSurgicalBridge` is a C++ Loadable module, you must build it against your Slicer installation:
```bash
mkdir SlicerSurgicalBridge-build && cd SlicerSurgicalBridge-build
cmake -DSlicer_DIR=/path/to/Slicer-SuperBuild/Slicer-build ../SlicerSurgicalBridge
make
```
## 🚀 How to Run

### Step 1: Start the Physics Server
Navigate to the `surgical_physics_bridge` directory and launch the standalone PyBullet server. You must provide a URDF file for the robot.

```bash
cd surgical_physics_bridge
python src/server.py --urdf <path_to_your_robot.urdf>
```
*Optional Flags:*
- `--port`: OpenIGTLink port (default is `18944`).
- `--gui`: Add this flag to open the native PyBullet GUI to visualize the physics engine alongside Slicer.

### Step 2: Load the Slicer Plugin
1. Build the C++ module as described in the Prerequisites.
2. Open **3D Slicer**.
3. Go to `Edit` -> `Application Settings` -> `Modules`.
4. Add your `SlicerSurgicalBridge-build/lib/Slicer-5.x/qt-loadable-modules` folder to your **Additional module paths**.
4. Restart 3D Slicer.
5. In the module drop-down, look for **Surgical Physics Bridge** (under the `Simulation` category).

### Step 3: Connect and Interact
1. In the Slicer module UI, click **"Connect to Physics Server"**. 
   *(Slicer will now start subscribing to the physics state and updating internal `vtkMRMLLinearTransformNode`s.)*
2. To send joint commands, enter a comma-separated list of joint angles (e.g., `0.1, -0.5, 1.2, ...`) matching your robot's controlled joints into the `q_des` field and click **"Send Command"**.
