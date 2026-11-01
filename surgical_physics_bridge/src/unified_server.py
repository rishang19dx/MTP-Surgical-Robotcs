"""
Unified Surgical Physics & Autonomous 3D Ultrasound Reconstruction Server.
Bridges 1 kHz PyBullet physical contact dynamics and GPU model-based acoustic ray tracing
directly to 3D Slicer via OpenIGTLink binary sockets.

Features:
1. Real Franka Emika Panda 7-DoF manipulator with attached ultrasound transducer probe.
2. Clinical operating room suite: examination table, patient lumbar torso, and soft-tissue contact dynamics.
3. Closed-loop physical contact force (F_N) and ultrasound acoustic coupling.
4. TotalSegmentator multi-vertebra NIfTI patient anatomy loading.
5. High-fidelity dual-window visualization:
   - PyBullet 3D Window: Robot arm scanning patient, contact force arrows, acoustic beam cone, and real-time HUD.
   - 3D Slicer Window: Synchronized 2D live ultrasound feed, 3D moving probe model, and reconstructed bone point cloud.
"""

import sys
import os
import time
import threading
import argparse
import json
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import pyigtl
import pyigtl.comm
import socket
import pybullet as p
import pybullet_data

# Hardened RequestHandler for OpenIGTLinkServer
class RobustTCPRequestHandler(pyigtl.comm.TCPRequestHandler):
    """
    Hardened RequestHandler for OpenIGTLinkServer:
    - 5 ms timeout for non-blocking command reception.
    - 2.0 s timeout for binary message transmission (US B-mode slices and point clouds).
    - Ignores momentary buffer timeouts without dropping client connection.
    """
    def handle(self):
        self.server._connected = True
        while not self.server.communication_thread_stop_requested:
            # 1. Non-blocking command reception
            self.request.settimeout(0.005)
            try:
                while self.server._receive_message_from_socket(self.request):
                    if self.server.communication_thread_stop_requested:
                        break
            except Exception:
                pass

            # 2. Generous send timeout for large binary images and point clouds
            self.request.settimeout(2.0)
            try:
                while self.server._send_queued_message_from_socket(self.request):
                    pass
            except (socket.timeout, TimeoutError):
                # Socket buffer momentarily saturated by Slicer UI rendering; keep client alive
                continue
            except (BrokenPipeError, ConnectionResetError):
                self.server._communication_error_occurred()
                break
            except Exception:
                self.server._communication_error_occurred()
                break

pyigtl.comm.TCPRequestHandler = RobustTCPRequestHandler

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from surgical_physics_bridge.src.pybullet_backend import PyBulletBackend
from surgical_physics_bridge.src.coordinates import convert_pose_pb_to_slicer, pybullet_to_slicer_ras_matrix
from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
from ReconstructionPipeline.planning.heuristic_planner import HeuristicScanPlanner
from ReconstructionPipeline.kinematics.kinova_kinematics import KinovaGen3Kinematics
from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator


class UnifiedSurgicalBridgeServer:
    """
    Unified Runtime Bridge synchronizing PyBullet physics, GPU acoustic slicing,
    and 3D Slicer OpenIGTLink communications.
    """

    def __init__(
        self,
        port: int = 18944,
        urdf_path: Optional[str] = None,
        patient_dir: Optional[str] = None,
        gui: bool = False,
        publish_fps: float = 30.0,
        scan_mode: str = "orbital",
        device: Optional[torch.device] = None,
    ):
        self.port = port
        self.gui = gui
        self.publish_fps = publish_fps
        self.publish_interval = 1.0 / publish_fps
        self.scan_mode = scan_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. OpenIGTLink Server
        print(f"[*] Initializing OpenIGTLink Server on port {self.port}...")
        self.server = pyigtl.OpenIGTLinkServer(port=self.port)

        # 2. PyBullet Physics Backend
        print(f"[*] Initializing PyBullet Physics Suite (1 kHz, GUI={gui})...")
        self.backend = PyBulletBackend(gui=gui)
        self.robot_id = None
        self.ee_idx = 11  # Default Franka Panda grasptarget link
        self.arm_joints = list(range(7))
        self.probe_body_id = None
        self.hud_text_id = -1

        # Build operating room scene in PyBullet
        self._build_pybullet_scene(urdf_path)

        # 3. Kinematics Controller
        self.kinematics = KinovaGen3Kinematics()

        # 4. TotalSegmentator Patient Asset Loading
        patient_source = self._resolve_patient_data(patient_dir)

        # 5. Ultrasound Simulation & Reconstruction Environment
        print("[*] Initializing Ultrasound Reconstruction Engine (Dorsal Spine Scanning)...")
        self.env = UltrasoundReconstructionEnv(
            patient_data=patient_source,
            max_steps=200,
            scan_approach="dorsal",
            pybullet_backend=self.backend,
            device=self.device,
        )
        self.obs = self.env.reset()
        self.update_robot_ik_and_visuals(teleport=True)

        # 6. Trajectory Planner
        self.planner = HeuristicScanPlanner()
        self.trajectory = self._build_trajectory(self.scan_mode)
        self.traj_idx = 0

        # State tracking
        self.is_running = True
        self.dt_physics = 0.001  # 1 kHz
        self.last_publish_time = time.time()
        self.total_frames_published = 0
        self.last_points_sent = 0
        self.last_points_publish_time = 0.0
        self.physics_lock = threading.Lock()

    def _safe_send(self, msg):
        """Dispatches an OpenIGTLink message non-blockingly, suppressing socket disconnect errors."""
        try:
            self.server.send_message(msg, wait=False)
        except Exception:
            pass

    def _resolve_patient_data(self, patient_dir: Optional[str]) -> Any:
        """Locates or synthesizes TotalSegmentator clinical patient data."""
        if patient_dir and os.path.isdir(patient_dir):
            print(f"[+] Using specified TotalSegmentator subject directory: {patient_dir}")
            return patient_dir

        default_totalseg = os.path.join(PROJECT_ROOT, "data", "sample_totalsegmentator_patient")
        if os.path.isdir(default_totalseg):
            print(f"[+] Using default TotalSegmentator patient dataset: {default_totalseg}")
            return default_totalseg

        print("[*] Sample patient dataset not found. Synthesizing TotalSegmentator patient assets...")
        PatientAssetGenerator.export_totalsegmentator_dataset(default_totalseg)
        return default_totalseg

    def _build_pybullet_scene(self, custom_urdf: Optional[str] = None):
        """Constructs clinical operating suite: robot, patient table, soft torso, and probe."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        if self.gui:
            # Clean presentation view
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.15,
                cameraYaw=48,
                cameraPitch=-24,
                cameraTargetPosition=[0.45, 0.0, 0.38]
            )

        # 1. Floor
        p.loadURDF("plane.urdf", [0, 0, 0])

        # 2. Patient Examination Bed
        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.42, 0.30, 0.15])
        table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.42, 0.30, 0.15], rgbaColor=[0.68, 0.72, 0.75, 1.0])
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[0.55, 0.0, 0.15]
        )

        # 3. Patient Torso Body (Soft Tissue Volume)
        torso_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.16, 0.24, 0.06])
        torso_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.16, 0.24, 0.06], rgbaColor=[0.88, 0.73, 0.61, 1.0])
        self.torso_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=torso_col,
            baseVisualShapeIndex=torso_vis,
            basePosition=[0.55, 0.0, 0.36]
        )
        self.backend.anatomy_body_id = self.torso_id

        # 4. Robot Manipulator
        robot_urdf = custom_urdf if (custom_urdf and os.path.exists(custom_urdf)) else "franka_panda/panda.urdf"
        try:
            self.robot_id = p.loadURDF(robot_urdf, [0.0, 0.0, 0.10], useFixedBase=True)
            self.backend.robot_id = self.robot_id
            self.backend._map_links()
            self.has_robot = True
            print(f"[+] Loaded robot manipulator from {robot_urdf} (ID: {self.robot_id})")
        except Exception as e:
            print(f"[!] Warning: Could not load robot URDF: {e}")
            self.has_robot = False

        # 5. Ultrasound Probe Tool rigidly attached to End-Effector
        if self.has_robot:
            probe_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.014, length=0.09, rgbaColor=[0.2, 0.5, 0.85, 1.0])
            probe_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.014, height=0.09)
            self.probe_body_id = p.createMultiBody(
                baseMass=0.2,
                baseCollisionShapeIndex=probe_col,
                baseVisualShapeIndex=probe_vis,
                basePosition=[0.55, 0.0, 0.45]
            )
            self.backend.probe_body_id = self.probe_body_id
            # Constrain probe tool firmly to Franka Panda hand
            p.createConstraint(
                self.robot_id, self.ee_idx,
                self.probe_body_id, -1,
                p.JOINT_FIXED,
                [0, 0, 0], [0, 0, 0], [0, 0, 0.045]
            )

    def _build_trajectory(self, mode: str) -> Optional[np.ndarray]:
        if mode == "orbital":
            return self.planner.generate_orbital_trajectory(radii=[18.0, 28.0, 36.0], points_per_orbit=30)
        elif mode == "raster":
            return self.planner.generate_linear_raster_trajectory(num_sweep_lines=8, points_per_line=15)
        return None

    def handle_incoming_messages(self):
        """Processes control signals, virtual fixtures, and ROI delineations from 3D Slicer."""
        messages = self.server.get_latest_messages() if hasattr(self.server, "get_latest_messages") else []
        for message in messages:
            if isinstance(message, pyigtl.StringMessage) and message.device_name == "CMD":
                try:
                    cmd_data = json.loads(message.string)
                    if cmd_data.get("reset"):
                        self.obs = self.env.reset()
                        self.update_robot_ik_and_visuals(teleport=True)
                        self.traj_idx = 0
                        print("[+] Reset command received from Slicer.")

                    if "mode" in cmd_data:
                        self.scan_mode = cmd_data["mode"].lower()
                        self.trajectory = self._build_trajectory(self.scan_mode)
                        self.traj_idx = 0
                        print(f"[+] Scan mode set to: {self.scan_mode.upper()}")

                    if "roi_target" in cmd_data:
                        roi_ras = np.array(cmd_data["roi_target"], dtype=np.float64)
                        t_conv_inv = np.linalg.inv(pybullet_to_slicer_ras_matrix())
                        target_pb = (t_conv_inv[:3, :3] @ roi_ras + t_conv_inv[:3, 3])
                        self.env.current_pose_params[0] = np.clip(target_pb[0], self.env.x_bounds[0], self.env.x_bounds[1])
                        self.env.current_pose_params[1] = np.clip(target_pb[2], self.env.z_bounds[0], self.env.z_bounds[1])
                        self.obs = self.env._observe_current_state()

                except json.JSONDecodeError as err:
                    print(f"[!] Malformed CMD JSON: {err}")

            elif isinstance(message, pyigtl.StringMessage) and message.device_name == "VirtualFixture":
                try:
                    vf_data = json.loads(message.string)
                    if "vertices" in vf_data and "indices" in vf_data:
                        self.backend.set_virtual_fixture(vf_data, vf_data.get("policy", "keep_out"))
                except json.JSONDecodeError as err:
                    print(f"[!] Malformed VirtualFixture JSON: {err}")

    def update_robot_ik_and_visuals(self, teleport: bool = False):
        """Solves 7-DoF IK and draws real-time contact force vectors & ultrasound beam cone in PyBullet."""
        if not self.has_robot or self.robot_id is None:
            return

        c1 = float(self.env.current_pose_params[0])
        c2 = float(self.env.current_pose_params[1])
        yaw = float(self.env.current_pose_params[2])
        pitch = float(self.env.current_pose_params[3])
        fn = float(self.env.current_normal_force)
        coupling = float(self.env.current_coupling)

        # Indentation depth from Kelvinlet elasticity
        delta_m = (fn / 2500.0)

        # PyBullet world frame coordinates:
        # Patient torso center is at (X=0.55, Y=0.0, Z=0.36), dorsal surface at Z=0.42
        x_pb = 0.55 + c1 * 1e-3
        y_pb = 0.0 + c2 * 1e-3
        z_contact = 0.42 - delta_m
        z_ee = z_contact + 0.09  # End-effector mount elevation

        target_pos = [x_pb, y_pb, z_ee]
        target_orn = p.getQuaternionFromEuler([np.pi + pitch, 0.0, yaw])

        # Inverse Kinematics solution
        joint_targets = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_idx,
            target_pos,
            target_orn,
            maxNumIterations=40,
            residualThreshold=1e-4,
        )

        # Command or teleport revolute joints
        if teleport:
            for j_idx, q_val in zip(self.arm_joints, joint_targets[:7]):
                p.resetJointState(self.robot_id, j_idx, q_val)
        else:
            p.setJointMotorControlArray(
                self.robot_id,
                self.arm_joints,
                p.POSITION_CONTROL,
                targetPositions=joint_targets[:7],
                forces=[150.0] * 7,
            )

        # PyBullet GUI visual feedback
        if self.gui:
            # 1. Contact Force Vector (Vertical Red Arrow)
            force_vec_end = [x_pb, y_pb, z_contact + fn * 0.02]
            p.addUserDebugLine(
                [x_pb, y_pb, z_contact],
                force_vec_end,
                [1.0, 0.1, 0.1],
                lineWidth=4,
                lifeTime=self.publish_interval * 1.5,
            )

            # 2. Ultrasound Acoustic Slicing Plane (Cyan Frustum)
            half_w = 0.018  # 36 mm transducer lateral width
            depth = 0.06    # 60 mm acoustic penetration depth
            # Project 4 corners into patient spine
            c_bl = [x_pb - half_w, y_pb, z_contact - depth]
            c_br = [x_pb + half_w, y_pb, z_contact - depth]
            c_tl = [x_pb - half_w, y_pb, z_contact]
            c_tr = [x_pb + half_w, y_pb, z_contact]
            cyan = [0.0, 0.85, 1.0]
            p.addUserDebugLine(c_tl, c_bl, cyan, lineWidth=2, lifeTime=self.publish_interval * 1.5)
            p.addUserDebugLine(c_tr, c_br, cyan, lineWidth=2, lifeTime=self.publish_interval * 1.5)
            p.addUserDebugLine(c_bl, c_br, cyan, lineWidth=2, lifeTime=self.publish_interval * 1.5)

            # 3. Dynamic On-Screen HUD Overlay
            pts_count = len(self.env.accumulator.get_points())
            hud_msg = (
                f"ROBOTIC ULTRASOUND OPERATING SUITE\n"
                f"Patient: TotalSegmentator Lumbar L1-L5\n"
                f"Normal Force: {fn:.2f} N (Elastic Coupling: {coupling*100:.1f}%)\n"
                f"Reconstructed Bone Points: {pts_count}\n"
                f"Slicer 3D Bridge: {self.publish_fps:.0f} FPS Streaming"
            )
            self.hud_text_id = p.addUserDebugText(
                hud_msg,
                textPosition=[0.25, -0.35, 0.65],
                textColorRGB=[0.1, 0.95, 0.4],
                textSize=1.1,
                lifeTime=self.publish_interval * 1.5,
                replaceItemUniqueId=self.hud_text_id if self.hud_text_id != -1 else -1,
            )

    def _execute_scan_step_and_publish(self, current_time: float):
        """Executes one scan step in the environment and publishes multimodality packets."""
        # Autonomous Scan Step
        if self.trajectory is not None and self.traj_idx < len(self.trajectory):
            target_p = self.trajectory[self.traj_idx]
            curr_p = self.env.current_pose_params

            dx = np.clip((target_p[0] - curr_p[0]) / self.env.action_scale_trans, -1.0, 1.0)
            dz = np.clip((target_p[1] - curr_p[1]) / self.env.action_scale_trans, -1.0, 1.0)
            dyaw = np.clip((target_p[2] - curr_p[2]) / self.env.action_scale_yaw, -1.0, 1.0)
            dpitch = np.clip((target_p[3] - curr_p[3]) / self.env.action_scale_pitch, -1.0, 1.0)

            action = np.array([dx, dz, dyaw, dpitch], dtype=np.float32)
            self.obs, reward, term, trunc, info = self.env.step(action)
            self.traj_idx += 1

            if term or trunc or self.traj_idx >= len(self.trajectory):
                total_pts = len(self.env.accumulator.get_points())
                print(
                    f"[+] Trajectory Pass Completed: "
                    f"Total Reconstructed Points={total_pts}, "
                    f"Coverage={info['coverage_ratio']*100:.1f}%, "
                    f"NormalForce={info['normal_force']:.2f}N, Coupling={info['acoustic_coupling']*100:.1f}%"
                )
                self.traj_idx = 0
                # Retain accumulator points across scanning passes to allow full multi-vertebra reconstruction
                self.obs = self.env.reset(clear_accumulator=False)

        # --- Network Publishing ---

        # A. Ultrasound Probe Transform in Slicer RAS
        t_probe = self.env._build_probe_matrix(*self.env.current_pose_params)
        msg_probe = pyigtl.TransformMessage(t_probe.astype(np.float64), device_name="UltrasoundProbe")
        self._safe_send(msg_probe)

        # B. 2D B-mode Ultrasound Image (US_Image)
        bmode = (np.clip(self.obs["bmode"], 0.0, 1.0) * 255.0).astype(np.uint8)
        img_msg = pyigtl.ImageMessage(bmode[:, :, np.newaxis], device_name="US_Image")
        self._safe_send(img_msg)

        # C. Incremental 3D Reconstructed Bone Points (throttled to avoid TCP buffer congestion)
        pts = self.env.accumulator.get_points()
        n_pts = len(pts)
        if n_pts > 0 and (n_pts >= self.last_points_sent + 5 or (current_time - self.last_points_publish_time >= 0.15)):
            pts_msg = pyigtl.PointMessage(positions=pts.astype(np.float32), device_name="ReconstructedBone")
            self._safe_send(pts_msg)
            self.last_points_sent = n_pts
            self.last_points_publish_time = current_time

        # D. Real-Time Contact Forces
        curr_f = float(self.env.current_normal_force)
        c_array = np.array(
            [[self.env.current_pose_params[0], self.env.get_skin_surface(*self.env.current_pose_params[:2]),
              self.env.current_pose_params[1], 0.0, -1.0, 0.0, curr_f]],
            dtype=np.float32,
        )
        force_msg = pyigtl.StringMessage(json.dumps(c_array.tolist()), device_name="ContactForces")
        self._safe_send(force_msg)

        self.last_publish_time = current_time
        self.total_frames_published += 1

    def _physics_worker(self):
        """Dedicated 1000 Hz physics simulation loop decoupled from Slicer streaming."""
        target_dt = self.dt_physics
        while self.is_running:
            t0 = time.perf_counter()
            with self.physics_lock:
                self.update_robot_ik_and_visuals()
                self.backend.step(target_dt)
            elapsed = time.perf_counter() - t0
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _network_streaming_worker(self):
        """Dedicated network streaming loop running at display FPS (30-60 Hz)."""
        while self.is_running:
            t0 = time.perf_counter()
            with self.physics_lock:
                self.handle_incoming_messages()
            self._execute_scan_step_and_publish(time.time())
            elapsed = time.perf_counter() - t0
            sleep_time = self.publish_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def step_simulation_and_stream(self):
        """Synchronously steps physics and publishes for unit tests and deterministic execution."""
        with self.physics_lock:
            self.handle_incoming_messages()
            self.update_robot_ik_and_visuals()
            self.backend.step(self.dt_physics)

        current_time = time.time()
        if current_time - self.last_publish_time >= self.publish_interval:
            self._execute_scan_step_and_publish(current_time)

        time.sleep(self.dt_physics)

    def run(self):
        """Main service loop with 1000 Hz physics decoupled from network streaming."""
        print(f"==================================================================")
        print(f"[*] Unified Surgical Physics & Ultrasound Reconstruction Server Running")
        print(f"[*] Port: {self.port} | Display FPS: {self.publish_fps} | Mode: {self.scan_mode.upper()} | GUI: {self.gui}")
        print(f"[*] Multi-Rate Decoupling: 1000 Hz Physics Worker + {self.publish_fps} Hz Slicer Streamer")
        print(f"==================================================================")
        physics_thread = threading.Thread(target=self._physics_worker, daemon=True, name="PyBulletPhysicsThread")
        physics_thread.start()
        try:
            self._network_streaming_worker()
        except KeyboardInterrupt:
            print("\n[*] Stopping Unified Server...")
        finally:
            self.is_running = False
            physics_thread.join(timeout=1.0)
            self.server.stop()
            self.backend.disconnect()
            print("[+] Unified Server cleanly shutdown.")


def main():
    parser = argparse.ArgumentParser(description="Unified Surgical Physics & Ultrasound Reconstruction Server")
    parser.add_argument("--port", type=int, default=18944, help="OpenIGTLink port (default 18944)")
    parser.add_argument("--urdf", type=str, default="", help="Path to robot URDF")
    parser.add_argument("--patient-dir", type=str, default="", help="Path to TotalSegmentator patient directory")
    parser.add_argument("--gui", action="store_true", help="Launch native PyBullet GUI")
    parser.add_argument("--fps", type=float, default=30.0, help="Publishing rate to Slicer (Hz)")
    parser.add_argument("--mode", type=str, choices=["orbital", "raster", "interactive"], default="orbital",
                        help="Scan execution mode: orbital, raster, or interactive")
    args = parser.parse_args()

    server = UnifiedSurgicalBridgeServer(
        port=args.port,
        urdf_path=args.urdf if args.urdf else None,
        patient_dir=args.patient_dir if args.patient_dir else None,
        gui=args.gui,
        publish_fps=args.fps,
        scan_mode=args.mode,
    )
    server.run()


if __name__ == "__main__":
    main()
