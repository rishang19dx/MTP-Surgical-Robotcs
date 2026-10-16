"""
Unified Surgical Physics & Autonomous 3D Ultrasound Reconstruction Server.
Bridges 1 kHz PyBullet physical contact dynamics and GPU model-based acoustic ray tracing
directly to 3D Slicer via OpenIGTLink binary sockets.

Features:
1. Closed-loop coupling between physical contact force (F_N) and ultrasound acoustic coupling.
2. Synchronous streaming of:
   - Robot link transforms and UltrasoundProbe pose (TransformMessage)
   - Live 2D simulated B-mode ultrasound feed (ImageMessage: US_Image)
   - Incremental 3D bone point cloud / surface mesh (PointMessage & PolyDataMessage)
   - Real-time contact force vectors for Slicer haptic/arrow visualization (NDArrayMessage: ContactForces)
3. Bidirectional clinical control:
   - Virtual keep-out fixtures (StringMessage: VirtualFixture)
   - Clinician ROI target delineation and automated IK pose initialization (StringMessage: CMD)
   - Interactive or autonomous trajectory scanning (raster, orbital, active DRL)
"""

import sys
import os
import time
import argparse
import json
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import pyigtl

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
        self.publish_fps = publish_fps
        self.publish_interval = 1.0 / publish_fps
        self.scan_mode = scan_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. OpenIGTLink Server
        print(f"[*] Initializing OpenIGTLink Server on port {self.port}...")
        self.server = pyigtl.OpenIGTLinkServer(port=self.port)

        # 2. PyBullet Physics Backend
        print("[*] Initializing PyBullet Physics Backend (1 kHz)...")
        self.backend = PyBulletBackend(gui=gui)
        self.has_robot = False
        if urdf_path and os.path.exists(urdf_path):
            self.has_robot = self.backend.load_robot(urdf_path)
            print(f"[+] Loaded robot description from {urdf_path}")
        else:
            # Fallback to loading standard manipulator from pybullet_data if available
            try:
                import pybullet_data
                fallback_urdf = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa", "model.urdf")
                if os.path.exists(fallback_urdf):
                    self.has_robot = self.backend.load_robot(fallback_urdf)
                    print(f"[+] Loaded fallback KUKA iiwa robot from pybullet_data")
            except Exception as e:
                print(f"[!] Robot loading skipped: {e}")

        # 3. Kinematics & IK Controller
        self.kinematics = KinovaGen3Kinematics()

        # 4. Ultrasound Simulation & Reconstruction Environment
        print("[*] Initializing Ultrasound Reconstruction Engine...")
        self.env = UltrasoundReconstructionEnv(
            patient_data=patient_dir if patient_dir else None,
            max_steps=200,
            pybullet_backend=self.backend,
            device=self.device,
        )
        self.obs = self.env.reset()

        # 5. Trajectory Planners
        self.planner = HeuristicScanPlanner()
        self.trajectory = self._build_trajectory(self.scan_mode)
        self.traj_idx = 0

        # State tracking
        self.is_running = True
        self.dt_physics = 0.001 # 1 kHz
        self.last_publish_time = time.time()
        self.total_frames_published = 0

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
            # 1. String Commands (CMD)
            if isinstance(message, pyigtl.StringMessage) and message.device_name == "CMD":
                try:
                    cmd_data = json.loads(message.string)
                    # Reset command
                    if cmd_data.get("reset"):
                        self.obs = self.env.reset()
                        self.traj_idx = 0
                        print("[+] Reset command received from Slicer.")

                    # Change Scan Mode
                    if "mode" in cmd_data:
                        self.scan_mode = cmd_data["mode"].lower()
                        self.trajectory = self._build_trajectory(self.scan_mode)
                        self.traj_idx = 0
                        print(f"[+] Scan mode set to: {self.scan_mode.upper()}")

                    # Manual Joint Angles
                    if "q_des" in cmd_data:
                        self.backend.set_joint_command(cmd_data["q_des"])

                    # Clinician ROI Target Markups in Slicer RAS coordinates
                    if "roi_target" in cmd_data:
                        roi_ras = np.array(cmd_data["roi_target"], dtype=np.float64)
                        print(f"[+] Received Slicer ROI target: {roi_ras}")
                        # Invert Slicer RAS matrix to obtain PyBullet frame target
                        t_conv_inv = np.linalg.inv(pybullet_to_slicer_ras_matrix())
                        target_pb = (t_conv_inv[:3, :3] @ roi_ras + t_conv_inv[:3, 3])
                        # Target pose in probe frame
                        self.env.current_pose_params[0] = np.clip(target_pb[0], self.env.x_bounds[0], self.env.x_bounds[1])
                        self.env.current_pose_params[1] = np.clip(target_pb[2], self.env.z_bounds[0], self.env.z_bounds[1])
                        self.obs = self.env._observe_current_state()

                except json.JSONDecodeError as err:
                    print(f"[!] Malformed CMD JSON from Slicer: {err}")

            # 2. Virtual Keep-out Fixtures
            elif isinstance(message, pyigtl.StringMessage) and message.device_name == "VirtualFixture":
                try:
                    vf_data = json.loads(message.string)
                    if "vertices" in vf_data and "indices" in vf_data:
                        self.backend.set_virtual_fixture(vf_data, vf_data.get("policy", "keep_out"))
                        print("[+] Virtual Fixture collision mesh updated in PyBullet.")
                except json.JSONDecodeError as err:
                    print(f"[!] Malformed VirtualFixture JSON: {err}")

            # 3. Interactive Delta Actions (cmd_action)
            elif isinstance(message, pyigtl.NDArrayMessage) and message.device_name == "cmd_action":
                action = message.ndarray.flatten().astype(np.float32)
                if len(action) == 4:
                    self.obs, reward, term, trunc, info = self.env.step(action)
                    if term or trunc:
                        self.obs = self.env.reset()

            # 4. Direct Joint Target Array (q_des)
            elif isinstance(message, pyigtl.NDArrayMessage) and message.device_name == "q_des":
                self.backend.set_joint_command(message.ndarray.tolist())

    def step_simulation_and_stream(self):
        """Steps 1 kHz PyBullet physics and publishes multimodality packets at target FPS."""
        # 1. Process network input
        self.handle_incoming_messages()

        # 2. Step 1 kHz PyBullet physics
        self.backend.step(self.dt_physics)

        # 3. Publish at Slicer Display Rate (30-60 Hz)
        current_time = time.time()
        if current_time - self.last_publish_time >= self.publish_interval:
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
                    print(
                        f"[+] Trajectory Pass Completed: "
                        f"Points={info['total_points']}, Coverage={info['coverage_ratio']*100:.1f}%, "
                        f"NormalForce={info['normal_force']:.2f}N, Coupling={info['acoustic_coupling']:.2f}"
                    )
                    self.traj_idx = 0
                    self.obs = self.env.reset()

            # --- Network Publishing ---

            # A. Robot Link Transforms
            if self.has_robot:
                link_poses = self.backend.get_link_poses()
                for link_name, pose_mat in link_poses.items():
                    msg = pyigtl.TransformMessage(np.array(pose_mat), device_name=link_name)
                    self.server.send_message(msg)

            # B. Ultrasound Probe Transform in Slicer RAS
            t_probe = self.env._build_probe_matrix(*self.env.current_pose_params)
            msg_probe = pyigtl.TransformMessage(t_probe.astype(np.float64), device_name="UltrasoundProbe")
            self.server.send_message(msg_probe)

            # C. 2D B-mode Ultrasound Image (US_Image)
            bmode = (self.obs["bmode"] * 255.0).astype(np.uint8)
            img_msg = pyigtl.ImageMessage(bmode[:, :, np.newaxis], device_name="US_Image")
            self.server.send_message(img_msg)

            # D. Incremental 3D Reconstructed Bone Points
            pts = self.env.accumulator.get_points()
            if len(pts) > 0:
                pts_msg = pyigtl.PointMessage(positions=pts.astype(np.float32), device_name="ReconstructedBone")
                self.server.send_message(pts_msg)

                # PolyData for Slicer Model node
                poly_msg = pyigtl.PolyDataMessage(points=pts.astype(np.float32), device_name="DeformedAnatomy")
                self.server.send_message(poly_msg)

            # E. Real-Time Contact Forces (Nx7: [px, py, pz, nx, ny, nz, fn])
            contacts = self.backend.get_contacts()
            if contacts:
                contact_list = []
                for c in contacts:
                    p = c["position_on_b"]
                    n = c["contact_normal_on_b"]
                    fn = c["normal_force"]
                    contact_list.append([p[0], p[1], p[2], n[0], n[1], n[2], fn])
                c_array = np.array(contact_list, dtype=np.float32)
            else:
                # Fallback to analytical contact normal force from environment
                curr_f = self.env.current_normal_force
                c_array = np.array(
                    [[self.env.current_pose_params[0], self.env.get_skin_surface_y(*self.env.current_pose_params[:2]),
                      self.env.current_pose_params[1], 0.0, -1.0, 0.0, curr_f]],
                    dtype=np.float32,
                )
            force_msg = pyigtl.NDArrayMessage(c_array, device_name="ContactForces")
            self.server.send_message(force_msg)

            self.last_publish_time = current_time
            self.total_frames_published += 1

        # Match real-time physics clock
        time.sleep(self.dt_physics)

    def run(self):
        """Main service loop."""
        print(f"==================================================================")
        print(f"[*] Unified Surgical Physics & Ultrasound Reconstruction Server Running")
        print(f"[*] Port: {self.port} | Display FPS: {self.publish_fps} | Mode: {self.scan_mode.upper()}")
        print(f"==================================================================")
        try:
            while self.is_running:
                self.step_simulation_and_stream()
        except KeyboardInterrupt:
            print("\n[*] Stopping Unified Server...")
        finally:
            self.server.stop()
            self.backend.disconnect()
            print("[+] Unified Server cleanly shutdown.")


def main():
    parser = argparse.ArgumentParser(description="Unified Surgical Physics & Ultrasound Reconstruction Server")
    parser.add_argument("--port", type=int, default=18944, help="OpenIGTLink port (default 18944)")
    parser.add_argument("--urdf", type=str, default="", help="Path to robot URDF")
    parser.add_argument("--gui", action="store_true", help="Launch native PyBullet GUI")
    parser.add_argument("--fps", type=float, default=30.0, help="Publishing rate to Slicer (Hz)")
    parser.add_argument("--mode", type=str, choices=["orbital", "raster", "interactive"], default="orbital",
                        help="Scan execution mode: orbital, raster, or interactive")
    args = parser.parse_args()

    server = UnifiedSurgicalBridgeServer(
        port=args.port,
        urdf_path=args.urdf if args.urdf else None,
        gui=args.gui,
        publish_fps=args.fps,
        scan_mode=args.mode,
    )
    server.run()


if __name__ == "__main__":
    main()
