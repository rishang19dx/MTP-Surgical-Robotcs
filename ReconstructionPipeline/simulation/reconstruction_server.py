"""
Real-time Standalone 3D Ultrasound Reconstruction & OpenIGTLink Bridge Server.
Simulates autonomous or interactive robotic ultrasound acquisition and streams:
1. Probe transform -> pyigtl.TransformMessage("UltrasoundProbe")
2. Real-time B-mode ultrasound feed -> pyigtl.ImageMessage("US_Image")
3. Reconstructed 3D bone point cloud -> pyigtl.NDArrayMessage("ReconstructedBone")
Directly visualizable within 3D Slicer via OpenIGTLink.
"""

import time
import argparse
import json
import numpy as np
import pyigtl

from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
from ReconstructionPipeline.planning.heuristic_planner import HeuristicScanPlanner


def main():
    parser = argparse.ArgumentParser(description="Autonomous Ultrasound Reconstruction OpenIGTLink Server")
    parser.add_argument("--port", type=int, default=18944, help="OpenIGTLink port (default 18944)")
    parser.add_argument("--mode", type=str, choices=["orbital", "raster", "interactive"], default="orbital",
                        help="Scan execution mode: orbital, raster, or interactive from Slicer")
    parser.add_argument("--fps", type=float, default=30.0, help="Publishing frame rate (Hz)")
    args = parser.parse_args()

    print(f"================================================================")
    print(f"Starting Ultrasound Reconstruction Bridge Server on port {args.port}...")
    print(f"Execution Mode: {args.mode.upper()} | Target FPS: {args.fps}")
    print(f"================================================================")

    # 1. Initialize OpenIGTLink Server
    server = pyigtl.OpenIGTLinkServer(port=args.port)

    # 2. Initialize Reconstruction Simulation Environment
    env = UltrasoundReconstructionEnv(max_steps=150)
    obs = env.reset()

    # 3. Trajectory Planner (if autonomous mode)
    planner = HeuristicScanPlanner()
    if args.mode == "orbital":
        trajectory = planner.generate_orbital_trajectory(radii=[18.0, 28.0, 36.0], points_per_orbit=30)
    elif args.mode == "raster":
        trajectory = planner.generate_linear_raster_trajectory(num_sweep_lines=8, points_per_line=15)
    else:
        trajectory = None

    traj_idx = 0
    dt_publish = 1.0 / args.fps
    last_publish_time = time.time()

    try:
        while True:
            # Handle incoming OpenIGTLink messages from 3D Slicer
            for message in server.get_messages():
                if isinstance(message, pyigtl.StringMessage) and message.device_name == "CMD":
                    try:
                        cmd = json.loads(message.string)
                        if "reset" in cmd and cmd["reset"]:
                            obs = env.reset()
                            traj_idx = 0
                            print("Reset command received from Slicer.")
                    except json.JSONDecodeError:
                        pass
                elif isinstance(message, pyigtl.NDArrayMessage) and message.device_name == "cmd_action":
                    # Interactive action from Slicer or external planner: [dx, dz, dyaw, dpitch]
                    action = message.ndarray.flatten()
                    if len(action) == 4:
                        obs, reward, term, trunc, info = env.step(action)
                        if term or trunc:
                            obs = env.reset()

            current_time = time.time()
            if current_time - last_publish_time >= dt_publish:
                # In autonomous mode, advance along scan trajectory
                if trajectory is not None and traj_idx < len(trajectory):
                    target_pose = trajectory[traj_idx]
                    current_p = env.current_pose_params
                    dx = np.clip((target_pose[0] - current_p[0]) / env.action_scale_trans, -1.0, 1.0)
                    dz = np.clip((target_pose[1] - current_p[1]) / env.action_scale_trans, -1.0, 1.0)
                    dyaw = np.clip((target_pose[2] - current_p[2]) / env.action_scale_yaw, -1.0, 1.0)
                    dpitch = np.clip((target_pose[3] - current_p[3]) / env.action_scale_pitch, -1.0, 1.0)

                    action = np.array([dx, dz, dyaw, dpitch], dtype=np.float32)
                    obs, reward, term, trunc, info = env.step(action)
                    traj_idx += 1

                    if term or trunc or traj_idx >= len(trajectory):
                        print(f"Scan trajectory complete! Points: {info['total_points']}, Coverage: {info['coverage_ratio']*100:.1f}%")
                        traj_idx = 0
                        obs = env.reset()  # Reset environment state for next loop

                # 1. Publish Probe Transform Matrix
                t_matrix = env._build_probe_matrix(*env.current_pose_params)
                tf_msg = pyigtl.TransformMessage(t_matrix.astype(np.float64), device_name="UltrasoundProbe")
                server.send_message(tf_msg)

                # 2. Publish Simulated 2D B-mode Ultrasound Image
                bmode_norm = obs["bmode"] # (H, W) in [0, 1]
                bmode_uint8 = (bmode_norm * 255.0).astype(np.uint8)
                # Format: (H, W, 1) for grayscale image message
                img_msg = pyigtl.ImageMessage(bmode_uint8[:, :, np.newaxis], device_name="US_Image")
                server.send_message(img_msg)

                # 3. Publish Incremental 3D Reconstructed Bone Points
                pts = env.accumulator.get_points()
                if len(pts) > 0:
                    pts_msg = pyigtl.PointMessage(positions=pts.astype(np.float32), device_name="ReconstructedBone")
                    server.send_message(pts_msg)
                    # Also stream as PolyDataMessage for existing Slicer model node observer
                    poly_msg = pyigtl.PolyDataMessage(points=pts.astype(np.float32), device_name="DeformedAnatomy")
                    server.send_message(poly_msg)

                last_publish_time = current_time

            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\nStopping Ultrasound Reconstruction Server...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
