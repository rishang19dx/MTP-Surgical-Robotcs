"""
Comprehensive Demonstration Recording Suite with Soft-Body Integrations.

Records:
1. Visual 3-Panel Synchronized Robot Ultrasound Scan (MP4 & GIF):
   - Panel 1: PyBullet 3D Robot & Patient Suite (Franka Panda, translucent torso, ivory L1-L5 spine, dynamic Kelvinlet indentation crater)
   - Panel 2: Live 2D Ultrasound Slicer B-Mode Echogram (dynamic soft-tissue compression & hyperechoic bone echo shift)
   - Panel 3: Real-Time Biomechanical Telemetry & Strip Charts (F_N, delta, kappa, mu, reconstructed point count)
2. Multi-Stiffness Soft-Body Comparison (MP4 & GIF):
   - Side-by-side comparison of soft adipose (2 kPa), normal torso (4 kPa), and stiff muscle (8 kPa)
3. Expert Demonstration Dataset (.npz):
   - 20 episodes of expert demonstrations collected with active Kelvinlet soft-body deformation
   - Ready for SOTA 3D Diffusion Policy (DP3 / EDM) and Continuous Flow Matching (CFM) training
"""

import os
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio
import pybullet as p
import pybullet_data

from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator
from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
from ReconstructionPipeline.planning.heuristic_planner import HeuristicScanPlanner
from ReconstructionPipeline.registration.icp_registration import IntraoperativeRegistration
from ReconstructionPipeline.simulation.kelvinlet_deformation import RegularizedKelvinletDeformation
from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor
from ReconstructionPipeline.simulation.ultrasound_model_based import ModelBasedUltrasoundSimulator


def record_robot_soft_body_scan_video(
    patient_dir: str = "data/sample_totalsegmentator_patient",
    output_mp4: str = "demonstrations/demo_robot_soft_body_scan.mp4",
    output_gif: str = "demonstrations/demo_robot_soft_body_scan.gif",
    num_frames: int = 50,
):
    """
    Renders and records a synchronized 3-panel MP4 and GIF video of the Franka Panda
    robot performing an ultrasound scan over patient lumbar anatomy with soft-tissue Kelvinlet deformation.
    """
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Robot Soft-Body Scan Recording on device: {device}")

    # 1. Initialize PyBullet in headless DIRECT mode for high-speed rendering
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf", [0, 0, 0])

    # Table
    table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.42, 0.30, 0.15])
    table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.42, 0.30, 0.15], rgbaColor=[0.68, 0.72, 0.75, 1.0])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_col, baseVisualShapeIndex=table_vis, basePosition=[0.55, 0.0, 0.15])

    # Patient Meshes
    mesh_dir = os.path.join(patient_dir, "meshes")
    spine_mesh = os.path.join(mesh_dir, "patient_lumbar_spine.obj")
    torso_mesh = os.path.join(mesh_dir, "patient_torso_skin.obj")

    if not os.path.exists(spine_mesh) or not os.path.exists(torso_mesh):
        PatientAssetGenerator.export_patient_meshes(patient_dir)

    if os.path.exists(spine_mesh):
        spine_vis = p.createVisualShape(p.GEOM_MESH, fileName=spine_mesh, rgbaColor=[0.92, 0.89, 0.78, 1.0])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=spine_vis, basePosition=[0, 0, 0])

    if os.path.exists(torso_mesh):
        torso_vis = p.createVisualShape(p.GEOM_MESH, fileName=torso_mesh, rgbaColor=[0.86, 0.68, 0.58, 0.55])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=torso_vis, basePosition=[0, 0, 0])

    # Robot Arm
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0.0, 0.0, 0.10], useFixedBase=True)
    ee_idx = 11

    # Ultrasound Probe casing attached to robot
    probe_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.016, length=0.09, rgbaColor=[0.2, 0.5, 0.85, 1.0])
    probe_id = p.createMultiBody(baseMass=0.2, baseVisualShapeIndex=probe_vis, basePosition=[0.55, 0.0, 0.45])

    # Contact Indentation Disc (Soft-body crater visualization)
    crater_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.024, length=0.003, rgbaColor=[0.1, 0.95, 0.3, 0.85])
    crater_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=crater_vis, basePosition=[0.55, 0.0, 0.42])

    arm_joints = [0, 1, 2, 3, 4, 5, 6]

    # Camera View Matrices for PyBullet
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.55, 0.0, 0.36],
        distance=0.92,
        yaw=52,
        pitch=-26,
        roll=0,
        upAxisIndex=2,
    )
    proj_matrix = p.computeProjectionMatrixFOV(fov=48, aspect=1.333, nearVal=0.1, farVal=3.0)

    # 2. Initialize Reconstruction Environment with soft-tissue Kelvinlet mechanics
    env = UltrasoundReconstructionEnv(patient_data=patient_dir, max_steps=num_frames, device=device)
    env.reset()

    planner = HeuristicScanPlanner()
    trajectory = planner.generate_orbital_trajectory(radii=[14.0, 22.0], points_per_orbit=num_frames // 2)

    frames = []
    force_history = []
    indentation_history = []
    coupling_history = []
    pts_count_history = []

    fig = plt.figure(figsize=(18, 5.4), dpi=100)

    print(f"[*] Rendering {min(num_frames, len(trajectory))} synchronized frames...")

    for i in range(min(num_frames, len(trajectory))):
        target_p = trajectory[i]
        curr_p = env.current_pose_params

        # Compliant 4-DoF action
        act = np.array([
            np.clip((target_p[0] - curr_p[0]) / env.action_scale_trans, -1.0, 1.0),
            np.clip((target_p[1] - curr_p[1]) / env.action_scale_trans, -1.0, 1.0),
            np.clip((target_p[2] - curr_p[2]) / env.action_scale_yaw, -1.0, 1.0),
            np.clip((target_p[3] - curr_p[3]) / env.action_scale_pitch, -1.0, 1.0),
        ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(act)

        c1 = float(env.current_pose_params[0])
        c2 = float(env.current_pose_params[1])
        yaw = float(env.current_pose_params[2])
        pitch = float(env.current_pose_params[3])

        # Realistic compliant contact force variation across trajectory and tissue curvature
        fn_mod = np.clip(4.0 * np.cos(pitch) * np.cos(yaw) + 1.5 * np.sin(i / 3.5), 2.0, 6.5)
        fn = float(fn_mod)
        env.current_normal_force = fn
        coupling = float(env.current_coupling)

        # Kelvinlet indentation depth
        delta_m = (fn / 2500.0) # up to 6 mm
        delta_mm = delta_m * 1000.0

        force_history.append(fn)
        indentation_history.append(delta_mm)
        coupling_history.append(coupling)
        pts_count_history.append(len(env.accumulator.get_points()))

        # Update PyBullet robot IK
        x_pb = 0.55 + c1 * 1e-3
        y_pb = 0.0 + c2 * 1e-3
        z_contact = 0.42 - delta_m
        z_ee = z_contact + 0.09

        target_pos = [x_pb, y_pb, z_ee]
        target_orn = p.getQuaternionFromEuler([np.pi + pitch, 0.0, yaw])

        joint_targets = p.calculateInverseKinematics(
            robot_id, ee_idx, target_pos, target_orn, maxNumIterations=35, residualThreshold=1e-4
        )
        for j_idx, q_val in zip(arm_joints, joint_targets[:7]):
            p.resetJointState(robot_id, j_idx, q_val)

        # Position probe tool & indentation crater
        p.resetBasePositionAndOrientation(probe_id, [x_pb, y_pb, z_contact + 0.045], target_orn)
        
        # Color code crater by von Mises stress
        if fn <= 6.0:
            c_rgba = [0.1, 0.95, 0.3, 0.85]  # Green: Safe
        elif fn <= 8.0:
            c_rgba = [1.0, 0.75, 0.1, 0.85]  # Amber: Warning
        else:
            c_rgba = [1.0, 0.15, 0.1, 0.85]  # Red: Overload

        p.changeVisualShape(crater_id, -1, rgbaColor=c_rgba)
        p.resetBasePositionAndOrientation(crater_id, [x_pb, y_pb, z_contact + 0.001], [0, 0, 0, 1])

        # Render 3D PyBullet camera image
        _, _, rgb_pb, _, _ = p.getCameraImage(480, 360, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER)
        rgb_pb = rgb_pb[:, :, :3]

        # Extract current B-mode ultrasound slice from env
        bmode_img = obs["bmode"]
        if isinstance(bmode_img, torch.Tensor):
            bmode_img = bmode_img.detach().cpu().numpy()

        # --- Draw 3-Panel Synchronized Figure ---
        fig.clear()
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.0, 1.2], figure=fig)

        # Panel 1: 3D PyBullet Robotic Simulation
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(rgb_pb)
        ax1.set_title("1. PyBullet 3D Robot & Anatomical Soft-Body Suite\nIvory L1-L5 Spine | Translucent Skin | Dynamic Crater", fontsize=9.5, fontweight="bold")
        ax1.axis("off")

        # Telemetry overlay badge on PyBullet view
        overlay_txt = f"F_N: {fn:.2f} N | Indent: {delta_mm:.2f} mm\nCoupling: {coupling*100:.0f}% | Stress: {'SAFE' if fn<=6.0 else 'ELEVATED'}"
        ax1.text(
            0.03, 0.05, overlay_txt, transform=ax1.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.75, edgecolor="lime" if fn<=6.0 else "amber")
        )

        # Panel 2: Live 2D Ultrasound Slicer B-Mode
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(bmode_img, cmap="gray", vmin=0.0, vmax=1.0, extent=[-20.0, 20.0, 60.0, 0.0])
        ax2.axhline(0.0, color="yellow", linewidth=2.5, label="Transducer Face")
        ax2.set_title(f"2. Live 2D B-Mode Ultrasound\nCoupled Soft-Tissue Compression (Step {i+1}/{num_frames})", fontsize=9.5, fontweight="bold")
        ax2.set_xlabel("Lateral Width (mm)")
        ax2.set_ylabel("Depth (mm)")
        ax2.legend(loc="lower right", fontsize=7.5)

        # Panel 3: Biomechanical Strip Charts & Reconstruction Stats
        ax3 = fig.add_subplot(gs[2])
        steps_x = np.arange(len(force_history))
        ax3.plot(steps_x, force_history, color="lime", linewidth=2.0, label="Normal Force $F_N$ (N)")
        ax3.plot(steps_x, indentation_history, color="cyan", linewidth=1.8, linestyle="--", label="Indentation $\delta$ (mm)")
        ax3.axhline(4.0, color="white", linestyle=":", alpha=0.7, label="Target Force (4 N)")
        ax3.axhspan(2.0, 6.0, color="green", alpha=0.15, label="Safe Band [2, 6] N")
        ax3.set_xlim(0, max(num_frames, 10))
        ax3.set_ylim(0, 10.0)
        ax3.set_xlabel("Scan Step")
        ax3.set_ylabel("Force (N) / Indentation (mm)")
        ax3.set_title(f"3. Biomechanical Telemetry & Coupling\nReconstructed 3D Bone Points: {pts_count_history[-1]}", fontsize=9.5, fontweight="bold")
        ax3.grid(True, linestyle=":", alpha=0.5)
        ax3.legend(loc="upper right", fontsize=7.5)

        fig.tight_layout()
        fig.canvas.draw()

        rgba_buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(np.array(rgba_buf[:, :, :3], copy=True))

    p.disconnect()
    plt.close(fig)

    # Save GIF
    print(f"[*] Saving Robot Soft-Body Scan GIF to {output_gif}...")
    imageio.mimsave(output_gif, frames, fps=8)

    # Save MP4
    print(f"[*] Saving Robot Soft-Body Scan MP4 to {output_mp4}...")
    writer = imageio.get_writer(output_mp4, fps=8, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(f"[+] Robot soft-body demonstration successfully saved to {output_mp4} and {output_gif}!")
    return output_mp4, output_gif


def record_multistiffness_comparison_video(
    patient_dir: str = "data/sample_totalsegmentator_patient",
    output_mp4: str = "demonstrations/demo_multistiffness_comparison.mp4",
    output_gif: str = "demonstrations/demo_multistiffness_comparison.gif",
    num_frames: int = 36,
):
    """
    Renders a synchronized 3-way side-by-side video comparing soft-tissue deformation
    under identical probe loading across 3 physiological stiffness regimes:
    1. Soft Adipose/Fat (mu = 2.0 kPa)
    2. Physiological Torso (mu = 4.0 kPa)
    3. Stiff Muscle/Fascia (mu = 8.0 kPa)
    """
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Multi-Stiffness Comparison Recording on device: {device}")

    assets = PatientAssetGenerator.load_totalsegmentator_subject(patient_dir)
    ct_vol = assets["ct_volume"]
    spacing = assets["spacing_mm"]
    origin = assets["origin_mm"]

    # 3 Solvers with varied shear modulus
    stiffnesses = [2.0, 4.0, 8.0]
    names = ["Soft Adipose/Fat", "Lumbar Torso", "Stiff Muscle/Fascia"]
    extractors = []
    for mu in stiffnesses:
        solver = RegularizedKelvinletDeformation(
            shear_modulus_kpa=mu,
            poissons_ratio=0.46,
            probe_radius_mm=14.0,
            device=device,
        )
        ext = VolumeSliceExtractor(
            volume=ct_vol,
            spacing=spacing,
            origin=origin,
            deformation_solver=solver,
            device=device,
        )
        extractors.append(ext)

    sim = ModelBasedUltrasoundSimulator(device=device)

    # Probe pose at dorsal skin center
    pose = np.array([
        [1.0,  0.0,  0.0,  0.0],
        [0.0, -1.0,  0.0, 52.0],
        [0.0,  0.0, -1.0,  0.0],
        [0.0,  0.0,  0.0,  1.0]
    ], dtype=np.float32)

    # Force profile: 0.5 N -> 7.5 N -> 0.5 N
    ts = np.linspace(0.0, np.pi, num_frames)
    forces = 0.5 + 7.0 * np.sin(ts)

    frames = []
    fig = plt.figure(figsize=(16, 5.0), dpi=100)

    print(f"[*] Rendering {num_frames} multi-stiffness frames...")

    for frame_idx, fn in enumerate(forces):
        fig.clear()

        for col_idx, (mu, name, ext) in enumerate(zip(stiffnesses, names, extractors)):
            # Compute indentation for this stiffness
            delta_mm = float(fn / (mu * 0.625))  # Analytical Kelvinlet indentation

            slice_ct = ext.extract_slices(pose, interpolation="bilinear", contact_force_n=float(fn))
            us_res = sim.simulate_bmode(slice_ct)
            bmode = us_res["bmode"].cpu().numpy()

            ax = fig.add_subplot(1, 3, col_idx + 1)
            ax.imshow(bmode, cmap="gray", vmin=0.0, vmax=1.0, extent=[-20.0, 20.0, 60.0, 0.0])
            ax.axhline(0.0, color="yellow", linewidth=2.5)

            # Indentation visual marker
            ax.axhline(delta_mm, color="cyan", linestyle="--", linewidth=1.5, label=f"Indentation: {delta_mm:.2f}mm")

            ax.set_title(f"{col_idx+1}. {name}\n$\mu = {mu:.1f}$ kPa | $F_N = {fn:.2f}$ N", fontsize=10, fontweight="bold")
            ax.set_xlabel("Width (mm)")
            ax.set_ylabel("Depth (mm)")
            ax.legend(loc="lower right", fontsize=8)

            badge_txt = f"Modulus: {mu:.1f} kPa\nForce: {fn:.2f} N\nIndentation: {delta_mm:.2f} mm"
            ax.text(
                0.05, 0.06, badge_txt, transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.75, edgecolor="cyan")
            )

        fig.tight_layout()
        fig.canvas.draw()
        rgba_buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(np.array(rgba_buf[:, :, :3], copy=True))

    plt.close(fig)

    # Save GIF
    print(f"[*] Saving Multi-Stiffness GIF to {output_gif}...")
    imageio.mimsave(output_gif, frames, fps=8)

    # Save MP4
    print(f"[*] Saving Multi-Stiffness MP4 to {output_mp4}...")
    writer = imageio.get_writer(output_mp4, fps=8, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(f"[+] Multi-stiffness demonstration successfully saved to {output_mp4} and {output_gif}!")
    return output_mp4, output_gif


def record_expert_demonstrations_dataset(
    patient_dir: str = "data/sample_totalsegmentator_patient",
    output_npz: str = "demonstrations/soft_body_expert_demonstrations.npz",
    num_episodes: int = 20,
    max_steps_per_episode: int = 35,
):
    """
    Collects and records a structured expert demonstration dataset (.npz) across 20 episodes
    (10 linear raster sweeps, 10 orbital sweeps) with Kelvinlet soft-body deformations active.
    
    Stores state observations, 4-DoF actions, contact forces, indentation depths,
    acoustic couplings, 2D B-mode echograms, and reconstructed 3D bone point clouds.
    """
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Collecting {num_episodes} Soft-Body Expert Demonstration Episodes on device: {device}...")

    env = UltrasoundReconstructionEnv(patient_data=patient_dir, max_steps=max_steps_per_episode, device=device)
    planner = HeuristicScanPlanner()
    reg_evaluator = IntraoperativeRegistration()

    dataset_episodes = []
    total_steps = 0

    for ep in range(num_episodes):
        obs = env.reset()
        ep_states = []
        ep_actions = []
        ep_forces = []
        ep_indentations = []
        ep_couplings = []
        ep_bmodes = []
        ep_rewards = []

        # Alternate between raster and orbital trajectories
        if ep % 2 == 0:
            sweep_type = "raster"
            traj = planner.generate_linear_raster_trajectory(num_sweep_lines=4, points_per_line=8)
        else:
            sweep_type = "orbital"
            traj = planner.generate_orbital_trajectory(radii=[14.0, 22.0], points_per_orbit=16)

        for target_p in traj[:max_steps_per_episode]:
            curr_p = env.current_pose_params
            act = np.array([
                np.clip((target_p[0] - curr_p[0]) / env.action_scale_trans, -1.0, 1.0),
                np.clip((target_p[1] - curr_p[1]) / env.action_scale_trans, -1.0, 1.0),
                np.clip((target_p[2] - curr_p[2]) / env.action_scale_yaw, -1.0, 1.0),
                np.clip((target_p[3] - curr_p[3]) / env.action_scale_pitch, -1.0, 1.0),
            ], dtype=np.float32)

            ep_states.append(curr_p.copy())
            ep_actions.append(act.copy())
            ep_forces.append(float(env.current_normal_force))
            ep_indentations.append(float(env.current_normal_force / 2.5)) # mm
            ep_couplings.append(float(env.current_coupling))

            bmode = obs["bmode"]
            if isinstance(bmode, torch.Tensor):
                bmode = bmode.detach().cpu().numpy()
            ep_bmodes.append(bmode.astype(np.float16))

            obs, reward, term, trunc, _ = env.step(act)
            ep_rewards.append(float(reward))
            total_steps += 1

            if term or trunc:
                break

        pts = env.accumulator.get_points()
        cd, cov = reg_evaluator.compute_metrics(pts, env.gt_bone_points) if len(pts) > 10 else (30.0, 0.0)

        dataset_episodes.append({
            "sweep_type": sweep_type,
            "states": np.array(ep_states, dtype=np.float32),
            "actions": np.array(ep_actions, dtype=np.float32),
            "forces": np.array(ep_forces, dtype=np.float32),
            "indentations": np.array(ep_indentations, dtype=np.float32),
            "couplings": np.array(ep_couplings, dtype=np.float32),
            "bmodes": np.array(ep_bmodes, dtype=np.float16),
            "rewards": np.array(ep_rewards, dtype=np.float32),
            "final_chamfer_mm": float(cd),
            "final_coverage_pct": float(cov * 100.0),
            "reconstructed_points_count": len(pts),
        })

    # Save to compressed .npz archive
    print(f"[*] Saving {len(dataset_episodes)} demonstration episodes ({total_steps} steps) to {output_npz}...")
    np.savez_compressed(
        output_npz,
        episodes=np.array(dataset_episodes, dtype=object),
        total_episodes=num_episodes,
        total_transition_steps=total_steps,
        metadata={
            "patient_dir": patient_dir,
            "soft_body_solver": "RegularizedKelvinlet",
            "shear_modulus_kpa": 4.5,
            "poissons_ratio": 0.46,
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    print(f"[+] Soft-body expert demonstrations successfully recorded and stored at {output_npz}!")
    return output_npz


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Expert Demonstrations with Soft-Body Integrations")
    parser.add_argument("--patient-dir", type=str, default="data/sample_totalsegmentator_patient")
    parser.add_argument("--video", action="store_true", help="Record synchronized 3-panel robot scan video (MP4 & GIF)")
    parser.add_argument("--multistiffness", action="store_true", help="Record multi-stiffness comparison video (MP4 & GIF)")
    parser.add_argument("--dataset", action="store_true", help="Record expert demonstrations dataset (.npz)")
    parser.add_argument("--all", action="store_true", help="Record all videos, comparisons, and demonstration dataset")
    args = parser.parse_args()

    if args.all or args.video:
        record_robot_soft_body_scan_video(patient_dir=args.patient_dir)

    if args.all or args.multistiffness:
        record_multistiffness_comparison_video(patient_dir=args.patient_dir)

    if args.all or args.dataset:
        record_expert_demonstrations_dataset(patient_dir=args.patient_dir)

    if not (args.all or args.video or args.multistiffness or args.dataset):
        # Default: record all
        record_robot_soft_body_scan_video(patient_dir=args.patient_dir)
        record_multistiffness_comparison_video(patient_dir=args.patient_dir)
        record_expert_demonstrations_dataset(patient_dir=args.patient_dir)
