"""
Soft-body demonstration recording (FEM tissue model, DEC-037).

1. Video (MP4 + GIF, ``--video``): the Franka Panda presses a transverse probe on the L4 midline with a
   force ramp 3 -> 10 -> 3 N and then slides laterally, all through ``UltrasoundReconstructionEnv``
   with the default GPU FEM tissue model. Four panels, every number from real model outputs:
     a. PyBullet render (TinyRenderer, DIRECT) of the robot, spine and the FEM-deformed skin patch
        (``FEMTissueModel.skin_surface_mesh(state)``, same P -> PyBullet mapping as the server);
     b. the simulated B-mode at that pose (bone echo moves up as the tissue is compressed);
     c. transverse skin profile under the probe (undeformed vs FEM-deformed skin vertices, true scale);
     d. force trace: FEM reaction force vs the admittance target, and the indentation.
2. Expert demonstration dataset (.npz, ``--dataset``, git-ignored): force and indentation are the values
   reported by the environment's tissue model (``env.current_normal_force``, ``env.tissue_state``).

Usage:
    ./.venv/bin/python -m ReconstructionPipeline.scripts.record_soft_body_demonstrations --video
"""

import os
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
import pybullet as p

from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator
from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
from ReconstructionPipeline.planning.heuristic_planner import HeuristicScanPlanner
from ReconstructionPipeline.registration.icp_registration import IntraoperativeRegistration
from surgical_physics_bridge.src.coordinates import P_TO_PB_ROT, patient_to_pybullet_points, probe_axes_to_pybullet_quaternion
from surgical_physics_bridge.src.pybullet_backend import PyBulletBackend


def _indentation_mm(env) -> float:
    ts = env.tissue_state
    return float(ts.indentation_mm) if ts is not None else float(env.current_indentation_mm)


def _demo_schedule(n_press: int, n_slide: int):
    """(c1, c2, target force) per frame: force ramp at the L4 midline, then a lateral slide at 6 N."""
    half = n_press // 2
    f_up = np.linspace(3.0, 10.0, half)
    f_dn = np.linspace(10.0, 3.0, n_press - half)
    sched = [(0.0, -32.5, float(f)) for f in np.concatenate([f_up, f_dn])]
    sched += [(float(x), -32.5, 6.0) for x in np.linspace(0.0, 24.0, n_slide)]
    return sched


def record_robot_soft_body_scan_video(
    patient_dir: str = "data/sample_totalsegmentator_patient",
    output_mp4: str = "demonstrations/demo_fem_soft_body.mp4",
    output_gif: str = "demonstrations/demo_fem_soft_body.gif",
    n_press: int = 36,
    n_slide: int = 16,
    fps: int = 8,
):
    os.makedirs(os.path.dirname(output_mp4) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_dir = os.path.join(patient_dir, "meshes")
    if not os.path.exists(os.path.join(mesh_dir, "patient_lumbar_spine.obj")):
        PatientAssetGenerator.export_patient_meshes(patient_dir)

    env = UltrasoundReconstructionEnv(patient_data=patient_dir, target_vertebra="vertebrae_L4",
                                      max_steps=10_000, device=device, seed=0)
    model = env.tissue_model
    if not hasattr(model, "skin_surface_mesh"):
        raise RuntimeError("the soft-body demo needs the FEM tissue backend")
    env.reset(seed=0)
    skin_ref, skin_tris = model.skin_surface_mesh(None)

    # PyBullet scene: same builder as the server (DIRECT client; rendered with TinyRenderer).
    pb = PyBulletBackend(gui=False, patient_dir=patient_dir)
    pb.set_skin_mesh(skin_tris)
    c = pb.cid
    if pb.torso_id >= 0:
        p.changeVisualShape(pb.torso_id, -1, rgbaColor=[0.86, 0.68, 0.58, 0.25], physicsClientId=c)
    cam_target = patient_to_pybullet_points(np.array([0.0, skin_ref[:, 1].max() - 15.0, -32.5]))
    view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=cam_target.tolist(), distance=0.2, yaw=60,
                                               pitch=-28, roll=0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=4 / 3, nearVal=0.02, farVal=3.0)

    sched = _demo_schedule(n_press, n_slide)
    hist = {"t": [], "F": [], "Ft": [], "d": []}
    frames = []
    fig = plt.figure(figsize=(10.0, 5.6), dpi=80)
    for k, (cx, cz, f_target) in enumerate(sched):
        env.target_force_n = f_target
        p_now = env.current_pose_params
        act = np.array([(cx - p_now[0]) / env.action_scale_trans, (cz - p_now[1]) / env.action_scale_trans,
                        -p_now[2] / env.action_scale_yaw, -p_now[3] / env.action_scale_pitch], dtype=np.float32)
        obs, _, _, _, info = env.step(np.clip(act, -1.0, 1.0))
        st = env.tissue_state
        V, _ = model.skin_surface_mesh(st)
        hist["t"].append(k / fps)
        hist["F"].append(info["normal_force"])
        hist["Ft"].append(f_target)
        hist["d"].append(info["indentation_mm"])

        # a. robot + deformed skin (probe face at the indented pose)
        t_skin = env.skin_pose(*env.current_pose_params)
        beam_p, lat_p = t_skin[:3, 1], t_skin[:3, 0]
        face = patient_to_pybullet_points(t_skin[:3, 3] + info["indentation_mm"] * beam_p)
        beam_pb = P_TO_PB_ROT @ beam_p
        pb.apply_target({"ee_pos": (face - beam_pb * pb.PROBE_LENGTH_M).tolist(),
                         "ee_orn": list(probe_axes_to_pybullet_quaternion(beam_p, lat_p)), "teleport": True})
        pb.update_skin(patient_to_pybullet_points(V))
        rgb = np.asarray(p.getCameraImage(400, 300, view, proj, renderer=p.ER_TINY_RENDERER,
                                          physicsClientId=c)[2], dtype=np.uint8).reshape(300, 400, 4)[:, :, :3]

        fig.clf()
        gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 0.8, 1.0], hspace=0.38, wspace=0.28)
        ax = fig.add_subplot(gs[:, 0])
        ax.imshow(rgb)
        ax.set_axis_off()
        ax.set_title(f"FEM skin patch under the probe\nF = {info['normal_force']:.1f} N (target {f_target:.1f} N)",
                     fontsize=10)
        ax = fig.add_subplot(gs[:, 1])
        ax.imshow(obs["bmode"], cmap="gray", vmin=0, vmax=1, aspect="auto",
                  extent=[-env.PHYSICAL_SIZE_MM[1] / 2, env.PHYSICAL_SIZE_MM[1] / 2, env.PHYSICAL_SIZE_MM[0], 0])
        ax.set_title("Simulated B-mode", fontsize=10)
        ax.set_xlabel("lateral mm", fontsize=8)
        ax.set_ylabel("depth from probe face mm", fontsize=8)
        ax.tick_params(labelsize=7)
        # c. transverse skin profile through the probe centre (nodes of the nearest lattice row)
        ax = fig.add_subplot(gs[0, 2])
        zrow = skin_ref[np.argmin(np.abs(skin_ref[:, 2] - cz)), 2]
        sel = np.abs(skin_ref[:, 2] - zrow) < 1e-3
        order = np.argsort(skin_ref[sel, 0])
        ax.plot(skin_ref[sel, 0][order], skin_ref[sel, 1][order], color="0.6", lw=1.2, label="undeformed")
        ax.plot(V[sel, 0][order], V[sel, 1][order], color="tab:red", lw=1.6, label="FEM deformed")
        x0 = env.current_pose_params[0]
        ax.axvspan(x0 - model.face_dims[0] / 2, x0 + model.face_dims[0] / 2, color="tab:blue", alpha=0.12,
                   label="probe face")
        ax.set_xlim(-60, 60)
        ax.set_title(f"Skin profile at z = {zrow:.0f} mm (true scale)", fontsize=9)
        ax.set_xlabel("x mm", fontsize=8)
        ax.set_ylabel("y mm", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="lower center")
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(hist["t"], hist["Ft"], "k--", lw=1, label="target")
        ax.plot(hist["t"], hist["F"], color="tab:red", lw=1.6, label="FEM force")
        ax.set_xlim(0, len(sched) / fps)
        ax.set_ylim(0, 12)
        ax.set_xlabel("time s", fontsize=8)
        ax.set_ylabel("N", fontsize=8)
        ax.tick_params(labelsize=7)
        ax2 = ax.twinx()
        ax2.plot(hist["t"], hist["d"], color="tab:blue", lw=1.2, label="indentation")
        ax2.set_ylabel("indentation mm", fontsize=8, color="tab:blue")
        ax2.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")
        ax.set_title("Force tracking (admittance on FEM force)", fontsize=9)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        print(f"  frame {k + 1}/{len(sched)}: F={info['normal_force']:.2f} N, d={info['indentation_mm']:.2f} mm")
    plt.close(fig)
    pb.disconnect()

    imageio.mimsave(output_mp4, frames, fps=fps, codec="libx264", quality=7)
    imageio.mimsave(output_gif, frames, duration=1000 / fps, loop=0)
    for path in (output_mp4, output_gif):
        print(f"[+] wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")
    return output_mp4, output_gif


def record_expert_demonstrations_dataset(
    patient_dir: str = "data/sample_totalsegmentator_patient",
    output_npz: str = "demonstrations/soft_body_expert_demonstrations.npz",
    num_episodes: int = 20,
    max_steps_per_episode: int = 35,
):
    """
    Collects and records a structured expert demonstration dataset (.npz) across 20 episodes
    (10 linear raster sweeps, 10 orbital sweeps) with the tissue model active.
    
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
            ep_indentations.append(_indentation_mm(env))
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
            "tissue_model": type(getattr(env, "tissue_model", None)).__name__,
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    print(f"[+] Soft-body expert demonstrations successfully recorded and stored at {output_npz}!")
    return output_npz


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Expert Demonstrations with Soft-Body Integrations")
    parser.add_argument("--patient-dir", type=str, default="data/sample_totalsegmentator_patient")
    parser.add_argument("--video", action="store_true", help="Record the FEM soft-body demo video (MP4 & GIF)")
    parser.add_argument("--dataset", action="store_true", help="Record expert demonstrations dataset (.npz)")
    parser.add_argument("--all", action="store_true", help="Record all videos, comparisons, and demonstration dataset")
    args = parser.parse_args()

    if args.all or args.video:
        record_robot_soft_body_scan_video(patient_dir=args.patient_dir)

    if args.all or args.dataset:
        record_expert_demonstrations_dataset(patient_dir=args.patient_dir)

    if not (args.all or args.video or args.dataset):
        # Default: record all
        record_robot_soft_body_scan_video(patient_dir=args.patient_dir)
        record_expert_demonstrations_dataset(patient_dir=args.patient_dir)
