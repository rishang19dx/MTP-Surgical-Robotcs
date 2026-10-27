"""
Generates visual demonstration (MP4 and GIF) of Track 2: Soft Body Deformation & Dynamic B-Mode Echo Compression.
Visualizes:
1. Deformed 2D CT slice under Kelvinlet backward Eulerian warping.
2. Simulated 2D B-mode ultrasound frame with bone echo compression.
3. Central A-line beam profile showing bone peak depth shift.
4. Dynamic mechanical telemetry (F_N, coupling kappa, indentation delta).
"""

import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator
from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor
from ReconstructionPipeline.simulation.ultrasound_model_based import ModelBasedUltrasoundSimulator
from ReconstructionPipeline.simulation.kelvinlet_deformation import RegularizedKelvinletDeformation


def generate_deformation_animation(
    output_gif_path: str = "demonstrations/demo_soft_body_deformation.gif",
    output_mp4_path: str = "demonstrations/demo_soft_body_deformation.mp4",
    num_frames: int = 40,
):
    os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing soft body deformation demo on device: {device}")

    # Generate patient phantom
    assets = PatientAssetGenerator.generate_lumbar_phantom(shape=(80, 80, 60))
    ct_vol = assets["ct_volume"]
    spacing = assets["spacing_mm"]
    origin = assets["origin_mm"]

    # Image specifications: 50 mm depth x 40 mm width
    img_h, img_w = 150, 100
    depth_mm, width_mm = 50.0, 40.0

    extractor = VolumeSliceExtractor(
        volume=ct_vol,
        spacing=spacing,
        origin=origin,
        image_size=(img_h, img_w),
        physical_size_mm=(depth_mm, width_mm),
        device=device,
    )
    sim = ModelBasedUltrasoundSimulator(device=device)

    # Dorsal scanning pose on skin (y=50, x=0, z=0) pointing downward into spine
    pose = np.array([
        [1.0,  0.0,  0.0,  0.0],
        [0.0, -1.0,  0.0, 50.0],
        [0.0,  0.0, -1.0,  0.0],
        [0.0,  0.0,  0.0,  1.0]
    ], dtype=np.float32)

    # Force profile: ramps smoothly from 0.0 N -> 7.5 N -> 0.0 N (sinusoidal loading cycle)
    ts = np.linspace(0.0, np.pi, num_frames)
    forces = 7.5 * np.sin(ts)

    frames = []
    depth_axis_mm = np.linspace(0.0, depth_mm, img_h)

    print(f"[*] Rendering {num_frames} frames of dynamic probe compression...")
    fig = plt.figure(figsize=(14, 5), dpi=100)

    for i, fn in enumerate(forces):
        # 1. Extract deformed CT slice under force fn
        slice_ct = extractor.extract_slices(
            pose, interpolation="bilinear", contact_force_n=float(fn)
        )

        # 2. Simulate B-mode ultrasound
        us_res = sim.simulate_bmode(slice_ct)
        bmode_raw = us_res["bmode"]

        # Compute acoustic coupling kappa(fn)
        if fn < 1.0:
            kappa = 0.05 + 0.15 * (fn / 1.0)
        elif fn <= 7.0:
            kappa = min(1.0, 0.20 + 0.80 * ((fn - 1.0) / 2.0))
        else:
            kappa = max(0.6, 1.0 - 0.05 * (fn - 7.0))

        # Modulate B-mode by coupling
        noise_floor = torch.rand_like(bmode_raw) * 0.06
        bmode_img = (bmode_raw * kappa + noise_floor * (1.0 - kappa)).cpu().numpy()
        slice_ct_np = slice_ct.cpu().numpy()

        # Compute indentation delta
        u_0 = extractor.deformation_solver.compute_displacement(
            torch.tensor([[0.0, 50.0, 0.0]], device=device),
            torch.tensor([0.0, 50.0, 0.0], device=device),
            torch.tensor([0.0, -float(fn), 0.0], device=device),
            bone_depth_mm=30.0,
        )
        delta_mm = float(torch.norm(u_0))

        # Central A-line profile
        center_col = bmode_img[:, img_w // 2]
        peak_idx = np.argmax(center_col[15:]) + 15
        peak_depth_mm = depth_axis_mm[peak_idx]

        # Draw frame
        fig.clear()

        # Subplot 1: Deformed CT Slice
        ax1 = fig.add_subplot(1, 3, 1)
        im1 = ax1.imshow(slice_ct_np, cmap="bone", vmin=-100, vmax=1200, extent=[-width_mm/2, width_mm/2, depth_mm, 0])
        ax1.set_title("Deformed CT Slice (Resliced)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Width (mm)")
        ax1.set_ylabel("Depth from Probe (mm)")
        ax1.axhline(0, color="cyan", linestyle="--", linewidth=2, label="Probe Face")
        ax1.legend(loc="upper right", fontsize=8)

        # Subplot 2: Deformed B-Mode Ultrasound
        ax2 = fig.add_subplot(1, 3, 2)
        im2 = ax2.imshow(bmode_img, cmap="gray", vmin=0.0, vmax=1.0, extent=[-width_mm/2, width_mm/2, depth_mm, 0])
        ax2.axvline(0, color="yellow", linestyle=":", alpha=0.6, label="Central A-line")
        ax2.axhline(peak_depth_mm, color="red", linestyle="--", linewidth=1.5, label=f"Bone Peak ({peak_depth_mm:.1f} mm)")
        ax2.set_title("Simulated B-Mode US ($\kappa$ modulated)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Width (mm)")
        ax2.set_ylabel("Depth (mm)")
        ax2.legend(loc="upper right", fontsize=8)

        # Subplot 3: A-line Depth Profile & Mechanical Telemetry
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(center_col, depth_axis_mm, color="yellow", linewidth=2, label="A-line Intensity")
        ax3.axhline(peak_depth_mm, color="red", linestyle="--", linewidth=1.5, label=f"Peak: {peak_depth_mm:.1f} mm")
        ax3.invert_yaxis()
        ax3.set_xlim(0.0, 1.05)
        ax3.set_ylim(depth_mm, 0.0)
        ax3.set_xlabel("RF Intensity")
        ax3.set_ylabel("Depth from Probe (mm)")
        ax3.set_title("A-Line Profile & Contact Telemetry", fontsize=11, fontweight="bold")
        ax3.grid(True, linestyle=":", alpha=0.6)

        # Telemetry text box
        telemetry_text = (
            f"Contact Force: $F_N$ = {fn:.2f} N\n"
            f"Coupling Factor: $\kappa$ = {kappa:.2f}\n"
            f"Skin Indentation: $\delta$ = {delta_mm:.2f} mm\n"
            f"Bone Cortex Depth: {peak_depth_mm:.1f} mm"
        )
        ax3.text(
            0.05, 0.08, telemetry_text,
            transform=ax3.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7, edgecolor="cyan")
        )
        ax3.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.canvas.draw()

        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        rgb_image = rgba_buffer[:, :, :3]
        frames.append(rgb_image)

    plt.close(fig)

    # Save GIF
    print(f"[*] Saving GIF to {output_gif_path}...")
    imageio.mimsave(output_gif_path, frames, fps=12)

    # Save MP4
    print(f"[*] Saving MP4 to {output_mp4_path}...")
    writer = imageio.get_writer(output_mp4_path, fps=12, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print("[+] Demonstration successfully generated!")
    return output_gif_path, output_mp4_path


if __name__ == "__main__":
    generate_deformation_animation()
