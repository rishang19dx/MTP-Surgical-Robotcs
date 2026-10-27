"""
Generates high-visibility visual demonstration (MP4 and GIF) of Track 2:
Soft Body Tissue Deformation & Dynamic B-Mode Echo Compression under Robotic Ultrasound Contact.

Provides a 4-panel synchronized multi-scale view:
1. Global Patient Anatomical Cross-Section: Probe descending, skin visibly indenting (0-10+ mm), warped Lagrangian grid, and strain field.
2. Local Deformed CT Reslice (US FOV): Internal anatomy visibly shifting shallower by 25-35 pixels with reference horizon.
3. Simulated 2D B-Mode Ultrasound: Acoustic coupling kappa(F_N) modulation and dynamic hyperechoic bone reflection shift.
4. Central A-Line Profile & Biomechanical Telemetry HUD: Real-time echo peak tracking and quantitative contact metrics.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    print(f"[*] Initializing high-visibility soft-body deformation demo on device: {device}")

    # Generate patient phantom
    assets = PatientAssetGenerator.generate_lumbar_phantom(shape=(100, 100, 80))
    ct_vol_raw = assets["ct_volume"]
    d_vol, h_vol, w_vol = ct_vol_raw.shape
    sx, sy, sz = assets["spacing_mm"]
    ox, oy, oz = assets["origin_mm"]

    ct_tensor = torch.from_numpy(ct_vol_raw).float().unsqueeze(0).unsqueeze(0).to(device)

    # Compliant soft tissue elasticity: mu = 4.0 kPa for abdominal/paraspinal tissue
    solver = RegularizedKelvinletDeformation(
        shear_modulus_kpa=4.0,
        poissons_ratio=0.46,
        probe_radius_mm=14.0,
        max_indentation_mm=14.0,
        device=device,
    )

    # Paramedian probe placement: X = 18 mm, on dorsal skin Y = 55 mm, pointing down (-Y)
    probe_x = 18.0
    skin_y = float(assets["skin_y_mm"]) # 55.0 mm
    contact_pt = torch.tensor([probe_x, skin_y, 0.0], device=device)

    # Local ultrasound FOV parameters: depth 50 mm, width 40 mm
    us_depth_mm = 50.0
    us_width_mm = 40.0
    img_h, img_w = 160, 100

    extractor = VolumeSliceExtractor(
        volume=ct_vol_raw,
        spacing=assets["spacing_mm"],
        origin=assets["origin_mm"],
        image_size=(img_h, img_w),
        physical_size_mm=(us_depth_mm, us_width_mm),
        deformation_solver=solver,
        device=device,
    )
    sim = ModelBasedUltrasoundSimulator(device=device)

    # Local probe pose matrix (at skin surface pointing -Y)
    pose = np.array([
        [1.0,  0.0,  0.0, probe_x],
        [0.0, -1.0,  0.0,  skin_y],
        [0.0,  0.0, -1.0,     0.0],
        [0.0,  0.0,  0.0,     1.0]
    ], dtype=np.float32)

    # Global 2D axial slice grid coordinates (Z = 0)
    y_coords = torch.linspace(oy, oy + (h_vol - 1) * sy, h_vol, device=device)
    x_coords = torch.linspace(ox, ox + (w_vol - 1) * sx, w_vol, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid_z = torch.zeros_like(grid_x)
    global_slice_pts = torch.stack([grid_x, grid_y, grid_z], dim=-1) # (H, W, 3)

    # Pre-extract undeformed reference slice to identify baseline bone depth
    ref_slice = extractor.extract_slices(pose, interpolation="bilinear", contact_force_n=0.0)
    ref_bmode = sim.simulate_bmode(ref_slice)["bmode"]
    depth_axis_mm = np.linspace(0.0, us_depth_mm, img_h)
    ref_col = ref_bmode[:, img_w // 2].cpu().numpy()
    baseline_peak_idx = int(np.argmax(ref_col[40:]) + 40)
    baseline_bone_depth_mm = float(depth_axis_mm[baseline_peak_idx])

    # Dynamic loading profile: smooth sinusoidal ramping 0 N -> 7.5 N -> 0 N
    ts = np.linspace(0.0, np.pi, num_frames)
    forces = 7.5 * np.sin(ts)

    frames = []
    print(f"[*] Rendering {num_frames} multi-scale frames of dynamic tissue deformation...")

    fig = plt.figure(figsize=(19, 5.2), dpi=100)

    for frame_idx, fn in enumerate(forces):
        force_vec = torch.tensor([0.0, -float(fn), 0.0], device=device)

        # 1. Global Axial Slice Deformation: X_ref = x - u(x)
        if fn > 0.05:
            u_global = solver.compute_displacement(
                global_slice_pts.view(-1, 3), contact_pt, force_vec, bone_depth_mm=42.0
            ).view(h_vol, w_vol, 3)
            pts_ref_global = global_slice_pts - u_global

            # Compute indentation at probe contact
            u_contact = solver.compute_displacement(
                contact_pt.unsqueeze(0), contact_pt, force_vec, bone_depth_mm=42.0
            )
            indentation_mm = float(torch.norm(u_contact))
        else:
            pts_ref_global = global_slice_pts
            u_global = torch.zeros_like(global_slice_pts)
            indentation_mm = 0.0

        # Sample deformed global CT
        vx_g = (pts_ref_global[..., 0] - ox) / sx
        vy_g = (pts_ref_global[..., 1] - oy) / sy
        vz_g = (pts_ref_global[..., 2] - oz) / sz
        gw_g = 2.0 * (vx_g / (w_vol - 1.0)) - 1.0
        gh_g = 2.0 * (vy_g / (h_vol - 1.0)) - 1.0
        gd_g = 2.0 * (vz_g / (d_vol - 1.0)) - 1.0
        samp_grid_g = torch.stack([gw_g, gh_g, gd_g], dim=-1).view(1, 1, h_vol, w_vol, 3)
        deformed_global_ct = F.grid_sample(
            ct_tensor, samp_grid_g, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze().cpu().numpy()

        disp_mag_global = torch.norm(u_global, dim=-1).cpu().numpy()

        # 2. Local Deformed Ultrasound Resliced CT
        slice_ct = extractor.extract_slices(
            pose, interpolation="bilinear", contact_force_n=float(fn), bone_depth_mm=42.0
        )
        slice_ct_np = slice_ct.cpu().numpy()

        # 3. Simulated B-Mode Ultrasound
        us_res = sim.simulate_bmode(slice_ct)
        bmode_raw = us_res["bmode"]

        # Force-acoustic coupling kappa(F_N)
        if fn < 1.0:
            kappa = 0.05 + 0.35 * (fn / 1.0)
        elif fn <= 6.5:
            kappa = min(1.0, 0.40 + 0.60 * ((fn - 1.0) / 2.0))
        else:
            kappa = max(0.7, 1.0 - 0.05 * (fn - 6.5))

        noise_floor = torch.rand_like(bmode_raw) * 0.05
        bmode_img = (bmode_raw * kappa + noise_floor * (1.0 - kappa)).cpu().numpy()

        # Central A-Line profile & peak detection
        center_col = bmode_img[:, img_w // 2]
        curr_peak_idx = int(np.argmax(center_col[30:]) + 30)
        curr_bone_depth_mm = float(depth_axis_mm[curr_peak_idx])
        shift_mm = baseline_bone_depth_mm - curr_bone_depth_mm
        axial_strain_pct = (indentation_mm / 42.0) * 100.0

        # --- Draw 4-Panel Visualization ---
        fig.clear()

        # PANEL 1: Global Patient Cross-Section & Probe Indentation
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(
            deformed_global_ct, cmap="bone", vmin=-150, vmax=1200,
            extent=[ox, ox + w_vol * sx, oy, oy + h_vol * sy], origin="lower"
        )
        # Overlay displacement magnitude contour heatmap
        if fn > 0.5:
            x_mesh = grid_x.cpu().numpy()
            y_mesh = grid_y.cpu().numpy()
            cs = ax1.contourf(x_mesh, y_mesh, disp_mag_global, levels=np.linspace(1.0, 12.0, 6), cmap="plasma", alpha=0.35)

        # Draw Ultrasound Probe Head descending with indentation
        probe_head_width = 24.0
        probe_face_y = skin_y - indentation_mm
        probe_rect = patches.Rectangle(
            (probe_x - probe_head_width / 2.0, probe_face_y), probe_head_width, 18.0,
            linewidth=2, edgecolor="cyan", facecolor="darkslategrey", alpha=0.9, zorder=5
        )
        ax1.add_patch(probe_rect)
        # Probe face line
        ax1.plot([probe_x - probe_head_width/2.0, probe_x + probe_head_width/2.0], [probe_face_y, probe_face_y], color="yellow", linewidth=3, zorder=6)
        # Acoustic beam frustum
        beam_poly = np.array([
            [probe_x - us_width_mm / 2.0, probe_face_y],
            [probe_x + us_width_mm / 2.0, probe_face_y],
            [probe_x + us_width_mm / 2.0, probe_face_y - us_depth_mm],
            [probe_x - us_width_mm / 2.0, probe_face_y - us_depth_mm]
        ])
        ax1.add_patch(patches.Polygon(beam_poly, closed=True, edgecolor="cyan", linestyle=":", fill=False, linewidth=1.5, alpha=0.8, zorder=4))
        
        # Original skin horizon line
        ax1.axhline(skin_y, color="white", linestyle="--", alpha=0.5, label="Original Skin (Y=55)")
        ax1.set_title(f"1. Patient Anatomy Cross-Section\nProbe Indentation: {indentation_mm:.1f} mm", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Lateral X (mm)")
        ax1.set_ylabel("Posterior-Anterior Y (mm)")
        ax1.set_xlim(-35.0, 38.0)
        ax1.set_ylim(-15.0, 72.0)

        # PANEL 2: Local Ultrasound Resliced CT (US FOV)
        ax2 = fig.add_subplot(1, 4, 2)
        ax2.imshow(slice_ct_np, cmap="bone", vmin=-100, vmax=1200, extent=[-us_width_mm/2, us_width_mm/2, us_depth_mm, 0])
        ax2.axhline(0, color="yellow", linewidth=2.5, label="Transducer Face")
        ax2.axhline(baseline_bone_depth_mm, color="cyan", linestyle="--", linewidth=1.5, label=f"Ref Bone ({baseline_bone_depth_mm:.1f}mm)")
        ax2.axhline(curr_bone_depth_mm, color="red", linestyle="-", linewidth=2.0, label=f"Current Bone ({curr_bone_depth_mm:.1f}mm)")
        if abs(shift_mm) > 0.8:
            ax2.annotate(
                f"Shift: -{shift_mm:.1f}mm", xy=(-12.0, curr_bone_depth_mm),
                xytext=(-12.0, baseline_bone_depth_mm),
                arrowprops=dict(facecolor="yellow", edgecolor="black", arrowstyle="->", lw=1.5),
                fontsize=8, color="yellow", fontweight="bold"
            )
        ax2.set_title("2. Deformed CT Reslice (US FOV)\nInternal Tissue Compressed", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Width (mm)")
        ax2.set_ylabel("Depth from Probe (mm)")
        ax2.legend(loc="lower right", fontsize=7)

        # PANEL 3: Simulated 2D B-Mode Ultrasound
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(bmode_img, cmap="gray", vmin=0.0, vmax=1.0, extent=[-us_width_mm/2, us_width_mm/2, us_depth_mm, 0])
        ax3.axvline(0, color="yellow", linestyle=":", alpha=0.5, label="Central A-line")
        ax3.axhline(curr_bone_depth_mm, color="red", linestyle="--", linewidth=1.8, label=f"Bone Echo ({curr_bone_depth_mm:.1f}mm)")
        ax3.set_title(f"3. Simulated B-Mode US\nCoupling: kappa={kappa:.2f}", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Width (mm)")
        ax3.set_ylabel("Depth (mm)")
        ax3.legend(loc="lower right", fontsize=7)

        # PANEL 4: Central A-Line Profile & Biomechanical HUD
        ax4 = fig.add_subplot(1, 4, 4)
        ax4.plot(center_col, depth_axis_mm, color="yellow", linewidth=2, label="A-Line RF Envelope")
        ax4.axhline(baseline_bone_depth_mm, color="cyan", linestyle="--", linewidth=1.5, label="Baseline Peak")
        ax4.axhline(curr_bone_depth_mm, color="red", linestyle="-", linewidth=2.0, label=f"Peak: {curr_bone_depth_mm:.1f}mm")
        ax4.invert_yaxis()
        ax4.set_xlim(0.0, 1.05)
        ax4.set_ylim(us_depth_mm, 0.0)
        ax4.set_xlabel("Acoustic Intensity")
        ax4.set_ylabel("Depth from Transducer (mm)")
        ax4.set_title("4. A-Line Tracking & Telemetry", fontsize=10, fontweight="bold")
        ax4.grid(True, linestyle=":", alpha=0.5)

        hud_text = (
            f"--- CONTACT MECHANICS ---\n"
            f"Contact Force: $F_N$ = {fn:.2f} N\n"
            f"Probe Indentation: delta = {indentation_mm:.2f} mm\n"
            f"Tissue Axial Strain: eps = {axial_strain_pct:.1f} %\n"
            f"Acoustic Coupling: kappa = {kappa:.2f}\n"
            f"Bone Echo Shift: Delta d = -{shift_mm:.2f} mm"
        )
        ax4.text(
            0.05, 0.06, hud_text, transform=ax4.transAxes, fontsize=8.5,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8, edgecolor="cyan")
        )
        ax4.legend(loc="upper right", fontsize=7)

        fig.tight_layout()
        fig.canvas.draw()

        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        rgb_image = rgba_buffer[:, :, :3]
        frames.append(rgb_image)

    plt.close(fig)

    # Save GIF
    print(f"[*] Saving enhanced GIF to {output_gif_path}...")
    imageio.mimsave(output_gif_path, frames, fps=10)

    # Save MP4
    print(f"[*] Saving enhanced MP4 to {output_mp4_path}...")
    writer = imageio.get_writer(output_mp4_path, fps=10, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(f"[+] High-visibility demonstration successfully written to {output_gif_path} and {output_mp4_path}!")
    return output_gif_path, output_mp4_path


if __name__ == "__main__":
    generate_deformation_animation()
