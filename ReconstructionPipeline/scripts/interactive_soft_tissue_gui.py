"""
Interactive Desktop Soft-Tissue Modelling & Ultrasound Reslicing Studio.
Allows real-time interactive exploration of:
- Regularized Kelvinlet soft-tissue indentation mechanics
- Backward Eulerian coordinate warping for acoustic volume slicing
- Dynamic hyperechoic bone echo shift and acoustic coupling kappa(F_N)
- Live central A-Scan oscilloscope

Usage:
    ./.venv/bin/python ReconstructionPipeline/scripts/interactive_soft_tissue_gui.py
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec

from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator
from ReconstructionPipeline.simulation.kelvinlet_deformation import RegularizedKelvinletDeformation
from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor
from ReconstructionPipeline.simulation.ultrasound_model_based import ModelBasedUltrasoundSimulator


def launch_interactive_gui():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Interactive Soft-Tissue Studio on device: {device}")

    # 1. Generate realistic patient anatomical volume
    assets = PatientAssetGenerator.generate_lumbar_phantom(shape=(100, 100, 80))
    ct_vol = assets["ct_volume"]
    d_vol, h_vol, w_vol = ct_vol.shape
    spacing = assets["spacing_mm"]
    origin = assets["origin_mm"]
    skin_y = float(assets["skin_y_mm"])  # 55.0 mm

    # 2. Local US FOV parameters
    us_depth_mm = 50.0
    us_width_mm = 40.0
    img_h, img_w = 160, 100

    # 3. Solvers
    initial_force = 5.0
    initial_mu = 4.0
    initial_x = 18.0
    initial_radius = 14.0

    solver = RegularizedKelvinletDeformation(
        shear_modulus_kpa=initial_mu,
        poissons_ratio=0.46,
        probe_radius_mm=initial_radius,
        max_indentation_mm=16.0,
        device=device,
    )

    extractor = VolumeSliceExtractor(
        volume=ct_vol,
        spacing=spacing,
        origin=origin,
        image_size=(img_h, img_w),
        physical_size_mm=(us_depth_mm, us_width_mm),
        deformation_solver=solver,
        device=device,
    )
    sim = ModelBasedUltrasoundSimulator(device=device)

    # Pre-extract baseline undeformed slice for reference
    ref_pose = np.array([
        [1.0,  0.0,  0.0, initial_x],
        [0.0, -1.0,  0.0,  skin_y],
        [0.0,  0.0, -1.0,     0.0],
        [0.0,  0.0,  0.0,     1.0]
    ], dtype=np.float32)
    ref_slice = extractor.extract_slices(ref_pose, interpolation="bilinear", contact_force_n=0.0)
    ref_bmode = sim.simulate_bmode(ref_slice)["bmode"]
    depth_axis_mm = np.linspace(0.0, us_depth_mm, img_h)
    ref_col = ref_bmode[:, img_w // 2].detach().cpu().numpy()
    baseline_peak_idx = int(np.argmax(ref_col[30:]) + 30)
    baseline_bone_depth_mm = float(depth_axis_mm[baseline_peak_idx])

    # 4. Setup Matplotlib Interactive Figure
    fig = plt.figure(figsize=(16, 9))
    try:
        fig.canvas.manager.set_window_title("MTP - Interactive Soft-Tissue & Ultrasound Deformation Studio")
    except Exception:
        pass
    fig.patch.set_facecolor("#12141a")

    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[4.5, 1.2],
        width_ratios=[1.2, 1.0, 1.0],
        left=0.06, right=0.96, top=0.92, bottom=0.08, wspace=0.28, hspace=0.35
    )

    ax_anatomy = fig.add_subplot(gs[0, 0])
    ax_bmode = fig.add_subplot(gs[0, 1])
    ax_ascan = fig.add_subplot(gs[0, 2])

    # Slider axes
    ax_f = fig.add_subplot(gs[1, 0])
    ax_mu = fig.add_subplot(gs[1, 1])
    ax_pos = fig.add_subplot(gs[1, 2])

    for ax in [ax_f, ax_mu, ax_pos]:
        ax.set_facecolor("#1b1f2b")

    slider_f = Slider(ax_f, "Normal Force $F_N$ (N)", 0.0, 10.0, valinit=initial_force, valstep=0.2, color="#00e5ff")
    slider_mu = Slider(ax_mu, "Shear Modulus $\\mu$ (kPa)", 2.0, 20.0, valinit=initial_mu, valstep=0.5, color="#ff9100")
    slider_pos = Slider(ax_pos, "Probe Lateral Pos $X$ (mm)", -25.0, 25.0, valinit=initial_x, valstep=1.0, color="#76ff03")

    for s in [slider_f, slider_mu, slider_pos]:
        s.label.set_color("white")
        s.label.set_fontsize(11)
        s.valtext.set_color("white")
        s.valtext.set_fontsize(11)

    # Pre-extract background CT cross-section for panel 1
    z_mid_idx = ct_vol.shape[0] // 2
    ct_slice_2d = ct_vol[z_mid_idx, :, :]
    extent_anatomy = [
        origin[0], origin[0] + ct_vol.shape[2] * spacing[0],
        origin[1], origin[1] + ct_vol.shape[1] * spacing[1]
    ]

    # Global slice points for Lagrangian grid deformation
    ox, oy, oz = origin
    sx, sy, sz = spacing
    y_coords = torch.linspace(oy, oy + (h_vol - 1) * sy, h_vol, device=device)
    x_coords = torch.linspace(ox, ox + (w_vol - 1) * sx, w_vol, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid_z = torch.zeros_like(grid_x)
    global_slice_pts = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    def compute_frame(fn_val, mu_val, x_val):
        solver.mu_n_per_mm2 = max(1e-4, mu_val * 1e-3)
        solver.prefactor = 1.0 / (4.0 * np.pi * solver.mu_n_per_mm2)

        probe_pose = np.array([
            [1.0,  0.0,  0.0, x_val],
            [0.0, -1.0,  0.0,  skin_y],
            [0.0,  0.0, -1.0,     0.0],
            [0.0,  0.0,  0.0,     1.0]
        ], dtype=np.float32)

        contact_pt = torch.tensor([x_val, skin_y, 0.0], device=device)
        force_vec = torch.tensor([0.0, -float(fn_val), 0.0], device=device)

        # 1. Indentation depth
        if fn_val > 0.05:
            u_contact = solver.compute_displacement(
                contact_pt.unsqueeze(0), contact_pt, force_vec, bone_depth_mm=42.0
            )
            indent_mm = float(torch.norm(u_contact).item())
        else:
            indent_mm = 0.0

        # 2. Extract warped slice using correct API
        slice_tensor = extractor.extract_slices(
            probe_pose,
            interpolation="bilinear",
            contact_force_n=float(fn_val),
            bone_depth_mm=42.0,
        )

        # 3. Simulate B-mode
        bmode_dict = sim.simulate_bmode(slice_tensor)
        bmode_raw = bmode_dict["bmode"]
        if bmode_raw.ndim == 3:
            bmode_raw = bmode_raw[0]
        bmode_np = bmode_raw.detach().cpu().numpy()

        # Acoustic coupling modulation (kappa drops if FN < 2.0 N)
        coupling = min(1.0, max(0.05, fn_val / 2.0)) if fn_val < 2.0 else 1.0
        bmode_np = bmode_np * coupling

        # 4. Central A-Scan
        center_col = img_w // 2
        a_line = bmode_np[:, center_col]

        return {
            "bmode": bmode_np,
            "a_line": a_line,
            "depth_axis": depth_axis_mm,
            "indent_mm": indent_mm,
            "coupling": coupling,
            "x_val": x_val,
        }

    data = compute_frame(initial_force, initial_mu, initial_x)

    # Title
    fig.suptitle("AUTONOMOUS ROBOTIC ULTRASOUND: REAL-TIME SOFT-BODY KELVINLET DEFORMATION",
                 fontsize=14, fontweight="bold", color="#00e5ff", y=0.97)

    # Panel 1: Global Anatomy
    ax_anatomy.set_title("1. Patient Anatomy & Probe Contact", color="white", fontsize=11, pad=8)
    im_anat = ax_anatomy.imshow(ct_slice_2d, cmap="bone", extent=extent_anatomy, origin="lower", vmin=-150, vmax=1100)
    ax_anatomy.set_xlabel("Lateral X (mm)", color="#aaaaaa", fontsize=9)
    ax_anatomy.set_ylabel("Anterior-Posterior Y (mm)", color="#aaaaaa", fontsize=9)
    ax_anatomy.tick_params(colors="#888888")

    probe_width_vis = 18.0
    probe_rect = plt.Rectangle(
        (initial_x - probe_width_vis / 2, skin_y - data["indent_mm"]),
        probe_width_vis, 10.0, color="#00e5ff", alpha=0.85, zorder=5
    )
    ax_anatomy.add_patch(probe_rect)

    # Beam FOV lines
    beam_lines = ax_anatomy.plot(
        [initial_x - us_width_mm / 2, initial_x, initial_x + us_width_mm / 2],
        [skin_y - data["indent_mm"] - us_depth_mm, skin_y - data["indent_mm"], skin_y - data["indent_mm"] - us_depth_mm],
        color="#ffeb3b", linestyle="--", linewidth=1.2, alpha=0.75, zorder=4
    )[0]

    # Panel 2: Simulated B-Mode
    ax_bmode.set_title("2. Dynamic B-Mode Echogram (US FOV)", color="white", fontsize=11, pad=8)
    im_bmode = ax_bmode.imshow(data["bmode"], cmap="gray", vmin=0.0, vmax=1.0,
                               extent=[-us_width_mm / 2, us_width_mm / 2, us_depth_mm, 0.0])
    ax_bmode.set_xlabel("Azimuthal Width (mm)", color="#aaaaaa", fontsize=9)
    ax_bmode.set_ylabel("Axial Depth from Transducer (mm)", color="#aaaaaa", fontsize=9)
    ax_bmode.tick_params(colors="#888888")

    ref_bone_line = ax_bmode.axhline(
        y=baseline_bone_depth_mm, color="#ff1744", linestyle=":", linewidth=1.5,
        label=f"Rest Bone Echo ({baseline_bone_depth_mm:.1f} mm)"
    )
    cur_peak_idx = int(np.argmax(data["a_line"][30:]) + 30)
    cur_bone_depth = float(depth_axis_mm[cur_peak_idx])
    cur_bone_line = ax_bmode.axhline(
        y=cur_bone_depth, color="#00e676", linestyle="-", linewidth=1.5,
        label=f"Compressed Echo ({cur_bone_depth:.1f} mm)"
    )
    ax_bmode.legend(loc="lower right", facecolor="#1b1f2b", edgecolor="#444444", labelcolor="white", fontsize=8)

    # Panel 3: Central A-Scan RF Profile
    ax_ascan.set_title("3. Central A-Scan Profile (Echo Shift)", color="white", fontsize=11, pad=8)
    ax_ascan.set_facecolor("#161a23")
    ascan_line = ax_ascan.plot(data["a_line"], data["depth_axis"], color="#00e5ff", linewidth=1.8, label="RF Intensity")[0]
    ax_ascan.set_xlim(-0.05, 1.1)
    ax_ascan.set_ylim(us_depth_mm, 0.0)
    ax_ascan.set_xlabel("Normalized Intensity", color="#aaaaaa", fontsize=9)
    ax_ascan.set_ylabel("Depth (mm)", color="#aaaaaa", fontsize=9)
    ax_ascan.tick_params(colors="#888888")
    ax_ascan.grid(True, color="#2b3142", linestyle="--", alpha=0.5)

    shift_mm = max(0.0, baseline_bone_depth_mm - cur_bone_depth)
    hud_text = ax_ascan.text(
        0.05, 0.92,
        f"Normal Force: {initial_force:.1f} N\n"
        f"Indentation:  {data['indent_mm']:.1f} mm\n"
        f"Coupling:     {data['coupling'] * 100:.0f}%\n"
        f"Echo Shift:   -{shift_mm:.1f} mm\n"
        f"Stiffness:    {initial_mu:.1f} kPa",
        transform=ax_ascan.transAxes,
        color="#00e5ff",
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1b1f2b", edgecolor="#00e5ff", alpha=0.85)
    )

    def update(val):
        fn = slider_f.val
        mu = slider_mu.val
        x = slider_pos.val

        res = compute_frame(fn, mu, x)

        # Update Panel 1
        probe_rect.set_xy((x - probe_width_vis / 2, skin_y - res["indent_mm"]))
        beam_lines.set_data(
            [x - us_width_mm / 2, x, x + us_width_mm / 2],
            [skin_y - res["indent_mm"] - us_depth_mm, skin_y - res["indent_mm"], skin_y - res["indent_mm"] - us_depth_mm]
        )

        # Update Panel 2
        im_bmode.set_data(res["bmode"])
        peak_idx = int(np.argmax(res["a_line"][30:]) + 30)
        bone_depth = float(res["depth_axis"][peak_idx])
        cur_bone_line.set_ydata([bone_depth, bone_depth])
        cur_bone_line.set_label(f"Compressed Echo ({bone_depth:.1f} mm)")

        # Update Panel 3
        ascan_line.set_xdata(res["a_line"])
        cur_shift = max(0.0, baseline_bone_depth_mm - bone_depth)
        hud_text.set_text(
            f"Normal Force: {fn:.1f} N\n"
            f"Indentation:  {res['indent_mm']:.1f} mm\n"
            f"Coupling:     {res['coupling'] * 100:.0f}%\n"
            f"Echo Shift:   -{cur_shift:.1f} mm\n"
            f"Stiffness:    {mu:.1f} kPa"
        )

        fig.canvas.draw_idle()

    slider_f.on_changed(update)
    slider_mu.on_changed(update)
    slider_pos.on_changed(update)

    print("[✓] Studio ready! Sliders connected.")
    plt.show()


if __name__ == "__main__":
    launch_interactive_gui()
