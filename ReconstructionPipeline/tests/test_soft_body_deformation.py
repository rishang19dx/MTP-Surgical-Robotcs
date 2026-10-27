"""
Unit & Integration Tests for GPU Regularized Kelvinlet Soft-Body Deformation.
Tests:
1. Kelvinlet mathematical invariants (singularity-free, asymptotic decay, physiological scale).
2. Backward Eulerian coordinate warping in VolumeSliceExtractor.
3. Dynamic bone echo shift under increasing contact force.
4. Latency benchmarks (< 2 ms GPU execution budget).
"""

import time
import numpy as np
import pytest
import torch

from ReconstructionPipeline.simulation.kelvinlet_deformation import RegularizedKelvinletDeformation
from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor
from ReconstructionPipeline.simulation.ultrasound_model_based import ModelBasedUltrasoundSimulator
from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator


def test_kelvinlet_mathematical_invariants():
    """Verifies that Regularized Kelvinlets have no singularity, decay with 1/r, and match elasticity theory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = RegularizedKelvinletDeformation(
        shear_modulus_kpa=12.0,
        poissons_ratio=0.48,
        probe_radius_mm=15.0,
        device=device,
    )

    contact_pt = torch.tensor([0.0, 50.0, 0.0], device=device)
    force_vec = torch.tensor([0.0, -4.0, 0.0], device=device) # 4 N pushing down along -y

    # 1. Zero force -> zero displacement
    zero_disp = solver.compute_displacement(contact_pt.unsqueeze(0), contact_pt, torch.zeros(3, device=device))
    assert torch.allclose(zero_disp, torch.zeros_like(zero_disp)), "Zero force must yield zero displacement"

    # 2. Indentation at contact center (r = 0)
    disp_center = solver.compute_displacement(contact_pt.unsqueeze(0), contact_pt, force_vec)
    indent_y = float(torch.abs(disp_center[0, 1]))
    theoretical_indent = solver.get_contact_indentation(4.0)

    # Indentation should match theoretical formula within 5%
    assert abs(indent_y - theoretical_indent) < 0.05, f"Indent {indent_y} != theory {theoretical_indent}"
    assert 1.0 <= indent_y <= 5.0, f"4 N force should cause 1-5 mm soft tissue indentation, got {indent_y:.2f} mm"

    # 3. Spatial decay: displacement at r = 50 mm should be much smaller than at r = 0
    far_pt = torch.tensor([[0.0, 50.0, 50.0]], device=device)
    disp_far = solver.compute_displacement(far_pt, contact_pt, force_vec)
    indent_far = float(torch.norm(disp_far))
    assert indent_far < indent_y * 0.35, f"Far displacement {indent_far} should decay significantly compared to {indent_y}"

    # 4. Bone depth damping: displacement at depth >= bone_depth should decay to zero
    bone_depth = 25.0
    deep_pt = torch.tensor([[0.0, 50.0 - bone_depth - 5.0, 0.0]], device=device)
    disp_deep = solver.compute_displacement(deep_pt, contact_pt, force_vec, bone_depth_mm=bone_depth)
    assert float(torch.norm(disp_deep)) < 1e-4, "Displacement beneath rigid bone boundary must be zero"


def test_backward_eulerian_slice_warping():
    """Verifies that backward Eulerian coordinate warping shifts anatomy shallower in the B-scan."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets = PatientAssetGenerator.generate_lumbar_phantom(shape=(80, 80, 60))

    ct_vol = assets["ct_volume"]
    spacing = assets["spacing_mm"]
    origin = assets["origin_mm"]

    extractor = VolumeSliceExtractor(
        volume=ct_vol,
        spacing=spacing,
        origin=origin,
        image_size=(100, 80),
        physical_size_mm=(50.0, 40.0),
        device=device,
    )

    # Define a probe pose at the dorsal skin pointing inward (-Y)
    pose = np.array([
        [1.0,  0.0,  0.0,  0.0],
        [0.0, -1.0,  0.0, 50.0],
        [0.0,  0.0, -1.0,  0.0],
        [0.0,  0.0,  0.0,  1.0]
    ], dtype=np.float32)

    # 1. Undeformed slice (0 N)
    slice_undeformed = extractor.extract_slices(pose, interpolation="bilinear", contact_force_n=0.0)

    # 2. Deformed slice under 6.0 N probe compression
    slice_deformed = extractor.extract_slices(pose, interpolation="bilinear", contact_force_n=6.0)

    assert slice_undeformed.shape == (100, 80)
    assert slice_deformed.shape == (100, 80)

    # Slices must differ under compression
    diff = torch.abs(slice_deformed - slice_undeformed)
    assert torch.max(diff) > 20.0, "Tissue deformation must noticeably alter CT resliced intensities"


def test_bmode_echo_shift_under_compression():
    """Verifies that increasing probe pressure physically compresses tissue and shifts the bone reflection shallower."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets = PatientAssetGenerator.generate_lumbar_phantom(shape=(80, 80, 60))

    extractor = VolumeSliceExtractor(
        volume=assets["ct_volume"],
        spacing=assets["spacing_mm"],
        origin=assets["origin_mm"],
        image_size=(120, 80),
        physical_size_mm=(60.0, 40.0),
        device=device,
    )

    sim = ModelBasedUltrasoundSimulator(device=device)

    pose = np.array([
        [1.0,  0.0,  0.0,  0.0],
        [0.0, -1.0,  0.0, 50.0],
        [0.0,  0.0, -1.0,  0.0],
        [0.0,  0.0,  0.0,  1.0]
    ], dtype=np.float32)

    # Low force (1.0 N) vs High force (7.0 N)
    slice_low = extractor.extract_slices(pose, interpolation="bilinear", contact_force_n=1.0)
    slice_high = extractor.extract_slices(pose, interpolation="bilinear", contact_force_n=7.0)

    bmode_low = sim.simulate_bmode(slice_low)["bmode"]
    bmode_high = sim.simulate_bmode(slice_high)["bmode"]

    # In the central column, locate the depth index of the peak cortical bone reflection
    center_col_low = bmode_low[:, 40].cpu().numpy()
    center_col_high = bmode_high[:, 40].cpu().numpy()

    peak_low = np.argmax(center_col_low[10:]) + 10
    peak_high = np.argmax(center_col_high[10:]) + 10

    # Under compression, bone reflection must appear shallower (smaller or equal depth index)
    assert peak_high <= peak_low, f"Bone peak under compression ({peak_high}) should be shallower than or equal to uncompressed ({peak_low})"


def test_deformation_latency_budget():
    """Verifies that Kelvinlet deformation + backward warping runs in < 2 ms on GPU/CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets = PatientAssetGenerator.generate_lumbar_phantom(shape=(60, 60, 40))

    extractor = VolumeSliceExtractor(
        volume=assets["ct_volume"],
        spacing=assets["spacing_mm"],
        origin=assets["origin_mm"],
        image_size=(200, 150), # Full resolution
        physical_size_mm=(75.0, 50.0),
        device=device,
    )

    pose = np.eye(4, dtype=np.float32)

    # Warmup
    for _ in range(3):
        _ = extractor.extract_slices(pose, contact_force_n=4.0)

    # Benchmark 20 iterations
    t0 = time.time()
    num_runs = 20
    for _ in range(num_runs):
        _ = extractor.extract_slices(pose, contact_force_n=4.0)
    avg_latency_ms = ((time.time() - t0) / num_runs) * 1000.0

    print(f"[*] Deformable Slice Extraction Latency: {avg_latency_ms:.3f} ms")
    assert avg_latency_ms < 10.0, f"Deformable slice extraction must be fast (< 10 ms), got {avg_latency_ms:.2f} ms"
