"""
Unit tests for the Model-Based Ultrasound Simulator & image-only Confidence Estimator.
"""

import math
import time
import warnings

import numpy as np
import pytest
import torch

from ReconstructionPipeline.simulation.ultrasound_model_based import ModelBasedUltrasoundSimulator, DB_TO_NEPER
from ReconstructionPipeline.simulation.confidence_map import UltrasoundConfidenceEstimator

CPU = torch.device("cpu")


def _bone_phantom(h=100, w=80, bone_row=60):
    ct = np.full((h, w), 40.0, dtype=np.float32)
    ct[bone_row:, :] = 1200.0
    lab = np.full((h, w), 2, dtype=np.int32)
    lab[bone_row:, :] = 1
    return torch.from_numpy(ct), torch.from_numpy(lab)


def test_ultrasound_simulation_pipeline():
    sim = ModelBasedUltrasoundSimulator(device=CPU, seed=0)
    ct, lab = _bone_phantom()
    res = sim.simulate_bmode(ct, label_slice=lab)
    bmode, energy = res["bmode"], res["energy"]
    assert set(res) == {"bmode", "reflection", "backscatter", "energy", "effective_energy", "bone_mask"}
    assert bmode.min() >= 0.0 and bmode.max() <= 1.0
    # Bone interface brighter than tissue above and than the shadow below
    assert bmode[58:63].mean() > bmode[30:50].mean() > bmode[80:95].mean()
    assert energy[90, 40] < 1e-3 * energy[55, 40]


def test_pixel_spacing_changes_attenuation_in_nepers():
    """Homogeneous soft tissue: energy(y) must equal exp(-2 * 0.1151 * alpha * f * depth_cm)."""
    ct = torch.full((200, 20), 40.0)
    for dy_mm in (0.2, 0.5):
        sim = ModelBasedUltrasoundSimulator(device=CPU, axial_resolution_mm=dy_mm, lateral_resolution_mm=0.3, seed=0)
        e = sim.simulate_bmode(ct)["energy"][:, 10]
        depth_cm = 150 * dy_mm / 10.0
        expected = math.exp(-2.0 * DB_TO_NEPER * sim.alpha_soft_tissue_db * sim.frequency * depth_cm)
        assert e[150].item() == pytest.approx(expected, rel=1e-4)
    # sanity: 0.6 dB/cm/MHz at 5 MHz over 1 cm round trip = 6 dB amplitude loss
    sim = ModelBasedUltrasoundSimulator(device=CPU, axial_resolution_mm=0.5, seed=0)
    e = sim.simulate_bmode(ct)["energy"][:, 0]
    assert 20 * math.log10(e[20].item()) == pytest.approx(-6.0, abs=1e-3)


def test_tgc_is_fixed_and_does_not_cancel_tissue_attenuation():
    ct = torch.full((200, 10), 40.0)
    ratios = []
    for alpha in (0.8, 1.2):  # round-trip loss 8 and 12 dB/cm > nominal TGC 5 dB/cm (no clamping)
        sim = ModelBasedUltrasoundSimulator(device=CPU, alpha_soft_tissue_db=alpha, seed=0)
        r = sim.simulate_bmode(ct)
        ratios.append(r["effective_energy"][:, 0] / r["energy"][:, 0])
        # TGC must not flatten the attenuation exactly
        assert abs(r["effective_energy"][190, 0].item() - r["effective_energy"][0, 0].item()) > 0.05
    assert torch.allclose(ratios[0], ratios[1], rtol=1e-4), "TGC must not depend on the tissue present"


def test_psf_sigmas_in_mm_and_deprecated_alias():
    sim = ModelBasedUltrasoundSimulator(device=CPU, axial_resolution_mm=0.25, lateral_resolution_mm=0.5,
                                        psf_sigma_axial_mm=0.5, psf_sigma_lateral_mm=1.0)
    assert sim.psf_sigma_axial_px == pytest.approx(2.0)
    assert sim.psf_sigma_lateral_px == pytest.approx(2.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old = ModelBasedUltrasoundSimulator(device=CPU, spatial_resolution_mm=0.5)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert old.axial_resolution_mm == old.lateral_resolution_mm == 0.5


def test_speckle_rng_is_seedable():
    ct, lab = _bone_phantom()
    a = ModelBasedUltrasoundSimulator(device=CPU, seed=3).simulate_bmode(ct, lab)["bmode"]
    b = ModelBasedUltrasoundSimulator(device=CPU, seed=3).simulate_bmode(ct, lab)["bmode"]
    c = ModelBasedUltrasoundSimulator(device=CPU, seed=4).simulate_bmode(ct, lab)["bmode"]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    g = torch.Generator(device=CPU).manual_seed(7)
    g2 = torch.Generator(device=CPU).manual_seed(7)
    sim = ModelBasedUltrasoundSimulator(device=CPU)
    assert torch.equal(sim.simulate_bmode(ct, lab, generator=g)["bmode"], sim.simulate_bmode(ct, lab, generator=g2)["bmode"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU latency target")
def test_gpu_latency_under_2ms():
    dev = torch.device("cuda")
    sim = ModelBasedUltrasoundSimulator(device=dev, axial_resolution_mm=75 / 199, lateral_resolution_mm=50 / 149, seed=0)
    ct = torch.full((200, 150), 40.0, device=dev)
    ct[120:] = 1200.0
    for _ in range(10):
        sim.simulate_bmode(ct)
    times = []
    for _ in range(50):
        torch.cuda.synchronize()
        t = time.perf_counter()
        sim.simulate_bmode(ct)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    assert np.min(times) * 1e3 < 2.0, np.median(times) * 1e3  # best-of-N (robust to contention)


def test_confidence_map_is_image_only():
    est = UltrasoundConfidenceEstimator(device=CPU)
    bmode = torch.rand((100, 80)) * 0.6
    conf = est.compute_confidence_map(bmode)
    assert conf.shape == (100, 80)
    assert conf.min() >= 0.0 and conf.max() <= 1.0
    # the simulator's privileged energy map cannot even be passed in (invariant 5)
    with pytest.raises(TypeError):
        est.compute_confidence_map(bmode, energy_map=torch.zeros((100, 80)))


def test_confidence_shadow_starts_below_bone_peak():
    sim = ModelBasedUltrasoundSimulator(device=CPU, seed=0)
    ct, lab = _bone_phantom()
    bmode = sim.simulate_bmode(ct, lab)["bmode"]
    conf = UltrasoundConfidenceEstimator(device=CPU, axial_resolution_mm=sim.axial_resolution_mm).compute_confidence_map(bmode)
    peak_rows = bmode.argmax(dim=0)
    cols = torch.arange(80)
    assert conf[peak_rows, cols].mean() > 0.7, "bone ridge must keep high confidence"
    assert conf[80:].mean() < 0.2, "region below bone must be low-confidence (shadow)"
    assert conf[30:50].mean() > 0.7
