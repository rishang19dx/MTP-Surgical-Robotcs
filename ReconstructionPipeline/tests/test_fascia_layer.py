"""Superficial fascia layer in the B-mode simulator (DEC-055, thread B measurability).

The thoracolumbar fascia over the erector spinae is a thin, hyperechoic dense-collagen layer (Pirri et al. 2023;
Langevin et al. 2009) that a CT-derived B-mode cannot show (DEC-051). `superficial_fascia_mask` places a layer of
literature thickness on the beam-facing side of a muscle label slice; `simulate_bmode(fascia_mask=...)` renders it
with a literature impedance and attenuation.
"""

import numpy as np
import pytest
import torch

from ReconstructionPipeline.simulation.ultrasound_model_based import (
    FASCIA_ALPHA_DB,
    FASCIA_IMPEDANCE_MRAYL,
    FASCIA_THICKNESS_MM,
    ModelBasedUltrasoundSimulator,
    superficial_fascia_mask,
)

AX, LAT = 75.0 / 199, 50.0 / 149
H, W = 200, 150
FAT_HU, MUSCLE_HU = -100.0, 50.0


def _phantom(muscle_top_px=80):
    """Fat above `muscle_top_px`, muscle below it; label = muscle."""
    ct = torch.full((1, H, W), FAT_HU)
    ct[:, muscle_top_px:, :] = MUSCLE_HU
    label = torch.zeros((1, H, W), dtype=torch.bool)
    label[:, muscle_top_px:, :] = True
    return ct, label


def _sim():
    return ModelBasedUltrasoundSimulator(device=torch.device("cpu"), axial_resolution_mm=AX, lateral_resolution_mm=LAT)


def test_literature_constants():
    """Pirri et al. 2023 Table 3 (healthy, L3, transverse: 1.6 and 1.7 mm); Miles 1996 (1650 m/s across collagen
    fibres, 11.3 dB/cm at 7 MHz) with a density of 1.11 g/cm^3."""
    assert FASCIA_THICKNESS_MM == pytest.approx(1.65)
    assert FASCIA_IMPEDANCE_MRAYL == pytest.approx(1.650 * 1.11, abs=0.005)
    assert FASCIA_ALPHA_DB == pytest.approx(11.3 / 7.0, abs=0.01)


def test_mask_covers_the_layer_thickness_just_above_the_muscle():
    _, label = _phantom(muscle_top_px=80)
    label[:, :, :10] = False                                   # columns without muscle get no fascia
    m = superficial_fascia_mask(label, AX)
    assert m.shape == label.shape
    col = m[0, :, 50].numpy()
    assert col.sum() == pytest.approx(FASCIA_THICKNESS_MM / AX, rel=1e-5)   # sub-pixel coverage, exact total
    assert col[80:].sum() == 0.0                               # nothing inside the muscle
    assert col[: 80 - int(np.ceil(FASCIA_THICKNESS_MM / AX))].sum() == 0.0
    assert m[0, :, :10].sum() == 0.0
    assert float(m.max()) <= 1.0 and float(m.min()) >= 0.0


def test_mask_follows_the_muscle_surface_per_column():
    _, label = _phantom(muscle_top_px=80)
    label[:, :100, 100:] = False                               # a deeper surface on the right
    m = superficial_fascia_mask(label, AX)
    assert m[0, 75:80, 50].sum() > 0 and m[0, 75:80, 120].sum() == 0
    assert m[0, 95:100, 120].sum() > 0


def test_no_mask_leaves_the_simulator_unchanged():
    ct, _ = _phantom()
    sim = _sim()
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    a = sim.simulate_bmode(ct, generator=g1)["bmode"]
    b = sim.simulate_bmode(ct, fascia_mask=None, generator=g2)["bmode"]
    assert torch.equal(a, b)


def test_fascia_makes_the_fat_muscle_boundary_echogenic():
    """The superficial face reflects with R(fat -> fascia) instead of R(fat -> muscle); a second, deeper echo comes
    from the fascia -> muscle face."""
    ct, label = _phantom(muscle_top_px=80)
    sim = _sim()
    plain = sim.simulate_bmode(ct, generator=torch.Generator().manual_seed(0))["reflection"][0, :, 75]
    fas = sim.simulate_bmode(ct, fascia_mask=superficial_fascia_mask(label, AX),
                             generator=torch.Generator().manual_seed(0))["reflection"][0, :, 75]
    z_fat = 1.54 + (FAT_HU - 40.0) / 1000.0
    z_mus = 1.54 + (MUSCLE_HU - 40.0) / 1000.0
    r_plain = abs(z_mus - z_fat) / (z_mus + z_fat)
    r_fas = abs(FASCIA_IMPEDANCE_MRAYL - z_fat) / (FASCIA_IMPEDANCE_MRAYL + z_fat)
    top = 80 - FASCIA_THICKNESS_MM / AX
    sup = float(fas[int(top) - 2:int(top) + 3].max())
    assert sup / float(plain.max()) == pytest.approx(r_fas / r_plain, rel=0.35)
    assert sup > 2.0 * float(plain.max())
    assert float(fas[78:84].max()) > 0.3 * sup                  # the deep face echoes too


def test_fascia_attenuates_what_lies_below():
    ct, label = _phantom(muscle_top_px=80)
    sim = _sim()
    e0 = sim.simulate_bmode(ct)["energy"][0, 150, 75]
    e1 = sim.simulate_bmode(ct, fascia_mask=superficial_fascia_mask(label, AX))["energy"][0, 150, 75]
    assert float(e1) < float(e0)
