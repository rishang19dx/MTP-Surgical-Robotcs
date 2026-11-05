"""
Soft-tissue model tests (contract C1): regularized Kelvinlet kernel, half-space correction, bone rigidity,
force-indentation behaviour, inverse warping and B-mode geometry under compression.
"""

import time

import numpy as np
import pytest
import torch

from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator
from ReconstructionPipeline.simulation.kelvinlet_deformation import (
    RegularizedKelvinletDeformation,
    kelvinlet_constants,
    regularized_kelvinlet,
)
from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor
from ReconstructionPipeline.simulation.tissue_model import KelvinletTissueModel, TissueModel

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def phantom():
    np.random.seed(0)
    return PatientAssetGenerator.generate_lumbar_phantom()


@pytest.fixture(scope="module")
def model(phantom):
    return KelvinletTissueModel.from_patient_data(phantom, device=DEV)


def dorsal_pose(model, x, z):
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])  # beam = -Y (into the back)
    T[:3, 3] = [x, model.skin.height(x, z), z]
    return T


# ----------------------------------------------------------------- kernel
def test_kelvinlet_centre_value_matches_closed_form():
    mu, nu, eps = 5.0, 0.45, 6.0
    a, b = kelvinlet_constants(mu, nu)
    f = torch.tensor([0.0, -2.0, 0.0], dtype=torch.float64)
    u0 = regularized_kelvinlet(torch.zeros(1, 3, dtype=torch.float64), f, a, b, eps)
    expected = a * (5 - 6 * nu) / (4 * (1 - nu) * eps) * 2.0  # de Goes & James, r = 0
    assert abs(float(-u0[0, 1]) - expected) / expected < 1e-9


def test_kelvinlet_far_field_is_classical_kelvin():
    mu, nu, eps = 5.0, 0.45, 2.0
    a, b = kelvinlet_constants(mu, nu)
    f = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    r = torch.tensor([[60.0, 40.0, 70.0]], dtype=torch.float64)
    u = regularized_kelvinlet(r, f, a, b, eps)[0].numpy()
    rn = np.linalg.norm(r.numpy())
    rh = r.numpy()[0] / rn
    mu_n = mu * 1e-3
    kel = (1.0 / (16 * np.pi * mu_n * (1 - nu) * rn)) * ((3 - 4 * nu) * np.eye(3) + np.outer(rh, rh)) @ f.numpy()
    assert np.allclose(u, kel, rtol=5e-3)


def test_half_space_centre_matches_boussinesq_circular_load():
    """Kelvinlet x half-space factor vs Boussinesq uniform circular load centre w0 = (1-nu) F / (pi mu a)."""
    mu, nu, eps, F = 6.0, 0.45, 8.0, 3.0
    solver = RegularizedKelvinletDeformation(mu, nu, eps, half_space=True, device=torch.device("cpu"))
    w0 = (1 - nu) * F / (np.pi * mu * 1e-3 * eps)
    u = solver.compute_displacement(np.zeros((1, 3), np.float32), np.zeros(3, np.float32),
                                    np.array([0, -F, 0], np.float32))
    assert abs(float(-u[0, 1]) - w0) / w0 < 0.02
    # Without the free-surface factor the full-space solution is ~2x too stiff
    solver_fs = RegularizedKelvinletDeformation(mu, nu, eps, half_space=False, device=torch.device("cpu"))
    u_fs = solver_fs.compute_displacement(np.zeros((1, 3), np.float32), np.zeros(3, np.float32),
                                          np.array([0, -F, 0], np.float32))
    assert float(-u_fs[0, 1]) < 0.6 * w0


# ------------------------------------------------------------ tissue model
def test_model_satisfies_protocol(model):
    assert isinstance(model, TissueModel)


def test_force_monotone_in_indentation(model):
    T = dorsal_pose(model, 0.0, 0.0)
    d_max = model.max_indentation(T)
    ds = np.linspace(0.0, d_max, 12)
    fs = np.array([model.force_at(T, d) for d in ds])
    assert fs[0] == 0.0
    assert np.all(np.diff(fs) > 0), fs


def test_indentation_at_4n_is_physiological(model):
    """Literature (paraspinal, linear probe): ~3 mm at 2 N, ~6 mm at 5 N; 4 N should give 3-9 mm."""
    T = dorsal_pose(model, 25.0, 0.0)  # off the spinous process: ~45 mm of tissue over bone
    d4 = model.indentation_for_force(T, 4.0)
    assert 3.0 <= d4 <= 9.0, d4
    assert abs(model.force_at(T, d4) - 4.0) < 1e-6


def test_force_depends_on_anatomy(model):
    """Thin tissue over the spinous process must be stiffer than thick tissue beside it."""
    thin = model.contact_mechanics(dorsal_pose(model, 0.0, 0.0))
    thick = model.contact_mechanics(dorsal_pose(model, 25.0, 0.0))
    assert thin["bone_depth_mm"] < thick["bone_depth_mm"]
    assert thin["k_lin"] > 1.1 * thick["k_lin"]  # only the central Kelvinlet sits over the 8 mm-wide process
    assert model.force_at(dorsal_pose(model, 0.0, 0.0), 5.0) > model.force_at(dorsal_pose(model, 25.0, 0.0), 5.0)


def test_stiffer_tissue_needs_less_indentation(phantom):
    soft = KelvinletTissueModel.from_patient_data(phantom, reference_k0_n_per_mm=0.3, device=DEV)
    stiff = KelvinletTissueModel.from_patient_data(phantom, reference_k0_n_per_mm=0.8, device=DEV)
    assert stiff.shear_modulus_kpa > soft.shear_modulus_kpa
    T = dorsal_pose(soft, 25.0, 0.0)
    assert stiff.indentation_for_force(T, 4.0) < soft.indentation_for_force(T, 4.0)


def test_bone_points_do_not_move(model, phantom):
    T = dorsal_pose(model, 0.0, 0.0)
    state = model.solve(T, model.indentation_for_force(T, 6.0))
    assert state.in_contact and state.indentation_mm > 1.0
    bone = phantom["bone_points"]
    near = bone[(np.abs(bone[:, 0]) < 30) & (np.abs(bone[:, 2]) < 20)]
    X = torch.tensor(near, dtype=torch.float32, device=DEV)
    u = (state.displace(X) - X).norm(dim=1)
    assert float(u.max()) < 1e-4
    # ...while soft tissue right under the probe moves by ~the indentation
    face = torch.tensor(T[:3, 3][None], dtype=torch.float32, device=DEV)
    assert float((state.displace(face) - face).norm()) > 0.5 * state.indentation_mm


def test_face_moves_by_indentation_and_warp_inverts(model):
    T = dorsal_pose(model, 25.0, 0.0)
    state = model.solve(T, 5.0)
    face = state.probe_face_pose(T)
    ys, xs = torch.meshgrid(torch.linspace(0, 75, 100), torch.linspace(-25, 25, 80), indexing="ij")
    loc = torch.stack([xs, ys, torch.zeros_like(xs), torch.ones_like(xs)], -1).view(-1, 4).to(DEV)
    x = (torch.tensor(face, dtype=torch.float32, device=DEV) @ loc.T).T[:, :3]
    X = state.warp_to_reference(x)
    err = (state.displace(X) - x).norm(dim=1)
    assert float(torch.quantile(err, 0.99)) < 0.05
    # probe-face centre maps back to (close to) the undeformed skin point
    centre = torch.tensor(face[:3, 3][None], dtype=torch.float32, device=DEV)
    assert float((state.warp_to_reference(centre)[0] - torch.tensor(T[:3, 3], device=DEV, dtype=torch.float32)).norm()) < 1.5


def test_no_contact_is_identity(model):
    T = dorsal_pose(model, 0.0, 0.0)
    state = model.solve(T, 0.0)
    assert not state.in_contact and state.reaction_force_n == 0.0
    x = torch.randn(10, 3, device=DEV)
    assert torch.equal(state.warp_to_reference(x), x)


# -------------------------------------------------------------- imaging
def _first_bone_row(label_slice, cols):
    col = (label_slice[:, cols] == 1).any(dim=1).cpu().numpy()
    return int(np.argmax(col)) if col.any() else -1


def test_bone_appears_shallower_by_the_indentation(model, phantom):
    """Bone is rigid, so its echo depth from the face must drop by exactly the extra indentation."""
    ext = VolumeSliceExtractor(phantom["label_volume"], spacing=phantom["spacing_mm"], origin=phantom["origin_mm"],
                               image_size=(301, 101), physical_size_mm=(75.0, 50.0), device=DEV)
    T = dorsal_pose(model, 0.0, 0.0)
    res = 75.0 / 300
    cols = slice(48, 53)
    rows = []
    for d in (1.0, 5.0):
        st = model.solve(T, d)
        lbl = ext.extract_slices(st.probe_face_pose(T), interpolation="nearest", tissue_state=st)
        rows.append(_first_bone_row(lbl, cols))
    assert rows[0] > 0 and rows[1] > 0
    shift_mm = (rows[0] - rows[1]) * res
    assert abs(shift_mm - 4.0) < 1.0, shift_mm


def test_solve_and_warp_latency(model):
    T = dorsal_pose(model, 10.0, 0.0)
    x = torch.randn(30000, 3, device=DEV) * 10 + torch.tensor(T[:3, 3], dtype=torch.float32, device=DEV)
    for _ in range(2):
        model.solve(T, 5.0).warp_to_reference(x)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        model.solve(T, 5.0).warp_to_reference(x)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 100
    print(f"[*] solve + 5-iteration warp of 30k points: {ms:.2f} ms")
    assert ms < (25.0 if DEV.type == "cuda" else 400.0)
