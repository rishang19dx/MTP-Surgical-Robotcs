"""Parity of the GPU-batched FEM solver with the single-env path (DEC-041, task T1).

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_fem_batched.py -q -s
The single-env `FEMTissueModel._solve` is the reference: every batched output must match it per env,
including envs that are out of contact, tilted, deeply indented, and envs with their own e_scale.
"""
import os
import time

import numpy as np
import pytest
import torch

from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator
from ReconstructionPipeline.simulation.fem_batched import BatchedFEMSolver
from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel

SUBJECT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_totalsegmentator_patient")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SITES = [(0.0, -32.5), (-20.0, -45.0), (20.0, -32.5), (0.0, -15.0), (-28.0, -20.0)]


@pytest.fixture(scope="module")
def model():
    if not os.path.isdir(SUBJECT):
        pytest.skip("sample patient not available")
    pat = PatientAssetGenerator.load_totalsegmentator_subject(SUBJECT, target_vertebrae=["vertebrae_L4"],
                                                              crop_to_spine=False)
    return FEMTissueModel.from_patient_data(pat, device=DEV, verbose=False)


@pytest.fixture(scope="module")
def solver(model):
    return BatchedFEMSolver(model, contact_dtype=torch.float64)   # strict parity with the CPU float64 path


def _pose(m, x, z, yaw=0.0, pitch=0.0, roll=0.0):
    T = m.dorsal_pose(x, z, yaw=yaw, pitch=pitch)
    c, s = np.cos(roll), np.sin(roll)
    T[:3, :3] = T[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return T


def _queries(m, rng, n):
    """Mixed queries: untilted/tilted, before first contact, light and deep indentation."""
    poses, ds = [], []
    for k in range(n):
        x, z = SITES[k % len(SITES)]
        x += rng.uniform(-4, 4)
        z += rng.uniform(-4, 4)
        T = _pose(m, x, z, yaw=rng.uniform(-0.26, 0.26), pitch=rng.uniform(-0.17, 0.17),
                  roll=rng.uniform(-0.17, 0.17) if k % 3 else 0.0)
        d0 = m.first_contact_indentation(T)
        d0 = 0.0 if not np.isfinite(d0) else d0
        ds.append(d0 + (rng.uniform(-2.0, -0.1) if k % 7 == 0 else rng.uniform(0.5, 12.0)))
        poses.append(T)
    return np.stack(poses), np.array(ds)


def _ref(m, T, d):
    m._cache_last = None
    return m.solve(T, d)


def test_batched_matches_single_env(model, solver):
    rng = np.random.default_rng(0)
    poses, ds = _queries(model, rng, 24)
    st = solver.solve(poses, ds)
    assert bool(st.active_set_converged.all())
    n_contact = 0
    for i in range(len(ds)):
        r = _ref(model, poses[i], ds[i])
        assert bool(st.in_contact[i]) == r.in_contact, i
        if not r.in_contact:
            assert float(st.reaction_force_n[i]) == 0.0
            assert float(st.u_grid[i].abs().max()) == 0.0
            continue
        n_contact += 1
        F_ref = r.reaction_force_n
        assert abs(float(st.reaction_force_n[i]) - F_ref) <= 1e-6 * max(1.0, F_ref), (i, F_ref)
        assert abs(float(st.linear_force_n[i]) - r.linear_force_n) <= 1e-6 * max(1.0, F_ref)
        assert abs(float(st.compression_mm[i]) - r.compression_mm) <= 1e-6
        np.testing.assert_allclose(st.force_vec_n[i].cpu().numpy(), r.force_vec_n, atol=1e-6 * max(1, F_ref))
        np.testing.assert_allclose(st.torque_nmm[i].cpu().numpy(), r.torque_nmm, atol=1e-4 * max(1, F_ref))
        assert abs(float(st.contact_stiffness_n_per_mm[i]) - r.contact_stiffness_n_per_mm) \
            <= 1e-4 * max(1.0, r.contact_stiffness_n_per_mm)
        assert int(st.num_active_nodes[i]) == r.num_active_nodes
        # field: same fp16 G_lat, so differences are only fp16 rounding of the force weights
        du = (st.u_grid[i] - r.u_grid[0]).abs().max().item()
        assert du <= 2e-3 * max(1e-3, r.u_grid.abs().max().item()) + 1e-4, (i, du)
    assert n_contact >= 15


def test_per_env_e_scale(model, solver):
    """Per-env stiffness randomisation equals re-running the single-env model with that e_scale."""
    T = _pose(model, -20.0, -45.0)
    base = model.e_scale
    scales = np.array([0.5, 1.0, 2.0]) * base
    st = solver.solve(np.stack([T] * 3), np.full(3, 5.0), e_scale=scales)
    try:
        for i, s in enumerate(scales):
            model.set_parameters(e_scale=float(s))
            r = _ref(model, T, 5.0)
            assert abs(float(st.reaction_force_n[i]) - r.reaction_force_n) <= 1e-6 * r.reaction_force_n
    finally:
        model.set_parameters(e_scale=base)
    F = st.reaction_force_n.cpu().numpy()
    assert F[0] < F[1] < F[2]   # stiffer tissue -> larger force at the same indentation


def test_displacement_lookup(model, solver):
    T = _pose(model, -20.0, -45.0)
    st = solver.solve(T, 6.0)
    r = _ref(model, T, 6.0)
    X = torch.randn(1, 500, 3, device=DEV) * 15 + torch.tensor(T[:3, 3], dtype=torch.float32, device=DEV)
    u_b = st.displacement(X)[0]
    u_r = r.displace(X[0]) - X[0]
    assert (u_b - u_r).abs().max().item() < 1e-3 * max(1.0, u_r.abs().max().item())


def test_float32_default_close_to_float64(model, solver):
    """The fp32 default must stay within 1e-5 relative force error and 0.05 mm field error of fp64."""
    poses, ds = _queries(model, np.random.default_rng(1), 64)
    r = solver.solve(poses, ds)
    s = BatchedFEMSolver(model).solve(poses, ds)
    assert torch.equal(s.in_contact, r.in_contact)
    rel = ((s.reaction_force_n.double() - r.reaction_force_n).abs() / r.reaction_force_n.clamp_min(1.0)).max()
    assert rel.item() < 1e-5
    assert (s.u_grid - r.u_grid).abs().max().item() < 0.05


@pytest.mark.parametrize("B", [1, 8, 32, 128])
def test_batch_sizes_run(model, B):
    """No Python loop over envs; prints the time per batched step (informational, not asserted)."""
    if DEV.type != "cuda" and B > 32:
        pytest.skip("large batches only on GPU")
    solver = BatchedFEMSolver(model)   # default float32, the training configuration
    rng = np.random.default_rng(B)
    poses, ds = _queries(model, rng, B)
    solver.solve(poses, ds)   # warm-up
    if DEV.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    reps = 5
    for _ in range(reps):
        st = solver.solve(poses, ds)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / reps * 1e3
    peak = torch.cuda.max_memory_allocated() / 1e6 if DEV.type == "cuda" else float("nan")
    print(f"\n[fem_batched] B={B}: {ms:.2f} ms/step ({ms / B:.3f} ms/env), active-set iters "
          f"{st.active_set_iters}, peak VRAM {peak:.0f} MB on {torch.cuda.get_device_name() if DEV.type == 'cuda' else 'cpu'}")
    assert st.reaction_force_n.shape == (B,)
    assert st.u_grid.shape[0] == B
    assert bool(st.active_set_converged.all())


def test_controller_queries_match_single_env(model, solver):
    """first_contact, tissue_thickness, max_indentation and force() equal the single-env methods."""
    poses, ds = _queries(model, np.random.default_rng(5), 20)
    d0 = solver.first_contact(poses).cpu().numpy()
    th = solver.tissue_thickness(poses).cpu().numpy()
    dm = solver.max_indentation(poses).cpu().numpy()
    Fb = solver.force(poses, ds).cpu().numpy()
    for i in range(len(ds)):
        np.testing.assert_allclose(d0[i], model.first_contact_indentation(poses[i]), atol=1e-6)
        np.testing.assert_allclose(th[i], model.tissue_thickness(poses[i]), atol=1e-6)
        np.testing.assert_allclose(dm[i], model.max_indentation(poses[i]), atol=1e-6)
        model._cache_last = None
        F_ref = model.force_at(poses[i], ds[i])
        assert abs(Fb[i] - F_ref) <= 1e-6 * max(1.0, F_ref)
