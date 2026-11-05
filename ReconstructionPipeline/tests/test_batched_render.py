"""Batched deformed slicing + B-mode (DEC-041, task T2).

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_batched_render.py -q -s
Reference: the single-env path the env uses today (env_reconstruction.py): `state.probe_face_pose`, then
`extract_slices(..., tissue_state=state)`, then `simulate_bmode(ct_slice)`.
"""
import os
import time

import numpy as np
import pytest
import torch

from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator
from ReconstructionPipeline.simulation.fem_batched import BatchedFEMSolver
from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel
from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor
from ReconstructionPipeline.simulation.ultrasound_model_based import ModelBasedUltrasoundSimulator
from ReconstructionPipeline.tests.test_fem_batched import SUBJECT, _queries

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE, PHYS = (200, 150), (75.0, 50.0)


@pytest.fixture(scope="module")
def setup():
    if not os.path.isdir(SUBJECT):
        pytest.skip("sample patient not available")
    pat = PatientAssetGenerator.load_totalsegmentator_subject(SUBJECT, target_vertebrae=["vertebrae_L4"],
                                                              crop_to_spine=False)
    model = FEMTissueModel.from_patient_data(pat, device=DEV, verbose=False)
    ext = VolumeSliceExtractor(pat["ct_volume"], spacing=pat["spacing_mm"], origin=pat["origin_mm"],
                               image_size=IMAGE_SIZE, physical_size_mm=PHYS, device=DEV)
    return model, ext


def _single(model, ext, T, d):
    model._cache_last = None
    st = model.solve(T, d)
    return ext.extract_slices(st.probe_face_pose(T), interpolation="bilinear", tissue_state=st)


def test_batched_slices_match_single_env(setup):
    model, ext = setup
    poses, ds = _queries(model, np.random.default_rng(3), 16)
    st = BatchedFEMSolver(model).solve(poses, ds)
    slices = ext.extract_slices(st.probe_face_poses(poses), interpolation="bilinear",
                                batched_warp_fn=st.warp_to_reference)
    assert slices.shape == (16, *IMAGE_SIZE)
    n_deformed = 0
    for i in range(16):
        ref = _single(model, ext, poses[i], ds[i])
        # same CT, same warp up to fp32-vs-fp64 contact and fp16 field rounding: compare in HU
        diff = (slices[i] - ref).abs()
        assert diff.mean().item() < 1.0, (i, diff.mean().item())
        assert torch.quantile(diff.flatten(), 0.99).item() < 25.0, i
        if bool(st.in_contact[i]):
            n_deformed += 1
    assert n_deformed >= 10


def test_face_samples_compressed_skin(setup):
    """Falsifiable physics of the batched warp: at the flat site the indented face sits d mm below the
    undeformed skin, yet the tissue just under it must map back to (within 0.5 mm of) the skin surface,
    because the face pushed the skin down.  A rigid slice at the same face pose would instead sample
    tissue d mm deep.  The CT under the face must therefore differ from the rigid slice."""
    model, ext = setup
    T = model.dorsal_pose(-20.0, -45.0)
    d0 = model.first_contact_indentation(T)
    solver = BatchedFEMSolver(model)
    ds = np.array([2.0, 5.0, 8.0]) + d0
    st = solver.solve(np.stack([T] * 3), ds)
    face = st.probe_face_poses(np.stack([T] * 3))
    c = face[:, :3, 3].float()[:, None, :]
    X = st.warp_to_reference(c)[:, 0].cpu().numpy()
    for i, d in enumerate(ds - d0):
        assert T[1, 3] - face[i, 1, 3].item() == pytest.approx(d, abs=1e-6)       # face is d below the skin
        depth_ref = model.skin_height(X[i, 0], X[i, 2]) - X[i, 1]
        assert abs(depth_ref) < 0.5, (d, depth_ref)                                   # ... but samples the skin
    deformed = ext.extract_slices(face, batched_warp_fn=st.warp_to_reference)
    rigid = ext.extract_slices(face)
    top = slice(0, 20)   # first 7.5 mm under the face
    assert (deformed[2, top] - rigid[2, top]).abs().mean().item() > 1.0


@pytest.mark.parametrize("B", [1, 32, 128])
def test_render_batch_timing(setup, B):
    """FEM solve + warp + slice + B-mode for B envs (informational timing, not asserted)."""
    model, ext = setup
    if DEV.type != "cuda" and B > 32:
        pytest.skip("large batches only on GPU")
    solver = BatchedFEMSolver(model)
    sim = ModelBasedUltrasoundSimulator(device=DEV, axial_resolution_mm=PHYS[0] / (IMAGE_SIZE[0] - 1),
                                        lateral_resolution_mm=PHYS[1] / (IMAGE_SIZE[1] - 1), seed=0)
    poses, ds = _queries(model, np.random.default_rng(B), B)

    def step():
        st = solver.solve(poses, ds)
        sl = ext.extract_slices(st.probe_face_poses(poses), batched_warp_fn=st.warp_to_reference)
        return st, sl, sim.simulate_bmode(sl)["bmode"]

    step()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    reps = 5
    t = time.perf_counter()
    for _ in range(reps):
        st, sl, bm = step()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / reps * 1e3
    print(f"\n[render_batch] B={B}: {ms:.2f} ms per step (FEM + warp + slice + B-mode), {ms / B:.3f} ms/env")
    assert bm.shape == (B, *IMAGE_SIZE)
    assert torch.isfinite(bm).all()
