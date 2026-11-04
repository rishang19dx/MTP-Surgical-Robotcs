"""FEM tissue model inside the RL / benchmark environment (DEC-037 integration).

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_fem_env.py -q
"""
import os

import numpy as np
import pytest
import torch

from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator
from ReconstructionPipeline.evaluation.benchmark_runner import detect_bone_echo_depth
from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel, FEMTissueState
from ReconstructionPipeline.simulation.tissue_model import KelvinletTissueModel

SUBJECT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_totalsegmentator_patient")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIDLINE = (0.0, -32.5)   # L4 spinous process: a flat probe touches the paraspinal ridges first


@pytest.fixture(scope="module")
def patient():
    if not os.path.isdir(SUBJECT):
        pytest.skip("sample patient not available")
    return PatientAssetGenerator.load_totalsegmentator_subject(SUBJECT, target_vertebrae=["vertebrae_L4"],
                                                               crop_to_spine=False)


@pytest.fixture(scope="module")
def env(patient):
    return UltrasoundReconstructionEnv(patient, target_vertebra="vertebrae_L4", device=DEV, seed=0)


def _env(patient, model, **kw):
    return UltrasoundReconstructionEnv(patient, target_vertebra="vertebrae_L4", device=DEV, seed=0,
                                       tissue_model=model, **kw)


def test_fem_is_default_backend_and_window_inside_region(env):
    m = env.tissue_model
    assert isinstance(m, FEMTissueModel) and env.tissue_backend == "fem"
    half = 0.5 * m.face_dims[0]
    for (a, b), i in ((env.c1_bounds, 0), (env.c2_bounds, 2)):
        assert a - half >= m.roi_min[i] + env.WINDOW_MARGIN_MM - 1e-9
        assert b + half <= m.roi_max[i] - env.WINDOW_MARGIN_MM + 1e-9


def test_kelvinlet_backend_still_selectable(patient):
    e = UltrasoundReconstructionEnv(patient, target_vertebra="vertebrae_L4", device=DEV, seed=0,
                                    tissue_backend="kelvinlet")
    assert isinstance(e.tissue_model, KelvinletTissueModel) and e.tissue_backend == "kelvinlet"
    with pytest.raises(ValueError):
        UltrasoundReconstructionEnv(patient, target_vertebra="vertebrae_L4", device=DEV, tissue_backend="sofa")


def test_midline_force_tracks_target_from_first_contact_with_negative_indentation(env):
    t = env.skin_pose(*MIDLINE, 0.0, 0.0)
    d_fc = env._first_contact(t)
    assert d_fc < -3.0, "flat probe over the midline groove must touch the paraspinal ridges first"
    assert env.tissue_model.force_at(t, d_fc) < 0.5           # just touching
    assert env.tissue_model.force_at(t, 0.0) > env.target_force_n  # centre-on-skin is already above target
    d, f = env.regulate_force(t, 0.0, iterations=15)          # start AT first contact
    assert abs(f - env.target_force_n) < 0.2, f
    assert d_fc < d < 0.0, (d_fc, d)                           # negative indentation is used, not clamped to 0


def test_env_steps_track_force_and_expose_fem_state(env):
    env.reset(seed=3)
    forces = []
    for _ in range(12):   # sweep across the midline and back: first contact changes by ~9 mm
        _, _, _, _, info = env.step(np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32))
        forces.append(info["normal_force"])
    for _ in range(12):
        _, _, _, _, info = env.step(np.array([-0.8, -0.2, 0.1, 0.0], dtype=np.float32))
        forces.append(info["normal_force"])
    st = env.tissue_state
    assert isinstance(st, FEMTissueState) and st.in_contact and st.contact_stiffness_n_per_mm > 0
    assert info["normal_force"] == pytest.approx(st.reaction_force_n)
    f = np.asarray(forces)
    assert np.all(np.abs(f - env.target_force_n) < 1.5), f
    assert np.std(f) > 1e-3, "force is a model output and must vary"


def test_bmode_bone_echo_shallower_at_higher_force(patient, env):
    """Same pose and speckle seed, FEM backend: a higher target force pushes the face closer to the rigid
    bone, so the bone echo detected in the B-mode image moves up by about the extra indentation."""
    pose = np.array([*MIDLINE, 0.0, 0.0], dtype=np.float32)
    cols = slice(env.IMAGE_SIZE[1] // 2 - 8, env.IMAGE_SIZE[1] // 2 + 8)
    out = {}
    for fn in (3.0, 10.0):
        e = _env(patient, env.tissue_model, target_force_n=fn)
        e.reset(seed=0)
        e.current_pose_params = pose.copy()
        t = e.skin_pose(*pose)
        e._indent_rel_mm, _ = e.regulate_force(t, 0.0, iterations=20)
        e._indent_rel_mm -= e._first_contact(t)
        torch.manual_seed(0)
        obs, _ = e._acquire(pose)
        out[fn] = (detect_bone_echo_depth(obs["bmode"], e.axial_res_mm, cols), obs["indentation_mm"],
                   obs["normal_force"])
    (z3, d3, f3), (z10, d10, f10) = out[3.0], out[10.0]
    assert abs(f3 - 3.0) < 0.3 and abs(f10 - 10.0) < 0.5, out
    assert np.isfinite(z3) and np.isfinite(z10), out
    assert d10 > d3 + 1.0
    assert z10 < z3 - 1.0, out
    assert abs((z3 - z10) - (d10 - d3)) < 1.5, out   # bone is rigid: echo shift = extra indentation
