import os

import numpy as np
import pytest

from surgical_physics_bridge.src.coordinates import (
    P_TO_PB_ROT,
    P_TO_RAS,
    patient_pose_to_ras,
    patient_to_pybullet_points,
    patient_to_ras_points,
    probe_axes_to_pybullet_quaternion,
    pybullet_to_patient_points,
    ras_to_patient_points,
    ras_to_pybullet_points,
)
from surgical_physics_bridge.src.obj_io import read_obj

PATIENT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_totalsegmentator_patient")


def test_p_to_ras_is_proper_rotation_with_expected_axes():
    r = P_TO_RAS[:3, :3]
    assert np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(r @ r.T, np.eye(3))
    # P is LPS: +X_P = Left -> -R, +Y_P = Posterior (dorsal) -> -A, +Z_P = Superior -> +S
    assert np.allclose(patient_to_ras_points([1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0])
    assert np.allclose(patient_to_ras_points([0.0, 1.0, 0.0]), [0.0, -1.0, 0.0])
    assert np.allclose(patient_to_ras_points([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])


def test_dorsal_skin_is_posterior_in_ras():
    """The dorsal skin (max P_y) must end up with the most negative A coordinate in RAS."""
    verts, _ = read_obj(os.path.join(PATIENT_DIR, "meshes", "patient_torso_skin_ras.obj"))
    ras = patient_to_ras_points(verts)
    dorsal = verts[:, 1] >= np.percentile(verts[:, 1], 99)
    assert ras[dorsal, 1].mean() < np.percentile(ras[:, 1], 5)


def test_ras_patient_roundtrip_and_pose():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-100, 100, size=(50, 3))
    assert np.allclose(ras_to_patient_points(patient_to_ras_points(pts)), pts)
    t = np.eye(4)
    t[:3, 3] = [10.0, 20.0, 30.0]
    t_ras = patient_pose_to_ras(t)
    assert np.allclose(t_ras[:3, 3], [-10.0, -20.0, 30.0])
    assert np.isclose(np.linalg.det(t_ras[:3, :3]), 1.0)


def test_patient_to_pybullet_matches_exported_meshes():
    """The P->PB mapping must reproduce the vertices of the exported PyBullet torso mesh."""
    v_p, f_p = read_obj(os.path.join(PATIENT_DIR, "meshes", "patient_torso_skin_ras.obj"))
    v_pb, f_pb = read_obj(os.path.join(PATIENT_DIR, "meshes", "patient_torso_skin.obj"))
    if len(v_p) != len(v_pb):
        pytest.skip("P and PyBullet torso meshes were decimated differently")
    assert np.allclose(patient_to_pybullet_points(v_p), v_pb, atol=1e-5)
    # dorsal skin at P_y = 55 mm sits at PB z = 0.42 m
    assert np.isclose(patient_to_pybullet_points([0.0, 55.0, 0.0])[2], 0.42)
    assert np.allclose(pybullet_to_patient_points(patient_to_pybullet_points(v_p)), v_p, atol=1e-4)


def test_ras_to_pybullet_composes_through_patient():
    ras = np.array([[-12.0, -55.0, 31.0]])
    assert np.allclose(ras_to_pybullet_points(ras), patient_to_pybullet_points(ras_to_patient_points(ras)))


def test_probe_orientation_is_proper_and_aligned():
    from scipy.spatial.transform import Rotation

    beam_p = np.array([0.0, -1.0, 0.0])  # into the patient from the dorsal skin
    lateral_p = np.array([1.0, 0.0, 0.0])
    q = probe_axes_to_pybullet_quaternion(beam_p, lateral_p)
    r = Rotation.from_quat(q).as_matrix()
    assert np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(r[:, 2], [0.0, 0.0, -1.0])  # EE z along beam = PyBullet down
    assert np.allclose(r[:, 0], P_TO_PB_ROT @ lateral_p)
    # tilted probe stays proper and z follows the beam
    beam_t = np.array([0.3, -0.9, 0.1])
    r2 = Rotation.from_quat(probe_axes_to_pybullet_quaternion(beam_t, lateral_p)).as_matrix()
    assert np.isclose(np.linalg.det(r2), 1.0)
    assert np.allclose(r2[:, 2], (P_TO_PB_ROT @ beam_t) / np.linalg.norm(beam_t))
