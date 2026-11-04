"""
Frame conventions and conversions for the Surgical Physics Bridge.

Three frames are involved:

* **P (patient / simulation frame, mm)** -- the CT array frame used by
  ``UltrasoundReconstructionEnv`` (contract C7). The sample patient was resampled by
  ``prepare_patient_assets.export_totalsegmentator_dataset`` from an LPS-oriented source
  CT (``case-0004.nii.gz`` header axcodes ``('L', 'P', 'S')``) with the source i/j/k axes
  copied 1:1 into P's x/y/z. Hence P is an **LPS** frame: +X = patient Left,
  +Y = Posterior (toward the dorsal skin), +Z = Superior. The resampled ``ct.nii.gz``
  header (identity rotation) and the ``*_ras.obj`` meshes nevertheless label it "RAS";
  their coordinates are P coordinates, not true RAS.
* **RAS (3D Slicer world, mm)** -- obtained from P with the fixed matrix
  :data:`P_TO_RAS` = diag(-1, -1, +1) (det +1, a pure rotation by pi about S).
  Slicer content is kept in P and parented under a transform node holding this matrix,
  so the flip is applied exactly once and never baked into data.
* **PB (PyBullet world, m, Z-up)** -- where the operating-room scene lives. The PyBullet
  meshes written by ``prepare_patient_assets.export_patient_meshes`` use
  ``pb = (0.55 + x, z, 0.42 + (y - 55)) * 1e-3`` (P in mm), i.e. a permutation with
  det -1 (a mirror). :func:`patient_to_pybullet_points` reproduces that mapping exactly
  so that the robot touches the same anatomy that the meshes show. Orientations are
  built from mapped axes with :func:`probe_axes_to_pybullet_quaternion`, which always
  returns a proper rotation.
"""

from typing import Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# P <-> RAS (3D Slicer)
# --------------------------------------------------------------------------------------

#: Homogeneous 4x4 mapping patient frame P (LPS, mm) to Slicer RAS (mm). det = +1.
P_TO_RAS = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def patient_to_ras_matrix() -> np.ndarray:
    """Returns a copy of the 4x4 P -> RAS matrix (det +1)."""
    return P_TO_RAS.copy()


def patient_to_ras_points(points_p: np.ndarray) -> np.ndarray:
    """(N, 3) or (3,) points in P (mm) -> RAS (mm)."""
    pts = np.asarray(points_p, dtype=np.float64)
    return pts @ P_TO_RAS[:3, :3].T + P_TO_RAS[:3, 3]


def ras_to_patient_points(points_ras: np.ndarray) -> np.ndarray:
    """(N, 3) or (3,) points in RAS (mm) -> P (mm)."""
    pts = np.asarray(points_ras, dtype=np.float64)
    r = P_TO_RAS[:3, :3]
    return (pts - P_TO_RAS[:3, 3]) @ r  # r is orthonormal: inverse = transpose


def patient_pose_to_ras(t_p: np.ndarray) -> np.ndarray:
    """4x4 pose expressed in P -> the same pose expressed in RAS."""
    return P_TO_RAS @ np.asarray(t_p, dtype=np.float64)


# --------------------------------------------------------------------------------------
# P <-> PyBullet world
# --------------------------------------------------------------------------------------

#: Linear part of P(mm) -> PB(m) before scaling: pb_x = P_x, pb_y = P_z, pb_z = P_y. det = -1.
P_TO_PB_ROT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
#: Translation (m): P origin lands at (0.55, 0, 0.42 - 0.055) so that P_y = 55 mm sits at PB z = 0.42 m.
P_TO_PB_OFFSET_M = np.array([0.55, 0.0, 0.42 - 0.055], dtype=np.float64)


def patient_to_pybullet_points(points_p_mm: np.ndarray) -> np.ndarray:
    """(N, 3) or (3,) points in P (mm) -> PyBullet world (m). Matches the exported PyBullet meshes."""
    pts = np.asarray(points_p_mm, dtype=np.float64)
    return (pts @ P_TO_PB_ROT.T) * 1e-3 + P_TO_PB_OFFSET_M


def pybullet_to_patient_points(points_pb_m: np.ndarray) -> np.ndarray:
    """(N, 3) or (3,) PyBullet world points (m) -> P (mm)."""
    pts = np.asarray(points_pb_m, dtype=np.float64)
    return ((pts - P_TO_PB_OFFSET_M) * 1e3) @ P_TO_PB_ROT  # P_TO_PB_ROT is symmetric & orthonormal


def ras_to_pybullet_points(points_ras_mm: np.ndarray) -> np.ndarray:
    """(N, 3) Slicer RAS points (mm) -> PyBullet world (m), via P."""
    return patient_to_pybullet_points(ras_to_patient_points(points_ras_mm))


def probe_axes_to_pybullet_quaternion(beam_p: np.ndarray, lateral_p: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Orientation (x, y, z, w) of the end-effector in PyBullet such that the EE +Z axis points
    along the acoustic beam and the EE +X axis along the probe's lateral (array) axis.

    Directions are mapped P -> PB with :data:`P_TO_PB_ROT`; the frame is re-orthogonalised and
    completed right-handed (y = z x x), so the result is always a proper rotation even though the
    P -> PB map itself is a mirror.
    """
    from scipy.spatial.transform import Rotation

    z = P_TO_PB_ROT @ np.asarray(beam_p, dtype=np.float64)
    z /= np.linalg.norm(z)
    x = P_TO_PB_ROT @ np.asarray(lateral_p, dtype=np.float64)
    x = x - np.dot(x, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.column_stack([x, y, z])
    return tuple(float(v) for v in Rotation.from_matrix(rot).as_quat())
