import numpy as np
import pytest
from surgical_physics_bridge.src.coordinates import (
    pybullet_to_slicer_ras_matrix,
    convert_pose_pb_to_slicer,
    convert_slicer_ras_to_pb,
    convert_points_slicer_to_pb,
)

def test_pybullet_to_slicer_ras_matrix_properties():
    T = pybullet_to_slicer_ras_matrix()
    R = T[:3, :3]
    # Check orthonormal and right-handed
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ R.T, np.eye(3))

def test_convert_pose_pb_to_slicer_scaling():
    # PyBullet pose: 1.0m Forward (X), 0.5m Left (Y), 0.2m Up (Z)
    pos_pb = [1.0, 0.5, 0.2]
    quat_pb = [0.0, 0.0, 0.0, 1.0] # Identity rotation
    
    T_slicer = convert_pose_pb_to_slicer(pos_pb, quat_pb, scale_to_mm=True)
    
    # Rotation part should be pure rotation (det = 1)
    assert np.isclose(np.linalg.det(T_slicer[:3, :3]), 1.0)
    
    # Position in Slicer RAS should be in millimeters:
    # Right = -Left * 1000 = -500.0 mm
    # Anterior = Forward * 1000 = +1000.0 mm
    # Superior = Up * 1000 = +200.0 mm
    assert np.isclose(T_slicer[0, 3], -500.0)
    assert np.isclose(T_slicer[1, 3], 1000.0)
    assert np.isclose(T_slicer[2, 3], 200.0)

def test_roundtrip_coordinate_conversion():
    pos_pb = np.array([0.45, -0.12, 0.78])
    quat_pb = [0.0, 0.0, 0.0, 1.0]
    
    T_slicer = convert_pose_pb_to_slicer(pos_pb, quat_pb, scale_to_mm=True)
    pos_ras = T_slicer[:3, 3]
    
    pos_pb_recovered = convert_slicer_ras_to_pb(pos_ras, scale_to_m=True)
    assert np.allclose(pos_pb, pos_pb_recovered)

def test_convert_points_slicer_to_pb():
    # Points in Slicer RAS (mm)
    points_ras = np.array([
        [-500.0, 1000.0, 200.0],
        [0.0, 500.0, 300.0]
    ])
    points_pb = convert_points_slicer_to_pb(points_ras, scale_to_m=True)
    
    assert np.allclose(points_pb[0], [1.0, 0.5, 0.2])
    assert np.allclose(points_pb[1], [0.5, 0.0, 0.3])
