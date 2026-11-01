import numpy as np

def pybullet_to_slicer_ras_matrix():
    """
    Returns a 4x4 transformation matrix that converts a pose or vector in the PyBullet 
    world frame (Z-up, X-forward, Y-left) to the 3D Slicer RAS coordinate system.
    
    PyBullet:
    X = Forward
    Y = Left
    Z = Up
    
    Slicer RAS:
    R (X) = Right
    A (Y) = Anterior (Forward)
    S (Z) = Superior (Up)
    
    Mapping:
    PyBullet X (Forward) -> Slicer Y (Anterior)
    PyBullet Y (Left)    -> Slicer -X (Left / -Right)
    PyBullet Z (Up)      -> Slicer Z (Superior)
    """
    T = np.eye(4, dtype=np.float64)
    # R column in Slicer corresponds to -Y in PyBullet
    T[0, 1] = -1.0
    T[0, 0] = 0.0
    
    # A column in Slicer corresponds to X in PyBullet
    T[1, 0] = 1.0
    T[1, 1] = 0.0
    
    # S column in Slicer corresponds to Z in PyBullet
    T[2, 2] = 1.0
    
    return T

def convert_pose_pb_to_slicer(pos, quat, scale_to_mm=True):
    """
    Convert a position and quaternion from PyBullet (meters) to Slicer RAS (millimeters).
    pos: [x, y, z] in meters
    quat: [x, y, z, w]
    scale_to_mm: If True, scales translation from meters to millimeters (1000x)
    Returns 4x4 homogenous matrix in Slicer RAS frame.
    """
    import scipy.spatial.transform as st
    
    R_pb = st.Rotation.from_quat(quat).as_matrix()
    pos_vec = np.asarray(pos, dtype=np.float64)
    if scale_to_mm:
        pos_vec = pos_vec * 1000.0
    
    T_conv = pybullet_to_slicer_ras_matrix()
    
    T_slicer = np.eye(4, dtype=np.float64)
    # Rotation part: R_conv @ R_pb (preserves unit determinant)
    T_slicer[:3, :3] = T_conv[:3, :3] @ R_pb
    # Translation part: R_conv @ pos_vec
    T_slicer[:3, 3] = T_conv[:3, :3] @ pos_vec
    
    return T_slicer

def convert_slicer_ras_to_pb(pos_ras, quat_ras=None, scale_to_m=True):
    """
    Convert position from Slicer RAS (millimeters) back to PyBullet world frame (meters).
    pos_ras: [R, A, S] in mm
    """
    pos = np.asarray(pos_ras, dtype=np.float64)
    if scale_to_m:
        pos = pos / 1000.0
        
    T_conv = pybullet_to_slicer_ras_matrix()
    R_inv = T_conv[:3, :3].T
    return R_inv @ pos

def convert_points_slicer_to_pb(points_ras, scale_to_m=True):
    """
    Convert an (N, 3) array of vertices from Slicer RAS (mm) to PyBullet world frame (m).
    """
    pts = np.asarray(points_ras, dtype=np.float64)
    if scale_to_m:
        pts = pts / 1000.0
        
    T_conv = pybullet_to_slicer_ras_matrix()
    R_inv = T_conv[:3, :3].T
    # pts shape (N, 3): pts @ R_inv.T == (R_inv @ pts.T).T
    return pts @ R_inv.T
