import numpy as np

def pybullet_to_slicer_ras_matrix():
    """
    Returns a 4x4 transformation matrix that converts a pose in the PyBullet 
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
    T = np.eye(4)
    # R column in Slicer corresponds to -Y in PyBullet
    T[0, 1] = -1.0
    T[0, 0] = 0.0
    
    # A column in Slicer corresponds to X in PyBullet
    T[1, 0] = 1.0
    T[1, 1] = 0.0
    
    # S column in Slicer corresponds to Z in PyBullet
    T[2, 2] = 1.0
    
    return T

def convert_pose_pb_to_slicer(pos, quat):
    """
    Convert a position and quaternion from PyBullet to Slicer RAS.
    pos: [x, y, z]
    quat: [x, y, z, w]
    Returns 4x4 homogenous matrix in Slicer RAS frame.
    """
    import scipy.spatial.transform as st
    
    # Convert pybullet quat to rotation matrix
    R_pb = st.Rotation.from_quat(quat).as_matrix()
    
    T_pb = np.eye(4)
    T_pb[:3, :3] = R_pb
    T_pb[:3, 3] = pos
    
    T_conv = pybullet_to_slicer_ras_matrix()
    
    # Apply transformation: T_slicer = T_conv * T_pb
    T_slicer = T_conv @ T_pb
    
    return T_slicer
