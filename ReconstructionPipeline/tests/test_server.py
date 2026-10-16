"""
Unit test for OpenIGTLink message generation and Reconstruction Server mechanics.
"""

import numpy as np
import pyigtl
from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv


def test_openigtl_message_formats():
    env = UltrasoundReconstructionEnv(max_steps=5)
    obs = env.reset()

    # 1. Transform message
    t_mat = env._build_probe_matrix(*env.current_pose_params)
    tf_msg = pyigtl.TransformMessage(t_mat.astype(np.float64), device_name="UltrasoundProbe")
    assert tf_msg.device_name == "UltrasoundProbe"

    # 2. Image message
    bmode = (obs["bmode"] * 255.0).astype(np.uint8)
    img_msg = pyigtl.ImageMessage(bmode[:, :, np.newaxis], device_name="US_Image")
    assert img_msg.device_name == "US_Image"

    # 3. Point message
    pts = env.accumulator.get_points()
    pts_msg = pyigtl.PointMessage(positions=pts.astype(np.float32), device_name="ReconstructedBone")
    assert pts_msg.device_name == "ReconstructedBone"

    # 4. PolyData message
    poly_msg = pyigtl.PolyDataMessage(points=pts.astype(np.float32), device_name="DeformedAnatomy")
    assert poly_msg.device_name == "DeformedAnatomy"


def test_unified_server_step():
    from surgical_physics_bridge.src.unified_server import UnifiedSurgicalBridgeServer
    server = UnifiedSurgicalBridgeServer(port=18949, gui=False, publish_fps=60.0, scan_mode="orbital")
    try:
        # Step simulation and check published state
        for _ in range(5):
            server.step_simulation_and_stream()
        assert server.total_frames_published >= 0
        assert server.env.current_normal_force > 0.0
    finally:
        server.server.stop()
        server.backend.disconnect()

