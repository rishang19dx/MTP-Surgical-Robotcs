import time
import argparse
import pyigtl
import numpy as np
import json
from pybullet_backend import PyBulletBackend

def main():
    parser = argparse.ArgumentParser(description="Standalone Surgical Physics Server")
    parser.add_argument('--urdf', type=str, required=True, help="Path to robot URDF")
    parser.add_argument('--port', type=int, default=18944, help="OpenIGTLink Server port")
    parser.add_argument('--gui', action='store_true', help="Enable PyBullet GUI")
    args = parser.parse_args()

    # OpenIGTLink Server Setup
    server = pyigtl.OpenIGTLinkServer(port=args.port)
    print(f"Starting Physics Server (OpenIGTLink) on port {args.port}...")

    # Physics Backend Setup
    backend = PyBulletBackend(gui=args.gui)
    if not backend.load_robot(args.urdf):
        print("Failed to load robot. Exiting.")
        return

    # 1 kHz physics loop -> 1ms dt
    dt = 0.001
    publish_rate = 50 # Hz, Slicer display rate
    publish_interval = 1.0 / publish_rate
    
    last_publish_time = time.time()
    
    try:
        while True:
            # 1. Receive control commands from IGTL
            for message in server.get_messages():
                if isinstance(message, pyigtl.StringMessage) and message.device_name == "CMD":
                    try:
                        cmd_data = json.loads(message.string)
                        if "q_des" in cmd_data:
                            backend.set_joint_command(cmd_data["q_des"])
                    except json.JSONDecodeError:
                        pass
                elif isinstance(message, pyigtl.StringMessage) and message.device_name == "VirtualFixture":
                    try:
                        vf_data = json.loads(message.string)
                        if "vertices" in vf_data and "indices" in vf_data:
                            backend.set_virtual_fixture(vf_data, vf_data.get("policy", "keep_out"))
                    except json.JSONDecodeError:
                        pass
                # Could also support NDArrayMessage for direct joint targets
                elif isinstance(message, pyigtl.NDArrayMessage) and message.device_name == "q_des":
                    backend.set_joint_command(message.ndarray.tolist())

            # 2. Step Physics
            backend.step(dt)
            
            # 3. Publish State at Slicer's target rate
            current_time = time.time()
            if current_time - last_publish_time >= publish_interval:
                poses = backend.get_link_poses()
                
                # Send Transforms for all links
                for link_name, pose_matrix in poses.items():
                    # pyigtl TransformMessage expects a 4x4 matrix
                    transform_msg = pyigtl.TransformMessage(np.array(pose_matrix), device_name=link_name)
                    server.send_message(transform_msg)
                
                # Send deformed vertices if soft anatomy is loaded
                vertices = backend.get_deformed_vertices()
                if vertices is not None:
                    # Send vertices as an NDArrayMessage
                    vert_msg = pyigtl.NDArrayMessage(vertices.astype(np.float32), device_name="DeformedAnatomy")
                    server.send_message(vert_msg)

                # Send contact forces for haptics/wrench rendering
                contacts = backend.get_contacts()
                if contacts:
                    contact_data = []
                    for c in contacts:
                        p = c["position_on_b"]
                        n = c["contact_normal_on_b"]
                        f = c["normal_force"]
                        contact_data.append([p[0], p[1], p[2], n[0], n[1], n[2], f])
                    contact_array = np.array(contact_data, dtype=np.float32)
                else:
                    # Send zero force to clear visualization
                    contact_array = np.zeros((1, 7), dtype=np.float32)
                contact_msg = pyigtl.NDArrayMessage(contact_array, device_name="ContactForces")
                server.send_message(contact_msg)

                last_publish_time = current_time
                
            # Sleep to match real-time
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("Stopping physics server...")
    finally:
        server.stop()
        backend.disconnect()

if __name__ == "__main__":
    main()
