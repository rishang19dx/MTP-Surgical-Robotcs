import time
import zmq
import json
import argparse
from pybullet_backend import PyBulletBackend

def np_encoder(object):
    import numpy as np
    if isinstance(object, np.generic):
        return object.item()
    elif isinstance(object, np.ndarray):
        return object.tolist()
    raise TypeError(f"Object of type {type(object)} is not JSON serializable")

def main():
    parser = argparse.ArgumentParser(description="Standalone Surgical Physics Server")
    parser.add_argument('--urdf', type=str, required=True, help="Path to robot URDF")
    parser.add_argument('--port', type=int, default=5555, help="ZMQ Publisher port")
    parser.add_argument('--gui', action='store_true', help="Enable PyBullet GUI")
    args = parser.parse_args()

    # ZMQ Setup
    context = zmq.Context()
    socket_pub = context.socket(zmq.PUB)
    socket_pub.bind(f"tcp://*:{args.port}")
    
    socket_sub = context.socket(zmq.SUB)
    socket_sub.bind(f"tcp://*:{args.port + 1}")
    socket_sub.setsockopt_string(zmq.SUBSCRIBE, "CMD")
    socket_sub.setsockopt(zmq.RCVTIMEO, 0)

    
    print(f"Starting Physics Server on port {args.port}...")

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
            # 1. Receive control commands
            try:
                while True:
                    msg = socket_sub.recv_string(flags=zmq.NOBLOCK)
                    if msg.startswith("CMD "):
                        cmd_data = json.loads(msg[4:])
                        if "q_des" in cmd_data:
                            backend.set_joint_command(cmd_data["q_des"])
            except zmq.Again:
                pass
            
            # 2. Step Physics
            backend.step(dt)
            
            # 3. Publish State at Slicer's target rate
            current_time = time.time()
            if current_time - last_publish_time >= publish_interval:
                poses = backend.get_link_poses()
                
                message = {
                    "timestamp": current_time,
                    "poses": poses
                }
                
                # Use JSON for simplicity in prototyping, could use MessagePack/Protobuf for scaling
                socket_pub.send_string("STATE " + json.dumps(message, default=np_encoder))
                last_publish_time = current_time
                
            # Sleep to match real-time
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("Stopping physics server...")
    finally:
        backend.disconnect()
        socket_pub.close()
        socket_sub.close()
        context.term()

if __name__ == "__main__":
    main()
