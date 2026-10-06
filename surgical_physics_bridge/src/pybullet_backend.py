import pybullet as p
import pybullet_data
from typing import Dict, List, Optional
import numpy as np
from backend_interface import SurgicalSimulationBackend
from coordinates import convert_pose_pb_to_slicer

class PyBulletBackend(SurgicalSimulationBackend):
    def __init__(self, gui=False):
        self.gui = gui
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        # Enable deformable physics support before loading anything
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.robot_id = None
        self.link_names_to_indices = {}
        self.anatomy_body_id = None
        self.is_soft_anatomy = False
        self.virtual_fixture_sdf = None
        self.virtual_fixture_policy = None
        
    def load_robot(self, robot_description: str) -> bool:
        """
        For PyBullet, robot_description is a path to a URDF file.
        """
        try:
            self.robot_id = p.loadURDF(robot_description, useFixedBase=True)
            self._map_links()
            return True
        except Exception as e:
            print(f"Error loading URDF: {e}")
            return False
            
    def _map_links(self):
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            link_name = info[12].decode('utf-8')
            self.link_names_to_indices[link_name] = i

    def load_anatomy(self, collision_mesh: str, visual_mesh: str, material_params: dict) -> bool:
        try:
            mass = material_params.get('mass', 0.0)
            self.is_soft_anatomy = material_params.get('is_soft', False)
            
            if self.is_soft_anatomy:
                # Load as a deformable soft body
                scale = material_params.get('scale', 1.0)
                self.anatomy_body_id = p.loadSoftBody(
                    collision_mesh, 
                    scale=scale, 
                    mass=mass, 
                    useNeoHookean=1, 
                    useBendingSprings=1, 
                    useMassSpring=1, 
                    springElasticStiffness=40, 
                    springDampingStiffness=.1, 
                    frictionCoeff=.5, 
                    useFaceContact=1
                )
            else:
                # Load as a rigid body
                col_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_mesh)
                vis_id = -1
                if visual_mesh:
                    vis_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_mesh)
                
                self.anatomy_body_id = p.createMultiBody(
                    baseMass=mass,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id
                )
            return True
        except Exception as e:
            print(f"Error loading anatomy: {e}")
            return False

    def set_joint_command(self, q_des: List[float], dq_des: Optional[List[float]] = None, tau: Optional[List[float]] = None):
        if self.robot_id is None: return
        num_joints = p.getNumJoints(self.robot_id)
        # Simplify: assume positional control for all revolute joints
        controlled_joints = [i for i in range(num_joints) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        
        if len(q_des) == len(controlled_joints):
            p.setJointMotorControlArray(
                bodyIndex=self.robot_id,
                jointIndices=controlled_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=q_des
            )

    def step(self, dt: float):
        p.setTimeStep(dt)
        p.stepSimulation()

    def get_link_poses(self) -> Dict[str, np.ndarray]:
        if self.robot_id is None: return {}
        poses = {}
        # Get base pose
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        poses["base_link"] = convert_pose_pb_to_slicer(base_pos, base_quat)
        
        for link_name, index in self.link_names_to_indices.items():
            state = p.getLinkState(self.robot_id, index)
            link_pos = state[4] # worldLinkFramePosition
            link_quat = state[5] # worldLinkFrameOrientation
            poses[link_name] = convert_pose_pb_to_slicer(link_pos, link_quat)
            
        return poses

    def get_tool_state(self) -> dict:
        if self.robot_id is None: return {}
        
        # Try to find a tool/tip link, fallback to the last link
        tool_index = -1
        for name, idx in self.link_names_to_indices.items():
            if any(keyword in name.lower() for keyword in ['tool', 'tip', 'end_effector', 'ee']):
                tool_index = idx
                break
                
        if tool_index == -1 and len(self.link_names_to_indices) > 0:
            tool_index = max(self.link_names_to_indices.values())
            
        if tool_index != -1:
            state = p.getLinkState(self.robot_id, tool_index, computeLinkVelocity=1)
            return {
                "position": list(state[4]),
                "orientation": list(state[5]),
                "linear_velocity": list(state[6]),
                "angular_velocity": list(state[7])
            }
        return {}

    def get_contacts(self) -> list:
        contacts = p.getContactPoints()
        return [
            {
                "body_a": c[1],
                "body_b": c[2],
                "link_a": c[3],
                "link_b": c[4],
                "position_on_a": c[5],
                "position_on_b": c[6],
                "contact_normal_on_b": c[7],
                "distance": c[8],
                "normal_force": c[9]
            } for c in contacts
        ]

    def get_deformed_vertices(self) -> Optional[np.ndarray]:
        if self.is_soft_anatomy and self.anatomy_body_id is not None:
            try:
                # MESH_DATA_SIMULATION_MESH retrieves the current simulation mesh vertices
                num_vertices, vertices = p.getMeshData(self.anatomy_body_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
                return np.array(vertices)
            except Exception as e:
                print(f"Error getting deformed vertices: {e}")
        return None

    def set_virtual_fixture(self, vf_data, policy):
        """
        Stores the virtual fixture definitions and creates a static collision body in PyBullet.
        """
        self.virtual_fixture_policy = policy
        
        vertices = vf_data.get("vertices", [])
        indices = vf_data.get("indices", [])
        
        if not vertices or not indices:
            print("Invalid Virtual Fixture data received.")
            return

        if policy == "keep_out":
            try:
                col_id = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=indices)
                vis_id = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices, rgbaColor=[1,0,0,0.3])
                self.virtual_fixture_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id)
                print(f"Virtual Fixture (Keep-Out) created with {len(vertices)} vertices and {len(indices)//3} faces.")
            except Exception as e:
                print(f"Error creating Virtual Fixture: {e}")

    def disconnect(self):
        p.disconnect(self.physics_client)
