import abc
from typing import Dict, List, Optional
import numpy as np

class SurgicalSimulationBackend(abc.ABC):
    """
    Abstract base class defining the contract for the physics engine backend 
    in the Slicer-Physics bridge.
    """
    
    @abc.abstractmethod
    def load_robot(self, robot_description: str) -> bool:
        """
        Load a robot model (e.g. URDF) into the simulation.
        Returns True if successful.
        """
        pass
        
    @abc.abstractmethod
    def load_anatomy(self, collision_mesh: str, visual_mesh: str, material_params: dict) -> bool:
        """
        Load an anatomical model for interaction.
        """
        pass

    @abc.abstractmethod
    def set_joint_command(self, q_des: List[float], dq_des: Optional[List[float]] = None, tau: Optional[List[float]] = None):
        """
        Send control commands to the robot joints.
        """
        pass

    @abc.abstractmethod
    def step(self, dt: float):
        """
        Step the physics simulation forward by dt seconds.
        """
        pass

    @abc.abstractmethod
    def get_link_poses(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary mapping link names to their 4x4 pose matrices 
        in the Slicer RAS coordinate frame.
        """
        pass

    @abc.abstractmethod
    def get_tool_state(self) -> dict:
        """
        Get the current state (pose, velocity) of the end effector / tool tip.
        """
        pass

    @abc.abstractmethod
    def get_contacts(self) -> list:
        """
        Get current contact points and forces.
        """
        pass

    @abc.abstractmethod
    def get_deformed_vertices(self) -> Optional[np.ndarray]:
        """
        If a deformable object is loaded, return its current Nx3 vertex array.
        Returns None for purely rigid simulations.
        """
        pass

    @abc.abstractmethod
    def set_virtual_fixture(self, signed_distance_field, policy):
        """
        Pass a safety boundary (SDF) and a reaction policy to the physics engine.
        """
        pass
