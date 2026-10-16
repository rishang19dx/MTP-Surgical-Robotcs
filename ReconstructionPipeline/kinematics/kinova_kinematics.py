"""
Robot-Agnostic Kinematics & Differential IK Controller with Kinova Gen3 7-DoF Support.
Computes forward kinematics, analytical Jacobians, and damped least-squares differential IK
with nullspace posture optimization and surface normal alignment constraints.
"""

from typing import Tuple, List, Optional
import numpy as np


class RobotKinematicsInterface:
    """Base abstract interface for robotic manipulators in the ultrasound pipeline."""
    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (position [x, y, z], rotation_matrix 3x3)."""
        raise NotImplementedError

    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Returns 6xN geometric Jacobian."""
        raise NotImplementedError

    def solve_differential_ik(
        self,
        q_current: np.ndarray,
        target_twist: np.ndarray,
        damping: float = 0.05,
    ) -> np.ndarray:
        """Computes joint velocity dq resolving target Cartesian velocity."""
        raise NotImplementedError


class KinovaGen3Kinematics(RobotKinematicsInterface):
    """
    Kinematic model and differential IK solver for the Kinova Gen3 7-DoF robotic arm.
    Standard modified Denavit-Hartenberg (DH) parameterization.
    """
    def __init__(self):
        self.num_dof = 7
        # Kinova Gen3 7-DoF DH parameters: [d, a, alpha, offset]
        # Distances in meters
        self.dh_d = np.array([0.2848, -0.0118, 0.4208, -0.0128, 0.3143, 0.0, 0.1674])
        self.dh_a = np.zeros(7)
        self.dh_alpha = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, 0.0])
        
        # Joint limits in radians
        self.q_min = np.array([-np.pi, -2.24, -np.pi, -2.57, -np.pi, -2.09, -np.pi])
        self.q_max = np.array([np.pi, 2.24, np.pi, 2.57, np.pi, 2.09, np.pi])
        self.q_nominal = np.array([0.0, 0.35, 0.0, 1.30, 0.0, 1.10, 0.0])

        # Velocity limits (rad/s)
        self.dq_max = np.array([1.39, 1.39, 1.39, 1.39, 1.22, 1.22, 1.22])

    def _dh_transform(self, q_i: float, d_i: float, a_i: float, alpha_i: float) -> np.ndarray:
        """Constructs 4x4 homogenous transformation matrix from DH parameters."""
        ct, st = np.cos(q_i), np.sin(q_i)
        ca, sa = np.cos(alpha_i), np.sin(alpha_i)

        return np.array([
            [ct, -st * ca,  st * sa, a_i * ct],
            [st,  ct * ca, -ct * sa, a_i * st],
            [0.0,      sa,       ca,      d_i],
            [0.0,     0.0,      0.0,      1.0]
        ], dtype=np.float64)

    def get_all_link_transforms(self, q: np.ndarray) -> List[np.ndarray]:
        """Computes successive homogeneous transforms from base to each link."""
        transforms = [np.eye(4)]
        t_accum = np.eye(4)
        for i in range(self.num_dof):
            t_i = self._dh_transform(q[i], self.dh_d[i], self.dh_a[i], self.dh_alpha[i])
            t_accum = t_accum @ t_i
            transforms.append(t_accum.copy())
        return transforms

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the end-effector pose for joint angles q.
        Returns:
            position: (3,) in meters.
            rot_matrix: (3, 3) orientation matrix.
        """
        t_all = self.get_all_link_transforms(q)
        t_ee = t_all[-1]
        return t_ee[:3, 3], t_ee[:3, :3]

    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Computes the 6x7 geometric Jacobian in the base frame.
        Top 3 rows: linear velocity (v = J_v * dq)
        Bottom 3 rows: angular velocity (omega = J_w * dq)
        """
        transforms = self.get_all_link_transforms(q)
        p_ee = transforms[-1][:3, 3]
        j = np.zeros((6, self.num_dof), dtype=np.float64)

        for i in range(self.num_dof):
            # z-axis of joint i in base frame
            z_i = transforms[i][:3, 2]
            p_i = transforms[i][:3, 3]

            # Linear velocity Jacobian: J_v = z_i x (p_ee - p_i)
            j[:3, i] = np.cross(z_i, p_ee - p_i)
            # Angular velocity Jacobian: J_w = z_i
            j[3:, i] = z_i

        return j

    def solve_differential_ik(
        self,
        q_current: np.ndarray,
        target_twist: np.ndarray,
        damping: float = 0.05,
        nullspace_gain: float = 0.5,
    ) -> np.ndarray:
        """
        Solves Damped Least Squares (DLS) differential IK with nullspace posture projection.
        
        Args:
            q_current: Current 7 joint angles in radians.
            target_twist: Desired end-effector twist [vx, vy, vz, wx, wy, wz] (6,).
            damping: Levenberg-Marquardt damping parameter lambda.
            nullspace_gain: Gain pulling arm toward nominal configuration.

        Returns:
            dq: Desired joint velocities (7,).
        """
        j = self.compute_jacobian(q_current) # (6, 7)

        # Damped Least Squares: J_dls = J^T (J J^T + lambda^2 I)^(-1)
        j_jt = j @ j.T # (6, 6)
        damped_inv = np.linalg.inv(j_jt + (damping ** 2) * np.eye(6))
        j_pseudo = j.T @ damped_inv # (7, 6)

        # Primary task joint velocity
        dq_task = j_pseudo @ target_twist

        # Nullspace projection: (I - J_pseudo @ J) * q_null
        eye_7 = np.eye(self.num_dof)
        null_projector = eye_7 - (j_pseudo @ j)

        # Secondary posture objective: return to comfortable neutral configuration
        q_null = -nullspace_gain * (q_current - self.q_nominal)
        dq_null = null_projector @ q_null

        dq_total = dq_task + dq_null

        # Clamp to joint velocity limits
        dq_clamped = np.clip(dq_total, -self.dq_max, self.dq_max)
        return dq_clamped

    def align_probe_with_surface_normal(
        self,
        current_rot: np.ndarray,
        surface_normal: np.ndarray,
        probe_beam_axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> np.ndarray:
        """
        Calculates corrective angular velocity omega to align the ultrasound probe axis
        with the local anatomical surface normal.
        """
        current_axis = current_rot @ probe_beam_axis
        # Error is the cross product between current beam axis and negative surface normal (pointing in)
        target_axis = -surface_normal / (np.linalg.norm(surface_normal) + 1e-6)
        rotation_error = np.cross(current_axis, target_axis)
        return rotation_error
