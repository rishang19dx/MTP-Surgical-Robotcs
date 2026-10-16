"""
Heuristic Scan Planners for Robotic Ultrasound Reconstruction Benchmarking.
Provides baseline open-loop trajectories:
1. Linear Raster Sweep (standard clinical zigzag pattern)
2. Orbital / Circular Sweep (orbits vertebra with inward acoustic tilt)
Based on SonoGym (Ao et al., 2025) and Li et al. (IJCARS 2023).
"""

from typing import List, Tuple
import numpy as np


class HeuristicScanPlanner:
    """Generates deterministic scanning patterns across the anatomical region of interest."""
    @staticmethod
    def generate_linear_raster_trajectory(
        x_range: Tuple[float, float] = (-35.0, 35.0),
        z_range: Tuple[float, float] = (-25.0, 25.0),
        num_sweep_lines: int = 7,
        points_per_line: int = 15,
    ) -> np.ndarray:
        """
        Generates a 2D zigzag raster scanning pattern on the patient's skin.
        Returns:
            trajectory: (N, 4) array of [x, z, yaw, pitch].
        """
        x_min, x_max = x_range
        z_min, z_max = z_range
        z_coords = np.linspace(z_min, z_max, num_sweep_lines)
        
        trajectory = []
        for i, z in enumerate(z_coords):
            # Alternating direction for zigzag
            if i % 2 == 0:
                xs = np.linspace(x_min, x_max, points_per_line)
            else:
                xs = np.linspace(x_max, x_min, points_per_line)
            
            for x in xs:
                # Linear sweep maintains probe perpendicular to skin (yaw=0, pitch=0)
                trajectory.append([x, z, 0.0, 0.0])

        return np.array(trajectory, dtype=np.float32)

    @staticmethod
    def generate_orbital_trajectory(
        center: Tuple[float, float] = (0.0, 0.0),
        radii: List[float] = [20.0, 32.0],
        points_per_orbit: int = 36,
        inward_pitch_deg: float = 12.0,
    ) -> np.ndarray:
        """
        Generates an elliptical orbital trajectory orbiting the vertebra center,
        with inward probe pitch tilt to capture steep lateral facets.
        Returns:
            trajectory: (N, 4) array of [x, z, yaw, pitch].
        """
        cx, cz = center
        pitch_rad = np.radians(inward_pitch_deg)
        trajectory = []

        for r in radii:
            thetas = np.linspace(0, 2 * np.pi, points_per_orbit, endpoint=False)
            for theta in thetas:
                x = cx + r * np.cos(theta)
                z = cz + (r * 0.75) * np.sin(theta) # Elliptical along spine
                # Yaw aligns along trajectory tangent
                yaw = theta + np.pi / 2.0
                # Inward pitch tilts probe towards anatomical center
                pitch = pitch_rad * np.cos(theta)
                trajectory.append([x, z, yaw, pitch])

        return np.array(trajectory, dtype=np.float32)
