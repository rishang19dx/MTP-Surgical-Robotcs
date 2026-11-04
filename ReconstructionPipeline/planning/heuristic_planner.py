"""
Heuristic open-loop scan planners (baselines).

Frame/pose convention (see env_reconstruction): waypoints are [c1, c2, yaw, pitch] with c1 = lateral X,
c2 = cranio-caudal Z (mm, on the dorsal skin), yaw = rotation about the beam, pitch = cranio-caudal tilt.
The dorsal image plane is TRANSVERSE (probe x = world X), so a useful raster moves the probe
ELEVATIONALLY (along Z) between frames; sweeping along X would only re-image the same plane.

1. Elevational raster: cranio-caudal sweeps at a few lateral offsets (standard clinical transverse sweep).
2. Orbital: elliptical orbit around the target with the beam tilted cranio-caudally toward the centre.
"""

from typing import List, Tuple, Optional
import numpy as np


class HeuristicScanPlanner:
    @staticmethod
    def generate_linear_raster_trajectory(
        x_range: Tuple[float, float] = (-20.0, 20.0),
        z_range: Tuple[float, float] = (-25.0, 25.0),
        num_sweep_lines: int = 3,
        points_per_line: int = 20,
    ) -> np.ndarray:
        """
        Elevational zigzag: `num_sweep_lines` lateral offsets spread over `x_range`; along each line the probe
        moves cranio-caudally over `z_range` with `points_per_line` frames (alternating direction).
        Returns (num_sweep_lines * points_per_line, 4) waypoints [x, z, yaw=0, pitch=0].
        """
        xs = np.linspace(x_range[0], x_range[1], num_sweep_lines) if num_sweep_lines > 1 else np.array([np.mean(x_range)])
        traj = []
        for i, x in enumerate(xs):
            zs = np.linspace(z_range[0], z_range[1], points_per_line)
            if i % 2 == 1:
                zs = zs[::-1]
            traj.extend([[x, z, 0.0, 0.0] for z in zs])
        return np.array(traj, dtype=np.float32)

    @staticmethod
    def generate_spine_raster_trajectory(
        x_range: Tuple[float, float] = (-18.0, 18.0),
        z_range: Tuple[float, float] = (-55.0, 55.0),
        step_distance_mm: float = 0.33,
        num_sweep_lines: int = 5,
    ) -> np.ndarray:
        """Dense elevational raster with a fixed cranio-caudal step (e.g. 0.33 mm at 30 Hz = 10 mm/s)."""
        n = max(10, int(np.ceil((z_range[1] - z_range[0]) / step_distance_mm)) + 1)
        return HeuristicScanPlanner.generate_linear_raster_trajectory(x_range, z_range, num_sweep_lines, n)

    @staticmethod
    def generate_orbital_trajectory(
        center: Tuple[float, float] = (0.0, 0.0),
        radii: List[float] = (20.0, 32.0),
        points_per_orbit: Optional[int] = None,
        step_distance_mm: float = 0.33,
        inward_pitch_deg: float = 12.0,
        yaw_amplitude_deg: float = 20.0,
    ) -> np.ndarray:
        """
        Elliptical orbits (semi-axes r laterally, 0.75 r cranio-caudally) around `center` = (x, z).
        pitch leans the beam cranio-caudally toward the centre (positive pitch leans the beam toward -Z);
        yaw rotates the image plane about the beam (|yaw| <= yaw_amplitude_deg) to vary insonation of the facets.
        """
        cx, cz = center
        tilt = np.radians(inward_pitch_deg)
        yaw_amp = np.radians(yaw_amplitude_deg)
        traj = []
        for r in radii:
            n = points_per_orbit if points_per_orbit is not None else max(36, int(np.ceil(2 * np.pi * r / step_distance_mm)))
            for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
                traj.append([cx + r * np.cos(th), cz + 0.75 * r * np.sin(th), yaw_amp * np.cos(th), tilt * np.sin(th)])
        return np.array(traj, dtype=np.float32)
