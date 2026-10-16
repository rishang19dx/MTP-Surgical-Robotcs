"""
Incremental 3D Bone Surface Point Cloud Accumulator and Spatial Aggregator.
Fuses tracked 2D ultrasound bone contours into a unified, downsampled 3D surface model
with normal estimation and statistical outlier filtering.
"""

from typing import Optional, Tuple, Dict
import numpy as np
import open3d as o3d
import torch


class PointCloudAccumulator:
    """
    Maintains and incrementally updates a 3D point cloud reconstructed from tracked 2D ultrasound scans.
    """
    def __init__(
        self,
        voxel_size_mm: float = 1.5,
        image_size: Tuple[int, int] = (200, 150),
        physical_size_mm: Tuple[float, float] = (75.0, 50.0),
        sor_nb_neighbors: int = 16,
        sor_std_ratio: float = 1.5,
    ):
        """
        Args:
            voxel_size_mm: Spatial resolution for voxel grid downsampling.
            image_size: (height_px, width_px) of the ultrasound frames.
            physical_size_mm: (depth_mm, width_mm) of the ultrasound scanning plane.
            sor_nb_neighbors: Number of neighbors for Statistical Outlier Removal.
            sor_std_ratio: Standard deviation multiplier for outlier filtering.
        """
        self.voxel_size = voxel_size_mm
        self.image_h, self.image_w = image_size
        self.depth_mm, self.width_mm = physical_size_mm
        
        self.scale_x = self.width_mm / float(self.image_w)
        self.scale_y = self.depth_mm / float(self.image_h)
        self.sor_neighbors = sor_nb_neighbors
        self.sor_ratio = sor_std_ratio

        # Internal Open3D PointCloud
        self.pcd = o3d.geometry.PointCloud()
        self.frame_count = 0
        self.total_points_ingested = 0

    def pixels_to_probe_coords(self, pixel_coords: np.ndarray) -> np.ndarray:
        """
        Converts 2D pixel coordinates (x_px, y_px) to 3D millimetric probe coordinates [x, y, z] in {U}.
        
        Args:
            pixel_coords: (N, 2) array of [x_px, y_px].
        Returns:
            probe_coords: (N, 3) array in probe frame {U}.
        """
        if len(pixel_coords) == 0:
            return np.empty((0, 3), dtype=np.float32)

        x_px = pixel_coords[:, 0]
        y_px = pixel_coords[:, 1]

        # Centered along lateral x, downwards along axial y, z is elevational thickness (0)
        x_mm = (x_px - self.image_w / 2.0) * self.scale_x
        y_mm = y_px * self.scale_y
        z_mm = np.zeros_like(x_mm)

        return np.stack([x_mm, y_mm, z_mm], axis=-1).astype(np.float32)

    def add_frame(
        self,
        pixel_coords: np.ndarray,
        t_world_probe: np.ndarray,
        filter_outliers: bool = True,
    ) -> int:
        """
        Back-projects a set of 2D bone boundary pixels into the 3D world frame
        and updates the accumulated point cloud.

        Args:
            pixel_coords: (N, 2) array of detected bone pixel positions.
            t_world_probe: 4x4 transformation matrix mapping probe frame {U} to world frame {W}.
            filter_outliers: Whether to run statistical outlier removal.

        Returns:
            current_total_points: Number of points currently in the accumulated model.
        """
        if len(pixel_coords) == 0:
            return len(self.pcd.points)

        # 1. Pixel -> Probe {U} frame
        pts_probe = self.pixels_to_probe_coords(pixel_coords) # (N, 3)

        # 2. Probe {U} -> World {W} frame
        # Homogeneous transform
        pts_homo = np.hstack([pts_probe, np.ones((len(pts_probe), 1), dtype=np.float32)]) # (N, 4)
        pts_world = (t_world_probe @ pts_homo.T).T[:, :3] # (N, 3)

        self.total_points_ingested += len(pts_world)
        self.frame_count += 1

        # 3. Concatenate with existing points
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(pts_world)

        self.pcd += new_pcd

        # 4. Voxel Grid Downsampling to maintain constant spatial density and prevent bloat
        self.pcd = self.pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # 5. Periodic Statistical Outlier Removal (every 5 frames)
        if filter_outliers and self.frame_count % 5 == 0 and len(self.pcd.points) > self.sor_neighbors:
            filtered_pcd, _ = self.pcd.remove_statistical_outlier(
                nb_neighbors=self.sor_neighbors,
                std_ratio=self.sor_ratio
            )
            self.pcd = filtered_pcd

        return len(self.pcd.points)

    def estimate_surface_normals(self, k_neighbors: int = 15):
        """Computes surface normals for the accumulated point cloud."""
        if len(self.pcd.points) >= k_neighbors:
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
            )

    def get_points(self) -> np.ndarray:
        """Returns the accumulated point cloud as an (N, 3) numpy float32 array."""
        return np.asarray(self.pcd.points, dtype=np.float32)

    def get_normals(self) -> Optional[np.ndarray]:
        """Returns the estimated normals as an (N, 3) numpy array if available."""
        if self.pcd.has_normals():
            return np.asarray(self.pcd.normals, dtype=np.float32)
        return None

    def get_reconstruction_area(self) -> float:
        """
        Approximates the total surface area of the current reconstruction |M_{0:t}|.
        Computed as point_count * (voxel_size)^2.
        """
        return float(len(self.pcd.points)) * (self.voxel_size ** 2)

    def clear(self):
        """Resets the accumulated point cloud."""
        self.pcd.clear()
        self.frame_count = 0
        self.total_points_ingested = 0
