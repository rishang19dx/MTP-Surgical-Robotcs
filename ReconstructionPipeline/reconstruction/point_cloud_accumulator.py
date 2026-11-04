"""
Incremental 3D Bone Surface Point Cloud Accumulator.

Back-projects tracked 2D ultrasound bone contours into 3D and fuses them on a sparse voxel grid.
Each occupied voxel stores the running mean of the points that fell into it, the number of points,
and the number of distinct frames that observed it (observation count), so callers can require
repeated observations (e.g. >= 2 frames) to suppress one-off false detections.

Pixel geometry matches `VolumeSliceExtractor`: lateral x = linspace(-W_mm/2, W_mm/2, n_w) and
axial y = linspace(0, depth_mm, n_h), i.e. spacing = size / (n - 1), x centred at (n_w - 1) / 2.
"""

from typing import Optional, Tuple
import numpy as np
import open3d as o3d

_BITS = 21
_OFF = 1 << (_BITS - 1)
_MASK = (1 << _BITS) - 1


def _voxel_keys(idx: np.ndarray) -> np.ndarray:
    """Pack (N, 3) int voxel indices (|i| < 2^20) into unique int64 keys."""
    i = (idx.astype(np.int64) + _OFF) & _MASK
    return (i[:, 0] << (2 * _BITS)) | (i[:, 1] << _BITS) | i[:, 2]


class PointCloudAccumulator:
    """Maintains and incrementally updates a 3D point cloud reconstructed from tracked 2D ultrasound scans."""

    def __init__(
        self,
        voxel_size_mm: float = 1.5,
        image_size: Tuple[int, int] = (200, 150),
        physical_size_mm: Tuple[float, float] = (75.0, 50.0),
        sor_nb_neighbors: int = 16,
        sor_std_ratio: float = 1.5,
        min_observations: int = 1,
    ):
        """
        Args:
            voxel_size_mm: Fusion voxel size.
            image_size: (height_px, width_px) of the ultrasound frames.
            physical_size_mm: (depth_mm, width_mm) of the imaging plane.
            sor_nb_neighbors / sor_std_ratio: Statistical outlier removal parameters (finalize()).
            min_observations: Default minimum number of frames that must observe a voxel for it to be
                returned by get_points() / counted in get_reconstruction_area().
        """
        self.voxel_size = float(voxel_size_mm)
        self.image_h, self.image_w = image_size
        self.depth_mm, self.width_mm = physical_size_mm
        self.scale_x = self.width_mm / float(max(self.image_w - 1, 1))
        self.scale_y = self.depth_mm / float(max(self.image_h - 1, 1))
        self.sor_neighbors = sor_nb_neighbors
        self.sor_ratio = sor_std_ratio
        self.min_observations = int(min_observations)
        self._normals: Optional[np.ndarray] = None
        self.clear()

    def pixels_to_probe_coords(self, pixel_coords: np.ndarray) -> np.ndarray:
        """(N, 2) [x_px, y_px] -> (N, 3) probe-frame {U} millimetres (z = 0 on the imaging plane)."""
        pixel_coords = np.asarray(pixel_coords, dtype=np.float64)
        if len(pixel_coords) == 0:
            return np.empty((0, 3), dtype=np.float32)
        x_mm = (pixel_coords[:, 0] - (self.image_w - 1) / 2.0) * self.scale_x
        y_mm = pixel_coords[:, 1] * self.scale_y
        return np.stack([x_mm, y_mm, np.zeros_like(x_mm)], axis=-1).astype(np.float32)

    def add_frame(
        self,
        pixel_coords: np.ndarray,
        t_world_probe: np.ndarray,
        filter_outliers: bool = False,
    ) -> int:
        """
        Back-projects 2D bone pixels with the 4x4 pose T_world<-probe and fuses them.
        `filter_outliers` is accepted for backward compatibility; outlier removal is done in finalize()
        so that the accumulated area stays monotone during an episode.

        Returns:
            Number of voxels currently satisfying `min_observations`.
        """
        if len(pixel_coords) == 0:
            return self.num_points()
        pts_probe = self.pixels_to_probe_coords(pixel_coords).astype(np.float64)
        t = np.asarray(t_world_probe, dtype=np.float64)
        pts_world = pts_probe @ t[:3, :3].T + t[:3, 3]

        self.total_points_ingested += len(pts_world)
        self.frame_count += 1

        keys = _voxel_keys(np.floor(pts_world / self.voxel_size))
        all_keys = np.concatenate([self._keys, keys])
        uniq, inv = np.unique(all_keys, return_inverse=True)
        n_old = len(self._keys)
        m = len(uniq)
        new_sum = np.zeros((m, 3))
        new_sum[inv[:n_old]] = self._sum  # old keys are unique -> plain assignment
        np.add.at(new_sum, inv[n_old:], pts_world)
        new_npts = np.bincount(inv, weights=np.concatenate([self._npts, np.ones(len(keys))]), minlength=m)
        new_obs = np.zeros(m)
        new_obs[inv[:n_old]] = self._obs
        new_obs[np.unique(inv[n_old:])] += 1.0  # one observation per frame per voxel

        self._keys, self._sum, self._npts, self._obs = uniq, new_sum, new_npts, new_obs
        self._normals = None
        return self.num_points()

    def _mask(self, min_observations: Optional[int]) -> np.ndarray:
        k = self.min_observations if min_observations is None else int(min_observations)
        return self._obs >= k

    def num_points(self, min_observations: Optional[int] = None) -> int:
        return int(self._mask(min_observations).sum())

    def get_points(self, min_observations: Optional[int] = None) -> np.ndarray:
        """(N, 3) float32 voxel-mean points observed in at least `min_observations` frames."""
        m = self._mask(min_observations)
        if not np.any(m):
            return np.empty((0, 3), dtype=np.float32)
        return (self._sum[m] / self._npts[m, None]).astype(np.float32)

    def get_observation_counts(self, min_observations: Optional[int] = None) -> np.ndarray:
        """Per-point number of frames that observed it (aligned with get_points())."""
        return self._obs[self._mask(min_observations)].astype(np.int32)

    @property
    def pcd(self) -> o3d.geometry.PointCloud:
        """Open3D view of the current points (rebuilt on access)."""
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.get_points().astype(np.float64)))
        if self._normals is not None and len(self._normals) == len(pcd.points):
            pcd.normals = o3d.utility.Vector3dVector(self._normals)
        return pcd

    def finalize(self) -> int:
        """Statistical outlier removal once at episode end (keeps online area monotone)."""
        m = self._mask(None)
        if m.sum() > self.sor_neighbors:
            _, keep_idx = self.pcd.remove_statistical_outlier(nb_neighbors=self.sor_neighbors, std_ratio=self.sor_ratio)
            visible = np.nonzero(m)[0]
            drop = np.ones(len(visible), bool)
            drop[np.asarray(keep_idx, dtype=np.int64)] = False
            keep_all = np.ones(len(self._keys), bool)
            keep_all[visible[drop]] = False
            self._keys, self._sum = self._keys[keep_all], self._sum[keep_all]
            self._npts, self._obs = self._npts[keep_all], self._obs[keep_all]
            self._normals = None
        return self.num_points()

    def estimate_surface_normals(self, k_neighbors: int = 15):
        """Estimates normals for the current points (available via get_normals())."""
        pcd = self.pcd
        if len(pcd.points) >= k_neighbors:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
            self._normals = np.asarray(pcd.normals).copy()

    def get_normals(self) -> Optional[np.ndarray]:
        if self._normals is not None and len(self._normals) == self.num_points():
            return self._normals.astype(np.float32)
        return None

    def get_reconstruction_area(self, min_observations: Optional[int] = None) -> float:
        """Approximate reconstructed surface area: occupied voxels * voxel_size^2 (mm^2)."""
        return float(self.num_points(min_observations)) * (self.voxel_size ** 2)

    def clear(self):
        """Resets the accumulated point cloud."""
        self._keys = np.empty(0, dtype=np.int64)
        self._sum = np.empty((0, 3))
        self._npts = np.empty(0)
        self._obs = np.empty(0)
        self._normals = None
        self.frame_count = 0
        self.total_points_ingested = 0
