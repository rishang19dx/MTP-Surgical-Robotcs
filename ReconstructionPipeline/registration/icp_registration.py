"""
Intraoperative Point-to-Plane ICP Registration and Quantitative Accuracy Evaluation.
Registers intraoperatively reconstructed ultrasound bone point clouds against pre-operative CT models.
Computes Chamfer Distance, Target Registration Error (TRE), and Surface Coverage Ratio.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import open3d as o3d


class IntraoperativeRegistration:
    """
    Handles rigid registration between reconstructed ultrasound point clouds
    and pre-operative CT segmented ground-truth anatomy.
    """
    def __init__(
        self,
        max_correspondence_distance_mm: float = 5.0,
        icp_convergence_threshold: float = 1e-6,
        max_iterations: int = 100,
    ):
        self.max_dist = max_correspondence_distance_mm
        self.threshold = icp_convergence_threshold
        self.max_iter = max_iterations

    def register_point_to_plane(
        self,
        source_us_points: np.ndarray,
        target_ct_points: np.ndarray,
        target_ct_normals: Optional[np.ndarray] = None,
        initial_transform: Optional[np.ndarray] = None,
    ) -> Dict[str, any]:
        """
        Executes Point-to-Plane ICP registration from intraoperative US (source) to pre-op CT (target).

        Args:
            source_us_points: (N, 3) reconstructed ultrasound surface points.
            target_ct_points: (M, 3) ground-truth CT surface points.
            target_ct_normals: Optional (M, 3) normals for the CT mesh.
            initial_transform: Optional 4x4 initial alignment guess.

        Returns:
            Dictionary containing:
                'transformation': 4x4 optimal transformation matrix
                'inlier_rmse': Root mean square error of matched correspondences
                'fitness': Inlier correspondence ratio
                'chamfer_distance': Symmetrical Chamfer Distance (mm)
                'coverage_ratio': Percentage of ground truth covered within tolerance
        """
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_us_points)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_ct_points)

        if target_ct_normals is not None:
            target_pcd.normals = o3d.utility.Vector3dVector(target_ct_normals)
        else:
            target_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
            )

        if initial_transform is None:
            # Initial centroid alignment
            source_center = np.mean(source_us_points, axis=0)
            target_center = np.mean(target_ct_points, axis=0)
            t_init = np.eye(4)
            t_init[:3, 3] = target_center - source_center
        else:
            t_init = initial_transform

        # Run Point-to-Plane ICP
        reg_result = o3d.pipelines.registration.registration_icp(
            source=source_pcd,
            target=target_pcd,
            max_correspondence_distance=self.max_dist,
            init=t_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iter,
                relative_fitness=self.threshold,
                relative_rmse=self.threshold
            )
        )

        # Apply transformation to source points
        t_opt = reg_result.transformation
        pts_aligned = (t_opt[:3, :3] @ source_us_points.T + t_opt[:3, 3:4]).T

        # Compute Chamfer Distance and Coverage Ratio
        chamfer_dist, coverage = self.compute_metrics(pts_aligned, target_ct_points)

        return {
            "transformation": t_opt,
            "inlier_rmse": reg_result.inlier_rmse,
            "fitness": reg_result.fitness,
            "chamfer_distance": chamfer_dist,
            "coverage_ratio": coverage,
            "aligned_points": pts_aligned,
        }

    def compute_metrics(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        coverage_tolerance_mm: float = 3.0,
    ) -> Tuple[float, float]:
        """
        Computes Chamfer Distance and Surface Coverage Ratio using KD-trees.

        Returns:
            chamfer_dist: Mean bidirectional nearest-neighbor distance (mm).
            coverage_ratio: Proportion of target points having a reconstructed point within tolerance.
        """
        if len(source_points) == 0 or len(target_points) == 0:
            return float("inf"), 0.0

        target_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points)))
        source_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points)))

        # 1. Source -> Target distances
        dists_s2t = []
        for p in source_points:
            _, idx, d2 = target_tree.search_knn_vector_3d(p, 1)
            dists_s2t.append(np.sqrt(d2[0]))

        # 2. Target -> Source distances
        dists_t2s = []
        covered_count = 0
        for p in target_points:
            _, idx, d2 = source_tree.search_knn_vector_3d(p, 1)
            dist = np.sqrt(d2[0])
            dists_t2s.append(dist)
            if dist <= coverage_tolerance_mm:
                covered_count += 1

        chamfer_distance = (np.mean(dists_s2t) + np.mean(dists_t2s)) / 2.0
        coverage_ratio = float(covered_count) / float(len(target_points))

        return float(chamfer_distance), float(coverage_ratio)
