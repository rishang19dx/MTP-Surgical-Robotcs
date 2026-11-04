"""
GPU-Accelerated 3D Volume Slice Extractor for Robotic Ultrasound Simulation.
Extracts 2D image slices from 3D CT and segmentation volumes using torch.nn.functional.grid_sample.
"""

from typing import Any, Callable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import numpy as np



class VolumeSliceExtractor:
    """
    Extracts 2D slice planes from 3D medical volumes given probe poses.
    Supports rigid slicing and soft-tissue deformation via backward Eulerian warping with a
    `TissueState` (contract C1) from any `TissueModel` backend.
    
    Transforms the probe imaging plane grid (in probe millimeters) into the 3D volume
    voxel coordinate frame, normalizes them to [-1, 1], and samples via grid_sample.
    """
    def __init__(
        self,
        volume: Union[torch.Tensor, np.ndarray],
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        image_size: Tuple[int, int] = (200, 150),
        physical_size_mm: Tuple[float, float] = (75.0, 50.0),
        deformation_solver: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            volume: 3D tensor of shape (D, H, W) or (1, 1, D, H, W) representing CT or labels.
            spacing: Physical voxel size (sx, sy, sz) in mm.
            origin: Physical coordinates (ox, oy, oz) of voxel (0, 0, 0) in mm.
            image_size: (height_px, width_px) of the 2D ultrasound image plane.
            physical_size_mm: (depth_mm, width_mm) of the ultrasound scanning plane.
            deformation_solver: Deprecated, ignored.
            device: torch.device ('cuda' or 'cpu').
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if isinstance(volume, np.ndarray):
            self.volume = torch.from_numpy(volume).float().to(self.device)
        else:
            self.volume = volume.float().to(self.device)

        if self.volume.ndim == 3:
            # Reshape (D, H, W) -> (1, 1, D, H, W) for grid_sample (N, C, D, H, W)
            self.volume = self.volume.unsqueeze(0).unsqueeze(0)
        elif self.volume.ndim == 4:
            self.volume = self.volume.unsqueeze(1)

        self.spacing = torch.tensor(spacing, dtype=torch.float32, device=self.device)
        self.origin = torch.tensor(origin, dtype=torch.float32, device=self.device)
        self.image_h, self.image_w = image_size
        self.depth_mm, self.width_mm = physical_size_mm

        # Deprecated: kept only for constructor backward compatibility (unused).
        self.deformation_solver = deformation_solver
        self._legacy_model = None

        # Precompute the local 2D pixel coordinates in probe frame {U} (in mm)
        # Probe convention:
        #   x_probe: lateral direction across array [-width_mm/2, width_mm/2]
        #   y_probe: axial direction down acoustic beam [0, depth_mm]
        #   z_probe: elevational normal (0 on imaging plane)
        self.local_grid = self._create_local_probe_grid()

    def _create_local_probe_grid(self) -> torch.Tensor:
        """
        Generates homogeneous coordinates (X, Y, Z, 1) for every pixel in the US probe frame.
        Shape: (image_h, image_w, 4)
        """
        xs = torch.linspace(-self.width_mm / 2.0, self.width_mm / 2.0, self.image_w, device=self.device)
        ys = torch.linspace(0.0, self.depth_mm, self.image_h, device=self.device)
        
        # Meshgrid: shape (image_h, image_w)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_z = torch.zeros_like(grid_x)
        grid_ones = torch.ones_like(grid_x)

        # Stack into (image_h, image_w, 4)
        local_coords = torch.stack([grid_x, grid_y, grid_z, grid_ones], dim=-1)
        return local_coords

    def _legacy_tissue_model(self):
        """Kelvinlet model built from this extractor's own volume (legacy force-driven API)."""
        if self._legacy_model is None:
            from ReconstructionPipeline.simulation.tissue_model import KelvinletTissueModel
            vol = self.volume[0, 0].detach().cpu().numpy()
            is_label = float(vol.max()) <= 16.0 and np.allclose(vol, np.round(vol))
            bone = (vol == 1) if is_label else (vol > 300.0)
            self._legacy_model = KelvinletTissueModel(
                bone, tuple(self.spacing.tolist()), tuple(self.origin.tolist()), device=self.device
            )
        return self._legacy_model

    def extract_slices(
        self,
        probe_poses: Union[torch.Tensor, np.ndarray],
        interpolation: str = "bilinear",
        tissue_state: Optional[Any] = None,
        warp_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        contact_force_n: float = 0.0,
        contact_normal: Optional[Union[torch.Tensor, np.ndarray]] = None,
        bone_depth_mm: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Extracts 2D slices for one or more probe poses with backward Eulerian warping:
        every image pixel at deformed position x is sampled from the CT at X = warp_to_reference(x).

        Args:
            probe_poses: 4x4 or (B,4,4) T_P<-U of the IMAGE plane (probe face).  When a
                `tissue_state` / `warp_fn` is given this must be the indented face pose
                (`TissueState.probe_face_pose`).  On the legacy `contact_force_n` path it is the pose
                at the undeformed skin and the face indentation is computed internally.
            interpolation: 'bilinear' (CT) or 'nearest' (labels).
            tissue_state: `TissueState` (contract C1); its `warp_to_reference` is applied.
            warp_fn: alternative callable (N,3) deformed -> (N,3) reference points (mm, frame P).
            contact_force_n: LEGACY. If > 0 and no state/warp is given, a `KelvinletTissueModel`
                built from this extractor's volume finds the indentation for this force and warps.
            contact_normal, bone_depth_mm: deprecated and ignored (beam = +Y_probe; the bone depth
                is ray-marched from the volume).

        Returns:
            (H, W) for a single pose, else (B, H, W).
        """
        if isinstance(probe_poses, np.ndarray):
            poses = torch.from_numpy(probe_poses).float().to(self.device)
        else:
            poses = probe_poses.float().to(self.device)

        is_single = False
        if poses.ndim == 2:
            poses = poses.unsqueeze(0)
            is_single = True
        batch_size = poses.shape[0]

        warps: List[Optional[Callable]] = [None] * batch_size
        if tissue_state is not None:
            warps = [tissue_state.warp_to_reference] * batch_size
        elif warp_fn is not None:
            warps = [warp_fn] * batch_size
        elif contact_force_n > 0.05:
            model = self._legacy_tissue_model()
            new_poses = []
            for bi in range(batch_size):
                T = poses[bi].detach().cpu().numpy().astype(np.float64)
                d = model.indentation_for_force(T, float(contact_force_n))
                state = model.solve(T, d)
                warps[bi] = state.warp_to_reference
                new_poses.append(torch.from_numpy(state.probe_face_pose(T)).float())
            poses = torch.stack(new_poses).to(self.device)

        # Local probe grid (H*W,4) -> world (B,H*W,3) deformed coordinates
        pts_local = self.local_grid.view(-1, 4).t()
        pts_world = torch.bmm(poses, pts_local.unsqueeze(0).expand(batch_size, -1, -1))
        pts_xyz = pts_world[:, :3, :].transpose(1, 2)

        if any(w is not None for w in warps):
            pts_xyz = torch.stack(
                [w(pts_xyz[bi]) if w is not None else pts_xyz[bi] for bi, w in enumerate(warps)], dim=0
            )

        voxel_coords = (pts_xyz - self.origin.view(1, 1, 3)) / self.spacing.view(1, 1, 3)
        _, _, d_vol, h_vol, w_vol = self.volume.shape
        grid_w = 2.0 * (voxel_coords[:, :, 0] / (w_vol - 1.0)) - 1.0
        grid_h = 2.0 * (voxel_coords[:, :, 1] / (h_vol - 1.0)) - 1.0
        grid_d = 2.0 * (voxel_coords[:, :, 2] / (d_vol - 1.0)) - 1.0
        sampling_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).view(batch_size, 1, self.image_h, self.image_w, 3)

        vol_batch = self.volume.expand(batch_size, -1, -1, -1, -1)
        sampled = F.grid_sample(vol_batch, sampling_grid, mode=interpolation, padding_mode="zeros", align_corners=True)
        slices = sampled.squeeze(1).squeeze(1)
        if is_single:
            slices = slices.squeeze(0)
        return slices
