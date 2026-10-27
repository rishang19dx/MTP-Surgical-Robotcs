"""
GPU-Accelerated 3D Volume Slice Extractor for Robotic Ultrasound Simulation.
Extracts 2D image slices from 3D CT and segmentation volumes using torch.nn.functional.grid_sample.
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np

from ReconstructionPipeline.simulation.kelvinlet_deformation import RegularizedKelvinletDeformation


class VolumeSliceExtractor:
    """
    Extracts 2D slice planes from 3D medical volumes given probe poses.
    Supports both rigid slicing and dynamic soft-tissue deformation via
    backward Eulerian coordinate warping coupled to Regularized Kelvinlets.
    
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
        deformation_solver: Optional[RegularizedKelvinletDeformation] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            volume: 3D tensor of shape (D, H, W) or (1, 1, D, H, W) representing CT or labels.
            spacing: Physical voxel size (sx, sy, sz) in mm.
            origin: Physical coordinates (ox, oy, oz) of voxel (0, 0, 0) in mm.
            image_size: (height_px, width_px) of the 2D ultrasound image plane.
            physical_size_mm: (depth_mm, width_mm) of the ultrasound scanning plane.
            deformation_solver: Optional RegularizedKelvinletDeformation solver for soft-tissue dynamics.
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

        # Deformation engine: instantiate default Kelvinlet solver if not provided
        self.deformation_solver = deformation_solver or RegularizedKelvinletDeformation(
            shear_modulus_kpa=12.0,
            poissons_ratio=0.48,
            probe_radius_mm=15.0,
            device=self.device,
        )

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

    def extract_slices(
        self,
        probe_poses: Union[torch.Tensor, np.ndarray],
        interpolation: str = "bilinear",
        contact_force_n: float = 0.0,
        contact_normal: Optional[Union[torch.Tensor, np.ndarray]] = None,
        bone_depth_mm: Optional[float] = 30.0,
    ) -> torch.Tensor:
        """
        Extracts 2D slices corresponding to one or more probe poses,
        accounting for soft-tissue deformation via backward Eulerian coordinate warping.

        Args:
            probe_poses: 4x4 matrix or batch of matrices (B, 4, 4) representing
                         transform from probe frame {U} to patient volume frame {P}.
                         P_point = T_patient_probe @ U_point.
            interpolation: 'bilinear' for continuous CT intensities, 'nearest' for labels.
            contact_force_n: Contact normal force (N). If > 0, tissue compresses along beam axis.
            contact_normal: Optional 3D contact normal vector. Defaults to probe beam axis (+Y_probe).
            bone_depth_mm: Depth of rigid bone cortex beneath transducer face where deformation decays.

        Returns:
            Extracted 2D slices tensor:
                If B == 1: shape (image_h, image_w)
                If B > 1: shape (B, image_h, image_w)
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

        # Local probe grid: (H, W, 4) -> flatten to (H*W, 4)
        pts_local = self.local_grid.view(-1, 4).t() # (4, H*W)

        # Batch transform: (B, 4, 4) @ (4, H*W) -> (B, 4, H*W)
        pts_world = torch.bmm(poses, pts_local.unsqueeze(0).expand(batch_size, -1, -1)) # (B, 4, H*W)
        pts_xyz = pts_world[:, :3, :].transpose(1, 2) # (B, H*W, 3) (deformed coordinates)

        # --- Backward Eulerian Soft-Tissue Deformation ---
        if contact_force_n > 0.05 and self.deformation_solver is not None:
            # Contact point is probe origin (transducer face center)
            contact_pt = poses[:, :3, 3] # (B, 3)

            # Normal vector: beam direction is column 1 (+Y_probe in world frame)
            if contact_normal is not None:
                if isinstance(contact_normal, np.ndarray):
                    norm_vec = torch.from_numpy(contact_normal).float().to(self.device)
                else:
                    norm_vec = contact_normal.float().to(self.device)
                if norm_vec.ndim == 1:
                    norm_vec = norm_vec.unsqueeze(0).expand(batch_size, -1)
            else:
                # Default normal along beam direction (+Y of probe frame)
                norm_vec = poses[:, :3, 1] # (B, 3)
                norm_vec = norm_vec / (torch.norm(norm_vec, dim=-1, keepdim=True) + 1e-6)

            force_vec = norm_vec * float(contact_force_n) # (B, 3)

            # Compute displacement field u(x) and probe indentation u_0 for each batch
            u_displacements = []
            for b in range(batch_size):
                u_b = self.deformation_solver.compute_displacement(
                    query_points=pts_xyz[b],
                    contact_point=contact_pt[b],
                    force_vector=force_vec[b],
                    bone_depth_mm=bone_depth_mm,
                )
                # Probe face indentation at contact point
                u_0 = self.deformation_solver.compute_displacement(
                    query_points=contact_pt[b:b+1],
                    contact_point=contact_pt[b],
                    force_vector=force_vec[b],
                    bone_depth_mm=bone_depth_mm,
                ) # (1, 3)
                # Relative displacement in probe frame: u_rel(x) = u(x) - u_0
                # Backward warping: X_ref = x - u_rel(x) = x - (u_b - u_0)
                u_displacements.append(u_b - u_0)
            u_tensor = torch.stack(u_displacements, dim=0) # (B, H*W, 3)

            # Backward coordinate warping in probe coordinate frame
            pts_sample_xyz = pts_xyz - u_tensor
        else:
            pts_sample_xyz = pts_xyz

        # Convert physical coordinates (mm) to voxel indices:
        # voxel = (world - origin) / spacing
        voxel_coords = (pts_sample_xyz - self.origin.unsqueeze(0).unsqueeze(0)) / self.spacing.unsqueeze(0).unsqueeze(0)

        # Volume shape is (1, 1, D, H_vol, W_vol)
        # grid_sample expects normalized coordinates [-1, 1] in order (x, y, z)
        _, _, d_vol, h_vol, w_vol = self.volume.shape
        grid_w = 2.0 * (voxel_coords[:, :, 0] / (w_vol - 1.0)) - 1.0
        grid_h = 2.0 * (voxel_coords[:, :, 1] / (h_vol - 1.0)) - 1.0
        grid_d = 2.0 * (voxel_coords[:, :, 2] / (d_vol - 1.0)) - 1.0

        # Stack into (B, 1, H, W, 3) for 5D grid_sample
        sampling_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)
        sampling_grid = sampling_grid.view(batch_size, 1, self.image_h, self.image_w, 3)

        # Expand volume to batch_size if batch_size > 1
        vol_batch = self.volume.expand(batch_size, -1, -1, -1, -1)

        # Sample 3D volume
        sampled = F.grid_sample(
            vol_batch,
            sampling_grid,
            mode=interpolation,
            padding_mode="zeros",
            align_corners=True
        )

        slices = sampled.squeeze(1).squeeze(1) # (B, H_out, W_out)
        if is_single:
            slices = slices.squeeze(0)

        return slices
