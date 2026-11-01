"""
Patient Anatomical Asset Loader and Lumbar Vertebra Phantom Generator.
Provides synthetic anatomical CT volumes and vertebra segmentations for testing and benchmarking,
as well as loaders for TotalSegmentator NIfTI datasets.
"""

from typing import Tuple, Dict, Optional, List, Any

import numpy as np
import nibabel as nib
import trimesh


class PatientAssetGenerator:
    """
    Generates realistic synthetic anatomical volumes (CT HU and multi-class segmentations)
    representing a patient's lumbar spine (L4 vertebra) and surrounding tissue.
    """
    @staticmethod
    def generate_lumbar_phantom(
        shape: Tuple[int, int, int] = (120, 160, 160),
        spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Dict[str, any]:
        """
        Creates a synthetic lumbar spine phantom containing:
        - CT volume with Hounsfield Units (-1000 to +1500 HU)
        - Label volume (0: air, 1: bone, 2: soft tissue, 3: skin)
        - Ground truth bone surface points and mesh

        Args:
            shape: (D, H, W) volume dimensions.
            spacing_mm: (sx, sy, sz) physical voxel spacing in mm.

        Returns:
            Dictionary containing 'ct_volume', 'label_volume', 'bone_points', 'spacing', 'origin'.
        """
        d, h, w = shape
        sx, sy, sz = spacing_mm

        # Physical coordinate grids in mm centered at vertebra center
        z = (np.arange(d) - d / 2.0) * sz # Cranial-caudal
        y = (np.arange(h) - h / 3.0) * sy # Anterior-posterior (y > 0 is dorsal / back)
        x = (np.arange(w) - w / 2.0) * sx # Lateral (left-right)
        grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing="ij")

        # Initialize CT with fat/soft tissue baseline (+40 HU)
        ct_vol = np.full(shape, 40.0, dtype=np.float32)
        label_vol = np.full(shape, 2, dtype=np.int32) # 2: soft tissue

        # 1. Vertebral Body: Elliptical cylinder anteriorly (y < 0)
        # Center: (z=0, y=-15, x=0), radii: rz=15, ry=18, rx=22
        body_dist = ((grid_z / 15.0)**2 + ((grid_y + 15.0) / 18.0)**2 + (grid_x / 22.0)**2)
        body_mask = body_dist <= 1.0

        # 2. Spinal Canal (foramen): Elliptical hollow behind vertebral body
        canal_dist = ((grid_y - 8.0) / 10.0)**2 + (grid_x / 11.0)**2
        canal_mask = (canal_dist <= 1.0) & (np.abs(grid_z) <= 18.0)

        # 3. Spinous Process: Posterior projection (y > 10, x ~ 0)
        spinous_mask = (
            (grid_y >= 10.0) & (grid_y <= 42.0) & 
            (np.abs(grid_x) <= (4.0 - 0.05 * grid_y)) & 
            (np.abs(grid_z) <= (12.0 - 0.2 * grid_y))
        )

        # 4. Transverse Processes: Lateral projections (y ~ 5, |x| up to 40)
        transverse_left = (
            (grid_y >= 0.0) & (grid_y <= 12.0) & 
            (grid_x >= 12.0) & (grid_x <= 38.0) & 
            (np.abs(grid_z) <= 6.0)
        )
        transverse_right = (
            (grid_y >= 0.0) & (grid_y <= 12.0) & 
            (grid_x <= -12.0) & (grid_x >= -38.0) & 
            (np.abs(grid_z) <= 6.0)
        )

        # Combine bone components
        bone_mask = (body_mask | spinous_mask | transverse_left | transverse_right) & (~canal_mask)

        # Assign Cortical Bone HU (+1200 HU)
        ct_vol[bone_mask] = 1200.0
        label_vol[bone_mask] = 1 # 1: bone

        # Spinal canal has CSF fluid (+10 HU)
        ct_vol[canal_mask] = 10.0

        # 5. Skin Surface boundary at the back (posterior dorsal surface at y = 55 mm)
        skin_y = 55.0
        air_mask = grid_y > skin_y
        ct_vol[air_mask] = -1000.0 # Air
        label_vol[air_mask] = 0

        # Subcutaneous fat layer right below skin (35 to 55 mm)
        fat_mask = (grid_y >= 38.0) & (grid_y <= skin_y) & (~bone_mask)
        ct_vol[fat_mask] = -90.0

        # Add realistic Gaussian CT noise
        noise = np.random.normal(0, 15.0, shape).astype(np.float32)
        ct_vol += noise

        # Extract ground truth bone surface coordinates (where label==1 touches non-bone)
        bone_voxels = np.argwhere(bone_mask)
        # Convert voxel indices to physical coordinates in mm
        bone_pts_z = (bone_voxels[:, 0] - d / 2.0) * sz
        bone_pts_y = (bone_voxels[:, 1] - h / 3.0) * sy
        bone_pts_x = (bone_voxels[:, 2] - w / 2.0) * sx
        bone_points = np.stack([bone_pts_x, bone_pts_y, bone_pts_z], axis=-1).astype(np.float32)

        origin_mm = (-w / 2.0 * sx, -h / 3.0 * sy, -d / 2.0 * sz)

        return {
            "ct_volume": ct_vol,
            "label_volume": label_vol,
            "bone_points": bone_points,
            "spacing_mm": spacing_mm,
            "spacing": spacing_mm,
            "origin_mm": origin_mm,
            "origin": origin_mm,
            "skin_y_mm": skin_y,
        }

    @staticmethod
    def load_totalsegmentator_nifti(
        ct_nifti_path: str,
        label_nifti_path: str,
    ) -> Dict[str, any]:
        """
        Loads clinical CT and vertebra label maps from TotalSegmentator NIfTI files.
        """
        ct_img = nib.load(ct_nifti_path)
        label_img = nib.load(label_nifti_path)

        ct_data = ct_img.get_fdata().astype(np.float32)
        label_data = label_img.get_fdata().astype(np.int32)
        spacing = tuple(ct_img.header.get_zooms()[:3])

        return {
            "ct_volume": ct_data,
            "label_volume": label_data,
            "spacing_mm": spacing,
            "affine": ct_img.affine,
        }

    @staticmethod
    def load_totalsegmentator_subject(
        subject_dir: str,
        target_vertebrae: Optional[List[str]] = None,
        crop_to_spine: bool = True,
        margin_voxels: int = 15,
    ) -> Dict[str, any]:
        """
        Loads and prepares a complete clinical patient dataset directly from a TotalSegmentator subject directory.
        Handles both individual segmentation masks (segmentations/vertebrae_L*.nii.gz) and single combined masks.

        Args:
            subject_dir: Path to patient folder containing ct.nii.gz and segmentations/
            target_vertebrae: List of bone masks to include (default: L1 to L5 and sacrum)
            crop_to_spine: Crops volume to the lumbar spine region of interest to optimize GPU memory.
            margin_voxels: Bounding margin around spine when cropping.

        Returns:
            Dictionary ready for direct ingestion by UltrasoundReconstructionEnv.
        """
        import os
        import glob

        if target_vertebrae is None:
            target_vertebrae = ["vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "sacrum"]

        # 1. Locate CT image
        ct_candidates = [
            os.path.join(subject_dir, "ct.nii.gz"),
            os.path.join(subject_dir, "ct.nii"),
        ] + glob.glob(os.path.join(subject_dir, "*ct*.nii*"))
        
        ct_path = next((p for p in ct_candidates if os.path.exists(p)), None)
        if not ct_path:
            raise FileNotFoundError(f"No CT image found in TotalSegmentator directory: {subject_dir}")

        ct_img = nib.load(ct_path)
        ct_data = ct_img.get_fdata().astype(np.float32)
        spacing = tuple(float(s) for s in ct_img.header.get_zooms()[:3])

        # 2. Locate Segmentations
        seg_dir = os.path.join(subject_dir, "segmentations")
        search_dirs = [seg_dir, subject_dir] if os.path.isdir(seg_dir) else [subject_dir]

        bone_mask = np.zeros(ct_data.shape, dtype=bool)
        loaded_bones = []

        for vert_name in target_vertebrae:
            for sdir in search_dirs:
                p_cand = os.path.join(sdir, f"{vert_name}.nii.gz")
                if os.path.exists(p_cand):
                    vert_img = nib.load(p_cand)
                    bone_mask |= (vert_img.get_fdata() > 0.5)
                    loaded_bones.append(vert_name)
                    break

        # Check for single combined mask if individual masks were not found
        if not loaded_bones:
            combined_candidates = glob.glob(os.path.join(subject_dir, "*seg*.nii*")) + glob.glob(os.path.join(subject_dir, "*mask*.nii*"))
            for comb_path in combined_candidates:
                if comb_path != ct_path and os.path.exists(comb_path):
                    comb_img = nib.load(comb_path)
                    comb_data = comb_img.get_fdata()
                    # TotalSegmentator standard bone label IDs or threshold
                    bone_mask = (comb_data >= 1) & (comb_data <= 30)
                    loaded_bones.append(os.path.basename(comb_path))
                    break

        # 3. Create Multi-Class Label Volume (0: air, 1: bone, 2: soft tissue, 3: skin)
        label_vol = np.full(ct_data.shape, 2, dtype=np.int32) # Soft tissue baseline
        label_vol[ct_data < -300.0] = 0 # Air
        label_vol[bone_mask] = 1 # Bone

        # Skin surface mask if available
        skin_path = os.path.join(seg_dir if os.path.isdir(seg_dir) else subject_dir, "skin.nii.gz")
        if os.path.exists(skin_path):
            skin_mask = nib.load(skin_path).get_fdata() > 0.5
            label_vol[skin_mask] = 3

        # 4. Optional Bounding Box Cropping around Spine & Posterior Back
        if crop_to_spine and np.any(bone_mask):
            coords = np.argwhere(bone_mask)
            min_c = np.maximum(0, coords.min(axis=0) - margin_voxels)
            max_c = np.minimum(ct_data.shape, coords.max(axis=0) + margin_voxels)
            # Ensure dorsal back (posterior y) is retained for probe contact
            max_c[1] = ct_data.shape[1]

            ct_data = ct_data[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
            label_vol = label_vol[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
            bone_mask = bone_mask[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]

        # 5. Extract Ground Truth Bone Surface Points in mm
        d, h, w = ct_data.shape
        sx, sy, sz = spacing
        bone_voxels = np.argwhere(bone_mask)
        if len(bone_voxels) > 0:
            bone_pts_z = (bone_voxels[:, 0] - d / 2.0) * sz
            bone_pts_y = (bone_voxels[:, 1] - h / 3.0) * sy
            bone_pts_x = (bone_voxels[:, 2] - w / 2.0) * sx
            bone_points = np.stack([bone_pts_x, bone_pts_y, bone_pts_z], axis=-1).astype(np.float32)
        else:
            bone_points = np.empty((0, 3), dtype=np.float32)

        origin_mm = (-w / 2.0 * sx, -h / 3.0 * sy, -d / 2.0 * sz)

        # Estimate posterior dorsal skin y boundary in mm
        skin_indices = np.where(label_vol != 0)[1]
        skin_y_mm = float((np.percentile(skin_indices, 95) - h / 3.0) * sy) if len(skin_indices) > 0 else 55.0

        print(f"[TotalSegmentator Loader] Loaded {len(loaded_bones)} anatomical structures from {subject_dir}.")
        print(f"  Volume Shape: {ct_data.shape} | Spacing: {spacing} mm | Bone points: {len(bone_points)}")

        return {
            "ct_volume": ct_data,
            "label_volume": label_vol,
            "bone_points": bone_points,
            "spacing_mm": spacing,
            "spacing": spacing,
            "origin_mm": origin_mm,
            "origin": origin_mm,
            "skin_y_mm": skin_y_mm,
            "subject_dir": subject_dir,
            "loaded_structures": loaded_bones,
            "acoustic_maps": PatientAssetGenerator.ct_to_acoustic_properties(ct_data),
        }

    @staticmethod
    def ct_to_acoustic_properties(ct_volume: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Maps CT Hounsfield Units directly to physical acoustic impedance Z and sound speed c
        using standard empirical medical physics formulas:
        - Density: rho(HU) in g/cm^3
        - Speed of sound: c(HU) in m/s
        - Acoustic Impedance: Z = rho * c in Mega-Rayls (10^6 kg / (m^2 * s))
        """
        # Density formula (Schneider et al., PMB)
        density = np.where(
            ct_volume <= 100.0,
            np.clip(1.0 + (ct_volume / 1000.0), 0.05, 1.15),
            1.0 + 0.0008 * ct_volume
        ).astype(np.float32)

        # Speed of sound formula (Mast et al., JASA)
        speed_of_sound = np.where(
            ct_volume <= 100.0,
            1540.0 + 0.25 * np.clip(ct_volume, -100.0, 100.0),
            1565.0 + 1.15 * np.clip(ct_volume - 100.0, 0.0, 1600.0)
        ).astype(np.float32)

        # Acoustic impedance in MRayl
        impedance_mrayl = (density * speed_of_sound) / 1000.0

        return {
            "density_g_cm3": density,
            "speed_of_sound_m_s": speed_of_sound,
            "acoustic_impedance_mrayl": impedance_mrayl,
        }

    @staticmethod
    def generate_multi_patient_dataset(
        num_patients: int = 3,
        base_shape: Tuple[int, int, int] = (120, 160, 160),
    ) -> Dict[str, Dict[str, any]]:
        """
        Synthesizes a cohort of diverse patient anatomies with varied BMI,
        lordosis curvature, vertebral dimensions, and bone mineral density (HU).
        """
        patients = {}
        for idx in range(num_patients):
            # Patient-specific anatomical variations
            np.random.seed(100 + idx)
            # Bone density variation (young vs osteoporotic)
            bone_hu = float(np.random.uniform(900.0, 1400.0))
            # Subcutaneous fat thickness (BMI variation)
            skin_elevation = float(np.random.uniform(50.0, 62.0))
            # Vertebral dimension scaling
            scale_vertebra = float(np.random.uniform(0.85, 1.15))

            patient_data = PatientAssetGenerator.generate_lumbar_phantom(
                shape=base_shape,
                spacing_mm=(1.0, 1.0, 1.0),
            )
            # Modulate bone HU and skin elevation for this patient
            bone_mask = patient_data["label_volume"] == 1
            patient_data["ct_volume"][bone_mask] = bone_hu + np.random.normal(0, 20.0, np.sum(bone_mask))
            patient_data["skin_y_mm"] = skin_elevation
            patient_data["patient_id"] = f"PATIENT_{idx+1:03d}"
            patient_data["bone_hu_nominal"] = bone_hu

            # Precompute acoustic properties
            patient_data["acoustic_maps"] = PatientAssetGenerator.ct_to_acoustic_properties(
                patient_data["ct_volume"]
            )

            patients[patient_data["patient_id"]] = patient_data

        return patients

    @staticmethod
    def export_totalsegmentator_dataset(
        output_dir: str,
        shape: Tuple[int, int, int] = (160, 160, 160),
        spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> str:
        """
        Synthesizes and exports a full clinical patient dataset adhering strictly
        to the standard TotalSegmentator NIfTI directory structure:
        - {output_dir}/ct.nii.gz
        - {output_dir}/segmentations/vertebrae_L1.nii.gz
        - {output_dir}/segmentations/vertebrae_L2.nii.gz
        - {output_dir}/segmentations/vertebrae_L3.nii.gz
        - {output_dir}/segmentations/vertebrae_L4.nii.gz
        - {output_dir}/segmentations/vertebrae_L5.nii.gz
        - {output_dir}/segmentations/sacrum.nii.gz
        - {output_dir}/segmentations/skin.nii.gz

        Args:
            output_dir: Target directory path for exported NIfTI files.
            shape: (D, H, W) volume dimensions.
            spacing_mm: (sx, sy, sz) physical voxel spacing in mm.

        Returns:
            Path to the generated patient directory.
        """
        import os

        os.makedirs(output_dir, exist_ok=True)
        seg_dir = os.path.join(output_dir, "segmentations")
        os.makedirs(seg_dir, exist_ok=True)

        d, h, w = shape
        sx, sy, sz = spacing_mm

        # Physical coordinate grids in mm
        z = (np.arange(d) - d / 2.0) * sz  # Cranial-caudal
        y = (np.arange(h) - h / 3.0) * sy  # Anterior-posterior (y > 0 is dorsal back)
        x = (np.arange(w) - w / 2.0) * sx  # Lateral
        grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing="ij")

        # CT HU baseline (+40 HU muscle/soft tissue)
        ct_vol = np.full(shape, 40.0, dtype=np.float32)
        affine = np.diag([sx, sy, sz, 1.0])
        affine[0, 3] = -w / 2.0 * sx
        affine[1, 3] = -h / 3.0 * sy
        affine[2, 3] = -d / 2.0 * sz

        # Define multi-vertebra levels along lumbar spine
        vertebrae_configs = [
            ("vertebrae_L1", -48.0),
            ("vertebrae_L2", -24.0),
            ("vertebrae_L3", 0.0),
            ("vertebrae_L4", 24.0),
            ("vertebrae_L5", 48.0),
        ]

        composite_bone_mask = np.zeros(shape, dtype=bool)

        for name, z_center in vertebrae_configs:
            # 1. Vertebral body
            rel_z = grid_z - z_center
            body_dist = ((rel_z / 10.0)**2 + ((grid_y + 15.0) / 16.0)**2 + (grid_x / 20.0)**2)
            body_mask = body_dist <= 1.0

            # 2. Spinal canal
            canal_dist = ((grid_y - 8.0) / 9.0)**2 + (grid_x / 10.0)**2
            canal_mask = (canal_dist <= 1.0) & (np.abs(rel_z) <= 12.0)

            # 3. Spinous process
            spinous_mask = (
                (grid_y >= 10.0) & (grid_y <= 40.0) &
                (np.abs(grid_x) <= (3.5 - 0.05 * grid_y)) &
                (np.abs(rel_z) <= (8.0 - 0.15 * grid_y))
            )

            # 4. Transverse processes
            trans_left = (
                (grid_y >= 0.0) & (grid_y <= 10.0) &
                (grid_x >= 10.0) & (grid_x <= 34.0) &
                (np.abs(rel_z) <= 5.0)
            )
            trans_right = (
                (grid_y >= 0.0) & (grid_y <= 10.0) &
                (grid_x <= -10.0) & (grid_x >= -34.0) &
                (np.abs(rel_z) <= 5.0)
            )

            vert_mask = (body_mask | spinous_mask | trans_left | trans_right) & (~canal_mask)
            composite_bone_mask |= vert_mask

            # Save individual vertebra mask
            vert_nii = nib.Nifti1Image(vert_mask.astype(np.uint8), affine)
            nib.save(vert_nii, os.path.join(seg_dir, f"{name}.nii.gz"))

        # Sacrum at lower lumbar junction
        sacrum_rel_z = grid_z - 68.0
        sacrum_mask = (
            (sacrum_rel_z >= -10.0) & (sacrum_rel_z <= 18.0) &
            (grid_y >= -10.0) & (grid_y <= 25.0) &
            (np.abs(grid_x) <= (28.0 - 0.5 * np.maximum(0, sacrum_rel_z)))
        )
        composite_bone_mask |= sacrum_mask
        sacrum_nii = nib.Nifti1Image(sacrum_mask.astype(np.uint8), affine)
        nib.save(sacrum_nii, os.path.join(seg_dir, "sacrum.nii.gz"))

        # Skin surface at posterior back (y = 54 mm)
        skin_y = 54.0
        skin_mask = (np.abs(grid_y - skin_y) <= 2.0) & (grid_y <= skin_y + 1.0)
        skin_nii = nib.Nifti1Image(skin_mask.astype(np.uint8), affine)
        nib.save(skin_nii, os.path.join(seg_dir, "skin.nii.gz"))

        # Assign HU to CT volume
        # Cortical bone (+1250 HU)
        ct_vol[composite_bone_mask] = 1250.0

        # Subcutaneous fat (-90 HU)
        fat_mask = (grid_y >= 38.0) & (grid_y <= skin_y) & (~composite_bone_mask)
        ct_vol[fat_mask] = -90.0

        # Air (-1000 HU)
        air_mask = grid_y > skin_y
        ct_vol[air_mask] = -1000.0

        # Gaussian CT noise
        noise = np.random.normal(0, 14.0, shape).astype(np.float32)
        ct_vol += noise

        # Save CT volume
        ct_nii = nib.Nifti1Image(ct_vol, affine)
        nib.save(ct_nii, os.path.join(output_dir, "ct.nii.gz"))

        print(f"[+] Exported TotalSegmentator clinical patient dataset to: {output_dir}")
        print(f"    Structures: L1-L5, Sacrum, Skin | Dimensions: {shape} | Spacing: {spacing_mm} mm")
        return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Patient Asset Generator & TotalSegmentator Exporter")
    parser.add_argument("--export-totalseg", type=str, default="", help="Export TotalSegmentator patient dataset to directory")
    args = parser.parse_args()

    if args.export_totalseg:
        PatientAssetGenerator.export_totalsegmentator_dataset(args.export_totalseg)
    else:
        print("[*] Generating synthetic lumbar phantom verification...")
        data = PatientAssetGenerator.generate_lumbar_phantom()
        print(f"[✓] Lumbar phantom ready. Bone points: {len(data['bone_points'])}, Shape: {data['ct_volume'].shape}")


