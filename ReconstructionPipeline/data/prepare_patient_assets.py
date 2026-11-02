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
        affine = ct_img.affine
        bone_voxels = np.argwhere(bone_mask)
        
        # Check if volume is in standard (X, Y, Z) ordering or legacy synthetic (Z, Y, X)
        if ct_data.ndim == 3 and ct_data.shape[0] < ct_data.shape[2]:
            # Standard NIfTI (nx, ny, nz) = (X, Y, Z) -> transpose to (nz, ny, nx) = (D, H, W) for VolumeSliceExtractor
            ct_data = ct_data.transpose(2, 1, 0)
            label_vol = label_vol.transpose(2, 1, 0)
            bone_pts_trans = nib.affines.apply_affine(affine, bone_voxels)
            bone_points = bone_pts_trans.astype(np.float32)
            origin_mm = (float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3]))
            spacing = (float(affine[0, 0]), float(affine[1, 1]), float(affine[2, 2]))
            skin_y_mm = 55.0
        elif len(bone_voxels) > 0:
            if np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3]))) and np.any(affine[:3, 3] != 0):
                bone_points = nib.affines.apply_affine(affine, bone_voxels).astype(np.float32)
                origin_mm = tuple(float(x) for x in affine[:3, 3])
                spacing = tuple(float(abs(affine[i, i])) for i in range(3))
                skin_y_mm = 55.0
            else:
                d, h, w = ct_data.shape
                sx, sy, sz = spacing
                bone_pts_z = (bone_voxels[:, 0] - d / 2.0) * sz
                bone_pts_y = (bone_voxels[:, 1] - h / 3.0) * sy
                bone_pts_x = (bone_voxels[:, 2] - w / 2.0) * sx
                bone_points = np.stack([bone_pts_x, bone_pts_y, bone_pts_z], axis=-1).astype(np.float32)
                origin_mm = (-w / 2.0 * sx, -h / 3.0 * sy, -d / 2.0 * sz)
                skin_indices = np.where(label_vol != 0)[1]
                skin_y_mm = float((np.percentile(skin_indices, 95) - h / 3.0) * sy) if len(skin_indices) > 0 else 55.0
        else:
            bone_points = np.empty((0, 3), dtype=np.float32)
            origin_mm = (0.0, 0.0, 0.0)
            skin_y_mm = 55.0

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

        # Check for real clinical CT dataset from huggingface or local cache
        cache_ct = "/home/rishang/.cache/huggingface/hub/datasets--SHPDRG--lumbar-ct-segmentation-human-sample-1/snapshots/328398893a6ae7f115f8d6ecd326bd1bc3da978b/data/images/case-0004.nii.gz"
        cache_lbl = "/home/rishang/.cache/huggingface/hub/datasets--SHPDRG--lumbar-ct-segmentation-human-sample-1/snapshots/328398893a6ae7f115f8d6ecd326bd1bc3da978b/data/labels/case-0004.nii.gz"

        if not (os.path.exists(cache_ct) and os.path.exists(cache_lbl)):
            try:
                from huggingface_hub import hf_hub_download
                cache_ct = hf_hub_download(
                    repo_id="SHPDRG/lumbar-ct-segmentation-human-sample-1",
                    filename="data/images/case-0004.nii.gz",
                    repo_type="dataset",
                )
                cache_lbl = hf_hub_download(
                    repo_id="SHPDRG/lumbar-ct-segmentation-human-sample-1",
                    filename="data/labels/case-0004.nii.gz",
                    repo_type="dataset",
                )
            except Exception as e:
                print(f"[!] Warning: Could not download clinical patient CT: {e}. Falling back to parametric synthesis.")
                cache_ct = ""

        if cache_ct and os.path.exists(cache_ct) and os.path.exists(cache_lbl):
            print(f"[*] Ingesting authentic clinical lumbar CT from {cache_ct}...")
            from scipy.ndimage import map_coordinates
            img_ct = nib.load(cache_ct)
            img_lbl = nib.load(cache_lbl)

            raw_ct = img_ct.get_fdata().astype(np.float32)
            raw_lbl = img_lbl.get_fdata().astype(np.int32)
            sx, sy, sz = img_ct.header.get_zooms()[:3]

            i_mid, j_skin, k_l3 = 187.0, 421.0, 150.0
            xs = np.arange(-70.0, 71.0, 1.0)
            ys = np.arange(-50.0, 66.0, 1.0)
            zs = np.arange(-90.0, 86.0, 1.0)
            nx, ny, nz = len(xs), len(ys), len(zs)

            grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
            sample_i = i_mid + grid_x / sx
            sample_j = j_skin - (55.0 - grid_y) / sy
            sample_k = k_l3 + grid_z / sz
            sample_coords = np.stack([sample_i, sample_j, sample_k], axis=0)

            ct_resampled = map_coordinates(raw_ct, sample_coords, order=1, mode="constant", cval=-1000.0).astype(np.float32)
            lbl_resampled = map_coordinates(raw_lbl, sample_coords, order=0, mode="constant", cval=0).astype(np.int32)

            affine_ras = np.diag([1.0, 1.0, 1.0, 1.0])
            affine_ras[0, 3] = -70.0
            affine_ras[1, 3] = -50.0
            affine_ras[2, 3] = -90.0

            nib.save(nib.Nifti1Image(ct_resampled, affine_ras), os.path.join(output_dir, "ct.nii.gz"))

            vert_map = {
                "vertebrae_L1": 1,
                "vertebrae_L2": 2,
                "vertebrae_L3": 3,
                "vertebrae_L4": 4,
                "vertebrae_L5": 5,
                "sacrum": 6,
            }
            for name, val in vert_map.items():
                mask = (lbl_resampled == val).astype(np.uint8)
                nib.save(nib.Nifti1Image(mask, affine_ras), os.path.join(seg_dir, f"{name}.nii.gz"))

            # Skin mask
            body_mask = ct_resampled >= -200.0
            skin_mask = np.zeros_like(body_mask, dtype=np.uint8)
            for ix in range(nx):
                for iz in range(nz):
                    col = np.where(body_mask[ix, :, iz])[0]
                    if len(col) > 0:
                        top_y = col.max()
                        skin_mask[ix, max(0, top_y - 2):top_y + 1, iz] = 1
            nib.save(nib.Nifti1Image(skin_mask, affine_ras), os.path.join(seg_dir, "skin.nii.gz"))

            print(f"[+] Successfully exported authentic clinical patient dataset to: {output_dir}")
            PatientAssetGenerator.export_patient_meshes(output_dir)
            return output_dir

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

        # Automatically export surface meshes for PyBullet and Slicer 3D visualization
        PatientAssetGenerator.export_patient_meshes(output_dir)
        return output_dir

    @staticmethod
    def export_patient_meshes(
        patient_dir: str,
        output_mesh_dir: Optional[str] = None,
        target_triangles_bone: int = 12000,
    ) -> Dict[str, str]:
        """
        Extracts 3D surface meshes (.obj) for lumbar spine (L1-L5 + sacrum) and patient torso skin
        from TotalSegmentator segmentations, aligned with PyBullet clinical coordinates.

        Args:
            patient_dir: Directory containing ct.nii.gz and segmentations/
            output_mesh_dir: Output folder for .obj meshes (defaults to patient_dir/meshes)
            target_triangles_bone: Target decimation triangle count for spine mesh.

        Returns:
            Dictionary mapping asset names ('spine', 'torso_skin') to absolute file paths.
        """
        import os
        import glob
        import skimage.measure
        import open3d as o3d

        if output_mesh_dir is None:
            output_mesh_dir = os.path.join(patient_dir, "meshes")
        os.makedirs(output_mesh_dir, exist_ok=True)

        seg_dir = os.path.join(patient_dir, "segmentations")
        if not os.path.isdir(seg_dir):
            seg_dir = patient_dir

        # 1. Bone Mesh (Lumbar Vertebrae L1-L5 + Sacrum)
        comb_bone = None
        bone_files = sorted(glob.glob(os.path.join(seg_dir, "vertebrae_*.nii.gz")))
        sacrum_file = os.path.join(seg_dir, "sacrum.nii.gz")
        if os.path.exists(sacrum_file):
            bone_files.append(sacrum_file)

        ct_path = os.path.join(patient_dir, "ct.nii.gz")
        has_ct = os.path.exists(ct_path)
        ct_affine = nib.load(ct_path).affine if has_ct else np.eye(4)

        for f in bone_files:
            data = nib.load(f).get_fdata() > 0.5
            if comb_bone is None:
                comb_bone = np.zeros(data.shape, dtype=bool)
            comb_bone |= data

        spine_path = os.path.join(output_mesh_dir, "patient_lumbar_spine.obj")
        spine_ras_path = os.path.join(output_mesh_dir, "patient_lumbar_spine_ras.obj")

        if comb_bone is not None and np.any(comb_bone):
            verts, faces, _, _ = skimage.measure.marching_cubes(
                comb_bone.astype(np.float32), level=0.5, spacing=(1.0, 1.0, 1.0)
            )

            # Check if volume has affine coordinate transform
            if has_ct and not np.allclose(ct_affine, np.eye(4)):
                ras_verts = nib.affines.apply_affine(ct_affine, verts)
            else:
                d, h, w = comb_bone.shape
                ras_verts = np.empty_like(verts)
                ras_verts[:, 0] = (verts[:, 2] - w / 2.0) * 1.0  # Lateral X (mm)
                ras_verts[:, 1] = (verts[:, 1] - h / 3.0) * 1.0  # Anterior Y (mm)
                ras_verts[:, 2] = (verts[:, 0] - d / 2.0) * 1.0  # Cranial Z (mm)

            # PyBullet world coordinates (meters, table at Z=0.30, skin at Z=0.42, X=0.55, Y=0.0)
            pb_verts = np.empty_like(ras_verts)
            pb_verts[:, 0] = 0.55 + ras_verts[:, 0] * 1e-3
            pb_verts[:, 1] = 0.0 + ras_verts[:, 2] * 1e-3
            pb_verts[:, 2] = 0.42 + (ras_verts[:, 1] - 55.0) * 1e-3

            # 1A. Export PyBullet Spine Mesh
            o3d_spine = o3d.geometry.TriangleMesh()
            o3d_spine.vertices = o3d.utility.Vector3dVector(pb_verts)
            o3d_spine.triangles = o3d.utility.Vector3iVector(faces)
            o3d_spine.compute_vertex_normals()
            if target_triangles_bone > 0 and len(faces) > target_triangles_bone:
                o3d_spine = o3d_spine.simplify_quadric_decimation(target_number_of_triangles=target_triangles_bone)
            o3d.io.write_triangle_mesh(spine_path, o3d_spine)
            print(f"[+] Exported patient lumbar spine mesh (PyBullet) to: {spine_path} ({len(o3d_spine.vertices)} verts, {len(o3d_spine.triangles)} faces)")

            # 1B. Export Slicer RAS Spine Mesh
            o3d_spine_ras = o3d.geometry.TriangleMesh()
            o3d_spine_ras.vertices = o3d.utility.Vector3dVector(ras_verts)
            o3d_spine_ras.triangles = o3d.utility.Vector3iVector(faces)
            o3d_spine_ras.compute_vertex_normals()
            if target_triangles_bone > 0 and len(faces) > target_triangles_bone:
                o3d_spine_ras = o3d_spine_ras.simplify_quadric_decimation(target_number_of_triangles=target_triangles_bone)
            o3d.io.write_triangle_mesh(spine_ras_path, o3d_spine_ras)
            print(f"[+] Exported patient lumbar spine mesh (Slicer RAS) to: {spine_ras_path}")

        # 2. Torso Skin Envelope Mesh
        torso_path = os.path.join(output_mesh_dir, "patient_torso_skin.obj")
        torso_ras_path = os.path.join(output_mesh_dir, "patient_torso_skin_ras.obj")

        body_extracted = False
        if has_ct:
            ct_vol = nib.load(ct_path).get_fdata()
            body_mask = ct_vol >= -200.0
            if np.any(body_mask):
                verts_t, faces_t, _, _ = skimage.measure.marching_cubes(
                    body_mask.astype(np.float32), level=0.5, spacing=(1.0, 1.0, 1.0)
                )
                if not np.allclose(ct_affine, np.eye(4)):
                    ras_verts_t = nib.affines.apply_affine(ct_affine, verts_t)
                else:
                    d, h, w = body_mask.shape
                    ras_verts_t = np.empty_like(verts_t)
                    ras_verts_t[:, 0] = (verts_t[:, 2] - w / 2.0) * 1.0
                    ras_verts_t[:, 1] = (verts_t[:, 1] - h / 3.0) * 1.0
                    ras_verts_t[:, 2] = (verts_t[:, 0] - d / 2.0) * 1.0

                mesh_torso_ras = o3d.geometry.TriangleMesh()
                mesh_torso_ras.vertices = o3d.utility.Vector3dVector(ras_verts_t)
                mesh_torso_ras.triangles = o3d.utility.Vector3iVector(faces_t)
                mesh_torso_ras = mesh_torso_ras.simplify_quadric_decimation(target_number_of_triangles=25000)
                mesh_torso_ras.compute_vertex_normals()
                o3d.io.write_triangle_mesh(torso_ras_path, mesh_torso_ras)

                v_t_dec = np.asarray(mesh_torso_ras.vertices)
                pb_verts_t = np.empty_like(v_t_dec)
                pb_verts_t[:, 0] = 0.55 + v_t_dec[:, 0] * 1e-3
                pb_verts_t[:, 1] = 0.0 + v_t_dec[:, 2] * 1e-3
                pb_verts_t[:, 2] = 0.42 + (v_t_dec[:, 1] - 55.0) * 1e-3

                mesh_torso_pb = o3d.geometry.TriangleMesh()
                mesh_torso_pb.vertices = o3d.utility.Vector3dVector(pb_verts_t)
                mesh_torso_pb.triangles = mesh_torso_ras.triangles
                mesh_torso_pb.compute_vertex_normals()
                o3d.io.write_triangle_mesh(torso_path, mesh_torso_pb)
                print(f"[+] Exported anatomical patient torso meshes to: {torso_path} and {torso_ras_path}")
                body_extracted = True

        if not body_extracted:
            # Parametric fallback if CT is missing
            x_center, y_center = 0.55, 0.0
            z_base, z_skin = 0.30, 0.42
            half_w, half_l = 0.15, 0.22
            nx, ny = 40, 50
            xs = np.linspace(-half_w, half_w, nx)
            ys = np.linspace(-half_l, half_l, ny)
            X, Y = np.meshgrid(xs, ys)
            Z_dorsal = z_skin - 0.035 * (X / half_w)**2 - 0.015 * (Y / half_l)**2
            verts = []
            for i in range(ny):
                for j in range(nx):
                    verts.append([x_center + X[i, j], y_center + Y[i, j], Z_dorsal[i, j]])
            for i in range(ny):
                for j in range(nx):
                    verts.append([x_center + X[i, j], y_center + Y[i, j], z_base])
            verts = np.array(verts)
            faces = []
            for i in range(ny - 1):
                for j in range(nx - 1):
                    v0, v1 = i * nx + j, i * nx + (j + 1)
                    v2, v3 = (i + 1) * nx + (j + 1), (i + 1) * nx + j
                    faces.append([v0, v1, v2])
                    faces.append([v0, v2, v3])
            offset = nx * ny
            for i in range(ny - 1):
                for j in range(nx - 1):
                    v0, v1 = offset + i * nx + j, offset + i * nx + (j + 1)
                    v2, v3 = offset + (i + 1) * nx + (j + 1), offset + (i + 1) * nx + j
                    faces.append([v0, v2, v1])
                    faces.append([v0, v3, v2])
            for i in range(ny - 1):
                top0, top1 = i * nx, (i + 1) * nx
                bot0, bot1 = offset + i * nx, offset + (i + 1) * nx
                faces.append([top0, bot0, bot1])
                faces.append([top0, bot1, top1])
                top0, top1 = i * nx + (nx - 1), (i + 1) * nx + (nx - 1)
                bot0, bot1 = offset + i * nx + (nx - 1), offset + (i + 1) * nx + (nx - 1)
                faces.append([top0, bot1, bot0])
                faces.append([top0, top1, bot1])
            for j in range(nx - 1):
                top0, top1 = (ny - 1) * nx + j, (ny - 1) * nx + (j + 1)
                bot0, bot1 = offset + (ny - 1) * nx + j, offset + (ny - 1) * nx + (j + 1)
                faces.append([top0, top1, bot1])
                faces.append([top0, bot1, bot0])
                top0, top1 = j, j + 1
                bot0, bot1 = offset + j, offset + (j + 1)
                faces.append([top0, bot1, top1])
                faces.append([top0, bot0, bot1])

            o3d_torso = o3d.geometry.TriangleMesh()
            o3d_torso.vertices = o3d.utility.Vector3dVector(verts)
            o3d_torso.triangles = o3d.utility.Vector3iVector(np.array(faces, dtype=np.int32))
            o3d_torso.compute_vertex_normals()
            o3d.io.write_triangle_mesh(torso_path, o3d_torso)
            print(f"[+] Exported patient torso skin mesh to: {torso_path} ({len(verts)} verts, {len(faces)} faces)")

        return {
            "spine": spine_path,
            "torso_skin": torso_path,
        }


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


