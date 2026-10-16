"""
Unit tests for GPU-accelerated 3D Volume Slice Extractor.
"""

import pytest
import numpy as np
import torch
from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor
from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator


def test_slice_extractor_shapes():
    # Generate test phantom
    phantom = PatientAssetGenerator.generate_lumbar_phantom(shape=(60, 80, 80))
    ct_vol = phantom["ct_volume"]

    extractor = VolumeSliceExtractor(
        volume=ct_vol,
        spacing=phantom["spacing_mm"],
        origin=phantom["origin_mm"],
        image_size=(200, 150),
        physical_size_mm=(75.0, 50.0)
    )

    # 1. Single pose test
    t_single = np.eye(4)
    t_single[1, 3] = phantom["skin_y_mm"] # Placed on skin
    slice_single = extractor.extract_slices(t_single)
    
    assert isinstance(slice_single, torch.Tensor)
    assert slice_single.shape == (200, 150), f"Expected (200, 150), got {slice_single.shape}"

    # 2. Batched poses test (batch of 4)
    t_batch = np.repeat(np.eye(4)[np.newaxis, ...], 4, axis=0)
    slices_batch = extractor.extract_slices(t_batch)
    assert slices_batch.shape == (4, 200, 150), f"Expected (4, 200, 150), got {slices_batch.shape}"


def test_slice_extractor_values():
    # Verify that sampling within bone region returns bone Hounsfield units
    phantom = PatientAssetGenerator.generate_lumbar_phantom(shape=(60, 80, 80))
    extractor = VolumeSliceExtractor(
        volume=phantom["ct_volume"],
        spacing=phantom["spacing_mm"],
        origin=phantom["origin_mm"],
        image_size=(100, 100),
        physical_size_mm=(60.0, 40.0)
    )

    t_probe = np.eye(4)
    t_probe[1, 3] = 40.0 # Near spine
    extracted = extractor.extract_slices(t_probe)

    # Max intensity should detect bone (> 500 HU)
    assert extracted.max() > 500.0, "Slice should contain high bone intensities"
