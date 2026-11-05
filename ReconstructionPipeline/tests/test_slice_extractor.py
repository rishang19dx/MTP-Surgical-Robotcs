"""Unit tests for the GPU volume slice extractor."""

import numpy as np
import torch

from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor
from ReconstructionPipeline.simulation.tissue_model import TissueState
from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator


def _dorsal(y, x=0.0, z=0.0):
    t = np.eye(4)
    t[:3, :3] = np.diag([1.0, -1.0, -1.0])
    t[:3, 3] = [x, y, z]
    return t


def test_slice_extractor_shapes():
    phantom = PatientAssetGenerator.generate_lumbar_phantom(shape=(60, 80, 80))
    ext = VolumeSliceExtractor(phantom["ct_volume"], spacing=phantom["spacing_mm"], origin=phantom["origin_mm"],
                               image_size=(200, 150), physical_size_mm=(75.0, 50.0))
    single = ext.extract_slices(_dorsal(phantom["skin_y_mm"]))
    assert isinstance(single, torch.Tensor) and single.shape == (200, 150)
    batch = ext.extract_slices(np.repeat(_dorsal(phantom["skin_y_mm"])[None], 4, axis=0))
    assert batch.shape == (4, 200, 150)


def test_slice_geometry_matches_phantom():
    """A dorsal probe on the skin sees tissue first and the spinous-process tip (y = 42 mm) at depth skin - 42."""
    phantom = PatientAssetGenerator.generate_lumbar_phantom(shape=(60, 80, 80))
    ext = VolumeSliceExtractor(phantom["label_volume"], spacing=phantom["spacing_mm"], origin=phantom["origin_mm"],
                               image_size=(301, 101), physical_size_mm=(75.0, 50.0))
    skin = phantom["skin_y_mm"]
    lbl = ext.extract_slices(_dorsal(skin), interpolation="nearest")
    assert int(lbl[0, 50]) == 2  # soft tissue directly under the probe
    first_bone = int(torch.argmax((lbl[:, 49:52] == 1).any(dim=1).int()))
    assert abs(first_bone * 0.25 - (skin - 42.0)) <= 1.0


def test_identity_state_equals_rigid():
    phantom = PatientAssetGenerator.generate_lumbar_phantom(shape=(60, 80, 80))
    ext = VolumeSliceExtractor(phantom["ct_volume"], spacing=phantom["spacing_mm"], origin=phantom["origin_mm"])
    t = _dorsal(phantom["skin_y_mm"])
    state = TissueState(0.0, 0.0, False, t[:3, 3], t[:3, 1])
    assert torch.equal(ext.extract_slices(t, tissue_state=state), ext.extract_slices(t))
    # A pure translation warp shifts the sampled anatomy
    shifted = ext.extract_slices(t, warp_fn=lambda x: x + torch.tensor([0.0, -5.0, 0.0], device=x.device))
    assert not torch.equal(shifted, ext.extract_slices(t))
