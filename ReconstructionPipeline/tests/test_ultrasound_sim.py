"""
Unit tests for Model-Based Ultrasound Simulator & Confidence Estimator.
"""

import pytest
import numpy as np
import torch
from ReconstructionPipeline.simulation.ultrasound_model_based import ModelBasedUltrasoundSimulator
from ReconstructionPipeline.simulation.confidence_map import UltrasoundConfidenceEstimator


def test_ultrasound_simulation_pipeline():
    sim = ModelBasedUltrasoundSimulator(kernel_size=7)

    # Create synthetic 2D slice: soft tissue top, bone bottom
    h, w = 100, 80
    ct_slice = np.full((h, w), 40.0, dtype=np.float32) # Soft tissue
    ct_slice[60:, :] = 1200.0 # Bone below depth y=60
    
    label_slice = np.full((h, w), 2, dtype=np.int32)
    label_slice[60:, :] = 1 # Bone label

    res = sim.simulate_bmode(torch.from_numpy(ct_slice), label_slice=torch.from_numpy(label_slice))

    bmode = res["bmode"]
    energy = res["energy"]

    # 1. Image intensity in normalized range [0, 1]
    assert bmode.min() >= 0.0
    assert bmode.max() <= 1.0

    # 2. Specular reflection peak should exist near bone boundary y=60
    reflection_band = bmode[55:65, :].mean().item()
    deep_shadow_band = bmode[85:95, :].mean().item()
    assert reflection_band > deep_shadow_band, "Bone interface must be brighter than shadowed area beneath it"

    # 3. Energy attenuation: energy deep inside/below bone should be significantly reduced
    assert energy[90, 40].item() < energy[10, 40].item()


def test_confidence_map():
    estimator = UltrasoundConfidenceEstimator()
    bmode = torch.rand((100, 80))
    energy = torch.ones((100, 80))

    conf = estimator.compute_confidence_map(bmode, energy_map=energy)
    assert conf.shape == (100, 80)
    assert conf.min() >= 0.0
    assert conf.max() <= 1.0
