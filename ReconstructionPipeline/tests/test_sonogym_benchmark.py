"""
Unit & Integration Tests for SonoGym Benchmark Suite with Soft-Body Deformations.
Verifies:
1. Complete SonoGym Metric Formulation (Coverage, Chamfer Distance, MAD, Safe Force Ratio, Force RMSE, Coupling).
2. Physical B-mode Bone Echo Shift under Contact Indentation (Kelvinlet Green's function).
3. Soft-Tissue Multi-Stiffness Generalization.
"""

import numpy as np
import pytest
import torch

from ReconstructionPipeline.evaluation.benchmark_runner import ReconstructionBenchmarkSuite
from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv


def test_sonogym_metric_calculation():
    suite = ReconstructionBenchmarkSuite(num_trials=1, device=torch.device("cpu"))
    env = UltrasoundReconstructionEnv(max_steps=10, device=torch.device("cpu"))
    env.reset()

    # Synthetic force & coupling history
    force_history = [3.5, 4.0, 4.2, 5.0, 3.8, 4.1]
    coupling_history = [0.95, 0.98, 1.0, 0.99, 0.94, 0.97]

    metrics = suite._evaluate_trial_metrics(env, force_history, coupling_history)

    assert "coverage_pct" in metrics
    assert "chamfer_mm" in metrics
    assert "mad_mm" in metrics
    assert "safe_force_ratio_pct" in metrics
    assert "max_force_n" in metrics
    assert "mean_force_n" in metrics
    assert "force_rmse_n" in metrics
    assert "mean_coupling" in metrics
    assert "path_length_mm" in metrics
    assert "rotation_deg" in metrics

    # All forces in [2.0, 8.0] N -> 100% compliance
    assert np.isclose(metrics["safe_force_ratio_pct"], 100.0)
    assert metrics["max_force_n"] == 5.0
    assert np.isclose(metrics["mean_force_n"], np.mean(force_history))
    assert metrics["force_rmse_n"] < 1.0
    assert metrics["mean_coupling"] > 0.90


def test_soft_body_echo_shift_under_pressure():
    """Verifies that probe pressure produces physical indentation and echo displacement."""
    suite = ReconstructionBenchmarkSuite(num_trials=1, device=torch.device("cpu"))
    res = suite.run_soft_body_deformation_benchmark()

    assert "echo_shifts" in res
    shifts = res["echo_shifts"]
    assert len(shifts) >= 4

    # Indentation must increase strictly monotonically with normal force
    for i in range(len(shifts) - 1):
        assert shifts[i + 1]["force_n"] > shifts[i]["force_n"]
        assert shifts[i + 1]["indentation_mm"] > shifts[i]["indentation_mm"]

    # Reconstruction metrics for rigid vs deformable must be recorded
    assert "cd_rigid_mm" in res
    assert "cd_deform_mm" in res
    assert res["cd_rigid_mm"] > 0.0
    assert res["cd_deform_mm"] > 0.0

    # Multi-stiffness generalization must cover soft, normal, and stiff tissue
    stiffness = res["stiffness_generalization"]
    assert "mu_2.0kPa" in stiffness
    assert "mu_4.0kPa" in stiffness
    assert "mu_8.0kPa" in stiffness
    for k, v in stiffness.items():
        assert v["safe_force_pct"] >= 95.0
