"""Benchmark-suite tests: metric bookkeeping, echo detection sanity checks, and the soft-body analyses."""

import numpy as np
import pytest
import torch

from ReconstructionPipeline.evaluation.benchmark_runner import (
    ReconstructionBenchmarkSuite,
    detect_bone_echo_depth,
)

PATIENT = "data/sample_totalsegmentator_patient"


@pytest.fixture(scope="module")
def suite():
    return ReconstructionBenchmarkSuite(num_trials=1, patient_dir=PATIENT, target_vertebra="vertebrae_L4")


def test_force_metrics_can_fail(suite):
    env = suite.make_env(5, seed=0)
    env.reset(seed=0)
    forces = [4.0, 4.5, 21.0, 3.0]
    m = suite._evaluate_trial_metrics(env, forces, [0.9, 0.9, 0.1, 0.9])
    assert m["safety_violations"] == 1.0
    assert m["max_force_n"] == 21.0
    assert np.isclose(m["force_rmse_n"], np.sqrt(np.mean((np.array(forces) - 4.0) ** 2)))
    assert m["force_in_band_pct"] == 75.0
    assert m["low_coupling_steps"] == 1.0
    for k in ("coverage2_pct", "coverage3_pct", "precision2_pct", "precision3_pct", "chamfer_mm", "mad_mm"):
        assert k in m


def test_echo_detector_rejects_non_bone():
    res = 0.25
    flat = np.full((300, 40), 0.3)
    assert np.isnan(detect_bone_echo_depth(flat, res, slice(10, 30)))
    edge = np.full((300, 40), 0.2)
    edge[-1] = 1.0  # brightest row at the window edge: must not be reported
    assert np.isnan(detect_bone_echo_depth(edge, res, slice(10, 30)))
    no_shadow = np.full((300, 40), 0.2)
    no_shadow[100] = 1.0
    no_shadow[101:] = 0.8  # bright tissue below the peak: no shadow
    assert np.isnan(detect_bone_echo_depth(no_shadow, res, slice(10, 30)))
    bone = np.full((300, 40), 0.2)
    bone[100] = 1.0
    bone[101:] = 0.02
    assert detect_bone_echo_depth(bone, res, slice(10, 30)) == pytest.approx(25.0)


def test_force_sweep_is_physical(suite):
    rows = suite.run_force_sweep(forces=(1.0, 2.0, 4.0, 6.0))
    ind = [r["indentation_mm"] for r in rows]
    assert all(b > a for a, b in zip(ind, ind[1:]))
    for r in rows:
        assert abs(r["model_force_n"] - r["target_force_n"]) < 0.05 or r["indentation_limited"]
        assert np.isfinite(r["echo_depth_mm"]), r
        # bone is rigid: the detected echo sits at (tissue thickness - indentation) within ~1.5 mm
        assert abs(r["echo_depth_mm"] - r["expected_echo_depth_mm"]) < 1.5, r
    depths = [r["echo_depth_mm"] for r in rows]
    assert depths[-1] < depths[0] - 2.0


def test_deformation_unaware_backprojection_is_worse(suite):
    # FEM backend (default): the indentation is measured from the skin under the probe centre and is only a
    # few mm (the face first touches the paraspinal ridges), so ignoring it costs less than with Kelvinlet.
    res = suite.run_deformation_ablation(seeds=[0])
    aware, unaware = res["aware"], res["unaware"]
    assert unaware["mean_mad_mm"] > aware["mean_mad_mm"] + 0.3
    assert unaware["mean_coverage2_pct"] < aware["mean_coverage2_pct"]
