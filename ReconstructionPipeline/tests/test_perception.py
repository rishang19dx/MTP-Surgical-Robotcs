"""
Perception pipeline tests: segmentation, back-projection geometry, accumulator observation counts,
and reconstruction metrics (contract C5, C6).
"""

import time

import numpy as np
import pytest
import torch

from ReconstructionPipeline.simulation.ultrasound_model_based import ModelBasedUltrasoundSimulator
from ReconstructionPipeline.simulation.confidence_map import UltrasoundConfidenceEstimator
from ReconstructionPipeline.reconstruction.bone_segmentation import BoneSurfaceSegmenter
from ReconstructionPipeline.reconstruction.point_cloud_accumulator import PointCloudAccumulator
from ReconstructionPipeline.reconstruction.metrics import compute_reconstruction_metrics
from ReconstructionPipeline.registration.icp_registration import IntraoperativeRegistration

CPU = torch.device("cpu")
H, W = 200, 150
DY, DX = 75.0 / (H - 1), 50.0 / (W - 1)


def _synthetic_frame(seed=0):
    """Soft tissue with a fascia (fat/muscle) interface at row 45 and a curved bone surface."""
    xs = np.arange(W)
    bone_top = (110 + 12 * np.sin(xs / 25.0)).astype(int)
    ct = np.full((H, W), 50.0, dtype=np.float32)
    ct[:45] = -100.0  # subcutaneous fat above a muscle fascia: a bright but non-shadowing distractor
    rows = np.arange(H)[:, None]
    ct[rows >= bone_top[None, :]] = 1200.0
    sim = ModelBasedUltrasoundSimulator(device=CPU, axial_resolution_mm=DY, lateral_resolution_mm=DX, seed=seed)
    bmode = sim.simulate_bmode(torch.from_numpy(ct))["bmode"]
    conf = UltrasoundConfidenceEstimator(device=CPU, axial_resolution_mm=DY).compute_confidence_map(bmode)
    return bmode.numpy(), conf.numpy(), bone_top


def test_segmentation_recovers_synthetic_bone_line():
    bmode, conf, bone_top = _synthetic_frame()
    pts = BoneSurfaceSegmenter(axial_resolution_mm=DY).segment_from_bmode(bmode, confidence_map=conf)
    assert pts.dtype == np.int32 and pts.shape[1] == 2
    assert len(np.unique(pts[:, 0])) == len(pts), "at most one surface pixel per scanline"
    err = np.abs(pts[:, 1] - bone_top[pts[:, 0]])
    precision = np.mean(err <= 2)
    recall = len(pts) / W
    assert precision >= 0.9, precision
    assert recall >= 0.8, recall
    assert not np.any(np.abs(pts[:, 1] - 45) <= 3), "fascia (no distal shadow) must not be detected"


def test_segmentation_rejects_isolated_responses():
    img = np.full((H, W), 0.45, dtype=np.float32)
    img[100:102, 20:23] = 0.95  # 3-column blip with shadow below -> isolated
    img[102:140, 20:23] = 0.05
    img[100:102, 60:100] = 0.95  # long continuous ridge with shadow
    img[102:140, 60:100] = 0.05
    pts = BoneSurfaceSegmenter(axial_resolution_mm=DY, lateral_smoothing_px=1).segment_from_bmode(img)
    assert set(pts[:, 0]) == set(range(60, 100))


def test_segmentation_is_deterministic_with_rng():
    bmode, conf, _ = _synthetic_frame()
    a = BoneSurfaceSegmenter(dropout_probability=0.3, rng=np.random.default_rng(1)).segment_from_bmode(bmode, conf)
    b = BoneSurfaceSegmenter(dropout_probability=0.3, rng=np.random.default_rng(1)).segment_from_bmode(bmode, conf)
    full = BoneSurfaceSegmenter().segment_from_bmode(bmode, conf)
    assert np.array_equal(a, b)
    assert 0 < len(a) < len(full)


def test_segmentation_latency_cpu():
    bmode, conf, _ = _synthetic_frame()
    seg = BoneSurfaceSegmenter()
    seg.segment_from_bmode(bmode, conf)
    times = []
    for _ in range(30):
        t = time.perf_counter()
        seg.segment_from_bmode(bmode, conf)
        times.append(time.perf_counter() - t)
    # best-of-N: robust to CPU contention from concurrent jobs; the old per-pixel loop took ~25 ms
    assert np.min(times) * 1e3 < 3.0, np.median(times) * 1e3


def test_label_oracle_first_hit():
    lab = np.zeros((10, 4), dtype=np.int32)
    lab[6:, 0] = 1
    lab[3:, 2] = 1
    pts = BoneSurfaceSegmenter().segment_from_label_map(lab)
    assert pts.tolist() == [[0, 6], [2, 3]]


def test_backprojection_matches_slice_extractor_grid():
    acc = PointCloudAccumulator(image_size=(H, W), physical_size_mm=(75.0, 50.0))
    p = acc.pixels_to_probe_coords(np.array([[0, 0], [W - 1, H - 1], [(W - 1) / 2.0, 0]]))
    np.testing.assert_allclose(p[0], [-25.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(p[1], [25.0, 75.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(p[2], [0.0, 0.0, 0.0], atol=1e-5)


def test_accumulator_observation_counts():
    acc = PointCloudAccumulator(voxel_size_mm=1.5)
    T = np.eye(4)
    line = np.column_stack([np.arange(10, 140, 4), np.full(33, 80)])
    acc.add_frame(line, T)
    acc.add_frame(line, T)
    acc.add_frame(np.array([[5, 190]]), T)  # one-off detection
    n_all = acc.num_points(min_observations=1)
    n_rep = acc.num_points(min_observations=2)
    assert n_all == n_rep + 1
    assert np.all(acc.get_observation_counts(min_observations=2) == 2)
    assert acc.get_reconstruction_area() == pytest.approx(n_all * 1.5 ** 2)


def test_metrics_exact_values():
    recon = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    target = np.array([[0.0, 0, 0], [0.0, 1.0, 0]])
    m = compute_reconstruction_metrics(recon, target, target, tol_mm=3.0)
    assert m["num_points"] == 2
    assert m["precision"] == pytest.approx(0.5)
    assert m["coverage"] == pytest.approx(1.0)
    assert m["f_score"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert m["mad_mm"] == pytest.approx(5.0)
    assert m["chamfer_mm"] == pytest.approx(0.5 * (5.0 + 0.5))
    m2 = compute_reconstruction_metrics(recon, target, target, tol_mm=[0.5, 3.0])
    assert m2["coverage"] == pytest.approx(0.5) and m2["coverage@3"] == pytest.approx(1.0)
    assert "precision@2" in m2 and "f_score@0.5" in m2
    # precision uses ALL bone, coverage uses the target ROI only
    bone = np.vstack([target, [[10.0, 0, 0]]])
    m3 = compute_reconstruction_metrics(recon, target, bone)
    assert m3["precision"] == pytest.approx(1.0) and m3["mad_mm"] == pytest.approx(0.0)
    empty = compute_reconstruction_metrics(np.empty((0, 3)), target, bone)
    assert empty["coverage"] == 0.0 and empty["num_points"] == 0 and np.isinf(empty["chamfer_mm"])


def test_icp_compute_metrics_backcompat():
    rng = np.random.default_rng(0)
    a = rng.uniform(-10, 10, (300, 3))
    b = a + np.array([1.0, 0.0, 0.0])
    cd, cov = IntraoperativeRegistration().compute_metrics(a, b, coverage_tolerance_mm=3.0)
    m = compute_reconstruction_metrics(a, b, b, tol_mm=3.0)
    assert cd == pytest.approx(m["chamfer_mm"]) and cov == pytest.approx(m["coverage"])
    assert 0.0 < cd <= 1.0 and cov == pytest.approx(1.0)
