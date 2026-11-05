"""Batched reconstruction task on the vector env (DEC-041, task T5).

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_vec_reconstruction.py -q -s
"""
import os

import numpy as np
import pytest
import torch

from mtp_gym.core import PatientSim, UltrasoundVectorEnv
from mtp_gym.perception import segment_bmode_batch
from mtp_gym.tasks import FreeScanTask, ReconstructionTask

SUBJECT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_totalsegmentator_patient")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytestmark = pytest.mark.skipif(not os.path.isdir(SUBJECT), reason="sample patient not available")


@pytest.fixture(scope="module")
def patient():
    return PatientSim(SUBJECT, DEV)


def _lawnmower(env, rows=6, cols=10):
    """Raster over the window: start at the low corner, sweep +c1, step +c2, sweep -c1, ..."""
    b = env.bounds[0].tolist()
    start = [b[0] + 2.0, b[2] + 2.0, 0.0, 0.0]
    acts = []
    for r in range(rows):
        acts += [[1.0 if r % 2 == 0 else -1.0, 0.0, 0.0, 0.0]] * cols
        acts.append([0.0, 1.0, 0.0, 0.0])
    return start, np.array(acts, dtype=np.float32)


def test_batched_segmenter_equals_numpy(patient):
    task = ReconstructionTask()
    env = UltrasoundVectorEnv(16, FreeScanTask(), [patient], seed=0)
    env.reset(seed=0)
    hits = 0
    for _ in range(3):
        env.step(np.random.default_rng(1).uniform(-1, 1, (16, 4)).astype(np.float32))
        fr = env.last_frame
        has, y = segment_bmode_batch(fr.bmode, fr.confidence, task.seg)
        for i in range(16):
            ref = task.seg.segment_from_bmode(fr.bmode[i].cpu().numpy(), confidence_map=fr.confidence[i].cpu().numpy())
            xs = has[i].nonzero()[:, 0].cpu().numpy()
            got = np.stack([xs, y[i, xs].cpu().numpy()], 1) if len(xs) else np.empty((0, 2))
            np.testing.assert_array_equal(got, ref)
            hits += len(ref)
    assert hits > 100            # the comparison is not vacuous


def test_raster_reaches_coverage_and_reward_is_area(patient):
    env = UltrasoundVectorEnv(2, ReconstructionTask(target_coverage=1.01), [patient], max_steps=500, seed=2)
    start, acts = _lawnmower(env)
    obs, _ = env.reset(seed=2, options={"init_pose": start})
    cov_hist, gains = [], []
    for a in acts:
        obs, r, term, trunc, info = env.step(np.repeat(a[None], 2, 0))
        cov_hist.append(info["coverage_ratio"].cpu().numpy())
        gains.append(info["marginal_gain"].cpu().numpy())
    cov = np.array(cov_hist)
    assert np.all(np.diff(cov, axis=0) >= -1e-6)                          # monotone within an episode
    assert cov[-1].min() > 0.5, cov[-1]                                   # a raster covers most of the L4 surface
    total = env.task._area(torch.arange(2, device=DEV)).cpu().numpy()
    first = env.task.prev_area.cpu().numpy() * 0                          # prev_area starts at 0 after reset
    np.testing.assert_allclose(np.sum(gains, 0) + first, total, rtol=1e-5)
    assert float(obs["recon_grid"].sum()) > 0
    print(f"\n[recon] raster final coverage {cov[-1]}")


def test_observations_do_not_use_ground_truth(patient):
    """Invariant 5: moving the ground-truth target changes coverage (reward) but not a single observation."""
    def run(shift):
        task = ReconstructionTask(target_coverage=1.01)
        env = UltrasoundVectorEnv(2, task, [patient], seed=3)
        task.targets = [t + shift for t in task.targets]
        start, acts = _lawnmower(env, rows=2, cols=6)
        obs, _ = env.reset(seed=3, options={"init_pose": start})
        out = []
        for a in acts:
            obs, r, term, trunc, info = env.step(np.repeat(a[None], 2, 0))
            out.append(({k: v.clone() for k, v in obs.items() if k != "coverage"}, info["coverage_ratio"].clone()))
        return out

    a, b = run(0.0), run(100.0)
    for (oa, ca), (ob, cb) in zip(a, b):
        for k in oa:
            assert torch.equal(oa[k], ob[k]), k
    assert float(a[-1][1].max()) > 0.1 and float(b[-1][1].max()) == 0.0


def test_unaware_mode_backprojects_from_skin(patient):
    env = UltrasoundVectorEnv(1, ReconstructionTask(), [patient], deformation_mode="unaware", seed=4)
    env.reset(seed=4)
    fr = env.last_frame
    assert torch.equal(fr.t_backproject, fr.t_skin) and not torch.equal(fr.t_img, fr.t_skin)


def test_raster_coverage_matches_single_env(patient):
    """The benchmark raster (72 frames) gives the same final coverage in the vector env as in env_reconstruction,
    within 5 points (different speckle draws; vector coverage uses raw points, the single env voxel means)."""
    from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
    from ReconstructionPipeline.planning.heuristic_planner import HeuristicScanPlanner
    traj = HeuristicScanPlanner.generate_linear_raster_trajectory(x_range=(-15.0, 15.0), z_range=(-52.0, -12.0),
                                                                  num_sweep_lines=3, points_per_line=24)
    single = UltrasoundReconstructionEnv(patient_data=patient.data, tissue_model=patient.model, device=DEV, seed=5,
                                         max_steps=200, target_coverage=1.01)
    vec = UltrasoundVectorEnv(1, ReconstructionTask(target_coverage=1.01), [patient], max_steps=200, seed=5)
    single.reset(options={"init_pose": traj[0]})
    vec.reset(seed=5, options={"init_pose": traj[0]})
    scale = np.array([5.0, 5.0, np.radians(10.0), np.radians(15.0)])
    for tgt in traj:
        a_s = np.clip((tgt - single.current_pose_params) / scale, -1, 1).astype(np.float32)
        _, _, _, _, info_s = single.step(a_s)
        a_v = np.clip((tgt - vec.pose[0].cpu().numpy()) / scale, -1, 1).astype(np.float32)
        _, _, _, _, info_v = vec.step(a_v[None])
    cs, cv = info_s["coverage_ratio"], float(info_v["coverage_ratio"][0])
    print(f"\n[recon] raster coverage single {cs:.3f} vs vector {cv:.3f}")
    assert abs(cs - cv) < 0.05
