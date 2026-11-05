"""
Tests for the EDM diffusion / CFM policies and their behaviour-cloning pipeline (dataset, episode split,
demo save/load with logged clouds, normalisation buffers, randomised raster expert).
"""

import numpy as np
import pytest
import torch

from ReconstructionPipeline.planning.diffusion_policy import SOTADiffusionPolicy
from ReconstructionPipeline.planning.flow_matching_policy import FlowMatchingPolicy
from ReconstructionPipeline.planning.train_diffusion_policy import (
    UltrasoundDemonstrationDataset,
    load_demos,
    randomized_raster,
    save_demos,
    split_episodes,
)


def _small_policy():
    return SOTADiffusionPolicy(action_dim=4, horizon=8, pointnet_latent_dim=64, pose_dim=4, hidden_dim=128,
                               num_layers=2, device=torch.device("cpu"))


def _episode(T=10, n=40):
    return {"points": [np.random.randn(n + t, 3).astype(np.float32) for t in range(T)],
            "poses": np.random.randn(T, 4).astype(np.float32),
            "actions": np.random.uniform(-1, 1, (T, 4)).astype(np.float32), "coverage": 0.5}


def test_diffusion_policy_loss_and_backprop():
    policy = _small_policy()
    loss = policy.compute_loss(torch.randn(2, 8, 4), torch.randn(2, 30, 3), torch.randn(2, 4))
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    for p in policy.parameters():
        if p.requires_grad:
            assert p.grad is not None and not torch.isnan(p.grad).any()


@pytest.mark.parametrize("method,steps", [("heun", 5), ("euler", 3), ("consistency", 1)])
def test_diffusion_policy_sampling_methods(method, steps):
    a = _small_policy().sample_actions(torch.randn(2, 25, 3), torch.randn(2, 4), num_steps=steps, method=method)
    assert a.shape == (2, 8, 4) and (a.abs() <= 1.0).all()


def test_normalisation_buffers_are_saved_and_used():
    policy = _small_policy()
    sd = policy.state_dict()
    for k in ("pose_center", "pose_scale", "pointnet.point_center", "pointnet.point_scale"):
        assert k in sd
    cfm = FlowMatchingPolicy(action_dim=4, chunk_horizon=8, condition_dim=132, device=torch.device("cpu"))
    cfm.set_pose_normalization([0.0, -32.5, 0.0, 0.0], [32.0, 22.5, 0.8, 0.6])
    assert torch.allclose(cfm.state_dict()["cond_center"][-4:], torch.tensor([0.0, -32.5, 0.0, 0.0]))
    assert torch.all(cfm.state_dict()["cond_scale"][:-4] == 1.0)


def test_dataset_chunks_pad_with_last_action():
    ep = _episode(T=5)
    ds = UltrasoundDemonstrationDataset([ep], horizon=8, max_points=32)
    assert len(ds) == 5
    s = ds[4]
    assert s["points"].shape == (32, 3) and s["chunk"].shape == (8, 4)
    assert torch.allclose(s["chunk"], torch.tensor(ep["actions"][4]).expand(8, 4))
    val = UltrasoundDemonstrationDataset([ep], horizon=8, max_points=32, deterministic=True)
    assert torch.equal(val[2]["points"], val[2]["points"])


def test_split_is_by_episode():
    eps = [dict(_episode(), coverage=float(i)) for i in range(20)]
    train, val = split_episodes(eps, 0.15, seed=0)
    assert len(val) == 3 and len(train) == 17
    assert not {e["coverage"] for e in train} & {e["coverage"] for e in val}


def test_demo_save_load_roundtrip_and_requires_clouds(tmp_path):
    eps = [_episode(T=4), _episode(T=6)]
    path = str(tmp_path / "demos.npz")
    save_demos(path, eps, {"expert": "test"})
    loaded, meta = load_demos(path)
    assert meta["expert"] == "test" and len(loaded) == 2
    for a, b in zip(eps, loaded):
        assert len(b["points"]) == len(a["points"])
        assert all(np.allclose(p, q) for p, q in zip(a["points"], b["points"]))
    bad = str(tmp_path / "no_points.npz")
    np.savez(bad, episodes=np.zeros(1))
    with pytest.raises(ValueError):
        load_demos(bad)


class _WindowEnv:
    is_l4_target = True


def test_randomized_raster_varies_and_starts_near_start_pose():
    rng = np.random.default_rng(0)
    trajs = [randomized_raster(_WindowEnv(), np.array([25.0, -12.0, 0, 0]), rng, 72) for _ in range(5)]
    assert all(len(t) >= 72 for t in trajs)
    assert len({round(float(t[0, 0]), 3) for t in trajs}) > 1           # randomised extent
    for t in trajs:
        assert t[0, 0] > 0 and t[0, 1] > -20                             # nearest corner to (25, -12)
