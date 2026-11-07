"""Navigation-to-plane task (DEC-041, task T6).

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_vec_navigation.py -q -s
Acceptance (tasks/todo.md T6): a scripted oracle succeeds in > 95 % of episodes, random actions in < 10 %.
"""
import os

import gymnasium as gym
import numpy as np
import pytest
import torch

import mtp_gym  # noqa: F401  (registers MTP/Nav-v0)
from mtp_gym.core import PatientSim, UltrasoundVectorEnv
from mtp_gym.tasks import NavigationTask
from mtp_gym.tasks.reconstruction import target_surface

SUBJECT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_totalsegmentator_patient")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytestmark = pytest.mark.skipif(not os.path.isdir(SUBJECT), reason="sample patient not available")
N, MAX_STEPS = 64, 40


@pytest.fixture(scope="module")
def patient():
    return PatientSim(SUBJECT, DEV)


def _run(env, policy):
    """One episode per env (no autoreset bookkeeping needed: count the first terminal flag of each env)."""
    env.reset(seed=0)
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=DEV)
    success = torch.zeros_like(done)
    steps = torch.zeros(env.num_envs, device=DEV)
    for _ in range(MAX_STEPS):
        a = policy(env)
        _, r, term, trunc, info = env.step(a)
        success |= ~done & info["success"]
        steps += (~done).float()
        done |= term | trunc
        if bool(done.all()):
            break
    return success.float().mean().item(), steps


def test_goal_is_l4_transverse_plane(patient):
    env = UltrasoundVectorEnv(2, NavigationTask(), [patient], seed=0)
    t = target_surface(patient.data, patient.is_l4)
    g = env.task.goal[0].cpu().numpy()
    np.testing.assert_allclose(g[:2], [t[:, 0].mean(), t[:, 2].mean()], atol=1e-4)
    assert g[2] == 0.0 and g[3] == 0.0
    assert env.single_action_space.shape == (3,)


def test_oracle_succeeds_random_does_not(patient):
    env = UltrasoundVectorEnv(N, NavigationTask(), [patient], max_steps=MAX_STEPS, seed=0)
    sr_oracle, steps = _run(env, lambda e: e.task.oracle_action())
    rng = np.random.default_rng(0)
    sr_random, _ = _run(env, lambda e: torch.as_tensor(rng.uniform(-1, 1, (N, 3)), dtype=torch.float32))
    print(f"\n[nav] success oracle {sr_oracle:.3f} (mean steps {steps.mean():.1f}), random {sr_random:.3f}")
    assert sr_oracle > 0.95
    assert sr_random < 0.10


def test_progress_reward_and_no_goal_in_observation(patient):
    env = UltrasoundVectorEnv(4, NavigationTask(), [patient], seed=1)
    obs, _ = env.reset(seed=1)
    assert set(obs) == {"bmode", "probe_pose", "force"}            # the goal is not observed
    _, r, term, _, info = env.step(env.task.oracle_action())
    assert bool((r[~term] > 0).all())                              # moving toward the goal is rewarded
    # moving away is penalised
    env.reset(seed=1)
    _, r, _, _, _ = env.step(-env.task.oracle_action())
    assert bool((r < 0.5).all())


def test_make_vec_nav(patient):
    envs = gym.make_vec("MTP/Nav-v0", num_envs=2, vectorization_mode="vector_entry_point", patients=[patient], seed=0)
    envs.reset(seed=0)
    envs.step(envs.action_space.sample())
