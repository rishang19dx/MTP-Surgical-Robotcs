"""mtp_gym: GPU-batched robotic-ultrasound environments with FEM soft tissue (DEC-041).

    import gymnasium as gym, mtp_gym
    envs = gym.make_vec("MTP/FreeScan-v0", num_envs=64, vectorization_mode="vector_entry_point",
                        patients=["data/sample_totalsegmentator_patient"])
"""

import gymnasium as gym

from mtp_gym.core import PatientSim, UltrasoundVectorEnv, load_patient
from mtp_gym.tasks import FreeScanTask, ReconstructionTask, Task

__all__ = ["UltrasoundVectorEnv", "PatientSim", "load_patient", "Task", "FreeScanTask", "ReconstructionTask"]


def _make_free_scan(num_envs: int = 1, **kwargs):
    return UltrasoundVectorEnv(num_envs, FreeScanTask(), **kwargs)


def _make_recon(num_envs: int = 1, task_kwargs=None, **kwargs):
    return UltrasoundVectorEnv(num_envs, ReconstructionTask(**(task_kwargs or {})), **kwargs)


def _register():
    for env_id, fn in (("MTP/FreeScan-v0", _make_free_scan), ("MTP/Recon-v0", _make_recon)):
        if env_id not in gym.registry:
            gym.register(id=env_id, vector_entry_point=fn)


_register()
