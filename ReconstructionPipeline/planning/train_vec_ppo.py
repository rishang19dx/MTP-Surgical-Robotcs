"""
PPO baselines on the batched gym (DEC-041, task T10) via Stable-Baselines3 and `mtp_gym.wrappers.SB3VecEnv`.

    ./.venv/bin/python ReconstructionPipeline/planning/train_vec_ppo.py --task recon --timesteps 200000 \
        --patients ../sample_totalsegmentator_patient,sonogym_s0015 --num-envs 32 --seed 0 [--deformation rigid]

--task recon | nav | force  ->  MTP/Recon-v0, MTP/Nav-v0, MTP/ForceScan-v0 (episodes of --max-steps steps).
--patients: names under data/patients (or ../sample_totalsegmentator_patient), or "train" / "test" from
data/splits.json.  Envs are split over patients in contiguous blocks.  A 4 GB GPU holds about 2-3 patients' FEM
fields; the full 9-patient training split needs a larger GPU.
--deformation aware | unaware | rigid: tissue/imaging condition during TRAINING (hypothesis H1 trains rigid and
evaluates aware).  --e-scale-range lo,hi randomises tissue stiffness per episode.

Outputs: checkpoints/vec_ppo/<run>.zip (gitignored) and checkpoints/vec_ppo/<run>.json (args, commit, wall time,
episode-return curve).  Evaluation is a separate script (evaluation/eval_vec_policies.py) that writes results/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import VecMonitor  # noqa: E402

from mtp_gym.core import PatientSim, UltrasoundVectorEnv  # noqa: E402
from mtp_gym.tasks import ForceScanTask, NavigationTask, ReconstructionTask  # noqa: E402
from mtp_gym.wrappers import SB3VecEnv  # noqa: E402
from ReconstructionPipeline.evaluation.compression_sweep import patient_dirs  # noqa: E402

TASKS = {"recon": ReconstructionTask, "nav": NavigationTask, "force": ForceScanTask}
DEFAULT_MAX_STEPS = {"recon": 72, "nav": 40, "force": 72}


def make_task(name: str):
    if name == "nav":
        return NavigationTask()
    return TASKS[name](target_coverage=0.90)


class CurveCallback(BaseCallback):
    """Records mean episode return / length every rollout (from VecMonitor's ep_info_buffer)."""

    def __init__(self):
        super().__init__()
        self.curve = []

    def _on_rollout_end(self) -> None:
        buf = self.model.ep_info_buffer
        if buf:
            self.curve.append({"timesteps": int(self.num_timesteps),
                               "ep_return": float(np.mean([e["r"] for e in buf])),
                               "ep_len": float(np.mean([e["l"] for e in buf]))})

    def _on_step(self) -> bool:
        return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--patients", default="../sample_totalsegmentator_patient,sonogym_s0015")
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--deformation", default="aware", choices=["aware", "unaware", "rigid", "kinematic"])
    ap.add_argument("--bmode-physics", default="depth", choices=["depth", "legacy"])
    ap.add_argument("--e-scale-range", default="")
    ap.add_argument("--augment", default="none", choices=["none", "fem", "warp", "imaging"],
                    help="training-time augmentation (DEC-048 thread A): fem preload, non-physical warp, imaging")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "checkpoints", "vec_ppo"))
    args = ap.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    patients = [PatientSim(p, dev) for p in patient_dirs(args.patients)]
    es = tuple(float(v) for v in args.e_scale_range.split(",")) if args.e_scale_range else None
    env = UltrasoundVectorEnv(args.num_envs, make_task(args.task), patients, deformation_mode=args.deformation,
                              bmode_physics=args.bmode_physics, max_steps=args.max_steps or DEFAULT_MAX_STEPS[args.task],
                              e_scale_range=es, seed=args.seed,
                              augment=None if args.augment == "none" else args.augment)
    venv = VecMonitor(SB3VecEnv(env))
    venv.seed(args.seed)
    model = PPO("MultiInputPolicy", venv, n_steps=args.n_steps, batch_size=args.batch_size, learning_rate=args.lr,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.0, device=dev, seed=args.seed, verbose=0)
    cb = CurveCallback()
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=False)
    wall = time.perf_counter() - t0

    run = f"{args.task}_{args.deformation}_{args.bmode_physics}_s{args.seed}" + \
        (f"_aug-{args.augment}" if args.augment != "none" else "") + (f"_{args.tag}" if args.tag else "")
    os.makedirs(args.out_dir, exist_ok=True)
    model.save(os.path.join(args.out_dir, run + ".zip"))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        commit += "-dirty" if subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) else ""
    except Exception:
        commit = "unknown"
    meta = {"run": run, "args": vars(args), "commit": commit, "wall_s": wall,
            "env_steps_per_s": args.timesteps / wall, "patients": [p.name for p in patients],
            "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", "curve": cb.curve}
    json.dump(meta, open(os.path.join(args.out_dir, run + ".json"), "w"), indent=2)
    last = cb.curve[-1] if cb.curve else {}
    print(f"{run}: {args.timesteps} steps in {wall / 60:.1f} min ({meta['env_steps_per_s']:.0f} steps/s); last mean "
          f"return {last.get('ep_return', float('nan')):.2f}, length {last.get('ep_len', float('nan')):.1f}")


if __name__ == "__main__":
    main()
