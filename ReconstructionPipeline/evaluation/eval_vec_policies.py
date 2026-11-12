"""
Evaluate policies on the batched gym, per patient (DEC-041, task T10).  Writes results/vec_eval_<task>_<stamp>.*.

    ./.venv/bin/python ReconstructionPipeline/evaluation/eval_vec_policies.py --task recon \
        --patients test --episodes 16 --ppo checkpoints/vec_ppo/recon_aware_depth_s0.zip [--deformation aware]

Every policy runs exactly one episode in each of --episodes envs per patient (the env's own random start poses,
seeded: the same starts for every policy).  Policies:
  random   uniform actions in [-1, 1]
  scripted recon / force: the benchmark raster (3 sweep lines x 24 waypoints, followed one waypoint per step from
           the random start, as `benchmark_runner.follow`); force task at a constant 4 N
  oracle   nav only: privileged goal-seeking expert (upper bound, uses the label-defined goal and the true pose)
  blind    nav only: drives to the patient's goal pose as seen in the OBSERVED pose frame, ignoring the image
           (DEC-045: must fail when the hidden placement offset is on; if it succeeds the task is solvable blind)
  ppo:<file> one entry per --ppo checkpoint (deterministic actions)
Metrics (at each env's first episode end):
  recon / force: coverage@3 (task coverage), mean sensed force, episode return; force task also mean commanded
  force and mean bone visibility;  nav: success rate, steps, final goal distance (mm) and yaw error (deg).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtp_gym.core import PatientSim, UltrasoundVectorEnv  # noqa: E402
from mtp_gym.wrappers import SB3VecEnv  # noqa: E402
from ReconstructionPipeline.evaluation.compression_sweep import patient_dirs  # noqa: E402
from ReconstructionPipeline.planning.heuristic_planner import HeuristicScanPlanner  # noqa: E402
from ReconstructionPipeline.planning.train_vec_ppo import DEFAULT_MAX_STEPS, make_task  # noqa: E402

RASTER = HeuristicScanPlanner.generate_linear_raster_trajectory(x_range=(-15.0, 15.0), z_range=(-52.0, -12.0),
                                                               num_sweep_lines=3, points_per_line=24)


def scripted_action(env, task_name: str, step: int) -> np.ndarray:
    wp = RASTER[min(step, len(RASTER) - 1)]
    a = ((torch.as_tensor(wp, device=env.device) - env.pose) / env.action_scale).clamp(-1, 1).cpu().numpy()
    if task_name == "force":
        f = np.full((env.num_envs, 1), env.task.force_action(4.0), np.float32)
        a = np.concatenate([a, f], 1)
    return a.astype(np.float32)


def run_policy(env: UltrasoundVectorEnv, venv: SB3VecEnv, task_name: str, policy: str, model, seed: int):
    n = env.num_envs
    obs, _ = env.reset(seed=seed)
    obs_sb = venv._convert(obs)
    rng = np.random.default_rng(seed)
    done = np.zeros(n, bool)
    rec = {k: np.full(n, np.nan) for k in ("coverage", "force_mean", "return", "success", "steps", "dist", "yaw_err",
                                           "target_force_mean", "visibility_mean")}
    ret, fsum, tfsum, vsum, cnt = (np.zeros(n) for _ in range(5))
    for t in range(env.max_steps):
        if policy == "random":
            a = rng.uniform(-1, 1, (n, env.task.action_dim)).astype(np.float32)
        elif policy == "scripted":
            a = scripted_action(env, task_name, t)
        elif policy == "oracle":
            a = env.task.oracle_action().cpu().numpy()
        elif policy == "blind":
            observed = env.pose - env.pose_offset
            a = ((env.task.goal - observed) / env.action_scale)[:, :3].clamp(-1, 1).cpu().numpy()
        else:
            a, _ = model.predict(obs_sb, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        obs_sb = venv._convert(obs)
        live = ~done
        r_np = r.cpu().numpy()
        ret[live] += r_np[live]
        fsum[live] += info["normal_force"].cpu().numpy()[live]
        if "target_force_n" in info:
            tfsum[live] += info["target_force_n"].cpu().numpy()[live]
            vsum[live] += info["bone_visibility"].cpu().numpy()[live]
        cnt[live] += 1
        end = live & (term | trunc).cpu().numpy()
        for i in np.nonzero(end)[0]:
            rec["return"][i], rec["steps"][i] = ret[i], cnt[i]
            rec["force_mean"][i] = fsum[i] / cnt[i]
            if "coverage_ratio" in info:
                rec["coverage"][i] = float(info["coverage_ratio"][i])
            if "success" in info:
                rec["success"][i] = float(info["success"][i])
                rec["dist"][i] = float(info["goal_distance_mm"][i])
                rec["yaw_err"][i] = float(info["goal_yaw_error_deg"][i])
            if "target_force_n" in info:
                rec["target_force_mean"][i] = tfsum[i] / cnt[i]
                rec["visibility_mean"][i] = vsum[i] / cnt[i]
        done |= end
        if done.all():
            break
    return {k: (float(np.nanmean(v)) if np.isfinite(v).any() else None) for k, v in rec.items()} | {
        "coverage_std": float(np.nanstd(rec["coverage"])) if np.isfinite(rec["coverage"]).any() else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["recon", "nav", "force"], required=True)
    ap.add_argument("--patients", default="test")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--ppo", nargs="*", default=[])
    ap.add_argument("--deformation", default="aware", choices=["aware", "unaware", "rigid", "kinematic"])
    ap.add_argument("--bmode-physics", default="depth", choices=["depth", "legacy"])
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from stable_baselines3 import PPO
    models = {f"ppo:{os.path.basename(p)}": PPO.load(p, device=dev) for p in args.ppo}
    policies = ["random", "scripted"] if args.task != "nav" else ["random", "blind", "oracle"]
    policies += list(models)
    rows = []
    for pdir in patient_dirs(args.patients):
        ps = PatientSim(pdir, dev)
        for pol in policies:
            env = UltrasoundVectorEnv(args.episodes, make_task(args.task), [ps], deformation_mode=args.deformation,
                                      bmode_physics=args.bmode_physics, max_steps=DEFAULT_MAX_STEPS[args.task],
                                      seed=args.seed)
            venv = SB3VecEnv(env)
            m = run_policy(env, venv, args.task, pol, models.get(pol), args.seed)
            rows.append({"patient": ps.name.split("/")[-1], "policy": pol, **m})
            key = "success" if args.task == "nav" else "coverage"
            print(f"{rows[-1]['patient']:34s} {pol:40s} {key} {m[key]:.3f}  force {m['force_mean']:.2f} N  "
                  f"return {m['return']:.2f}", flush=True)
            del env, venv
        del ps
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        commit += "-dirty" if subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) else ""
    except Exception:
        commit = "unknown"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"script": "ReconstructionPipeline/evaluation/eval_vec_policies.py", "args": vars(args), "commit": commit,
            "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"}
    base = os.path.join(args.out_dir, f"vec_eval_{args.task}_{stamp}")
    json.dump({"meta": meta, "rows": rows}, open(base + ".json", "w"), indent=2)
    key = "success" if args.task == "nav" else "coverage"
    L = [f"# Vector-env evaluation: {args.task} ({stamp})", "", f"commit `{commit}` · {meta['gpu']} · patients "
         f"`{args.patients}` · {args.episodes} episodes per patient · eval deformation `{args.deformation}` · B-mode "
         f"`{args.bmode_physics}` · seed {args.seed}", "",
         f"| Policy | Mean {key} | Mean force (N) | Mean return |" + (" Final pos. error (mm) | Final yaw error (deg) |"
                                                                    if args.task == "nav" else ""),
         "| :--- | ---: | ---: | ---: |" + (" ---: | ---: |" if args.task == "nav" else "")]
    for pol in policies:
        rs = [r for r in rows if r["policy"] == pol]
        line = (f"| {pol} | {np.mean([r[key] for r in rs]):.3f} | {np.mean([r['force_mean'] for r in rs]):.2f} | "
                f"{np.mean([r['return'] for r in rs]):.2f} |")
        if args.task == "nav":   # SonoGym reports final position / rotation error for navigation
            line += f" {np.mean([r['dist'] for r in rs]):.1f} | {np.mean([r['yaw_err'] for r in rs]):.1f} |"
        L.append(line)
    L += ["", "| Patient | Policy | " + key + " | force (N) | return |", "| :--- | :--- | ---: | ---: | ---: |"]
    for r in rows:
        L.append(f"| {r['patient']} | {r['policy']} | {r[key]:.3f} | {r['force_mean']:.2f} | {r['return']:.2f} |")
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("wrote", base + ".json / .md")


if __name__ == "__main__":
    main()
