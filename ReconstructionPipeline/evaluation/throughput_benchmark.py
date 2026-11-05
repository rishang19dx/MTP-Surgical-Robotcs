"""
Throughput of the batched vector env (DEC-041, task T8).  Writes results/throughput_<stamp>.json and .md.

    ./.venv/bin/python ReconstructionPipeline/evaluation/throughput_benchmark.py \
        --patient-dir data/sample_totalsegmentator_patient --num-envs 1,8,32,128 --modes rigid,aware

Measures env-steps per second of `UltrasoundVectorEnv.step` with the zero-reward FreeScan task, so the time
is the simulation itself: admittance force control (2 FEM force solves), the tissue solve with field
(deformable modes), the backward-warped slice, B-mode, coupling and confidence.  Random actions in [-1, 1].
Rigid mode still runs the force controller (force is simulated; only the image is undeformed).
"""

from __future__ import annotations

import argparse
import datetime as _dt
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

from mtp_gym.core import PatientSim, UltrasoundVectorEnv  # noqa: E402
from mtp_gym.tasks import FreeScanTask  # noqa: E402


def _commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def measure(patient: PatientSim, n: int, mode: str, steps: int, warmup: int, seed: int) -> dict:
    dev = patient.device
    env = UltrasoundVectorEnv(n, FreeScanTask(), [patient], deformation_mode=mode, max_steps=10 ** 9, seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    acts = rng.uniform(-1, 1, (warmup + steps, n, 4)).astype(np.float32)
    for k in range(warmup):
        env.step(acts[k])
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    forces = []
    for k in range(warmup, warmup + steps):
        _, _, _, _, info = env.step(acts[k])
        forces.append(info["normal_force"])
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t
    f = torch.stack(forces).float()
    return {
        "num_envs": n, "mode": mode, "steps": steps, "seconds": dt,
        "ms_per_step": 1e3 * dt / steps, "env_steps_per_s": n * steps / dt,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1e6 if dev.type == "cuda" else None,
        "force_mean_n": float(f.mean()), "force_std_n": float(f.std()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient-dir", default=os.path.join(ROOT, "data", "sample_totalsegmentator_patient"))
    ap.add_argument("--num-envs", default="1,8,32,128")
    ap.add_argument("--modes", default="rigid,aware")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient = PatientSim(args.patient_dir, dev)
    rows = []
    for mode in args.modes.split(","):
        for n in (int(v) for v in args.num_envs.split(",")):
            r = measure(patient, n, mode, args.steps, args.warmup, args.seed)
            rows.append(r)
            print(f"{mode:7s} N={n:4d}: {r['ms_per_step']:8.1f} ms/step  {r['env_steps_per_s']:8.1f} env-steps/s  "
                  f"peak {r['peak_vram_mb'] or float('nan'):.0f} MB  F {r['force_mean_n']:.2f}±{r['force_std_n']:.2f} N",
                  flush=True)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "script": "ReconstructionPipeline/evaluation/throughput_benchmark.py", "args": vars(args),
        "commit": _commit(), "timestamp": stamp, "device": str(dev),
        "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else None,
        "torch": torch.__version__, "patient": os.path.relpath(args.patient_dir, ROOT),
        "task": "FreeScan (zero reward; simulation cost only)",
    }
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"throughput_{stamp}")
    with open(base + ".json", "w") as fh:
        json.dump({"meta": meta, "rows": rows}, fh, indent=2)
    lines = [f"# Vector-env throughput ({stamp})", "",
             f"GPU: {meta['gpu']} · commit `{meta['commit']}` · torch {meta['torch']} · patient `{meta['patient']}`",
             f"Command: `./.venv/bin/python {meta['script']} --num-envs {args.num_envs} --modes {args.modes} "
             f"--steps {args.steps}`", "",
             "| Mode | Envs | ms / step | env-steps / s | Peak VRAM (MB) |", "| :--- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['num_envs']} | {r['ms_per_step']:.1f} | {r['env_steps_per_s']:.1f} | "
                     f"{r['peak_vram_mb']:.0f} |" if r["peak_vram_mb"] is not None else
                     f"| {r['mode']} | {r['num_envs']} | {r['ms_per_step']:.1f} | {r['env_steps_per_s']:.1f} | - |")
    with open(base + ".md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {base}.json / .md")


if __name__ == "__main__":
    main()
