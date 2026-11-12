"""
Interleaved physics-rate comparison (DEC-052): MuJoCo and PyBullet at HEAD, and optionally PyBullet at a reference
commit (default: ec88541, the parent of the backend refactor), each measured with the server's own benchmark.

    ./.venv/bin/python scripts/physics_rate_compare.py [--runs 3] [--seconds 20] [--ref-commit ec88541|none]
                                                       [--max-load 1.0] [--label idle]

Each run is `unified_server.py --benchmark-seconds N --physics-backend B` (default mode orbital_l4, FEM tissue,
OpenIGTLink drain client), and its `[benchmark] json` line is collected. The runs are interleaved (mujoco, pybullet,
pybullet@ref, mujoco, ...) so slow drifts in machine load hit every configuration alike. The reference commit is run
from a temporary `git worktree` (removed afterwards) with this checkout's `.venv` and patient data.

Before each run the script waits until the 1-minute load average is below --max-load (or 120 s pass) and records
the load average, CPU governor and the CPU the physics process pins to. Writes
results/physics_rate_<stamp>_<label>.json.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")


def _git(*args, cwd=ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _load() -> list:
    return [float(v) for v in open("/proc/loadavg").read().split()[:3]]


def _governor() -> str:
    try:
        return open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except OSError:
        return "unknown"


def _wait_for_quiet(max_load: float, timeout_s: float = 120.0) -> list:
    t0 = time.time()
    while _load()[0] > max_load and time.time() - t0 < timeout_s:
        time.sleep(5)
    return _load()


def _run(tree: str, backend: str, seconds: float, port: int) -> dict:
    cmd = [PY, os.path.join(tree, "surgical_physics_bridge", "src", "unified_server.py"),
           "--benchmark-seconds", str(seconds), "--port", str(port)]
    if backend != "pybullet-legacy":
        cmd += ["--physics-backend", backend]
    env = dict(os.environ, PYTHONPATH=tree)
    out = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=seconds + 600)
    for line in out.stdout.splitlines():
        if line.startswith("[benchmark] json "):
            return json.loads(line[len("[benchmark] json "):])
    raise RuntimeError(f"no benchmark line ({backend} in {tree}):\n{out.stdout[-1500:]}\n{out.stderr[-1500:]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--ref-commit", default="ec88541", help="'none' to skip the reference commit")
    ap.add_argument("--max-load", type=float, default=1.0)
    ap.add_argument("--label", default="compare")
    ap.add_argument("--port", type=int, default=18960)
    args = ap.parse_args(argv)

    head = _git("rev-parse", "--short", "HEAD")
    dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) != 0
    configs = [("mujoco", ROOT, head), ("pybullet", ROOT, head)]
    worktree = None
    if args.ref_commit != "none":
        worktree = tempfile.mkdtemp(prefix="mtp_rate_ref_")
        os.rmdir(worktree)
        subprocess.check_call(["git", "worktree", "add", "--detach", worktree, args.ref_commit], cwd=ROOT,
                              stdout=subprocess.DEVNULL)
        # untracked patient data (CT volumes, FEM cache) are shared with this checkout
        for rel in ("data", os.path.join("ReconstructionPipeline", ".fem_cache")):
            src, dst = os.path.join(ROOT, rel), os.path.join(worktree, rel)
            if os.path.exists(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                os.symlink(src, dst)
        configs.append(("pybullet-legacy", worktree, _git("rev-parse", "--short", args.ref_commit)))
    runs = []
    try:
        for i in range(args.runs):
            for backend, tree, commit in configs:
                before = _wait_for_quiet(args.max_load)
                r = _run(tree, backend, args.seconds, args.port)
                r.update(run=i + 1, config=backend if backend != "pybullet-legacy" else f"pybullet@{commit}",
                         commit=commit, loadavg_before=before, loadavg_after=_load())
                runs.append(r)
                print(f"run {i + 1} {r['config']:>18}: {r['achieved_hz']:7.1f} Hz, compute p99 "
                      f"{r['compute_p99_ms']:.3f} ms, overruns {r['overruns']}, stream {r['stream_hz']:.1f} Hz, "
                      f"load {before[0]:.2f}", flush=True)
    finally:
        if worktree:
            subprocess.call(["git", "worktree", "remove", "--force", worktree], cwd=ROOT)
    try:
        import mujoco
        mj = mujoco.__version__
    except Exception:
        mj = "unknown"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"script": "scripts/physics_rate_compare.py", "args": vars(args), "commit": head + ("-dirty" if dirty else ""),
            "date": _dt.date.today().isoformat(), "mujoco_version": mj, "governor": _governor(),
            "cpus": os.cpu_count(), "pinned_cpu": "highest-numbered (physics_common.pin_to_dedicated_cpu)",
            "protocol": f"{args.runs} interleaved runs per configuration, {args.seconds:.0f} s each, default server "
                        "mode orbital_l4, FEM tissue, OpenIGTLink drain client; each run waits for 1-min load "
                        f"< {args.max_load} (max 120 s)"}
    path = os.path.join(ROOT, "results", f"physics_rate_{stamp}_{args.label}.json")
    json.dump({**meta, "runs": runs}, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
