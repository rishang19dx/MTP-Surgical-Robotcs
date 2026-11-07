"""
Legacy vs depth B-mode physics on the benchmark raster, per patient (DEC-042 / DEC-043).

    ./.venv/bin/python ReconstructionPipeline/evaluation/bmode_physics_compare.py [--patients all|train|test|<names>]

For each patient and each `bmode_physics` in {legacy, depth}, the vector env (`MTP/Recon-v0`, `aware` mode, shared
moduli) follows the benchmark raster (3 sweep lines x 24 frames over x in [-15, 15], z in [-52, -12], the
single-env benchmark's L4 raster) from its first waypoint in `--envs` parallel envs (different speckle draws).
Reported per (patient, physics), mean over envs:
  * coverage@3 mm of the US-visible L4 surface (the task's coverage, raw back-projected points);
  * segmentation precision@2 mm: fraction of back-projected B-mode bone points within 2 mm of ANY bone surface in
    the volume (`_all_bone_surface`), the single env's Prec@2 definition; ground truth is used as a metric only;
  * points per frame (frames with force >= 1 N).
Writes results/bmode_physics_compare_<stamp>.json and .md.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtp_gym.core import IMAGE_SIZE, PHYSICAL_SIZE_MM, PatientSim, UltrasoundVectorEnv  # noqa: E402
from mtp_gym.perception import backproject, segment_bmode_batch  # noqa: E402
from mtp_gym.tasks import ReconstructionTask  # noqa: E402
from ReconstructionPipeline.evaluation.compression_sweep import patient_dirs  # noqa: E402
from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv  # noqa: E402
from ReconstructionPipeline.planning.heuristic_planner import HeuristicScanPlanner  # noqa: E402

SCALE = np.array([5.0, 5.0, np.radians(10.0), np.radians(15.0)])


def run(ps: PatientSim, physics: str, n_envs: int, seed: int, bone_tree: cKDTree):
    traj = HeuristicScanPlanner.generate_linear_raster_trajectory(x_range=(-15.0, 15.0), z_range=(-52.0, -12.0),
                                                                  num_sweep_lines=3, points_per_line=24)
    task = ReconstructionTask(target_coverage=1.01)
    env = UltrasoundVectorEnv(n_envs, task, [ps], max_steps=500, seed=seed, bmode_physics=physics)
    env.reset(seed=seed, options={"init_pose": traj[0]})
    pts_all = [[] for _ in range(n_envs)]
    frames = np.zeros(n_envs)
    for t in traj:
        a = np.clip((t - env.pose.cpu().numpy()) / SCALE, -1, 1).astype(np.float32)
        _, _, _, _, info = env.step(a)
        fr = env.last_frame
        has, y = segment_bmode_batch(fr.bmode, fr.confidence, task.seg)
        ok = (fr.force_n >= task.min_force)
        has &= ok[:, None]
        pts = backproject(has, y, fr.t_backproject, IMAGE_SIZE, PHYSICAL_SIZE_MM).cpu().numpy()
        frames += ok.cpu().numpy()
        for i in range(n_envs):
            p = pts[i][np.isfinite(pts[i, :, 0])]
            if len(p):
                pts_all[i].append(p)
    cov = info["coverage_ratio"].cpu().numpy()
    prec, ppf = [], []
    for i in range(n_envs):
        p = np.concatenate(pts_all[i]) if pts_all[i] else np.empty((0, 3))
        prec.append(float(np.mean(bone_tree.query(p, k=1)[0] <= 2.0)) if len(p) else float("nan"))
        ppf.append(len(p) / max(frames[i], 1))
    return {"coverage3": float(cov.mean()), "coverage3_std": float(cov.std()),
            "precision2": float(np.nanmean(prec)), "points_per_frame": float(np.mean(ppf))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", default="all")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for pdir in patient_dirs(args.patients):
        ps = PatientSim(pdir, dev)
        d = ps.data
        bone = UltrasoundReconstructionEnv._all_bone_surface(d["ct_volume"], d.get("label_volume"), d["origin_mm"],
                                                             d["spacing_mm"])
        tree = cKDTree(bone)
        row = {"patient": os.path.basename(pdir)}
        for phys in ("legacy", "depth"):
            row[phys] = run(ps, phys, args.envs, args.seed, tree)
        rows.append(row)
        print(f"{row['patient']:34s} cov legacy {row['legacy']['coverage3']:.3f} depth {row['depth']['coverage3']:.3f} | "
              f"prec@2 legacy {row['legacy']['precision2']:.3f} depth {row['depth']['precision2']:.3f} | pts/frame "
              f"{row['legacy']['points_per_frame']:.1f} / {row['depth']['points_per_frame']:.1f}", flush=True)
        del ps, tree
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        commit += "-dirty" if subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) else ""
    except Exception:
        commit = "unknown"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"script": "ReconstructionPipeline/evaluation/bmode_physics_compare.py", "args": vars(args),
            "commit": commit, "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
            "depth_physics": "DEC-042 (fat 0.48 / muscle 1.09 dB/cm/MHz, receiver noise before TGC) + DEC-043 "
                             "(fixed TGC 10.9 dB/cm, cap 81.75 dB)"}
    base = os.path.join(args.out_dir, f"bmode_physics_compare_{stamp}")
    json.dump({"meta": meta, "rows": rows}, open(base + ".json", "w"), indent=2)
    lines = [f"# Legacy vs depth B-mode physics on the benchmark raster ({stamp})", "",
             f"commit `{commit}` · {meta['gpu']} · {args.envs} envs per patient · 72-frame raster · aware mode · "
             "shared moduli", "", f"Depth physics: {meta['depth_physics']}.", "",
             "| Patient | Cov@3 legacy | Cov@3 depth | Prec@2 legacy | Prec@2 depth | Points/frame legacy | Points/frame depth |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        L, D = r["legacy"], r["depth"]
        lines.append(f"| {r['patient']} | {L['coverage3']:.3f} | {D['coverage3']:.3f} | {L['precision2']:.3f} | "
                     f"{D['precision2']:.3f} | {L['points_per_frame']:.1f} | {D['points_per_frame']:.1f} |")
    for k in ("legacy", "depth"):
        c = [r[k]["coverage3"] for r in rows]
        p = [r[k]["precision2"] for r in rows]
        lines.append(f"| **mean {k}** | {np.mean(c):.3f} | | {np.nanmean(p):.3f} | | | |")
    open(base + ".md", "w").write("\n".join(lines) + "\n")
    print("wrote", base + ".json / .md")


if __name__ == "__main__":
    main()
