"""
Re-tune image-only perception for the depth B-mode on the training patients (DEC-044; protocol fixed before running).

    ./.venv/bin/python ReconstructionPipeline/evaluation/tune_perception.py

1. Record, per patient, one benchmark-raster episode (72 frames, `aware`, depth physics, seed 0): B-mode,
   back-projection pose and force.
2. Score the 54-point grid (DEC-044) on the TRAIN patients only.  Metrics: Cov@3 of the US-visible L4 target
   (raw back-projected points) and Prec@2 against every bone surface; frames below 1 N are excluded.
3. Select: max mean train Cov@3 s.t. mean train Prec@2 >= 0.95 and every train patient's Prec@2 >= 0.90
   (no points = satisfied); ties within 0.005 -> fewest values changed from the defaults.
4. Only then score the TEST patients, for the defaults and the selected configuration.
Writes results/perception_tuning_<stamp>.json / .md and prints the selected parameters.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import itertools
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
from mtp_gym.tasks import FreeScanTask  # noqa: E402
from mtp_gym.tasks.reconstruction import target_surface  # noqa: E402
from ReconstructionPipeline.evaluation.compression_sweep import patient_dirs  # noqa: E402
from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv  # noqa: E402
from ReconstructionPipeline.planning.heuristic_planner import HeuristicScanPlanner  # noqa: E402
from ReconstructionPipeline.reconstruction.bone_segmentation import BoneSurfaceSegmenter  # noqa: E402
from ReconstructionPipeline.simulation.confidence_map import UltrasoundConfidenceEstimator  # noqa: E402

AXIAL_MM = PHYSICAL_SIZE_MM[0] / (IMAGE_SIZE[0] - 1)
SCALE = np.array([5.0, 5.0, np.radians(10.0), np.radians(15.0)])
DEFAULTS = {"shadow_threshold": 0.70, "attenuation_decay_rate": 1.6, "intensity_threshold": 0.75,
            "min_confidence": 0.20}
GRID = {"shadow_threshold": [0.70, 0.80, 0.90], "attenuation_decay_rate": [1.6, 0.8, 0.4],
        "intensity_threshold": [0.75, 0.85], "min_confidence": [0.20, 0.10, 0.05]}


@torch.no_grad()
def record(pdir: str, dev, seed: int = 0):
    ps = PatientSim(pdir, dev)
    traj = HeuristicScanPlanner.generate_linear_raster_trajectory(x_range=(-15.0, 15.0), z_range=(-52.0, -12.0),
                                                                  num_sweep_lines=3, points_per_line=24)
    env = UltrasoundVectorEnv(1, FreeScanTask(), [ps], max_steps=500, seed=seed, bmode_physics="depth")
    env.reset(seed=seed, options={"init_pose": traj[0]})
    bm, T, F = [], [], []
    for t in traj:
        a = np.clip((t - env.pose.cpu().numpy()) / SCALE, -1, 1).astype(np.float32)
        env.step(a)
        fr = env.last_frame
        bm.append(fr.bmode[0].clone())
        T.append(fr.t_backproject[0].clone())
        F.append(float(fr.force_n[0]))
    d = ps.data
    targets = target_surface(d, ps.is_l4)
    bone = UltrasoundReconstructionEnv._all_bone_surface(d["ct_volume"], d.get("label_volume"), d["origin_mm"],
                                                         d["spacing_mm"])
    out = {"bmode": torch.stack(bm), "T": torch.stack(T), "force": np.array(F),
           "target_tree": cKDTree(targets), "n_targets": len(targets), "bone_tree": cKDTree(bone)}
    del env, ps
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def score(rec, cfg, dev):
    conf = UltrasoundConfidenceEstimator(device=dev, axial_resolution_mm=AXIAL_MM,
                                         shadow_threshold=cfg["shadow_threshold"],
                                         attenuation_decay_rate=cfg["attenuation_decay_rate"]
                                         ).compute_confidence_map(rec["bmode"])
    seg = BoneSurfaceSegmenter(axial_resolution_mm=AXIAL_MM, intensity_threshold=cfg["intensity_threshold"],
                               min_confidence=cfg["min_confidence"])
    has, y = segment_bmode_batch(rec["bmode"], conf, seg)
    has &= torch.as_tensor(rec["force"] >= 1.0, device=has.device)[:, None]
    pts = backproject(has, y, rec["T"], IMAGE_SIZE, PHYSICAL_SIZE_MM).cpu().numpy().reshape(-1, 3)
    pts = pts[np.isfinite(pts[:, 0])]
    if len(pts) == 0:
        return {"cov3": 0.0, "prec2": float("nan"), "points": 0}
    d_t, _ = cKDTree(pts).query(rec["target_tree"].data, k=1)
    covered = d_t <= 3.0
    d_b, _ = rec["bone_tree"].query(pts, k=1)
    return {"cov3": float(covered.mean()), "prec2": float(np.mean(d_b <= 2.0)), "points": int(len(pts))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.load(open(os.path.join(ROOT, "data", "splits.json")))
    train_dirs, test_dirs = patient_dirs("train"), patient_dirs("test")
    name = lambda p: os.path.basename(p)  # noqa: E731

    train = {name(p): record(p, dev) for p in train_dirs}
    print(f"recorded {len(train)} train patients", flush=True)
    keys = list(GRID)
    configs = [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]
    table = []
    for cfg in configs:
        per = {n: score(r, cfg, dev) for n, r in train.items()}
        cov = float(np.mean([v["cov3"] for v in per.values()]))
        precs = [v["prec2"] for v in per.values() if np.isfinite(v["prec2"])]
        mprec = float(np.mean(precs)) if precs else float("nan")
        ok = bool(precs) and mprec >= 0.95 and min(precs) >= 0.90
        changed = sum(cfg[k] != DEFAULTS[k] for k in keys)
        table.append({"config": cfg, "train_cov3": cov, "train_prec2": mprec, "train_min_prec2": min(precs) if precs
                      else float("nan"), "feasible": ok, "changed": changed, "per_patient": per})
    feas = [t for t in table if t["feasible"]]
    best_cov = max(t["train_cov3"] for t in feas)
    near = [t for t in feas if t["train_cov3"] >= best_cov - 0.005]
    chosen = min(near, key=lambda t: (t["changed"], -t["train_cov3"]))
    default_row = next(t for t in table if t["config"] == DEFAULTS)
    print(f"defaults: train Cov@3 {default_row['train_cov3']:.3f} Prec@2 {default_row['train_prec2']:.3f}")
    print(f"selected {chosen['config']}: train Cov@3 {chosen['train_cov3']:.3f} Prec@2 {chosen['train_prec2']:.3f}",
          flush=True)
    del train
    gc.collect()

    # --- held-out patients, scored only now
    test = {}
    for p in test_dirs:
        rec = record(p, dev)
        test[name(p)] = {"defaults": score(rec, DEFAULTS, dev), "selected": score(rec, chosen["config"], dev)}
        del rec
        gc.collect()
    for k in ("defaults", "selected"):
        c = np.mean([v[k]["cov3"] for v in test.values()])
        pr = np.nanmean([v[k]["prec2"] for v in test.values()])
        print(f"TEST {k}: Cov@3 {c:.3f} Prec@2 {pr:.3f}  " +
              " ".join(f"{n}={v[k]['cov3']:.2f}" for n, v in test.items()))

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        commit += "-dirty" if subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) else ""
    except Exception:
        commit = "unknown"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"script": "ReconstructionPipeline/evaluation/tune_perception.py", "commit": commit,
            "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", "protocol": "DEC-044",
            "train": split["train"], "test": split["test"], "grid": GRID, "defaults": DEFAULTS}
    out = {"meta": meta, "selected": chosen["config"], "table": table, "test": test}
    base = os.path.join(args.out_dir, f"perception_tuning_{stamp}")
    json.dump(out, open(base + ".json", "w"), indent=2)
    L = [f"# Perception re-tuning for the depth B-mode ({stamp})", "",
         f"commit `{commit}` · {meta['gpu']} · protocol DEC-044 · 54 configurations scored on 9 train patients; "
         "test patients scored only for the defaults and the selected configuration", "",
         f"Selected: `{json.dumps(chosen['config'])}`", "",
         "| Split | Configuration | Mean Cov@3 | Mean Prec@2 |", "| :--- | :--- | ---: | ---: |",
         f"| train | defaults | {default_row['train_cov3']:.3f} | {default_row['train_prec2']:.3f} |",
         f"| train | selected | {chosen['train_cov3']:.3f} | {chosen['train_prec2']:.3f} |"]
    for k in ("defaults", "selected"):
        L.append(f"| **test** | {k} | {np.mean([v[k]['cov3'] for v in test.values()]):.3f} | "
                 f"{np.nanmean([v[k]['prec2'] for v in test.values()]):.3f} |")
    L += ["", "| Test patient | Cov@3 defaults | Cov@3 selected | Prec@2 defaults | Prec@2 selected |",
          "| :--- | ---: | ---: | ---: | ---: |"]
    for n, v in test.items():
        L.append(f"| {n} | {v['defaults']['cov3']:.3f} | {v['selected']['cov3']:.3f} | {v['defaults']['prec2']:.3f} | "
                 f"{v['selected']['prec2']:.3f} |")
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("wrote", base + ".json / .md")


if __name__ == "__main__":
    main()
