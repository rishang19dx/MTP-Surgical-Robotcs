"""
Thread B v2 (DEC-054): paraspinal muscle thickness read from the B-mode under probe force, with the superficial fascia
rendered (DEC-055) and the image-window rule (DEC-054). Pre-registration: the DEC entry named in PREREGISTRATION.

    ./.venv/bin/python ReconstructionPipeline/evaluation/muscle_thickness_study.py            # full pre-registered run
    ./.venv/bin/python ReconstructionPipeline/evaluation/muscle_thickness_study.py --dev      # TRAIN patients only

Per patient (the SonoGym subjects), the 6 paraspinal sites of DEC-050 and forces 1-5 N (8 and 10 N reported with the
DEC-047 caveat). Indentation per (site, force) by bisection on the batched FEM (shared moduli). Two conditions:
  * deformable: CT, muscle label and fascia sliced through the FEM warp at the indented face pose;
  * rigid (control): the same indented face pose, tissue NOT warped (the probe presses kinematically into rigid
    tissue, as in DEC-051 thread C).
Ground truth (metric only): fascia-to-bone thickness = muscle-label top -> CT bone surface (HU > 300) along the beam,
central +-5 mm columns, median over columns (`_gt`). Reference: the same at first contact (no compression).
Window rule (DEC-054): a site is scored only if, at every force and in both conditions, the muscle is not cut off by
the image window and bone is present (`_gt` "scorable").
Image thickness (B-mode only): `measure_thickness` (detector v3, frozen in DETECTOR_V3).
`--dev` restricts to TRAIN patients and writes to .agents/ (never to results/), for detector development.

DEC-067 (scale-up; the measurement above is unchanged): `--splits data/splits_v2.json` runs every CT-derived patient of
that split, `--shard i/n` one GPU's share (merge with `muscle_merge_shards.py`), `--e-scale-draw lo,hi` multiplies each
patient's shared modulus scale by a log-uniform draw seeded by (`--e-scale-seed`, patient name) (S1.2), and
`--prereg` names the pre-registration recorded in the output.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtp_gym.core import IMAGE_SIZE, PHYSICAL_SIZE_MM, PatientSim  # noqa: E402
from mtp_gym.perception import PERCEPTION_PRESETS, segment_bmode_batch  # noqa: E402
from ReconstructionPipeline.evaluation.compression_sweep import indentation_for_force  # noqa: E402
from ReconstructionPipeline.evaluation.muscle_compression_study import AX, LAT, SITES, central_cols, muscle_volume  # noqa: E402
from ReconstructionPipeline.reconstruction.bone_segmentation import BoneSurfaceSegmenter  # noqa: E402
from ReconstructionPipeline.simulation.confidence_map import UltrasoundConfidenceEstimator  # noqa: E402
from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor  # noqa: E402
from ReconstructionPipeline.simulation.ultrasound_model_based import (  # noqa: E402
    ModelBasedUltrasoundSimulator,
    superficial_fascia_mask,
)

PREREGISTRATION = "DEC-056"
# Imaging = the gym's default (DEC-042/043 depth-physics B-mode) and its perception preset (DEC-044, selected on the
# training patients on 7 Nov, before this study). Not tuned for this study.
BMODE_PHYSICS = "depth"
PERCEPTION = PERCEPTION_PRESETS[BMODE_PHYSICS]
WINDOW_BOTTOM_ROWS = 3
WINDOW_COL_FRACTION = 0.5
# Detector v3 (developed on TRAIN patients only; frozen at the pre-registration commit).
DETECTOR_V3 = {"fascia_min_depth_mm": 5.0, "fascia_half_width_mm": 10.0, "smooth_px": 5, "bone_margin_mm": 5.0,
               "min_bone_cols": 3}


def measure_thickness(bm: torch.Tensor, det: dict = DETECTOR_V3, components: bool = False):
    """Image-only muscle thickness (mm) per B-mode frame, NaN where not detected.
    Deep boundary = B-mode bone surface (median over the central columns); superficial boundary = the strongest
    local maximum of the column-averaged, smoothed profile between `fascia_min_depth_mm` and `bone_margin_mm` above
    the bone (the fascia echo)."""
    dev = bm.device
    conf = UltrasoundConfidenceEstimator(device=dev, axial_resolution_mm=AX,
                                         **PERCEPTION["confidence"]).compute_confidence_map(bm)
    has, y = segment_bmode_batch(bm, conf, BoneSurfaceSegmenter(axial_resolution_mm=AX, **PERCEPTION["segmenter"]))
    cs = central_cols()
    k = int(round(det["fascia_half_width_mm"] / LAT))
    c = (IMAGE_SIZE[1] - 1) // 2
    sp = int(det["smooth_px"])
    prof = torch.nn.functional.avg_pool1d(bm[:, :, c - k:c + k + 1].mean(2)[:, None], sp, 1, sp // 2)[:, 0].cpu().numpy()
    h_c, y_c = has[:, cs].cpu().numpy(), y[:, cs].cpu().numpy()
    out = np.full(len(bm), np.nan)
    fascia_mm = np.full(len(bm), np.nan)
    bone_mm = np.full(len(bm), np.nan)
    for i in range(len(bm)):
        if h_c[i].sum() < det["min_bone_cols"]:
            continue
        bone = float(np.median(y_c[i][h_c[i]])) * AX
        bone_mm[i] = bone
        lo, hi = int(round(det["fascia_min_depth_mm"] / AX)), int(round((bone - det["bone_margin_mm"]) / AX))
        if hi - lo < 5:
            continue
        p = prof[i, lo:hi]
        peaks = np.nonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]))[0] + 1
        if len(peaks) == 0:
            continue
        fascia_mm[i] = (lo + peaks[np.argmax(p[peaks])]) * AX
        out[i] = bone - fascia_mm[i]
    if components:
        return out, fascia_mm, bone_mm
    return out


BONE_HU = 300.0          # the simulator's bone rule (ultrasound_model_based: ct > 300)


def _gt(lab: torch.Tensor, ct: torch.Tensor):
    """Ground truth from the (warped) muscle label and CT slices, central +-5 mm columns (metric only, never fed to
    perception). Thickness = muscle-label top -> CT bone surface along the beam (the fascia-to-bone distance that
    clinical ultrasound measures); per column, median over the columns where both exist.
    Returns (thickness, scorable, top, bone) per frame. `scorable` = the muscle is not cut off by the image window
    (fewer than half of the columns with muscle in the last 3 rows) AND bone is found in at least half the columns."""
    cs = central_cols()
    L = lab[:, :, cs].cpu().numpy()
    B = (ct[:, :, cs] > BONE_HU).cpu().numpy()
    rows = L.shape[1]
    top = np.where(L.any(1), np.argmax(L, 1), np.nan) * AX
    bone = np.where(B.any(1), np.argmax(B, 1), np.nan) * AX
    thick = bone - top
    with np.errstate(all="ignore"):
        t_med = np.nanmedian(np.where(thick > 0, thick, np.nan), 1)
        top_med, bone_med = np.nanmedian(top, 1), np.nanmedian(bone, 1)
    truncated = L[:, rows - WINDOW_BOTTOM_ROWS:, :].any(1).mean(1) >= WINDOW_COL_FRACTION
    bone_ok = B.any(1).mean(1) >= 0.5
    return t_med, (~truncated) & bone_ok & np.isfinite(t_med), top_med, bone_med


@torch.no_grad()
def measure(ps: PatientSim, mus: VolumeSliceExtractor, forces, seed):
    dev = ps.device
    poses = np.stack([ps.model.dorsal_pose(x, z) for x, z in SITES])
    nP, nF = len(poses), len(forces)
    T = torch.as_tensor(np.repeat(poses, nF, 0), dtype=torch.float64, device=dev)
    Ft = torch.as_tensor(np.tile(forces, nP), dtype=torch.float64, device=dev)
    d, d0 = indentation_for_force(ps, T, Ft)
    st = ps.solver.solve(T, d, stiffness=False)
    face = st.probe_face_poses(T)
    sim = ModelBasedUltrasoundSimulator.depth_physics(device=dev, axial_resolution_mm=AX, lateral_resolution_mm=LAT)
    res = {"indent_mm": (d - d0).view(nP, nF).cpu().numpy(), "force_n": st.reaction_force_n.view(nP, nF).cpu().numpy()}
    for cond, warp in (("deformable", st.warp_to_reference), ("rigid", None)):
        kw = {"batched_warp_fn": warp} if warp is not None else {}
        lab = mus.extract_slices(face, interpolation="nearest", **kw) > 0.5
        ct = ps.ct.extract_slices(face, interpolation="bilinear", **kw)
        fas = superficial_fascia_mask(lab, AX)
        bm = []
        for i in range(len(ct)):
            g = torch.Generator(device=dev)
            g.manual_seed(seed)
            bm.append(sim.simulate_bmode(ct[i:i + 1], fascia_mask=fas[i:i + 1], generator=g)["bmode"])
        bm = torch.cat(bm)
        gt, scorable, gt_top, gt_bone = _gt(lab, ct)
        img, fas_mm, bone_mm = measure_thickness(bm, components=True)
        res[cond] = {"gt": gt.reshape(nP, nF), "scorable": scorable.reshape(nP, nF), "img": img.reshape(nP, nF),
                     "gt_top_mm": gt_top.reshape(nP, nF), "gt_bone_mm": gt_bone.reshape(nP, nF),
                     "img_fascia_mm": fas_mm.reshape(nP, nF), "img_bone_mm": bone_mm.reshape(nP, nF)}
    T0 = torch.as_tensor(poses, dtype=torch.float64, device=dev)
    d0p = ps.solver.first_contact(T0)
    d0p = torch.where(torch.isfinite(d0p), d0p, torch.zeros_like(d0p))
    face0 = T0.clone()
    face0[:, :3, 3] += d0p[:, None] * T0[:, :3, 1] / T0[:, :3, 1].norm(dim=1, keepdim=True)
    ref, _, _, _ = _gt(mus.extract_slices(face0, interpolation="nearest") > 0.5,
                       ps.ct.extract_slices(face0, interpolation="bilinear"))
    res["ref"] = ref
    res["scored_site"] = res["deformable"]["scorable"].all(1) & res["rigid"]["scorable"].all(1)
    return res


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return np.asarray(x).tolist()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forces", default="1,2,3,4,5,8,10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev", action="store_true", help="TRAIN patients only; output to .agents/ (detector development)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--splits", default=os.path.join(ROOT, "data", "splits.json"))
    ap.add_argument("--shard", default="", help="i/n: every n-th patient starting at i")
    ap.add_argument("--e-scale-draw", default="", help="lo,hi: per-patient log-uniform modulus multiplier (S1.2)")
    ap.add_argument("--e-scale-seed", type=int, default=1)
    ap.add_argument("--prereg", default=PREREGISTRATION)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    forces = np.array([float(v) for v in args.forces.split(",")])
    split = json.load(open(args.splits))
    names = [n for n in split["train"] + ([] if args.dev else split["test"])
             if n.startswith(("sonogym_", "../patients_ts2/"))]
    if args.shard:
        si, sn = (int(v) for v in args.shard.split("/"))
        names = names[si::sn]
    draw = tuple(float(v) for v in args.e_scale_draw.split(",")) if args.e_scale_draw else None
    out_dir = args.out_dir or os.path.join(ROOT, ".agents" if args.dev else "results")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {}
    for n in names:
        pdir = os.path.normpath(os.path.join(ROOT, split["patients_root"], n))
        ps = PatientSim(pdir, dev)
        mult = 1.0
        if draw:
            h = int(hashlib.sha256(f"{args.e_scale_seed}:{n}".encode()).hexdigest()[:8], 16)
            mult = float(np.exp(np.random.default_rng(h).uniform(np.log(draw[0]), np.log(draw[1]))))
            ps.model.set_parameters(e_scale=ps.model.e_scale * mult)
        d = ps.data
        mus = VolumeSliceExtractor(muscle_volume(pdir), spacing=d["spacing_mm"], origin=d["origin_mm"],
                                   image_size=IMAGE_SIZE, physical_size_mm=PHYSICAL_SIZE_MM, device=dev)
        r = measure(ps, mus, forces, args.seed)
        r["split"] = "train" if n in split["train"] else "test"
        r["e_scale_multiplier"] = mult
        r["e_scale"] = float(ps.model.e_scale)
        rows[n] = r
        ok = r["scored_site"]
        low = forces <= 2.0
        e = r["deformable"]["img"][ok][:, low] - r["deformable"]["gt"][ok][:, low]
        print(f"{n:16s} {r['split']:5s} scored sites {int(ok.sum())}/6  detect<=2N "
              f"{np.isfinite(e).mean() if e.size else float('nan'):.2f}  median err<=2N "
              f"{np.nanmedian(e) if np.isfinite(e).any() else float('nan'):+.2f} mm  |err| "
              f"{np.nanmedian(np.abs(e)) if np.isfinite(e).any() else float('nan'):.2f} mm", flush=True)
        del ps, mus
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        commit += "-dirty" if subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) else ""
    except Exception:
        commit = "unknown"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"script": "ReconstructionPipeline/evaluation/muscle_thickness_study.py", "commit": commit,
            "args": vars(args), "forces_n": forces.tolist(), "sites": SITES, "detector": DETECTOR_V3,
            "bmode_physics": BMODE_PHYSICS, "perception": PERCEPTION,
            "preregistration": args.prereg, "dev": bool(args.dev), "splits": os.path.relpath(args.splits, ROOT),
            "shard": args.shard, "e_scale_draw": draw, "e_scale_seed": args.e_scale_seed}
    os.makedirs(out_dir, exist_ok=True)
    suffix = (f"_{args.tag}" if args.tag else "") + (f"_shard{args.shard.replace('/', 'of')}" if args.shard else "")
    path = os.path.join(out_dir, f"muscle_thickness_{'dev_' if args.dev else ''}{stamp}{suffix}.json")
    json.dump({"meta": meta, "patients": _jsonable(rows)}, open(path, "w"), indent=1)
    print("wrote", path)
    return path


if __name__ == "__main__":
    main()
