"""
Vein lumen calibration (DEC-061, pre-registered): find the lumen modulus with which a probe halves a vein of the
measured popliteal geometry at the measured force.

    ./.venv/bin/python ReconstructionPipeline/evaluation/vein_calibration.py

Target (Sato & Karino 2022, Cureus 14:e32596, doi 10.7759/cureus.32596, Table 1; one healthy subject): popliteal vein,
depth 11.5 mm, diameter 5.0 mm, compressed to 50 % of its diameter at 191 +- 65 g = 1.87 N.
Phantom (no patient data): a fat block (HU -100; the popliteal fossa is fat), flat skin, a 5 mm cylindrical lumen
along z whose CENTRE is 11.5 mm below the skin (the paper does not say whether depth is to the top or the centre; the
centre is assumed and stated). FEM as in the patient study: DEC-040/041 linear FEM, shared e_scale, lumen = region
modulus with Poisson ratio 0 (blood can leave, the lumen volume can vanish), but h = 1 mm (2.5 cells per radius).
Probe untilted at the centre, transverse to the vessel. Outcome: AP diameter ratio on the central beam,
(d0 + u_y(top wall) - u_y(bottom wall)) / d0, at 1.87 N (force by bisection on the FEM).
The effective lumen modulus is found by bisection in log space so that the ratio is 0.50 +- 0.01.
Writes results/vein_calibration_<stamp>.json / .md.

`--fit fat` (DEC-062, pre-registered after the lumen fit proved unreachable): the lumen is held near-empty
(E_LUMEN_VOID kPa, Poisson 0: the blood leaves) and the FAT modulus is bisected instead, so that the same 5 mm vein
halves at 1.87 N. Muscle keeps its modulus. Writes results/vein_calibration_fat_<stamp>.json / .md.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtp_gym.core import SHARED_E_SCALE  # noqa: E402
from ReconstructionPipeline.evaluation.compression_sweep import indentation_for_force  # noqa: E402
from ReconstructionPipeline.simulation.fem_batched import BatchedFEMSolver  # noqa: E402
from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel  # noqa: E402

TARGET_FORCE_N = 0.191 * 9.80665
TARGET_RATIO, TOL = 0.50, 0.01
DEPTH_MM, DIAM_MM = 11.5, 5.0
HALF_X, HALF_Z, DEPTH_BLOCK, SKIN_Y = 32.5, 10.0, 30.0, 30.0
H = 1.0
NU_LUMEN = 0.0
E_LUMEN_VOID = 0.1          # kPa effective, DEC-062: near-empty lumen
E_FAT_DEFAULT = 10.0        # kPa before e_scale, the FEMTissueModel default


class _PS:
    def __init__(self, solver):
        self.solver = solver


def phantom():
    xs = np.arange(-HALF_X, HALF_X + 0.5, 1.0)
    ys = np.arange(0.0, SKIN_Y + 4.0 + 0.5, 1.0)
    zs = np.arange(-HALF_Z, HALF_Z + 0.5, 1.0)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    ct = np.where(Y <= SKIN_Y, -100.0, -1000.0).astype(np.float32)
    lumen = (X ** 2 + (Y - (SKIN_Y - DEPTH_MM)) ** 2) <= (DIAM_MM / 2) ** 2
    ct[lumen] = 40.0
    data = {"ct_volume": ct, "origin_mm": (float(xs[0]), float(ys[0]), float(zs[0])), "spacing_mm": (1.0, 1.0, 1.0)}
    return data, lumen


@torch.no_grad()
def ratio_at(data, lumen, e_eff, dev, e_fat_eff=None):
    lo = np.array([-HALF_X, 0.0, -HALF_Z])
    hi = np.array([HALF_X, SKIN_Y + 4.0, HALF_Z])
    fat = {} if e_fat_eff is None else {"e_fat_kpa": e_fat_eff / SHARED_E_SCALE}
    model = FEMTissueModel(data, roi_min_mm=lo, roi_max_mm=hi, element_size_mm=H, cache_dir=None, device=dev,
                           verbose=False, region_modulus=(lumen, e_eff / SHARED_E_SCALE, NU_LUMEN), **fat)
    model.set_parameters(e_scale=SHARED_E_SCALE)
    s = BatchedFEMSolver(model)
    T = torch.as_tensor(model.dorsal_pose(0.0, 0.0)[None], dtype=torch.float64, device=dev)
    d, _ = indentation_for_force(_PS(s), T, torch.tensor([TARGET_FORCE_N], dtype=torch.float64, device=dev))
    st = s.solve(T, d, stiffness=False)
    yc = SKIN_Y - DEPTH_MM
    X = torch.tensor([[[0.0, yc + DIAM_MM / 2, 0.0], [0.0, yc - DIAM_MM / 2, 0.0]]], dtype=torch.float32, device=dev)
    u = st.displacement(X)[0].cpu().numpy()
    r = (DIAM_MM + u[0, 1] - u[1, 1]) / DIAM_MM
    out = {"e_eff_kpa": e_eff, "e_fat_eff_kpa": e_fat_eff if e_fat_eff is not None else E_FAT_DEFAULT * SHARED_E_SCALE,
           "ratio": float(r), "force_n": float(st.reaction_force_n[0]),
           "indent_mm": float(d[0]), "build_s": model.timings.get("pcg_s", float("nan"))}
    del model, s, st
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", choices=("lumen", "fat"), default="lumen")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    data, lumen = phantom()
    fat_mode = args.fit == "fat"
    # lumen mode: bisect the lumen modulus; fat mode: bisect the fat modulus with a near-empty lumen.
    f = (lambda e: ratio_at(data, lumen, E_LUMEN_VOID, dev, e_fat_eff=e)) if fat_mode else \
        (lambda e: ratio_at(data, lumen, e, dev))
    lo_e, hi_e = (0.3, 12.0) if fat_mode else (0.05, 20.0)
    log = []
    r_hi = f(hi_e)
    print(f"E {hi_e} kPa -> ratio {r_hi['ratio']:.3f}", flush=True)
    r_lo = f(lo_e)
    log += [r_lo, r_hi]
    print(f"bracket: E {lo_e} kPa -> ratio {r_lo['ratio']:.3f}; E {hi_e} kPa -> ratio {r_hi['ratio']:.3f}", flush=True)
    if not (r_lo["ratio"] < TARGET_RATIO < r_hi["ratio"]):
        raise SystemExit("target ratio not bracketed; widen the bracket (disclose)")
    best = None
    for _ in range(14):
        e = float(np.sqrt(lo_e * hi_e))
        r = f(e)
        r["fitted_e_kpa"] = e
        log.append(r)
        print(f"E {e:.4f} kPa -> ratio {r['ratio']:.4f} at {r['force_n']:.3f} N", flush=True)
        best = r
        if abs(r["ratio"] - TARGET_RATIO) <= TOL:
            break
        if r["ratio"] < TARGET_RATIO:
            lo_e = e
        else:
            hi_e = e
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        commit += "-dirty" if subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) else ""
    except Exception:
        commit = "unknown"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    res = {"meta": {"script": "ReconstructionPipeline/evaluation/vein_calibration.py", "commit": commit,
                    "preregistration": "DEC-062" if fat_mode else "DEC-061", "fit": args.fit,
                    "lumen_e_void_kpa": E_LUMEN_VOID if fat_mode else None, "device": str(dev), "target": {"force_n": TARGET_FORCE_N, "ratio": TARGET_RATIO,
                                                             "depth_mm": DEPTH_MM, "diameter_mm": DIAM_MM},
                    "phantom": {"half_x": HALF_X, "half_z": HALF_Z, "skin_y": SKIN_Y, "h_mm": H, "nu_lumen": NU_LUMEN,
                                "tissue_hu": -100.0}, "shared_e_scale": SHARED_E_SCALE},
           "converged": bool(best and abs(best["ratio"] - TARGET_RATIO) <= TOL),
           "lumen_e_eff_kpa": (E_LUMEN_VOID if fat_mode else best["e_eff_kpa"]) if best else None,
           "fat_e_eff_kpa": (best["fitted_e_kpa"] if fat_mode else E_FAT_DEFAULT * SHARED_E_SCALE) if best else None,
           "log": log}
    base = os.path.join(ROOT, "results", f"vein_calibration_{'fat_' if fat_mode else ''}{stamp}")
    json.dump(res, open(base + ".json", "w"), indent=1)
    L = [f"# Vein lumen calibration ({stamp})", "",
         f"pre-registration DEC-061 · commit `{commit}` · target: 50 % AP diameter at {TARGET_FORCE_N:.2f} N, 5 mm vein, "
         "centre 11.5 mm deep (Sato & Karino 2022, Table 1) · fat phantom, h = 1 mm, lumen Poisson ratio 0", "",
         f"fit: **{args.fit}** · lumen modulus {res['lumen_e_eff_kpa']} kPa · **fat modulus {res['fat_e_eff_kpa']} kPa** "
         f"(effective) · converged: {res['converged']}", "",
         "| E_lumen (kPa) | E_fat (kPa) | AP ratio at target force | force (N) | indentation (mm) |",
         "| ---: | ---: | ---: | ---: | ---: |"]
    L += [f"| {r['e_eff_kpa']:.4f} | {r['e_fat_eff_kpa']:.3f} | {r['ratio']:.4f} | {r['force_n']:.3f} | {r['indent_mm']:.2f} |"
          for r in log]
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", base + ".json / .md")


if __name__ == "__main__":
    main()
