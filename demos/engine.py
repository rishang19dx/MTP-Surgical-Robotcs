"""
Deformation demo engine: one scene, one scripted probe, the project's FEM, and per-frame outputs for every viewer
(PyBullet, 3D Slicer, video, web export).

    from demos.engine import DemoEngine
    eng = DemoEngine("demos/scenes/liver")
    frames = eng.run(n_frames=240)          # or eng.frame(t) for live use

A frame holds, all in the scene frame (mm, N):
  * probe pose on the undeformed skin and at the indented face, target and reached force, indentation;
  * for every deforming display mesh: deformed vertices and |u| per vertex (only the vertices inside the FEM ROI
    move; the ROI faces are fixed, so the field is continuous and zero outside);
  * the B-mode drawn from the CT warped by the same field (depth physics; the fascia layer in the muscle scene and
    the liver capsule in the liver scene, both as the top boundary of the organ mask;
    vein/artery lumen rendered anechoic).
Physics = the DEC-040/041 linear FEM (+ DEC-060 sliding for the liver scene), shared e_scale, and the DEC-062 fitted
fat modulus when a fat calibration file is given. Nothing here changes a registered study.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtp_gym.core import IMAGE_SIZE, PHYSICAL_SIZE_MM, SHARED_E_SCALE  # noqa: E402
from ReconstructionPipeline.evaluation.compression_sweep import indentation_for_force  # noqa: E402
from ReconstructionPipeline.simulation.fem_batched import BatchedFEMSolver  # noqa: E402
from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel  # noqa: E402
from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor  # noqa: E402
from ReconstructionPipeline.simulation.ultrasound_model_based import (  # noqa: E402
    ModelBasedUltrasoundSimulator, superficial_fascia_mask)

AX = PHYSICAL_SIZE_MM[0] / (IMAGE_SIZE[0] - 1)
LAT = PHYSICAL_SIZE_MM[1] / (IMAGE_SIZE[1] - 1)
FACE_HALF = (22.5, 5.0)
ROI_MARGIN = 30.0
E_LIVER = 4.1          # kPa effective (Fung et al. 2013 mean, DEC-059)


def read_obj(path):
    v, f = [], []
    for line in open(path):
        if line.startswith("v "):
            v.append([float(a) for a in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(a.split("/")[0]) - 1 for a in line.split()[1:4]])
    return np.asarray(v, np.float64), np.asarray(f, np.int64)


@dataclass
class Frame:
    t: float
    pose_skin: np.ndarray            # (4,4)
    pose_face: np.ndarray            # (4,4)
    force_target_n: float
    force_n: float
    indent_mm: float
    verts: Dict[str, np.ndarray] = field(default_factory=dict)      # deformed vertices of deforming meshes
    disp: Dict[str, np.ndarray] = field(default_factory=dict)       # |u| per vertex (mm)
    bmode: Optional[np.ndarray] = None                               # (H, W) in [0,1]
    max_u_mm: float = 0.0


class DemoEngine:
    def __init__(self, scene_dir: str, fat_calibration: str = "", sliding: Optional[bool] = None,
                 device: Optional[torch.device] = None, press=(1.0, 6.0), press_period_s=4.0, sweep_period_s=16.0,
                 build_fem: bool = True):
        self.dir = scene_dir
        self.sc = json.load(open(os.path.join(scene_dir, "scene.json")))
        self.dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.press, self.press_T, self.sweep_T = press, float(press_period_s), float(sweep_period_s)
        crop = os.path.join(ROOT, self.sc["fem_crop"])
        self.data, self.masks = self._load_crop(crop)
        self.e_fat = None
        if fat_calibration:
            cal = json.load(open(fat_calibration))
            if cal.get("meta", {}).get("fit") == "fat" and cal.get("converged"):
                self.e_fat = float(cal["fat_e_eff_kpa"])
        self.sliding = (self.sc["scene"] == "liver") if sliding is None else sliding
        self._build_fem(build_fem)
        self._load_meshes()
        if not build_fem:                        # geometry only (replaying cached frames)
            return
        self.ct = VolumeSliceExtractor(self.data["ct_volume"], spacing=self.data["spacing_mm"],
                                       origin=self.data["origin_mm"], image_size=IMAGE_SIZE,
                                       physical_size_mm=PHYSICAL_SIZE_MM, device=self.dev)
        self.lumen = None
        if "lumen" in self.masks:
            self.lumen = VolumeSliceExtractor(self.masks["lumen"].astype(np.float32), spacing=self.data["spacing_mm"],
                                              origin=self.data["origin_mm"], image_size=IMAGE_SIZE,
                                              physical_size_mm=PHYSICAL_SIZE_MM, device=self.dev)
        self.muscle = None                       # the top boundary of this mask is drawn as a bright interface:
        layer = "muscle" if "muscle" in self.masks else ("liver" if "liver" in self.masks else None)
        if layer is not None:                    # the fascia over the muscle, or the liver capsule
            self.muscle = VolumeSliceExtractor(self.masks[layer].astype(np.float32), spacing=self.data["spacing_mm"],
                                               origin=self.data["origin_mm"], image_size=IMAGE_SIZE,
                                               physical_size_mm=PHYSICAL_SIZE_MM, device=self.dev)
        self.sim = ModelBasedUltrasoundSimulator.depth_physics(device=self.dev, axial_resolution_mm=AX,
                                                               lateral_resolution_mm=LAT)

    # ------------------------------------------------------------------ data
    def _load_crop(self, crop):
        scene = self.sc["scene"]
        masks = {}
        if self.sc["frame"] == "P":
            from mtp_gym.core import load_patient
            from ReconstructionPipeline.evaluation.muscle_compression_study import muscle_volume
            data = load_patient(crop)
            try:
                masks["muscle"] = muscle_volume(crop) > 0.5
            except FileNotFoundError:
                pass
            return data, masks

        def arr(name):
            img = nib.load(os.path.join(crop, name))
            return np.asarray(img.dataobj).transpose(2, 1, 0), img.affine
        ct, aff = arr("ct.nii.gz")
        data = {"ct_volume": ct.astype(np.float32), "origin_mm": tuple(float(v) for v in aff[:3, 3]),
                "spacing_mm": (1.0, 1.0, 1.0)}
        if scene == "liver":
            masks["liver"] = arr("liver.nii.gz")[0] > 0
        if scene == "vein":
            v = arr("vein.nii.gz")[0] > 0
            a = arr("artery.nii.gz")[0] > 0
            masks["vein"], masks["lumen"] = v, v | a
        return data, masks

    def _build_fem(self, build=True):
        path = np.asarray(self.sc["probe_path"], float)
        ct = self.data["ct_volume"]
        o = np.asarray(self.data["origin_mm"], float)
        s = np.asarray(self.data["spacing_mm"], float)
        vmax = o + s * (np.array([ct.shape[2], ct.shape[1], ct.shape[0]]) - 1)
        lo = np.array([path[:, 0].min() - FACE_HALF[0] - ROI_MARGIN, o[1], path[:, 1].min() - FACE_HALF[1] - ROI_MARGIN])
        hi = np.array([path[:, 0].max() + FACE_HALF[0] + ROI_MARGIN, vmax[1] + 4.0,
                       path[:, 1].max() + FACE_HALF[1] + ROI_MARGIN])
        lo[[0, 2]] = np.maximum(lo[[0, 2]], o[[0, 2]])
        hi[[0, 2]] = np.minimum(hi[[0, 2]], vmax[[0, 2]])
        if self.sc["frame"] == "P":          # lumbar crop: deep face at the usual depth below the spine window
            lo[1] = max(o[1], -10.0)
        self.roi = (lo, hi)
        if not build:
            return
        kw = {}
        if self.e_fat is not None:
            kw["e_fat_kpa"] = self.e_fat / SHARED_E_SCALE
        if "liver" in self.masks:
            kw["region_modulus"] = (self.masks["liver"], E_LIVER / SHARED_E_SCALE)
            if self.sliding:
                kw["sliding_region"] = self.masks["liver"]
        if "vein" in self.masks:
            kw["region_modulus"] = (self.masks["vein"], 0.1 / SHARED_E_SCALE, 0.0)     # near-empty lumen (DEC-062)
        self.model = FEMTissueModel(self.data, roi_min_mm=lo, roi_max_mm=hi, cache_dir=None, device=self.dev,
                                    verbose=False, **kw)
        self.model.set_parameters(e_scale=SHARED_E_SCALE)
        self.solver = BatchedFEMSolver(self.model)
        self.roi = (lo, hi)

    def _load_meshes(self):
        lo, hi = self.roi
        self.meshes = {}
        for m in self.sc["meshes"]:
            v, f = read_obj(os.path.join(self.dir, m["file"]))
            inside = np.all((v >= lo - 1e-6) & (v <= hi + 1e-6), axis=1)
            inner = False
            if "liver" in self.masks and m["name"] == "liver":
                inner = True
            self.meshes[m["name"]] = {"v": v, "f": f, "meta": m, "moving": inside & bool(m.get("deforms")),
                                      "inner": inner}

    # ------------------------------------------------------------------ motion
    def probe_params(self, t):
        """Probe position along the path (triangle wave, sweep period) and target force (press cycles)."""
        path = np.asarray(self.sc["probe_path"], float)
        ph = (t % self.sweep_T) / self.sweep_T
        a = 2 * ph if ph < 0.5 else 2 - 2 * ph
        a = 0.5 - 0.5 * math.cos(math.pi * a)                        # ease in/out
        x, z = path[0] + a * (path[-1] - path[0])
        f0, f1 = self.press
        F = f0 + (f1 - f0) * (0.5 - 0.5 * math.cos(2 * math.pi * t / self.press_T))
        return float(x), float(z), float(F)

    # ------------------------------------------------------------------ physics
    @torch.no_grad()
    def frames(self, times: List[float], bmode: bool = True, seed: int = 0) -> List[Frame]:
        P = [self.probe_params(t) for t in times]
        T = torch.as_tensor(np.stack([self.model.dorsal_pose(x, z) for x, z, _ in P]), dtype=torch.float64,
                            device=self.dev)
        F = torch.as_tensor([p[2] for p in P], dtype=torch.float64, device=self.dev)
        d, d0 = indentation_for_force(type("PS", (), {"solver": self.solver})(), T, F)
        st = self.solver.solve(T, d, stiffness=False)
        face = st.probe_face_poses(T)
        out = []
        B = len(times)
        mesh_u = {}
        for name, m in self.meshes.items():
            idx = np.nonzero(m["moving"])[0]
            if len(idx) == 0:
                continue
            X = torch.as_tensor(m["v"][idx], dtype=torch.float32, device=self.dev)[None].expand(B, -1, -1)
            u = st.displacement(X, inner=m["inner"]).cpu().numpy()
            mesh_u[name] = (idx, u)
        bm = None
        if bmode:
            sl = self.ct.extract_slices(face, interpolation="bilinear", batched_warp_fn=st.warp_to_reference)
            fas = None
            if self.muscle is not None:
                ms = self.muscle.extract_slices(face, interpolation="nearest", batched_warp_fn=st.warp_to_reference) > 0.5
                fas = superficial_fascia_mask(ms, AX)
            ims = []
            for i in range(B):
                g = torch.Generator(device=self.dev)
                g.manual_seed(seed)
                ims.append(self.sim.simulate_bmode(sl[i:i + 1], fascia_mask=None if fas is None else fas[i:i + 1],
                                                   generator=g)["bmode"])
            bm = torch.cat(ims)
            if self.lumen is not None:                               # blood is anechoic
                lm = self.lumen.extract_slices(face, interpolation="nearest", batched_warp_fn=st.warp_to_reference)
                bm = bm * (1.0 - 0.9 * (lm > 0.5).float())
            bm = bm.clamp(0, 1).cpu().numpy()
        Tn, Fn = T.cpu().numpy(), face.cpu().numpy()
        for i, t in enumerate(times):
            fr = Frame(t=float(t), pose_skin=Tn[i], pose_face=Fn[i], force_target_n=float(F[i]),
                       force_n=float(st.reaction_force_n[i]), indent_mm=float(d[i] - d0[i]),
                       bmode=None if bm is None else bm[i])
            mx = 0.0
            for name, m in self.meshes.items():
                if not m["meta"].get("deforms"):
                    continue
                v = m["v"].copy()
                mag = np.zeros(len(v))
                if name in mesh_u:
                    idx, u = mesh_u[name]
                    v[idx] += u[i]
                    mag[idx] = np.linalg.norm(u[i], axis=1)
                    mx = max(mx, float(mag.max()))
                fr.verts[name], fr.disp[name] = v, mag
            fr.max_u_mm = mx
            out.append(fr)
        return out

    def run(self, n_frames=240, fps=15.0, batch=8, bmode=True):
        times = [k / fps for k in range(n_frames)]
        out = []
        for k in range(0, n_frames, batch):
            out += self.frames(times[k:k + batch], bmode=bmode)
        return out
