"""
Build the project architecture diagram (draw.io) with the CLI-Anything draw.io harness:

    cli-hub install drawio            # once: provides `cli-anything-drawio`
    python3 Docs/architecture/make_architecture.py [--out Docs/architecture/mtp_architecture.drawio]

Every shape, connector and style is added through `cli-anything-drawio` calls (it saves the project after each
call). Page 1: the whole system. Page 2: one simulation step (Flow C in FLOW.md). Study outcomes are the recorded
DECISIONS.md verdicts (DEC-056 to DEC-063).
"""

from __future__ import annotations

import argparse
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = "cli-anything-drawio"

# palette: one tint per layer; study verdicts use semantic colours
LAYER = {
    "data": ("#eef4fb", "#5b84b1"), "phys": ("#fdf1e7", "#c7743b"), "img": ("#f0f7ee", "#5e9a55"),
    "env": ("#f4effa", "#8a6bb8"), "plan": ("#fbf0f4", "#b5577c"), "rt": ("#eef7f8", "#3f8f99"),
    "study": ("#f7f7f2", "#7d7d6a"), "demo": ("#f2f2fb", "#5d5fb0"), "frame": ("#fffbe6", "#b59b2a"),
}
VERDICT = {"pos": ("#d9f2d9", "#3b8a3b"), "fail": ("#f8dada", "#b23b3b"), "null": ("#ececec", "#7a7a7a"),
           "mixed": ("#fdebd3", "#c27a1c")}


class Diagram:
    def __init__(self, path):
        self.path = path

    def run(self, *args):
        r = subprocess.run([CLI, "--project", self.path, *map(str, args)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(map(str, args))}\n{r.stdout}\n{r.stderr}")
        return r.stdout

    def style(self, cid, page, **kv):
        for k, v in kv.items():
            self.run("shape", "style", cid, k, v, "--page", page)

    def group(self, cid, title, x, y, w, h, layer, page=0):
        fill, stroke = LAYER[layer]
        self.run("shape", "add", "rounded", "-l", title, "--x", x, "--y", y, "-w", w, "-h", h, "--id", cid, "--page", page)
        self.style(cid, page, fillColor=fill, strokeColor=stroke, strokeWidth=2, verticalAlign="top", align="left",
                   spacingLeft=12, spacingTop=6, fontSize=17, fontStyle=1, fontColor=stroke, arcSize=3)

    def node(self, cid, label, x, y, w, h, layer=None, kind="rounded", page=0, verdict=None, dashed=False, size=12):
        self.run("shape", "add", kind, "-l", label, "--x", x, "--y", y, "-w", w, "-h", h, "--id", cid, "--page", page)
        if verdict:
            fill, stroke = VERDICT[verdict]
        else:
            fill, stroke = "#ffffff", LAYER[layer][1]
        kv = dict(fillColor=fill, strokeColor=stroke, fontSize=size, fontColor="#1f2430", strokeWidth=1.5)
        if dashed:
            kv["dashed"] = 1
        self.style(cid, page, **kv)

    def edge(self, a, b, label="", page=0, style="orthogonal", color="#555b66", dashed=False, width=1.5):
        out = self.run("connect", "add", a, b, "--style", style, "-l", label, "--page", page)
        eid = next(l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("id:"))
        for k, v in dict(strokeColor=color, strokeWidth=width, fontSize=11, fontColor="#3a3f4a",
                         labelBackgroundColor="#ffffff", endArrow="block", endFill=1).items():
            self.run("connect", "style", eid, k, v, "--page", page)
        if dashed:
            self.run("connect", "style", eid, "dashed", 1, "--page", page)
        return eid

    def text(self, cid, label, x, y, w, h, page=0, size=14, color="#1f2430", bold=False, align="left"):
        self.run("shape", "add", "text", "-l", label, "--x", x, "--y", y, "-w", w, "-h", h, "--id", cid, "--page", page)
        self.style(cid, page, fontSize=size, fontColor=color, align=align, fontStyle=1 if bold else 0, autosize=0,
                   whiteSpace="wrap")


def b(title, *lines):
    return f"<b>{title}</b>" + "".join(f"<br>{l}" for l in lines)


def page1(d):
    d.text("title", b("MTP: patient-specific soft-tissue simulator for robotic ultrasound") +
           "<br><span style='font-size:13px'>SonoGym-style gym with a GPU linear FEM, model-based B-mode, learned "
           "and heuristic scan policies, a 1 kHz robot runtime streaming to 3D Slicer, pre-registered studies "
           "and live demos</span>", 20, 10, 1700, 60, size=22)

    # ---------------- row 1: data | physics | imaging | environments
    d.group("g_data", "1 · Data and patient assets", 20, 90, 540, 610, "data")
    d.node("hf", b("SonoGym assets (Hugging Face)", "12 TotalSegmentator CTs + labels, STLs", "CC BY 4.0"),
           40, 130, 240, 90, "data", kind="cylinder")
    d.node("hf2", b("Sample lumbar case", "HF case-0004 (CT + L1-L5, sacrum, skin)"), 300, 130, 240, 90, "data",
           kind="cylinder")
    d.node("imp_p", b("import_sonogym_subjects", "data/patients: frame P (LPS), L4 anchor"), 40, 250, 240, 60, "data",
           kind="process")
    d.node("prep", b("prepare_patient_assets", "resample, meshes, skin height map"), 300, 250, 240, 60, "data",
           kind="process")
    d.node("imp_l", b("import_liver_subjects [--wide]", "data/liver(_wide): frame Q"), 40, 330, 240, 60, "data",
           kind="process")
    d.node("imp_v", b("import_vein_subjects", "data/vein: femoral vein/artery, frame Q"), 300, 330, 240, 60, "data",
           kind="process")
    d.node("bfem", b("build_patient_fem", "FEM build + QA per patient"), 40, 410, 240, 60, "data", kind="process")
    d.node("split", b("data/splits.json", "fixed train / held-out split"), 300, 410, 240, 60, "data", kind="document")
    d.node("pd", b("Patient dict", "ct_volume [z,y,x], origin, spacing,", "labels, bone points, skin height map"),
           40, 500, 500, 70, "data")
    d.node("cal", b("Calibrations", "shared e_scale (force-indentation curve, k0 0.5 N/mm, β 0.15/mm)",
                    "fitted fat 3.38 kPa (popliteal vein, DEC-062)"), 40, 600, 500, 80, "data", kind="note")

    d.group("g_phys", "2 · Soft-tissue physics (GPU, PyTorch)", 590, 90, 560, 610, "phys")
    d.node("fem", b("FEMTissueModel", "linear tetrahedral FEM, 4 mm elements",
                    "Dirichlet: ROI deep + side faces, bone cells",
                    "precomputed Green's functions (3-direction skin compliance)",
                    "options: region_modulus · sliding interface · e_fat"), 610, 130, 520, 120, "phys")
    d.node("bsolve", b("BatchedFEMSolver", "N probe poses at once · active-set contact along the face normal",
                       "force by bisection on indentation · displacement field u(X)",
                       "warp_to_reference (backward Eulerian warp)"), 610, 275, 520, 105, "phys")
    d.node("adm", b("Admittance control", "δ in [first contact, max] so F ≈ target (4 N)"), 610, 405, 250, 70, "phys")
    d.node("stiff", b("Strain stiffening", "β = 0.15/mm on the force law"), 880, 405, 250, 70, "phys")
    d.node("kel", b("Kelvinlet model", "regularised, alternative backend"), 610, 500, 250, 60, "phys")
    d.node("flex", b("MuJoCo flex", "demo only (DEC-053, gate failed)"), 880, 500, 250, 60, "phys", dashed=True)
    d.node("lim", b("Known limits", "small strain only · no vein collapse mechanics",
                    "clamped box walls bias deep motion down ~30 % (DEC-063)"), 610, 590, 520, 80, "phys", kind="note")

    d.group("g_img", "3 · Imaging and perception", 1180, 90, 560, 610, "img")
    d.node("slice", b("VolumeSliceExtractor", "grid_sample of CT / labels at the indented face pose",
                      "through the deformed tissue"), 1200, 130, 520, 75, "img")
    d.node("us", b("ModelBasedUltrasoundSimulator", "dB→Np attenuation, interface reflection, Rayleigh speckle, fixed TGC",
                   "200 × 150 px over 75 × 50 mm, target < 2 ms/frame"), 1200, 230, 520, 80, "img")
    d.node("conf", b("Confidence estimator", "image-only"), 1200, 335, 250, 60, "img")
    d.node("seg", b("segment_from_bmode", "ridge + distal shadow + run length"), 1470, 335, 250, 60, "img")
    d.node("bp", b("Back-projection", "with the tracked (indented) face pose"), 1200, 420, 250, 60, "img")
    d.node("acc", b("PointCloudAccumulator", "1.5 mm voxels"), 1470, 420, 250, 60, "img")
    d.node("met", b("Metrics", "coverage @2/3 mm of the visible L4 ROI, precision, F, Chamfer"), 1200, 505, 520, 60,
           "img")
    d.node("oracle", b("Oracle only (never feeds perception)", "segment_from_label_map, bone_mask, energy"),
           1200, 595, 520, 70, "img", kind="note")

    d.group("g_env", "4 · Environments", 1770, 90, 600, 610, "env")
    d.node("env1", b("UltrasoundReconstructionEnv", "single env, used by the server and the benchmark"),
           1790, 130, 560, 65, "env")
    d.node("vec", b("mtp_gym vector env (GPU, no PyBullet)", "MTP/FreeScan-v0 · MTP/Recon-v0",
                    "one PatientSim per patient (FEM, compliance, CT); per-env e_scale",
                    "autoreset in the same step"), 1790, 220, 560, 95, "env")
    d.node("modes", b("Deformation modes", "aware (default) · unaware · rigid / kinematic"), 1790, 340, 270, 70, "env")
    d.node("tasks", b("tasks/", "labels feed rewards and metrics only"), 2080, 340, 270, 70, "env")
    d.node("act", b("Action", "[dc1, dc2, dyaw, dpitch] (+ target force)", "×5 mm, ×10° yaw, ×15° pitch"),
           1790, 435, 560, 65, "env")
    d.node("obs", b("Observation", "B-mode, confidence, reconstructed points, probe pose, force"), 1790, 525, 560, 60, "env")
    d.node("par", b("Parity test", "vector env = single env (force, indentation, pose)"), 1790, 610, 560, 60, "env",
           kind="note")

    # ---------------- row 2: planning | runtime | frames
    d.group("g_plan", "5 · Planning and learning", 20, 730, 840, 430, "plan")
    d.node("pn", b("PointNet encoder", "per-point LayerNorm"), 40, 775, 250, 60, "plan")
    d.node("ppo", b("Submodular PPO", "coverage-gain reward"), 310, 775, 250, 60, "plan")
    d.node("cfm", b("Flow Matching (CFM)", "8-step chunks, execute 2"), 580, 775, 260, 60, "plan")
    d.node("edm", b("EDM diffusion policy", "behaviour cloning from rasters"), 580, 860, 260, 60, "plan")
    d.node("heur", b("Heuristics", "raster · orbital"), 40, 860, 250, 60, "plan")
    d.node("nbv", b("Greedy NBV", "next best view"), 310, 860, 250, 60, "plan")
    d.node("ckpt", b("checkpoints/", "ppo, flow_matching, diffusion"), 40, 950, 250, 70, "plan", kind="cylinder")
    d.node("bench", b("benchmark_runner", "6 policies × trials, same start poses", "→ results/benchmark_*.json"),
           310, 950, 530, 70, "plan")
    d.node("sb3", b("Training scripts", "train_submodular_ppo.py · train_diffusion_policy.py"), 40, 1050, 800, 60,
           "plan")

    d.group("g_rt", "6 · Real-time runtime (unified_server)", 890, 730, 1000, 430, "rt")
    d.node("stream", b("Streaming loop (30 Hz)", "env.step → B-mode, recon points, force", "never imports a physics engine"),
           910, 775, 300, 85, "rt")
    d.node("physp", b("Physics process (1 kHz)", "PyBullet or MuJoCo · Franka Panda URDF, IK",
                      "F = F0 + K_c(δ − δ0) · virtual keep-out fixtures"), 1250, 775, 320, 85, "rt")
    d.node("viewer", b("Viewer process", "--gui, deformed skin"), 1610, 775, 260, 85, "rt")
    d.node("slicer", b("3D Slicer bridge", "PatientToRAS transform", "in-place VTK buffers"), 910, 940, 300, 85, "rt")
    d.node("user", b("Clinician / operator", "ROI target, mode, reset"), 1250, 940, 320, 85, "rt", kind="rounded")
    d.node("rate", b("Physics-rate monitor", "--benchmark-seconds → results/"), 1610, 940, 260, 85, "rt")
    d.node("igtl", "OpenIGTLink :18944: UltrasoundProbe, US_Image, ReconstructedBone, DeformedSkin, Telemetry",
           910, 1060, 960, 50, "rt", kind="note", size=11)

    d.group("g_frames", "Coordinate frames", 1920, 730, 450, 430, "frame")
    d.node("frames", b("P (simulation)", "LPS, mm; volumes [z, y, x]") + "<br><br>" +
           b("RAS (3D Slicer)", "p_RAS = diag(−1, −1, 1) · p_P, applied once") + "<br><br>" +
           b("PB (PyBullet / MuJoCo)", "Z-up, metres") + "<br><br>" +
           b("Q (liver and vein crops)", "P rotated 180° about z (anterior skin on +y)") + "<br><br>" +
           b("U (probe)", "lateral, beam (+Y into body), elevation"),
           1940, 775, 410, 360, "frame", kind="rectangle", size=12)
    d.style("frames", 0, align="left", spacingLeft=10, verticalAlign="top")

    # ---------------- row 3: studies | demos
    d.group("g_study", "7 · Pre-registered studies (DECISIONS.md) → results/", 20, 1190, 1180, 460, "study")
    s = [("st56", b("Muscle compression (DEC-056)", "POSITIVE: thickness changes 1.0–1.3 mm over 1–5 N"), "pos"),
         ("st57", b("Navigation transfer (DEC-057)", "null: policy succeeds with or without deformation"), "null"),
         ("st58", b("Force headroom, spine (DEC-058)", "0.28–1.13 mm: too small"), "null"),
         ("st59", b("Liver target shift (DEC-059, 062)", "FAIL: 0/10 patients ≥ 2 mm at 5 N"), "fail"),
         ("st60", b("Liver sliding (DEC-060)", "partial: sliding slightly reduces motion"), "null"),
         ("st61", b("Vein compression (DEC-061, 062)", "fat fit 3.38 kPa; FAIL: no vein halves by 20 N"), "fail"),
         ("st63", b("Box-margin sensitivity (DEC-063)", "inconclusive: clamped walls bias motion −30 %"), "mixed")]
    for i, (cid, lab, v) in enumerate(s):
        col, row = i % 2, i // 2
        d.node(cid, lab, 40 + col * 380, 1235 + row * 95, 360, 75, verdict=v)
    d.node("res", b("results/", "JSON + Markdown per run,", "named next to every documented number"),
           800, 1235, 380, 90, "study", kind="cylinder")
    d.node("rules", b("Rules", "gate committed before the run · deviations disclosed",
                      "metrics must be able to fail · .agents/ never cited"), 800, 1350, 380, 90, "study", kind="note")
    d.node("legend", "<b>Legend</b><br>green = positive · red = fail · grey = null / partial · orange = inconclusive",
           800, 1470, 380, 60, "study", kind="rectangle", size=11)

    d.group("g_demo", "8 · Deformation demos (demos/, demonstration only)", 1230, 1190, 1140, 460, "demo")
    d.node("bscene", b("build_scenes", "whole-body meshes from SonoGym", "spine · muscle · liver · vein"),
           1250, 1235, 260, 85, "demo")
    d.node("deng", b("DemoEngine", "probe sweep + 1–6 N press", "FEM, mesh displacements, B-mode"), 1540, 1235, 260, 85,
           "demo")
    d.node("rdemo", b("run_demo", "frame cache (.npz)", "OpenIGTLink :18945"), 1830, 1235, 250, 85, "demo")
    d.node("web", b("export_web", "exact FEM states on a", "position × force grid"), 2110, 1235, 240, 85, "demo")
    d.node("sdemo", b("slicer_demo", "clip-node cutaway at the image plane", "live B-mode plane, force plot"),
           1540, 1360, 260, 85, "demo")
    d.node("pbv", b("pybullet_view", "displacement bands, skin profile,", "force arrow (EGL-free snapshots)"),
           1830, 1360, 250, 85, "demo")
    d.node("viewer3", b("three.js web viewer", "interactive probe, published artifact"), 2110, 1360, 240, 85, "demo")
    d.node("vid", b("make_video", "Slicer + PyBullet → mp4 per scene"), 1540, 1480, 540, 60, "demo")

    # ---------------- connectors
    E = [("hf", "imp_p", "CT, labels"), ("hf", "imp_l", ""), ("hf2", "prep", ""), ("imp_p", "bfem", ""),
         ("prep", "pd", ""), ("bfem", "pd", ""), ("pd", "fem", "patient anatomy"), ("cal", "fem", "moduli"),
         ("fem", "bsolve", "G, compliance"), ("bsolve", "slice", "u(X), face pose"), ("slice", "us", "deformed CT"),
         ("us", "conf", ""), ("us", "seg", "B-mode"), ("seg", "bp", ""), ("bp", "acc", ""), ("acc", "met", ""),
         ("bsolve", "vec", "force, field"), ("us", "obs", ""), ("act", "adm", "pose + target force"),
         ("obs", "pn", "points + pose"), ("pn", "ppo", ""), ("pn", "cfm", ""), ("ppo", "ckpt", ""),
         ("edm", "ckpt", ""), ("bench", "res", "benchmark"), ("env1", "stream", "env.step"),
         ("stream", "physp", "targets (queue)"), ("physp", "stream", "contact snapshot"),
         ("stream", "viewer", "skin"), ("stream", "slicer", "OpenIGTLink"), ("user", "slicer", ""),
         ("physp", "rate", ""), ("bsolve", "st56", "FEM"), ("st63", "res", ""), ("hf", "bscene", "full CTs"),
         ("bscene", "deng", ""), ("bsolve", "deng", "FEM"), ("deng", "rdemo", ""), ("rdemo", "sdemo", ":18945"),
         ("rdemo", "pbv", ""), ("deng", "web", ""), ("web", "viewer3", ""), ("sdemo", "vid", ""), ("pbv", "vid", "")]
    for a, c, lab in E:
        d.edge(a, c, lab)


def page2(d):
    d.run("page", "add", "--name", "One simulation step", "--width", 1400, "--height", 1300)
    P = 1
    d.text("t2", b("One simulation step (Flow C)") + "<br><span style='font-size:13px'>single env and batched vector "
           "env follow the same chain; a parity test checks they agree</span>", 20, 10, 1200, 60, page=P, size=22)
    chain = [
        ("c1", b("Action", "[dc1, dc2, dyaw, dpitch] ∈ [−1, 1] (+ optional target force)"), "env"),
        ("c2", b("Probe pose on the undeformed skin", "skin height map, frame P, clipped to the scan window"), "env"),
        ("c3", b("First contact", "FEM first_contact_indentation (can be < 0)"), "phys"),
        ("c4", b("Admittance loop", "δ ∈ [first contact, max indentation] so F_tissue(δ) ≈ 4 N"), "phys"),
        ("c5", b("FEM solve", "active-set contact on the precomputed skin compliance",
                 "→ force F, contact stiffness K_c, displacement field u(X), deformed skin"), "phys"),
        ("c6", b("Backward warp + slice", "X = x − u(X) (fixed point, GPU), grid_sample CT and labels at the face pose"),
         "img"),
        ("c7", b("B-mode", "model-based: attenuation, interfaces, speckle, TGC; coupling κ(force, tilt) as dB"), "img"),
        ("c8", b("Perception (image only)", "confidence → segment_from_bmode → back-projection → point cloud"), "img"),
        ("c9", b("Task evaluation", "reward, termination, metrics (labels allowed here only)"), "env"),
    ]
    y = 90
    for cid, lab, layer in chain:
        d.node(cid, lab, 120, y, 640, 80, layer, page=P, size=13)
        y += 120
    for (a, _, _), (c, _, _) in zip(chain, chain[1:]):
        d.edge(a, c, page=P, style="straight")
    d.node("m1", b("Deformation modes (ablation)", "aware: deformed image, tracked face pose",
                   "unaware: deformed image, nominal pose", "rigid: no deformation, probe on the skin"),
           820, 450, 420, 120, "env", kind="note", page=P)
    d.node("m2", b("Physics side (server only)", "1 kHz loop applies F = F0 + K_c(δ − δ0) to the robot;",
                   "it never waits on the FEM"), 820, 610, 420, 100, "rt", kind="note", page=P)
    d.node("m3", b("Privileged outputs", "energy, effective_energy, bone_mask:", "metrics only, never perception"),
           820, 930, 420, 100, "img", kind="note", page=P)
    d.edge("m1", "c6", page=P, dashed=True)
    d.edge("m2", "c5", page=P, dashed=True)
    d.edge("m3", "c8", page=P, dashed=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "mtp_architecture.drawio"))
    a = ap.parse_args()
    if os.path.exists(a.out):
        os.remove(a.out)
    subprocess.run([CLI, "project", "new", "--width", 2400, "--height", 1680, "-o", a.out][:3] +
                   ["--width", "2400", "--height", "1680", "-o", a.out], check=True, capture_output=True)
    d = Diagram(a.out)
    d.run("page", "rename", 0, "System architecture")
    page1(d)
    page2(d)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
