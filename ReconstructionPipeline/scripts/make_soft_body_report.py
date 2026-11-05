"""
Figures and videos for the soft-tissue progress report (DEC-040 era).

Every image is produced from live model outputs (FEMTissueModel / UltrasoundReconstructionEnv) or from the
benchmark JSON in results/; nothing is drawn by hand.  Output: demonstrations/report/.

    ./.venv/bin/python -m ReconstructionPipeline.scripts.make_soft_body_report [--only NAME[,NAME]]

Names: deformation, tilt, bmode, echo, recon_video, recon_3d, results.
Ground-truth bone labels appear only as evaluation overlays (grey), never as input to the reconstruction.
"""

import argparse
import json
import os

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from ReconstructionPipeline.evaluation.benchmark_runner import ReconstructionBenchmarkSuite, detect_bone_echo_depth

OUT = "demonstrations/report"
PATIENT = "data/sample_totalsegmentator_patient"
RESULTS_JSON = "results/benchmark_20261105_120357.json"
MIDLINE = (0.0, -32.5)
FLAT = (-20.0, -45.0)

# reference categorical palette (dataviz skill, light mode), text inks and surface
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "lines.linewidth": 2.0, "font.family": "DejaVu Sans",
})


# ---------------------------------------------------------------------------- setup
class Ctx:
    def __init__(self):
        self.suite = ReconstructionBenchmarkSuite(num_trials=1, patient_dir=PATIENT, target_vertebra="vertebrae_L4")
        self.patient = self.suite.patient_data()
        self.model = self.suite.tissue_model("fem")
        self.dev = self.model.device
        self.ct = np.asarray(self.patient["ct_volume"], dtype=np.float32)
        self.lab = np.asarray(self.patient["label_volume"])
        self.o = np.asarray(self.patient["origin_mm"], dtype=np.float64)
        self.sp = np.asarray(self.patient["spacing_mm"], dtype=np.float64)

    def env(self, seed=0, **kw):
        return self.suite.make_env(200, seed, **kw)

    def sample_ct(self, P):
        """CT HU at points P (..., 3) in frame P (mm); volume indexed [z, y, x]."""
        idx = (P - self.o) / self.sp
        coords = np.stack([idx[..., 2], idx[..., 1], idx[..., 0]])
        return map_coordinates(self.ct, coords.reshape(3, -1), order=1, cval=-1000.0).reshape(P.shape[:-1])

    def sample_lab(self, P):
        idx = np.rint((P - self.o) / self.sp).astype(int)
        z, y, x = (np.clip(idx[..., i], 0, self.lab.shape[j] - 1) for i, j in ((2, 0), (1, 1), (0, 2)))
        return self.lab[z, y, x]

    def force_indentation(self, T, target):
        m = self.model
        lo = m.first_contact_indentation(T)
        hi = lo + 25.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if m.force_at(T, mid) < target else (lo, mid)
        return hi

    def section(self, st, plane, fixed, a_rng, y_rng, res=0.5):
        """Undeformed and deformed CT on a plane: 'xy' (transverse, z = fixed) or 'zy' (sagittal, x = fixed).
        Deformed image I(x) = CT(warp_to_reference(x)) (backward warping, as the B-mode pipeline does)."""
        a = np.arange(a_rng[0], a_rng[1] + 1e-6, res)
        y = np.arange(y_rng[1], y_rng[0] - 1e-6, -res)        # top row = most posterior (skin up)
        A, Y = np.meshgrid(a, y)
        F = np.full_like(A, fixed)
        P = np.stack([A, Y, F], -1) if plane == "xy" else np.stack([F, Y, A], -1)
        ref = self.sample_ct(P)
        if st is None or not st.in_contact:
            return ref, ref, P, np.zeros_like(P)
        Xt = torch.tensor(P.reshape(-1, 3), dtype=torch.float32, device=self.dev)
        Xr = st.warp_to_reference(Xt).cpu().numpy().reshape(P.shape)
        u = (st.displace(Xt) - Xt).cpu().numpy().reshape(P.shape)
        return ref, self.sample_ct(Xr), P, u


def _ct_show(ax, img, ext):
    ax.imshow(np.clip(img, -250, 600), cmap="gray", vmin=-250, vmax=600, extent=ext, aspect="equal")


def _face_line(T, d, half, axis_col):
    o, b, e = T[:3, 3], T[:3, 1], T[:3, axis_col]
    c = o + d * b
    return c - half * e, c + half * e


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] {path}")


def _write_video(frames, stem, fps):
    mp4, gif = os.path.join(OUT, stem + ".mp4"), os.path.join(OUT, stem + ".gif")
    imageio.mimsave(mp4, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    imageio.mimsave(gif, frames[::2] if len(frames) > 60 else frames, duration=1000 / fps * (2 if len(frames) > 60 else 1),
                    loop=0)
    for p in (mp4, gif):
        print(f"[+] {p} ({os.path.getsize(p) / 1e6:.1f} MB)")


def _fig_frame(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    h, w = img.shape[:2]
    return img[: h - h % 8, : w - w % 8]


# ---------------------------------------------------------------------------- 1. deformation field
def fig_deformation(c: Ctx):
    m = c.model
    x, z = MIDLINE
    T = m.dorsal_pose(x, z)
    ytop = float(m.skin_height(x, z)) + 12
    rows = []
    for F in (4.0, 8.0):
        d = c.force_indentation(T, F)
        st = m.solve(T, d)
        ref, dfm, P, u = c.section(st, "xy", z, (-50, 50), (ytop - 50, ytop))
        rows.append((F, d, st, ref, dfm, P, u))
    ext = [-50, 50, ytop - 50, ytop]
    xs = np.linspace(-50, 50, 201)
    skin0 = np.stack([xs, m.skin_height(xs, np.full_like(xs, z)), np.full_like(xs, z)], 1)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), constrained_layout=True)
    for r, (F, d, st, ref, dfm, P, u) in enumerate(rows):
        _ct_show(axes[r, 0], ref, ext)
        axes[r, 0].set_title("CT, undeformed\n(transverse section, L4 midline)")
        _ct_show(axes[r, 1], dfm, ext)
        p0, p1 = _face_line(T, d, m.face_dims[0] / 2, 0)
        axes[r, 1].plot([p0[0], p1[0]], [p0[1], p1[1]], color=ORANGE, lw=3, solid_capstyle="round")
        sk = torch.tensor(skin0, dtype=torch.float32, device=c.dev)
        skd = st.displace(sk).cpu().numpy()
        for ax in axes[r, :2]:
            ax.plot(skin0[:, 0], skin0[:, 1], color=AQUA, lw=1.4, ls="--")
        axes[r, 1].plot(skd[:, 0], skd[:, 1], color=AQUA, lw=1.6)
        axes[r, 1].set_title(f"FEM-deformed, F = {st.reaction_force_n:.1f} N\nprobe face orange, skin before (dashed) / after")
        mag = np.linalg.norm(u, axis=-1)
        im = axes[r, 2].imshow(mag, cmap="Blues", vmin=0, vmax=max(1e-3, np.percentile(mag, 99.5)), extent=ext,
                               aspect="equal")
        s = 6
        axes[r, 2].quiver(P[::s, ::s, 0], P[::s, ::s, 1], u[::s, ::s, 0], u[::s, ::s, 1], color=INK, angles="xy",
                          scale_units="xy", scale=0.35, width=0.003)
        bone = c.sample_lab(P) == 1
        axes[r, 2].contour(P[..., 0], P[..., 1], bone.astype(float), levels=[0.5], colors=[ORANGE], linewidths=1.0)
        axes[r, 2].set_title(f"Displacement |u| (arrows x2.9), max {mag.max():.1f} mm\nL4 outline orange")
        fig.colorbar(im, ax=axes[r, 2], shrink=0.8, label="|u| mm")
        for ax in axes[r]:
            ax.set_xlabel("x mm (patient left +)")
            ax.set_ylabel("y mm (posterior +)")
    fig.suptitle("Probe pressing on the L4 midline: the flat face touches the paraspinal ridges first, "
                 "the bone (fixed) does not move", fontsize=11)
    _save(fig, "fig_deformation_field.png")


# ---------------------------------------------------------------------------- 2. tilt shear video
def video_tilt(c: Ctx):
    """Probe pitch sweep at 4 N (pitch = the env's tilt action): sagittal section through the probe,
    deformed CT + displacement, and the contact force vector against the beam (DEC-040)."""
    m = c.model
    x, z = FLAT
    pitches = np.concatenate([np.linspace(0, 12, 13), np.linspace(12, -12, 25), np.linspace(-12, 0, 13)])
    ytop = float(m.skin_height(x, z)) + 22
    ext = [z - 40, z + 40, ytop - 60, ytop]
    hist = {"p": [], "ang": [], "Fz": [], "Fy": []}
    frames = []
    fig = plt.figure(figsize=(13.0, 5.2), dpi=90)
    for p_deg in pitches:
        T = m.dorsal_pose(x, z, pitch=np.radians(p_deg))
        d = c.force_indentation(T, 4.0)
        st = m.solve(T, d)
        ref, dfm, P, u = c.section(st, "zy", x, (z - 40, z + 40), (ytop - 60, ytop))
        b = T[:3, 1]
        Fv = np.asarray(st.force_vec_n)
        ang = np.degrees(np.arccos(np.clip(Fv @ -b / np.linalg.norm(Fv), -1, 1)))
        hist["p"].append(p_deg); hist["ang"].append(ang); hist["Fz"].append(Fv[2]); hist["Fy"].append(Fv[1])
        fig.clf()
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.9], wspace=0.3)
        ax = fig.add_subplot(gs[0])
        _ct_show(ax, dfm, [ext[0], ext[1], ext[2], ext[3]])
        p0, p1 = _face_line(T, d, m.face_dims[1] / 2, 2)
        ax.plot([p0[2], p1[2]], [p0[1], p1[1]], color=ORANGE, lw=4, solid_capstyle="round")
        cpt = T[:3, 3] + d * b
        ax.annotate("", xy=(cpt[2] + 18 * Fv[2] / 4, cpt[1] + 18 * Fv[1] / 4), xytext=(cpt[2], cpt[1]),
                    arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=3), annotation_clip=False)
        ax.annotate("", xy=(cpt[2] + 18 * b[2], cpt[1] + 18 * b[1]), xytext=(cpt[2], cpt[1]),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.5, ls="--"), annotation_clip=False)
        ax.set_title(f"Deformed CT, sagittal, pitch {p_deg:+.0f} deg\nbeam (blue, dashed) vs force on probe (aqua)",
                     fontsize=9)
        ax.set_xlabel("z mm (superior +)")
        ax.set_ylabel("y mm (posterior +)")
        ax = fig.add_subplot(gs[1])
        lim = 1.5
        im = ax.imshow(u[..., 2], cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext, aspect="equal")
        s = 7
        ax.quiver(P[::s, ::s, 2], P[::s, ::s, 1], u[::s, ::s, 2], u[::s, ::s, 1], color=INK, angles="xy",
                  scale_units="xy", scale=0.35, width=0.004)
        ax.set_title("Sideways (z) tissue displacement u_z (mm)\nred = superior, blue = inferior", fontsize=9)
        ax.set_xlabel("z mm")
        fig.colorbar(im, ax=ax, shrink=0.75)
        ax = fig.add_subplot(gs[2])
        ax.plot(hist["p"], hist["Fz"], color=AQUA, label="force on probe, z component (N)")
        ax.plot(hist["p"], hist["ang"], color=BLUE, label="angle force vs beam (deg)")
        ax.axhline(0, color=GRID, lw=1)
        ax.set_xlim(-13, 13)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("probe pitch (deg)")
        ax.set_title(f"F = {st.reaction_force_n:.2f} N along the beam\n(before DEC-040 the force was always vertical)",
                     fontsize=9)
        ax.legend(loc="lower left", fontsize=7)
        frames.append(_fig_frame(fig))
    plt.close(fig)
    _write_video(frames, "tilt_shear_fem", fps=8)


# ---------------------------------------------------------------------------- 3. B-mode pipeline
def _frame_parts(env, c1, c2, yaw=0.0, pitch=0.0, force=4.0, state_override="aware"):
    env.target_force_n = force
    t_skin = env.skin_pose(c1, c2, yaw, pitch)
    d, _ = env.regulate_force(t_skin, 0.0, iterations=15)
    st = env.tissue_model.solve(t_skin, d)
    t_img = st.probe_face_pose(t_skin)
    ct_rigid = env.slice_extractor_ct.extract_slices(t_skin, interpolation="bilinear", tissue_state=None)
    ct_def = env.slice_extractor_ct.extract_slices(t_img, interpolation="bilinear", tissue_state=st)
    us = env.us_sim.simulate_bmode(ct_def)
    bm = us["bmode"]
    conf = env.conf_estimator.compute_confidence_map(bm)
    bm_np, conf_np = bm.detach().cpu().numpy().squeeze(), conf.detach().cpu().numpy().squeeze()
    pts = env.segmenter.segment_from_bmode(bm_np, confidence_map=conf_np)
    sq = lambda t: t.detach().cpu().numpy().squeeze()
    return dict(st=st, d=d, ct_rigid=sq(ct_rigid), ct_def=sq(ct_def), refl=sq(us["reflection"]),
                back=sq(us["backscatter"]), bmode=bm_np, conf=conf_np, pts=pts)


def fig_bmode(c: Ctx):
    env = c.env()
    env.reset(seed=0)
    f = _frame_parts(env, *MIDLINE)
    H, W = env.PHYSICAL_SIZE_MM
    ext = [-W / 2, W / 2, H, 0]
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.6), constrained_layout=True)
    panels = [
        (f["ct_rigid"], "1. CT slice at the skin pose\n(no deformation)", dict(cmap="gray", vmin=-250, vmax=600)),
        (f["ct_def"], f"2. CT slice, backward-warped\n(FEM, F = {f['st'].reaction_force_n:.1f} N, face {f['d']:+.1f} mm)",
         dict(cmap="gray", vmin=-250, vmax=600)),
        (f["ct_def"] - f["ct_rigid"], "3. Difference 2 - 1 (HU)\ntissue moved toward the probe",
         dict(cmap="RdBu_r", vmin=-400, vmax=400)),
        (f["refl"], "4. Interface reflection\n(impedance jumps, specular)", dict(cmap="gray")),
        (f["back"], "5. Backscatter\n(Rayleigh speckle)", dict(cmap="gray")),
        (f["bmode"], "6. B-mode\nattenuation + TGC + log compression", dict(cmap="gray", vmin=0, vmax=1)),
        (f["conf"], "7. Confidence map, dark = trusted\n(from the B-mode image only)", dict(cmap="Blues", vmin=0, vmax=1)),
        (f["bmode"], f"8. Bone detection: ridge + distal shadow\n{len(f['pts'])} points (orange)",
         dict(cmap="gray", vmin=0, vmax=1)),
    ]
    for ax, (img, title, kw) in zip(axes.ravel(), panels):
        ax.imshow(img, extent=ext, aspect="auto", **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("lateral mm")
        ax.set_ylabel("depth from probe face mm")
    if len(f["pts"]):
        ph, pw = env.IMAGE_SIZE
        px = f["pts"][:, 0] / (pw - 1) * W - W / 2
        py = f["pts"][:, 1] / (ph - 1) * H
        axes[1, 3].scatter(px, py, s=9, color=ORANGE, edgecolors="none")
    fig.suptitle("From CT to bone points: one frame at the L4 midline, 4 N (FEM)", fontsize=11)
    _save(fig, "fig_bmode_pipeline.png")


def fig_echo(c: Ctx):
    """The bone echo moves up the image as the force grows: deformation reaches the image."""
    env = c.env()
    env.reset(seed=0)
    H, W = env.PHYSICAL_SIZE_MM
    ar = H / (env.IMAGE_SIZE[0] - 1)
    cols = slice(env.IMAGE_SIZE[1] // 2 - 8, env.IMAGE_SIZE[1] // 2 + 8)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), constrained_layout=True)
    for ax, F in zip(axes, (1.0, 4.0, 8.0)):
        f = _frame_parts(env, *MIDLINE, force=F)
        e = detect_bone_echo_depth(f["bmode"], ar, cols)
        ax.imshow(f["bmode"], cmap="gray", vmin=0, vmax=1, extent=[-W / 2, W / 2, H, 0], aspect="auto")
        if np.isfinite(e):
            ax.axhline(e, color=ORANGE, lw=1.5, ls="--")
        ax.set_title(f"F = {f['st'].reaction_force_n:.1f} N: bone echo at {e:.1f} mm", fontsize=9)
        ax.set_xlabel("lateral mm")
        ax.set_ylabel("depth mm")
    sw = json.load(open(RESULTS_JSON))["results"]["soft_body"]
    ax = axes[3]
    for be, col in (("fem", BLUE), ("kelvinlet", ORANGE)):
        rows = sw[be]["force_sweep"]
        Fm = [r["model_force_n"] for r in rows]
        ax.plot(Fm, [r["echo_depth_mm"] for r in rows], "o-", color=col, label=f"{be}: detected in B-mode",
                markersize=5)
        ax.plot(Fm, [r["expected_echo_depth_mm"] for r in rows], "--", color=col, lw=1.2, label=f"{be}: expected")
    ax.set_xlabel("force N")
    ax.set_ylabel("bone echo depth mm")
    ax.set_title("Echo depth vs force (results/benchmark_20261105_120357.json)", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("Pressing harder squashes the tissue, so the bone echo appears shallower (FEM, L4 midline)",
                 fontsize=11)
    _save(fig, "fig_echo_vs_force.png")


# ---------------------------------------------------------------------------- 4. reconstruction
def _raster(c: Ctx):
    return c.suite.heuristic_trajectory("raster")


def video_recon(c: Ctx):
    env = c.env(seed=7000, deformation_mode="aware")
    env.reset(seed=7000)
    traj = _raster(c)
    gt = env.target_roi_bone_points
    tree_all = cKDTree(getattr(env, "all_bone_points", env.gt_bone_points))
    H, W = env.PHYSICAL_SIZE_MM
    ph, pw = env.IMAGE_SIZE
    cov, frames = [], []
    fig = plt.figure(figsize=(13, 5.0), dpi=85)
    for k, target in enumerate(traj):
        p = env.current_pose_params
        a = np.array([(target[0] - p[0]) / env.action_scale_trans, (target[1] - p[1]) / env.action_scale_trans,
                      (target[2] - p[2]) / env.action_scale_yaw, (target[3] - p[3]) / env.action_scale_pitch])
        obs, _, _, _, info = env.step(np.clip(a, -1, 1).astype(np.float32))
        cov.append(100 * obs["coverage_ratio"])
        pts = env.segmenter.segment_from_bmode(obs["bmode"], confidence_map=obs["confidence_map"])
        recon = obs["reconstructed_points"]
        fig.clf()
        gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.35, 0.9], wspace=0.25)
        ax = fig.add_subplot(gs[0])
        ax.imshow(obs["bmode"], cmap="gray", vmin=0, vmax=1, extent=[-W / 2, W / 2, H, 0], aspect="auto")
        if len(pts):
            ax.scatter(pts[:, 0] / (pw - 1) * W - W / 2, pts[:, 1] / (ph - 1) * H, s=6, color=ORANGE,
                       edgecolors="none")
        ax.set_title(f"Frame {k + 1}/{len(traj)}: B-mode + detected bone\nF = {obs['normal_force']:.1f} N", fontsize=9)
        ax.set_xlabel("lateral mm")
        ax.set_ylabel("depth mm")
        ax = fig.add_subplot(gs[1])
        ax.scatter(gt[:, 0], gt[:, 2], s=0.6, color="#cfcec9", edgecolors="none")
        if len(recon):
            err, _ = tree_all.query(recon)
            ax.scatter(recon[:, 0], recon[:, 2], s=2.0, c=np.clip(err, 0, 3), cmap="Blues_r", vmin=0, vmax=3,
                       edgecolors="none")
        t = env.skin_pose(*env.current_pose_params)
        o = t[:3, 3] + obs["indentation_mm"] * t[:3, 1]
        e = t[:3, 0] * 22.5
        ax.plot([o[0] - e[0], o[0] + e[0]], [o[2] - e[2], o[2] + e[2]], color=ORANGE, lw=3, solid_capstyle="round")
        ax.set_xlim(-45, 45); ax.set_ylim(-75, 5); ax.set_aspect("equal")
        ax.set_xlabel("x mm (patient left +)"); ax.set_ylabel("z mm (superior +)")
        ax.set_title("Top view: reconstructed points (blue, darker = closer to bone)\non the visible L4 surface (grey); probe face orange",
                     fontsize=9)
        ax = fig.add_subplot(gs[2])
        ax.plot(np.arange(1, len(cov) + 1), cov, color=BLUE)
        ax.set_xlim(0, len(traj))
        ax.set_ylim(0, 100)
        ax.set_xlabel("frame")
        ax.set_ylabel("Cov@3 %")
        ax.set_title(f"Coverage of the visible L4 surface: {cov[-1]:.1f} %", fontsize=9)
        frames.append(_fig_frame(fig))
        if k + 1 >= len(traj):
            break
    plt.close(fig)
    _write_video(frames, "reconstruction_raster_fem", fps=6)


def fig_recon_3d(c: Ctx):
    traj = _raster(c)
    rows = []
    for backend in ("fem", "kelvinlet"):
        for mode in ("rigid", "aware", "unaware"):
            env = c.suite.make_env(len(traj), 7000, tissue_backend=backend, deformation_mode=mode)
            env.reset(seed=7000)
            c.suite.follow(env, traj, {})
            recon = env.accumulator.get_points()
            rows.append((backend, mode, recon, 100 * env.coverage_history[-1], env))
    fig, axes = plt.subplots(4, 3, figsize=(14, 15.5), gridspec_kw=dict(height_ratios=[1.25, 0.8, 1.25, 0.8]),
                             constrained_layout=True)
    sc = None
    for i, (backend, mode, recon, cov, env) in enumerate(rows):
        gt = env.target_roi_bone_points
        tree = cKDTree(getattr(env, "all_bone_points", env.gt_bone_points))
        err = tree.query(recon)[0] if len(recon) else np.zeros(0)
        p2 = 100 * float(np.mean(err <= 2.0)) if len(err) else 0.0
        r, col = divmod(i, 3)
        ax, ax2 = axes[2 * r, col], axes[2 * r + 1, col]
        ax.scatter(gt[:, 0], gt[:, 2], s=0.8, color="#cfcec9", edgecolors="none")
        if len(recon):
            sc = ax.scatter(recon[:, 0], recon[:, 2], s=3, c=np.clip(err, 0, 4), cmap="Blues_r", vmin=0, vmax=4,
                            edgecolors="none")
        ax.set_xlim(-45, 45); ax.set_ylim(-62, -2); ax.set_aspect("equal")
        ax.set_title(f"{backend} / {mode}\nCov@3 {cov:.1f} %, {p2:.0f} % of points < 2 mm from bone", fontsize=10)
        ax.set_xlabel("x mm (top view)"); ax.set_ylabel("z mm (superior +)")
        sel = np.abs(gt[:, 2] + 32.5) < 3
        ax2.scatter(gt[sel, 0], gt[sel, 1], s=4, color="#9e9d97", edgecolors="none", label="visible L4 surface (truth)")
        if len(recon):
            selr = np.abs(recon[:, 2] + 32.5) < 3
            ax2.scatter(recon[selr, 0], recon[selr, 1], s=5, color=BLUE, edgecolors="none", label="reconstruction")
        ax2.set_xlim(-35, 35); ax2.set_ylim(5, 45); ax2.set_aspect("equal")
        ax2.set_title("side cut through L4 centre (z = -32.5 ± 3 mm)", fontsize=9)
        ax2.set_xlabel("x mm"); ax2.set_ylabel("y mm (posterior +)")
    axes[1, 0].legend(loc="upper left", fontsize=8, markerscale=3)
    if sc is not None:
        fig.colorbar(sc, ax=axes[0::2, :].ravel().tolist(), shrink=0.4, label="distance to nearest bone (mm)")
    fig.suptitle("Final reconstructions after the 72-frame raster (one episode, seed 7000). Grey: visible L4 "
                 "surface (evaluation only).\nrigid = no deformation; aware = deformed images, back-projected with "
                 "the indented probe pose; unaware = deformed images, back-projected as if the tissue were rigid",
                 fontsize=10)
    _save(fig, "fig_reconstruction_aware_vs_unaware.png")


# ---------------------------------------------------------------------------- 5. results plots
def fig_results(c: Ctx = None):
    res = json.load(open(RESULTS_JSON))["results"]
    pol = res["policies"]
    names = sorted(pol, key=lambda k: -pol[k]["mean_coverage3_pct"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    ax = axes[0]
    y = np.arange(len(names))
    ax.barh(y, [pol[n]["mean_coverage3_pct"] for n in names], xerr=[pol[n]["std_coverage3_pct"] for n in names],
            color=BLUE, height=0.6, error_kw=dict(ecolor=INK2, lw=1))
    for i, n in enumerate(names):
        ax.text(pol[n]["mean_coverage3_pct"] + pol[n]["std_coverage3_pct"] + 1, i,
                f"{pol[n]['mean_coverage3_pct']:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Cov@3 % (mean ± std, 3 trials)")
    ax.set_title("Policies, 72 frames each, FEM at 4 N")
    sb = res["soft_body"]
    ax = axes[1]
    modes = ["rigid", "aware", "unaware"]
    xx = np.arange(len(modes))
    for j, (be, col) in enumerate((("fem", BLUE), ("kelvinlet", ORANGE))):
        v = [sb[be]["ablation"][m]["mean_coverage3_pct"] for m in modes]
        ax.bar(xx + (j - 0.5) * 0.36, v, width=0.34, color=col, label=be)
        for k, val in enumerate(v):
            ax.text(xx[k] + (j - 0.5) * 0.36, val + 1.5, f"{val:.0f}", ha="center", fontsize=8)
    ax.set_xticks(xx, modes)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Cov@3 %")
    ax.set_title("Ignoring deformation (raster): coverage")
    ax.legend(loc="lower left")
    ax = axes[2]
    for j, (be, col) in enumerate((("fem", BLUE), ("kelvinlet", ORANGE))):
        v = [sb[be]["ablation"][m]["mean_precision2_visible_pct"] for m in modes]
        ax.bar(xx + (j - 0.5) * 0.36, v, width=0.34, color=col, label=be)
        for k, val in enumerate(v):
            ax.text(xx[k] + (j - 0.5) * 0.36, val + 1.5, f"{val:.0f}", ha="center", fontsize=8)
    ax.set_xticks(xx, modes)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Prec@2 vis %")
    ax.set_title("Ignoring deformation (raster): precision")
    ax.legend(loc="lower left")
    fig.suptitle(f"Benchmark ({RESULTS_JSON})", fontsize=10)
    _save(fig, "fig_benchmark.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    ax = axes[0]
    for be, col in (("fem", BLUE), ("kelvinlet", ORANGE)):
        rows = sb[be]["force_sweep"]
        ax.plot([r["indentation_from_first_contact_mm"] for r in rows], [r["model_force_n"] for r in rows], "o-",
                color=col, label=be, markersize=5)
    dd = np.linspace(0, 10, 50)
    ax.plot(dd, 0.5 / 0.15 * np.expm1(0.15 * dd), "--", color=INK2, lw=1.2, label="literature fit (k0 0.5, β 0.15)")
    ax.set_xlabel("indentation from first contact (mm)")
    ax.set_ylabel("force N")
    ax.set_title("Force vs indentation, L4 midline, untilted")
    ax.legend()
    ax = axes[1]
    ks = ["k0_0.3", "k0_0.5", "k0_0.8"]
    for be, col in (("fem", BLUE), ("kelvinlet", ORANGE)):
        ax.plot([sb[be]["stiffness_sweep"][k]["modulus_kpa"] for k in ks],
                [sb[be]["stiffness_sweep"][k]["mean_indentation_mm"] for k in ks], "o-", color=col, label=be,
                markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("muscle modulus kPa (log)")
    ax.set_ylabel("mean indentation at 4 N (mm)")
    ax.set_title("Stiffer tissue, shallower probe (coverage stays 76–77 %)")
    ax.legend()
    _save(fig, "fig_force_and_stiffness.png")


JOBS = {"deformation": fig_deformation, "tilt": video_tilt, "bmode": fig_bmode, "echo": fig_echo,
        "recon_video": video_recon, "recon_3d": fig_recon_3d, "results": fig_results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(JOBS))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    names = [n for n in args.only.split(",") if n]
    ctx = None if names == ["results"] else Ctx()
    for n in names:
        JOBS[n](ctx)


if __name__ == "__main__":
    main()
