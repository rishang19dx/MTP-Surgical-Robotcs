"""
PyBullet rendering of a demo scene: the whole body (skin, bones, organs) with the deforming region rebuilt every
frame, a displacement heat map in colour bands, the probe, and (in snapshots) the image plane, a force arrow, the
deformed vs undeformed skin profile in the image plane and live text. Scene frame (mm) -> PyBullet (m):
(x, -z, y) / 1000, a proper rotation that puts the scanned skin (+y) up.

PyBullet cannot recolour vertices or update a mesh in place (resetMeshData is for soft bodies only, and loading a
12k-vertex surface as a soft body hangs), so the moving faces are rebuilt per frame, split into colour bands by |u|.
Snapshots use the hardware renderer of the GUI client (the EGL plugin cannot convert in-memory meshes and crashes;
without a GUI the tiny CPU renderer is used, which ignores transparency). Debug lines and text are not part of camera
images, so snapshots draw the overlays themselves by projecting the 3D points with the camera matrices.

snapshot(cut=True) hides the static skin for that one image: the deforming skin patch is drawn over the organs and
bones underneath.
"""

from __future__ import annotations

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

BANDS_MM = [0.25, 0.5, 1.0, 2.0, 4.0]                    # band edges: < 0.25 mm is shown in the base colour
BAND_RGB = [(0.99, 0.91, 0.55), (0.99, 0.70, 0.30), (0.95, 0.45, 0.20), (0.85, 0.20, 0.25), (0.55, 0.05, 0.35)]
BG_TOP, BG_BOTTOM = np.array([13, 15, 20]), np.array([34, 39, 50])


def to_pb(v):
    v = np.asarray(v, float)
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], -1) / 1000.0


def _font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


class PyBulletView:
    def __init__(self, engine, gui=True, magnify=1.0, cutaway=False, width=1280, height=720):
        self.eng, self.mag, self.cutaway, self.gui = engine, float(magnify), cutaway, gui
        self.cid = p.connect(p.GUI if gui else p.DIRECT,
                             options=f"--width={width} --height={height} --background_color_red=0.06 "
                                     f"--background_color_green=0.08 --background_color_blue=0.10" if gui else "")
        self.egl = False
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.cid)
        self.W, self.H = width, height
        self.dyn_bodies, self.debug_ids = [], []
        self.fr = None
        self._static()
        self._probe()
        lo, hi = engine.roi
        c = to_pb(0.5 * (lo + hi))
        if gui:
            p.resetDebugVisualizerCamera(0.42, 35, -28, c.tolist(), physicsClientId=self.cid)
        self.font, self.small = _font(26), _font(20)

    # ------------------------------------------------------------ static scene
    def _body(self, v, f, rgba):
        if len(f) == 0:
            return None
        vis = p.createVisualShape(p.GEOM_MESH, vertices=to_pb(v).tolist(), indices=np.asarray(f).ravel().tolist(),
                                  rgbaColor=list(rgba), physicsClientId=self.cid)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, physicsClientId=self.cid)

    def _skin_alpha(self, dynamic):
        return 0.8 if dynamic else 0.45

    def _static(self):
        for name, m in self.eng.meshes.items():
            meta = m["meta"]
            dyn = m["moving"][m["f"]].any(1) if meta.get("deforms") else np.zeros(len(m["f"]), bool)
            m["f_dyn"] = m["f"][dyn]
            f_static = m["f"][~dyn]
            alpha = self._skin_alpha(False) if name == "skin" else meta["opacity"]
            b = self._body(m["v"], f_static, (*meta["color"], alpha))
            if name == "skin":
                self.skin_static, self.skin_rgba = b, [*meta["color"], alpha]

    def _probe(self):
        hx, hz, hy = 0.0225, 0.005, 0.03
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hz, hy], rgbaColor=[0.15, 0.55, 0.95, 0.75],
                                  physicsClientId=self.cid)
        self.probe = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, physicsClientId=self.cid)
        self._probe_half_y = hy

    # ------------------------------------------------------------ per frame
    def update(self, fr):
        self.fr = fr
        for b in self.dyn_bodies:
            p.removeBody(b, physicsClientId=self.cid)
        self.dyn_bodies = []
        for name, m in self.eng.meshes.items():
            if name not in fr.verts or len(m.get("f_dyn", [])) == 0:
                continue
            v = m["v"] + self.mag * (fr.verts[name] - m["v"])
            fd, d = m["f_dyn"], fr.disp[name][m["f_dyn"]].max(1)
            base = (*m["meta"]["color"], self._skin_alpha(True) if name == "skin" else m["meta"]["opacity"])
            b = self._body(v, fd[d < BANDS_MM[0]], base)
            if b is not None:
                self.dyn_bodies.append(b)
            for k in range(len(BANDS_MM)):
                hi = BANDS_MM[k + 1] if k + 1 < len(BANDS_MM) else np.inf
                sel = (d >= BANDS_MM[k]) & (d < hi)
                b = self._body(v, fd[sel], (*BAND_RGB[k], 0.95))
                if b is not None:
                    self.dyn_bodies.append(b)
        # probe: face at the indented pose, body above it along -beam
        T = fr.pose_face
        R = T[:3, :3]
        b = R[:, 1] / np.linalg.norm(R[:, 1])
        centre = T[:3, 3] - b * self._probe_half_y * 1000.0
        M = np.stack([R[:, 0], -R[:, 2], R[:, 1]], 1)
        Rpb = np.stack([to_pb(M[:, 0]) * 1000, to_pb(M[:, 1]) * 1000, to_pb(-b) * 1000], 1)
        p.resetBasePositionAndOrientation(self.probe, to_pb(centre).tolist(), _quat_from_matrix(Rpb),
                                          physicsClientId=self.cid)
        if self.gui:
            self._debug_overlays(fr)

    def _geometry(self, fr):
        """Overlay geometry in scene mm: image plane corners, force arrow, undeformed / deformed skin profiles."""
        T = fr.pose_face
        R = T[:3, :3]
        lat, beam = R[:, 0] / np.linalg.norm(R[:, 0]), R[:, 1] / np.linalg.norm(R[:, 1])
        o = T[:3, 3]
        plane = [o - 25 * lat, o + 25 * lat, o + 25 * lat + 75 * beam, o - 25 * lat + 75 * beam]
        arrow = (o - beam * (12.0 * fr.force_n) - beam * 60.0, o - beam * 60.0)
        m = self.eng.meshes["skin"]
        v0, v1 = m["v"], fr.verts["skin"]
        rel = v0 - o
        near = (np.abs(rel @ np.cross(lat, beam)) < 1.5) & (np.abs(rel @ lat) < 45)
        idx = np.nonzero(near)[0]
        prof = None
        if len(idx) > 3:
            order = idx[np.argsort(rel[idx] @ lat)]
            prof = (v0[order], (v0 + self.mag * (v1 - v0))[order])
        return plane, arrow, prof

    def _debug_overlays(self, fr):
        for i in self.debug_ids:
            p.removeUserDebugItem(i, physicsClientId=self.cid)
        self.debug_ids = []
        plane, arrow, prof = self._geometry(fr)
        for a, b2 in zip(plane, plane[1:] + plane[:1]):
            self.debug_ids.append(p.addUserDebugLine(to_pb(a), to_pb(b2), [0.3, 0.8, 1.0], 2, physicsClientId=self.cid))
        self.debug_ids.append(p.addUserDebugLine(to_pb(arrow[0]), to_pb(arrow[1]), [1.0, 0.85, 0.2], 5,
                                                 physicsClientId=self.cid))
        if prof is not None:
            for arr, col in zip(prof, ([0.9, 0.9, 0.9], [1.0, 0.45, 0.1])):
                for a, b2 in zip(arr[:-1], arr[1:]):
                    self.debug_ids.append(p.addUserDebugLine(to_pb(a), to_pb(b2), col, 3, physicsClientId=self.cid))

    # ------------------------------------------------------------ snapshots
    def snapshot(self, yaw=35, pitch=-28, dist=0.42, label="", cut=False):
        if cut and getattr(self, "skin_static", None) is not None:
            p.resetBasePositionAndOrientation(self.skin_static, [0, 0, 50], [0, 0, 0, 1], physicsClientId=self.cid)
        try:
            return self._snapshot(yaw, pitch, dist, label)
        finally:
            if cut and getattr(self, "skin_static", None) is not None:
                p.resetBasePositionAndOrientation(self.skin_static, [0, 0, 0], [0, 0, 0, 1], physicsClientId=self.cid)

    def _snapshot(self, yaw, pitch, dist, label):
        lo, hi = self.eng.roi
        c = to_pb(0.5 * (lo + hi))
        view = p.computeViewMatrixFromYawPitchRoll(c.tolist(), dist, yaw, pitch, 0, 2, physicsClientId=self.cid)
        proj = p.computeProjectionMatrixFOV(45, self.W / self.H, 0.01, 3.0, physicsClientId=self.cid)
        hw = self.gui or self.egl
        img = p.getCameraImage(self.W, self.H, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL if hw else
                               p.ER_TINY_RENDERER, lightDirection=[0.4, -0.3, 1.0], shadow=0,
                               physicsClientId=self.cid)
        rgb = np.reshape(img[2], (self.H, self.W, 4))[..., :3].astype(np.uint8)
        seg = np.reshape(img[4], (self.H, self.W))
        w = np.linspace(0, 1, self.H)[:, None, None]
        bg = (BG_TOP * (1 - w) + BG_BOTTOM * w).astype(np.uint8)
        rgb = np.where((seg < 0)[..., None], np.broadcast_to(bg, rgb.shape), rgb)
        im = Image.fromarray(rgb)
        if self.fr is not None:
            self._draw_overlays(im, np.asarray(view).reshape(4, 4).T, np.asarray(proj).reshape(4, 4).T, label)
        return np.asarray(im)

    def _project(self, V, P, pts_mm):
        X = to_pb(np.atleast_2d(pts_mm))
        h = np.c_[X, np.ones(len(X))] @ (P @ V).T
        ndc = h[:, :3] / h[:, 3:4]
        return np.c_[(ndc[:, 0] + 1) * 0.5 * self.W, (1 - ndc[:, 1]) * 0.5 * self.H]

    def _draw_overlays(self, im, V, P, label):
        fr = self.fr
        d = ImageDraw.Draw(im, "RGBA")
        plane, arrow, prof = self._geometry(fr)
        q = [tuple(v) for v in self._project(V, P, plane)]
        d.polygon(q, fill=(80, 200, 255, 40), outline=(80, 200, 255, 230))
        if prof is not None:
            for arr, col in zip(prof, ((235, 235, 235, 255), (255, 120, 30, 255))):
                d.line([tuple(v) for v in self._project(V, P, arr)], fill=col, width=3)
        a0, a1 = self._project(V, P, np.stack(arrow))
        d.line([tuple(a0), tuple(a1)], fill=(255, 215, 60, 255), width=6)
        u = (a1 - a0) / (np.linalg.norm(a1 - a0) + 1e-9)
        n = np.array([-u[1], u[0]])
        d.polygon([tuple(a1 + 14 * u), tuple(a1 - 4 * u + 9 * n), tuple(a1 - 4 * u - 9 * n)], fill=(255, 215, 60, 255))
        d.text(tuple(a0 + np.array([10, -10])), f"{fr.force_n:.1f} N", font=self.small, fill=(255, 215, 60, 255))
        lines = [self.eng.sc["title"] + (f"  ·  {label}" if label else ""),
                 f"force {fr.force_n:4.1f} N    indentation {fr.indent_mm:4.1f} mm",
                 f"max tissue displacement {fr.max_u_mm:5.2f} mm" + (f"  (shown x{self.mag:g})" if self.mag != 1 else "")]
        wmax = max(d.textlength(t, font=self.font) for t in lines)
        d.rounded_rectangle([12, 12, 32 + wmax, 22 + 36 * len(lines)], 8, fill=(0, 0, 0, 150))
        for i, t in enumerate(lines):
            d.text((22, 18 + 36 * i), t, font=self.font if i else self.font, fill=(240, 242, 245, 255))
        # colour key for the bands
        x0, y0 = self.W - 230, self.H - 40 - 26 * (len(BANDS_MM) + 1)
        d.rounded_rectangle([x0 - 12, y0 - 40, self.W - 12, self.H - 20], 8, fill=(0, 0, 0, 150))
        d.text((x0, y0 - 34), "tissue displacement", font=self.small, fill=(240, 242, 245, 255))
        edges = BANDS_MM + [np.inf]
        for k, rgb in enumerate(BAND_RGB):
            y = y0 + 26 * k
            d.rectangle([x0, y, x0 + 30, y + 18], fill=tuple(int(255 * c) for c in rgb) + (255,))
            hi = edges[k + 1]
            txt = f"{BANDS_MM[k]:g}-{hi:g} mm" if np.isfinite(hi) else f"> {BANDS_MM[k]:g} mm"
            d.text((x0 + 40, y - 1), txt, font=self.small, fill=(230, 232, 236, 255))
        y = y0 + 26 * len(BAND_RGB)
        d.line([x0, y + 9, x0 + 30, y + 9], fill=(235, 235, 235, 255), width=3)
        d.text((x0 + 40, y - 1), "skin, at rest", font=self.small, fill=(230, 232, 236, 255))
        y += 26
        d.line([x0, y + 9, x0 + 30, y + 9], fill=(255, 120, 30, 255), width=3)
        d.text((x0 + 40, y - 1), "skin, pressed", font=self.small, fill=(230, 232, 236, 255))


def _quat_from_matrix(R):
    R = np.asarray(R, float)
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return [(R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q
