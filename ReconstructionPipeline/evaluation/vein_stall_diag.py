"""DEC-075 addendum, POST-HOC diagnosis of the DEC-075 patient stall (TRAINING veins already swept by the gate; test veins untouched).

    ./.venv/bin/python ReconstructionPipeline/evaluation/vein_stall_diag.py <vein name> <e_fat_eff_kpa> [<edge_radius_mm> <psd 0|1>]   (run from ~/ds404_work/MTP)

Repeats the gate sweep (cavity lumen, h = 2 mm, DEC-061 ROI). At the first failed increment it re-runs Newton on that
increment with instrumentation and prints, per iteration: residual, line-search step, min J and where it is (material,
depth below skin, distance to the probe face edge, distance to the vein, box faces), the dofs with the largest
residual (same location breakdown), and the contact state (probe penetration, active columns, wall pairs active).
Writes <vein>_r<r>_p<psd>_diag.json into $DIAG_OUT (default: the working directory); summarised by hand into results/vein_cavity_stall_diag_20261128.json / .md.
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla
import torch

sys.path.insert(0, os.path.expanduser("~/ds404_work/MTP"))
os.chdir(os.path.expanduser("~/ds404_work/MTP"))
torch.set_num_threads(4)
from mtp_gym.core import SHARED_E_SCALE  # noqa: E402
from ReconstructionPipeline.evaluation import vein_compress_study as vs  # noqa: E402
from ReconstructionPipeline.evaluation.vein_nl_study import E_LUMEN_VOID, NU_LUMEN  # noqa: E402
from ReconstructionPipeline.simulation.fem_contact_nl import NeoHookeanContact, _frame  # noqa: E402
from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel  # noqa: E402

name, e_fat = sys.argv[1], float(sys.argv[2])
edge_r = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
psd = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
data, vein, fov = vs.load(os.path.join("data", "vein_ts2", name))
site, _ = vs.site_and_path(data, vein, fov)
ox, oy, oz = data["origin_mm"]
ct = data["ct_volume"]
lo = np.array([max(site["x"] - vs.ROI_HALF_X, ox), oy, -vs.ROI_HALF_Z])
hi = np.array([min(site["x"] + vs.ROI_HALF_X, ox + ct.shape[2] - 1), oy + ct.shape[1] - 1 + 4.0, vs.ROI_HALF_Z])
m = FEMTissueModel(data, roi_min_mm=lo, roi_max_mm=hi, element_size_mm=vs.H_MM, cache_dir=None,
                   device=torch.device("cpu"), verbose=False, precompute=False, e_fat_kpa=e_fat / SHARED_E_SCALE,
                   region_modulus=(vein, E_LUMEN_VOID / SHARED_E_SCALE, NU_LUMEN))
m.set_parameters(e_scale=SHARED_E_SCALE)
nh = NeoHookeanContact(m, cavity=True, edge_radius_mm=edge_r, psd_fallback=psd)
T = m.dorsal_pose(site["x"], site["z"])
o, ex, b, ez = _frame(T)
X = nh.X.numpy()
tets = nh.tets.numpy()
cent = X[tets].mean(1)
mesh = m.mesh
E_cell = np.where(mesh.cell_hu < m.fat_hu, 1, 2)                       # 1 fat, 2 muscle
reg = m.cells_in_region()
keep = ~reg[mesh.tet_cell]
mat = E_cell[mesh.tet_cell][keep]
vein_c = 0.5 * (np.asarray(site["top_wall"], float) + np.asarray(site["bottom_wall"], float))
cav_nodes = np.unique(nh._cavity_faces)
from scipy.spatial import cKDTree  # noqa: E402
cav_tree = cKDTree(X[cav_nodes])
skin_tree = cKDTree(nh.skin_ref)
bmin, bmax = X.min(0), X.max(0)
L, W = m.face_dims


def where(p):
    r = p - o
    lat, ele, depth_b = r @ ex, r @ ez, r @ b
    edge = min(abs(abs(lat) - L / 2), abs(abs(ele) - W / 2)) if (abs(lat) <= L / 2 + 4 and abs(ele) <= W / 2 + 4) else None
    return {"under_face": bool(abs(lat) <= L / 2 and abs(ele) <= W / 2), "lat": round(float(lat), 1),
            "ele": round(float(ele), 1), "depth_from_probe_origin": round(float(depth_b), 1),
            "dist_skin": round(float(skin_tree.query(p)[0]), 1), "dist_face_edge": None if edge is None else round(edge, 1),
            "dist_cavity": round(float(cav_tree.query(p)[0]), 1),
            "dist_box_side": round(float(min(np.min(p - bmin), np.min(bmax - p))), 1)}


def jac(u):
    xe = (nh.X + torch.as_tensor(u).view(-1, 3))[nh.tets]
    Ds = torch.stack([xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0], xe[:, 3] - xe[:, 0]], 2)
    return torch.linalg.det(Ds @ nh.Dm_inv).numpy()


def snapshot(u, d, tag):
    J = jac(u)
    k = int(np.argmin(J))
    low = np.argsort(J)[:20]
    g, K, p, w = nh.grad_hess(u, T, d, None)
    gf = np.zeros_like(g)
    gf[nh.fidx] = g[nh.fidx]
    nodes_g = np.argsort(-np.linalg.norm(gf.reshape(-1, 3), axis=1))[:5]
    q = nh._cavity_terms(u)[3] if nh._pairs is not None else []
    act = p > 0
    return {"tag": tag, "minJ": float(J[k]), "minJ_mat": ["", "fat", "muscle"][int(mat[k])], "minJ_at": where(cent[k]),
            "n_J_below_0.2": int((J < 0.2).sum()), "low20_mats": [int(v) for v in np.bincount(mat[low], minlength=3)[1:]],
            "low20_under_face": int(sum(where(cent[i])["under_face"] for i in low)),
            "low20_near_edge": int(sum((where(cent[i])["dist_face_edge"] or 99) <= 3 for i in low)),
            "low20_near_cavity": int(sum(where(cent[i])["dist_cavity"] <= 3 for i in low)),
            "res_norm": float(np.linalg.norm(gf)),
            "top_res_nodes": [{"node": int(n), "g": round(float(np.linalg.norm(gf.reshape(-1, 3)[n])), 4),
                               "skin": bool(n in set(nh.skin.tolist())), **where(X[n])} for n in nodes_g],
            "probe_active_cols": int(act.sum()), "probe_max_pen_mm": float(p.max()), "probe_force_n": float(nh.k * (w * p).sum()),
            "wall_pairs_active": int(len(q))}


hist, t0 = [], time.time()
d0 = float(m.first_contact_indentation(T))
u = np.zeros(nh.n)
nh._pairs = nh.cavity_pairs(T)
dd, step, out = 0.0, 0.5, {"vein": name, "e_fat": e_fat, "tets": int(len(tets)), "site": {k: (np.asarray(v).tolist() if hasattr(v, "__len__") else v) for k, v in site.items()}}
while dd < 40:
    u_try, it, res = nh._newton(u, T, d0 + dd + step, 30, 1e-6)
    if res < 1e-6:
        u, dd = u_try, dd + step
        F = nh.force(u, T, d0 + dd)
        hist.append((round(dd, 3), round(F, 3), round(nh.cavity_ratio(u, vein_c), 3)))
        print("ok", hist[-1], f"{time.time() - t0:.0f}s psd_steps {nh.stats['psd_steps']}", flush=True)
        if hist[-1][2] <= 0.1 or F >= 40:
            break
        step = min(0.5, step * 1.5)
        continue
    step /= 2
    if step >= 1 / 32:
        continue
    break
out["curve"] = hist
out["edge_radius_mm"], out["psd_fallback"], out["psd_steps"] = edge_r, psd, nh.stats["psd_steps"]
out["collapsed"] = bool(hist and hist[-1][2] <= 0.1)
if out["collapsed"] or (hist and hist[-1][1] >= 40):
    json.dump(out, open(os.path.join(os.environ.get("DIAG_OUT", "."), f"{name}_r{edge_r:g}_p{int(psd)}_diag.json"), "w"), indent=1)
    print("reached", "collapse" if out["collapsed"] else "force cap", hist[-1])
    sys.exit(0)
out["at_last_converged"] = snapshot(u, d0 + dd, "last converged")
# instrumented Newton on the failing increment
dfail = d0 + dd + 1 / 32
it_log = []
uu = u.copy()
fi = nh.fidx
scale = None
for itn in range(12):
    g, K, p, w = nh.grad_hess(uu, T, dfail, None)
    gf = g[fi]
    scale = scale or max(float(np.linalg.norm(gf)), nh.k * 1e-3)
    du = np.zeros(nh.n)
    du[fi] = spla.spsolve(K[fi][:, fi].tocsc(), -gf)
    e0, a = nh.energy(uu, T, dfail, w), 1.0
    reasons = []
    while a > 1e-6:
        e1 = nh.energy(uu + a * du, T, dfail, w)
        if e1 <= e0 + 1e-12 * abs(e0):
            break
        reasons.append("inverted" if not np.isfinite(e1) else "energy_up")
        a *= 0.5
    J_try = jac(uu + du)
    it_log.append({"it": itn, "rel_res": float(np.linalg.norm(gf) / scale), "alpha": a, "max_du_mm": float(np.abs(du).max()),
                   "full_step_inverts_tets": int((J_try <= 0).sum()),
                   "full_step_inverted_mats": [int(v) for v in np.bincount(mat[J_try <= 0], minlength=3)[1:]],
                   "first_rejections": reasons[:3]})
    print("it", it_log[-1], flush=True)
    if a <= 1e-6:
        break
    uu = uu + a * du
out["failing_increment_newton"] = it_log
out["at_failure"] = snapshot(uu, dfail, "inside failing increment")
print(json.dumps({k: out[k] for k in ("at_last_converged", "at_failure")}, indent=1))
json.dump(out, open(os.path.join(os.environ.get("DIAG_OUT", "."), f"{name}_r{edge_r:g}_p{int(psd)}_diag.json"), "w"), indent=1)
