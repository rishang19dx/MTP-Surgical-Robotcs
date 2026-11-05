"""
Voxel-lattice tetrahedral mesh of CT soft tissue for the patient-specific FEM tissue model (DEC-037).

Frame / units
-------------
Frame P (contract C7): the CT array frame as loaded by `PatientAssetGenerator`; volume indexed
[z, y, x], physical point = origin + index * spacing (mm), +Y toward the dorsal skin.

Construction
------------
* A regular lattice of cubic cells (edge `h` mm, default 4 mm) is laid over a region of interest
  (ROI) box in P.  Each cell is classified by sub-sampling the CT on an m^3 sub-grid
  (m = ceil(h / voxel_spacing)):
    - bone  : fraction of bone samples >= `bone_fraction` (bone = label == 1 OR HU > `bone_hu`;
              the HU test catches vertebrae that are not in the label volume, e.g. when the env
              loads only L4),
    - soft  : fraction of body samples (HU > `body_hu`) >= 0.5 and not bone,
    - empty : otherwise (air, or outside the CT).
* Every soft cell is split into 6 Kuhn tetrahedra (main diagonal corner 0 -> 7).  The split is
  identical in every cell, so the mesh is conforming and all elements are one of 6 reference tets.
* Dirichlet (fixed) nodes: every node of a bone cell, plus all nodes on the ROI's deep face (j = 0)
  and four side faces (i = 0 / NX-1, k = 0 / NZ-1).  The dorsal (top) face is free.
* Soft components (6-connected cells) that touch fewer than 4 fixed nodes are removed (they would be
  floating / rigid-body modes).
* Skin nodes: the top-most mesh node of every lattice column (i, k).  The rendering surface is the
  height field over the column grid (two triangles per quad of 4 columns that all have a skin node)
  with each vertex lifted to the sub-voxel skin height map y_skin(x, z), not the staircased node.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Kuhn split of the unit cube along the 0-7 diagonal; corner c = (bit0: x, bit1: y, bit2: z)
CUBE_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.int64
)
KUHN_TETS = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])


@dataclass
class TetMesh:
    h: float                       # lattice spacing (mm)
    lattice_origin: np.ndarray     # (3,) P mm of lattice node (0,0,0)
    lattice_shape: Tuple[int, int, int]  # node counts (NX, NY, NZ)
    node_ijk: np.ndarray           # (Nn,3) lattice indices of mesh nodes
    nodes: np.ndarray              # (Nn,3) reference node positions, P mm
    tets: np.ndarray               # (Nt,4) node indices
    tet_ref: np.ndarray            # (Nt,) index 0..5 of the Kuhn reference tet
    tet_cell: np.ndarray           # (Nt,) index into cell_ijk
    cell_ijk: np.ndarray           # (Nc,3) soft cells
    cell_hu: np.ndarray            # (Nc,) mean HU of body samples in the cell
    fixed: np.ndarray              # (Nn,) bool Dirichlet
    bone_node: np.ndarray          # (Nn,) bool node belongs to a bone cell
    skin_nodes: np.ndarray         # (S,) node indices, top node of each column
    skin_column_ik: np.ndarray     # (S,2) column (i,k) of each skin node
    column_skin: np.ndarray        # (NX,NZ) skin index into skin_nodes, -1 if none
    skin_vertices_ref: np.ndarray  # (S,3) render vertices (x, y_skin(x,z), z)
    skin_tris: np.ndarray          # (T,3) triangles indexing skin_vertices_ref / skin_nodes
    cell_class: np.ndarray = field(repr=False, default=None)  # (nx,ny,nz) 0 empty, 1 soft, 2 bone

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def summary(self) -> str:
        return (f"h={self.h} mm lattice {self.lattice_shape}: {len(self.cell_ijk)} soft cells, {len(self.tets)} tets, "
                f"{self.num_nodes} nodes ({int(self.fixed.sum())} fixed), {len(self.skin_nodes)} skin nodes")


# ---------------------------------------------------------------------------
# Skin height map
# ---------------------------------------------------------------------------
def skin_height_map_from_patient(patient_data: Dict[str, Any]) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """(height (nz,nx) mm, origin_xz, spacing_xz).  Uses patient_data['skin_height_map'] if present
    (the ENV agent's map, so the env's probe origin and the FEM gap agree), otherwise computes a
    sub-voxel map from the CT body mask (HU = -200 crossing, linear interpolation along +Y)."""
    if "skin_height_map" in patient_data:
        return (np.asarray(patient_data["skin_height_map"], dtype=np.float64),
                tuple(patient_data["skin_map_origin_xz"]), tuple(patient_data["skin_map_spacing_xz"]))
    ct = np.asarray(patient_data["ct_volume"], dtype=np.float32)
    ox, oy, oz = (float(v) for v in patient_data["origin_mm"])
    sx, sy, sz = (float(v) for v in patient_data["spacing_mm"])
    thr = -200.0
    body = ct > thr
    ny = ct.shape[1]
    valid = body.any(axis=1)
    top = ny - 1 - np.argmax(body[:, ::-1, :], axis=1)            # (nz,nx)
    zi, xi = np.indices(top.shape)
    hu_in = ct[zi, top, xi]
    hu_out = ct[zi, np.minimum(top + 1, ny - 1), xi]
    frac = np.where((top < ny - 1) & (hu_in > hu_out), (hu_in - thr) / np.maximum(hu_in - hu_out, 1e-6), 0.0)
    height = oy + (top + np.clip(frac, 0.0, 1.0)) * sy
    if not valid.all() and valid.any():
        from scipy.ndimage import distance_transform_edt
        _, (iz, ix) = distance_transform_edt(~valid, return_indices=True)
        height = height[iz, ix]
    return height.astype(np.float64), (ox, oz), (sx, sz)


def bilinear_height(height: np.ndarray, origin_xz, spacing_xz, x, z) -> np.ndarray:
    nz, nx = height.shape
    fx = np.clip((np.asarray(x, dtype=np.float64) - origin_xz[0]) / spacing_xz[0], 0, nx - 1)
    fz = np.clip((np.asarray(z, dtype=np.float64) - origin_xz[1]) / spacing_xz[1], 0, nz - 1)
    x0 = np.clip(np.floor(fx).astype(np.int64), 0, max(nx - 2, 0))
    z0 = np.clip(np.floor(fz).astype(np.int64), 0, max(nz - 2, 0))
    x1 = np.minimum(x0 + 1, nx - 1)
    z1 = np.minimum(z0 + 1, nz - 1)
    tx, tz = fx - x0, fz - z0
    return (height[z0, x0] * (1 - tx) * (1 - tz) + height[z0, x1] * tx * (1 - tz)
            + height[z1, x0] * (1 - tx) * tz + height[z1, x1] * tx * tz)


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------
def default_roi(patient_data: Dict[str, Any],
                c1_bounds: Tuple[float, float] = (-32.0, 32.0),
                c2_bounds: Tuple[float, float] = (-55.0, -10.0),
                footprint_margin_mm: float = 30.0,
                deep_y_mm: float = -10.0) -> Tuple[np.ndarray, np.ndarray]:
    """ROI box (min, max) in P mm around a dorsal scan window (default: the env's L4 window
    x in [-32,32], z in [-55,-10]) padded by `footprint_margin_mm` (probe half-length + boundary
    distance) and clipped to the CT bounds.  Deep face at y = `deep_y_mm`; top 4 mm above the CT."""
    shape = np.asarray(patient_data["ct_volume"]).shape   # (nz,ny,nx)
    o = np.asarray(patient_data["origin_mm"], dtype=np.float64)
    s = np.asarray(patient_data["spacing_mm"], dtype=np.float64)
    vmax = o + s * (np.array([shape[2], shape[1], shape[0]]) - 1)
    lo = np.array([c1_bounds[0] - footprint_margin_mm, deep_y_mm, c2_bounds[0] - footprint_margin_mm])
    hi = np.array([c1_bounds[1] + footprint_margin_mm, vmax[1] + 4.0, c2_bounds[1] + footprint_margin_mm])
    lo = np.maximum(lo, o)
    hi[[0, 2]] = np.minimum(hi[[0, 2]], vmax[[0, 2]])
    return lo, hi


def volume_hash(patient_data: Dict[str, Any]) -> str:
    """Stable content hash of CT + labels + geometry (cache key component)."""
    hsh = hashlib.sha1()
    ct = np.ascontiguousarray(np.asarray(patient_data["ct_volume"], dtype=np.float32))
    hsh.update(str(ct.shape).encode())
    hsh.update(ct[::2, ::2, ::2].tobytes())
    lbl = patient_data.get("label_volume")
    if lbl is not None:
        hsh.update(np.ascontiguousarray(np.asarray(lbl)[::2, ::2, ::2] == 1).tobytes())
    hsh.update(np.asarray(patient_data["origin_mm"], dtype=np.float64).tobytes())
    hsh.update(np.asarray(patient_data["spacing_mm"], dtype=np.float64).tobytes())
    if "skin_height_map" in patient_data:
        hsh.update(np.ascontiguousarray(np.asarray(patient_data["skin_height_map"], dtype=np.float32)).tobytes())
    return hsh.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Mesh builder
# ---------------------------------------------------------------------------
def build_tet_mesh(
    patient_data: Dict[str, Any],
    roi_min_mm: Optional[np.ndarray] = None,
    roi_max_mm: Optional[np.ndarray] = None,
    h: float = 4.0,
    body_hu: float = -200.0,
    bone_hu: float = 250.0,
    bone_fraction: float = 0.5,
) -> TetMesh:
    ct = np.asarray(patient_data["ct_volume"], dtype=np.float32)       # (nz,ny,nx)
    lbl = patient_data.get("label_volume")
    o = np.asarray(patient_data["origin_mm"], dtype=np.float64)
    sp = np.asarray(patient_data["spacing_mm"], dtype=np.float64)
    if roi_min_mm is None or roi_max_mm is None:
        lo, hi = default_roi(patient_data)
        roi_min_mm = lo if roi_min_mm is None else roi_min_mm
        roi_max_mm = hi if roi_max_mm is None else roi_max_mm
    lo = np.asarray(roi_min_mm, dtype=np.float64)
    hi = np.asarray(roi_max_mm, dtype=np.float64)
    ncell = np.maximum(np.floor((hi - lo) / h + 1e-6).astype(np.int64), 1)   # (nx,ny,nz) cells
    nx, ny, nz = (int(v) for v in ncell)

    # --- classify cells by sub-sampling the CT (nearest voxel) ---
    m = int(np.ceil(h / float(sp.min())))
    sub = (np.arange(m) + 0.5) * (h / m)
    ax = [lo[d] + (np.arange(int(ncell[d]))[:, None] * h + sub[None, :]).ravel() for d in range(3)]
    idx = [np.rint((ax[d] - o[d]) / sp[d]).astype(np.int64) for d in range(3)]
    dims = (ct.shape[2], ct.shape[1], ct.shape[0])
    inb = [(idx[d] >= 0) & (idx[d] < dims[d]) for d in range(3)]
    ci = [np.clip(idx[d], 0, dims[d] - 1) for d in range(3)]
    # sample array (z, y, x) -> transpose to (x, y, z)
    hu = ct[np.ix_(ci[2], ci[1], ci[0])].transpose(2, 1, 0)
    inside = (inb[0][:, None, None] & inb[1][None, :, None] & inb[2][None, None, :])
    hu = np.where(inside, hu, -1000.0).astype(np.float32)
    bone_s = hu > bone_hu
    if lbl is not None:
        lb = np.asarray(lbl)[np.ix_(ci[2], ci[1], ci[0])].transpose(2, 1, 0) == 1
        bone_s |= lb & inside
    body_s = hu > body_hu

    def pool(a):
        return a.reshape(nx, m, ny, m, nz, m).mean(axis=(1, 3, 5))

    bone_f = pool(bone_s.astype(np.float32))
    body_f = pool(body_s.astype(np.float32))
    hu_sum = pool(np.where(body_s & ~bone_s, hu, 0.0))
    soft_cnt = pool((body_s & ~bone_s).astype(np.float32))
    cell_hu_all = np.where(soft_cnt > 0, hu_sum / np.maximum(soft_cnt, 1e-9), 0.0)
    is_bone = bone_f >= bone_fraction
    soft = (body_f >= 0.5) & ~is_bone

    NX, NY, NZ = nx + 1, ny + 1, nz + 1

    def nid(ijk):
        return (ijk[..., 0] * NY + ijk[..., 1]) * NZ + ijk[..., 2]

    # Lattice nodes that are fixed: nodes of bone cells and on ROI deep/side faces
    fixed_lat = np.zeros((NX, NY, NZ), dtype=bool)
    for c in CUBE_CORNERS:
        fixed_lat[c[0]:c[0] + nx, c[1]:c[1] + ny, c[2]:c[2] + nz] |= is_bone
    bone_lat = fixed_lat.copy()
    fixed_lat[0, :, :] = fixed_lat[-1, :, :] = True
    fixed_lat[:, :, 0] = fixed_lat[:, :, -1] = True
    fixed_lat[:, 0, :] = True

    # Remove soft components with < 4 fixed nodes (floating islands)
    from scipy.ndimage import label as cc_label
    comp, ncomp = cc_label(soft)
    if ncomp > 0:
        node_fixed_cnt = np.zeros(ncomp + 1, dtype=np.int64)
        cells = np.argwhere(soft)
        cid = comp[soft]
        cn = cells[:, None, :] + CUBE_CORNERS[None]
        fx = fixed_lat[cn[..., 0], cn[..., 1], cn[..., 2]]           # (C,8)
        # count unique fixed nodes per component (approx via cell-corner counts is enough)
        np.add.at(node_fixed_cnt, cid, fx.sum(1))
        cell_cnt = np.bincount(cid, minlength=ncomp + 1)
        keep = (node_fixed_cnt >= 4) & (cell_cnt >= 8)
        keep[0] = False
        soft = keep[comp] & soft

    cells = np.argwhere(soft)                                         # (C,3)
    cn = cells[:, None, :] + CUBE_CORNERS[None]                       # (C,8,3)
    cnid = nid(cn)                                                    # (C,8)
    tets_lat = cnid[:, KUHN_TETS].reshape(-1, 4)
    used, inv = np.unique(tets_lat, return_inverse=True)
    tets = inv.reshape(-1, 4).astype(np.int64)
    node_ijk = np.stack(np.unravel_index(used, (NX, NY, NZ)), 1).astype(np.int64)
    nodes = lo[None, :] + node_ijk * h
    fixed = fixed_lat[node_ijk[:, 0], node_ijk[:, 1], node_ijk[:, 2]]
    bone_node = bone_lat[node_ijk[:, 0], node_ijk[:, 1], node_ijk[:, 2]]
    tet_ref = np.tile(np.arange(6), len(cells))
    tet_cell = np.repeat(np.arange(len(cells)), 6)

    # Skin nodes: top node per column
    col_key = node_ijk[:, 0] * NZ + node_ijk[:, 2]
    order = np.lexsort((-node_ijk[:, 1], col_key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = col_key[order[1:]] != col_key[order[:-1]]
    skin_nodes = order[first]
    skin_ik = node_ijk[skin_nodes][:, [0, 2]]
    column_skin = -np.ones((NX, NZ), dtype=np.int64)
    column_skin[skin_ik[:, 0], skin_ik[:, 1]] = np.arange(len(skin_nodes))

    hm, hm_o, hm_s = skin_height_map_from_patient(patient_data)
    sx = nodes[skin_nodes, 0]
    sz = nodes[skin_nodes, 2]
    sy = bilinear_height(hm, hm_o, hm_s, sx, sz)
    skin_vertices_ref = np.stack([sx, sy, sz], 1)

    # Height-field triangulation over columns
    a = column_skin[:-1, :-1]; b = column_skin[1:, :-1]; c = column_skin[:-1, 1:]; d = column_skin[1:, 1:]
    ok = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    tris = np.concatenate([np.stack([a[ok], b[ok], d[ok]], 1), np.stack([a[ok], d[ok], c[ok]], 1)], 0)

    cell_class = np.zeros((nx, ny, nz), dtype=np.uint8)
    cell_class[soft] = 1
    cell_class[is_bone] = 2
    return TetMesh(
        h=float(h), lattice_origin=lo.copy(), lattice_shape=(NX, NY, NZ), node_ijk=node_ijk, nodes=nodes,
        tets=tets, tet_ref=tet_ref, tet_cell=tet_cell, cell_ijk=cells,
        cell_hu=cell_hu_all[cells[:, 0], cells[:, 1], cells[:, 2]].astype(np.float32),
        fixed=fixed, bone_node=bone_node, skin_nodes=skin_nodes, skin_column_ik=skin_ik, column_skin=column_skin,
        skin_vertices_ref=skin_vertices_ref, skin_tris=tris.astype(np.int64), cell_class=cell_class,
    )
