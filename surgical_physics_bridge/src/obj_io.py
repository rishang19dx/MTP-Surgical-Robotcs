"""
Minimal, order-preserving Wavefront OBJ reader (numpy only).

Used by both the unified server and the 3D Slicer bridge so that the vertex indices carried in
``DeformedSkin`` messages refer to exactly the same vertex order on both sides. Generic readers
(trimesh with processing, vtkOBJReader with per-face normals) may merge or split vertices.
Only ``v`` and triangular/polygonal ``f`` records are read; polygons are fan-triangulated.
"""

from typing import Tuple

import numpy as np


def read_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Returns ``(vertices (N, 3) float32, faces (M, 3) int64)`` in file order (0-based faces)."""
    verts = []
    faces = []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    v = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return v, f
