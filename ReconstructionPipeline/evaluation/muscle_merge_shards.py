"""
Merge the shards of a muscle-thickness run (DEC-067) into one study JSON.

    ./.venv/bin/python ReconstructionPipeline/evaluation/muscle_merge_shards.py --out results/muscle_thickness_<tag>.json \
        results/muscle_thickness_<stamp>_<tag>_shard0of4.json ...

All shards must share commit, forces, sites, detector, imaging, split, pre-registration and stiffness draw; patients
must not repeat. The merged meta keeps shard 0's meta plus the list of shard files.
"""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAME = ("commit", "forces_n", "sites", "detector", "bmode_physics", "preregistration", "splits", "e_scale_draw",
        "e_scale_seed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args(argv)
    ds = [json.load(open(p)) for p in a.shards]
    meta = dict(ds[0]["meta"])
    for p, d in zip(a.shards, ds):
        for k in SAME:
            if d["meta"].get(k) != meta.get(k):
                raise SystemExit(f"{p}: meta '{k}' differs from {a.shards[0]}")
    patients = {}
    for p, d in zip(a.shards, ds):
        for n, r in d["patients"].items():
            if n in patients:
                raise SystemExit(f"{n} appears in more than one shard")
            patients[n] = r
    meta["shards"] = [os.path.relpath(p, ROOT) for p in a.shards]
    meta.pop("shard", None)
    json.dump({"meta": meta, "patients": patients}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(patients)} patients from {len(ds)} shards")


if __name__ == "__main__":
    main()
