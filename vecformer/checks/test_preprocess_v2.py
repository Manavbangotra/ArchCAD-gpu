#!/usr/bin/env python3
"""The line preprocessor on both FloorPlanCAD SVG dialects.

    python vecformer/checks/test_preprocess_v2.py

Runs without torch_scatter (stubbed; the preprocessor never calls it).
"""

import os
import os.path as osp
import sys
import tempfile
import types

sys.modules.setdefault("torch_scatter", types.SimpleNamespace(scatter=None))
HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, ".."))

from data.floorplancad import preprocess as P  # noqa: E402
from utils.svg_util import get_t_values  # noqa: E402

SVG = """<?xml version="1.0" encoding="utf-8"?>
<svg viewBox="0 0 100.0 100.0" xmlns="http://www.w3.org/2000/svg">
 <g id="layer0">
  <path d="M 10,10 L 30,10" stroke="rgb(0,0,0)" stroke-width="0.1" {sem}="27" {ins}="4"/>
  <path d="M 30,10 A 20,20 0 0,0 50,30" stroke="rgb(0,0,0)" stroke-width="0.1" {sem}="1" {ins}="7"/>
  <circle cx="60" cy="60" r="5" stroke="rgb(0,0,0)" stroke-width="0.1"/>
  <text x="40" y="45" font-size="2" transform="rotate(-90.0,40,45)">
   BEDROOM
  </text>
 </g>
</svg>
"""

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def parse(sem, ins):
    with tempfile.TemporaryDirectory() as d:
        path = osp.join(d, "x.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(SVG.format(sem=sem, ins=ins))
        return P.parse_svg(path, get_t_values(2), get_t_values(9), connect_lines=True,
                           dynamic_sampling=True, dynamic_sampling_ratio=0.01)


def main():
    for sem, ins in (("semanticId", "instanceId"), ("semantic-id", "instance-id")):
        d = parse(sem, ins)
        check(len(d.semantic_ids) == 3, f"{sem}: 3 geometric primitives, got {len(d.semantic_ids)}")
        check(d.semantic_ids[:2] == [26, 0], f"{sem}: labels read and shifted, got {d.semantic_ids[:2]}")
        check(d.instance_ids[:2] == [4, 7], f"{sem}: instances read, got {d.instance_ids[:2]}")
        check(d.semantic_ids[2] == 35 and d.instance_ids[2] == -1, f"{sem}: unlabelled circle is background")
        check(len(d.texts) == 1 and d.texts[0]["text"] == "BEDROOM" and d.texts[0]["angle"] == -90.0,
              f"{sem}: text kept, got {d.texts}")
        # Dynamic sampling: segments no longer than 1% of the 100-unit extent.
        segs = [((c[2] - c[0]) ** 2 + (c[3] - c[1]) ** 2) ** 0.5 for c in d.coords]
        check(max(segs) <= 1.0 + 1e-6, f"{sem}: max segment {max(segs):.3f} <= 1.0")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("ok - preprocessor reads both SVG dialects and keeps text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
