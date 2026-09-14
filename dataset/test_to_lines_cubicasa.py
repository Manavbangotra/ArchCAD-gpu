#!/usr/bin/env python3
"""CubiCasa -> line JSON converter on a synthetic model.svg.

    python dataset/test_to_lines_cubicasa.py
"""

import math
import os.path as osp
import sys
import tempfile

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import to_lines_cubicasa as C  # noqa: E402

SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
 <g class="Wall"><polygon points="0,0 1000,0 1000,20 0,20"/></g>
 <g class="Door">
  <g class="Panel"><path d="M 100,20 L 190,20 A 90,90 0 0,1 100,110 Z"/></g>
 </g>
 <g class="Door"><polygon points="300,0 390,0 390,20 300,20"/></g>
 <g class="FixedFurniture Toilet"><circle cx="600" cy="600" r="30"/></g>
 <g class="Space Bedroom"><polygon points="0,20 500,20 500,500 0,500"/>
  <text x="200" y="250">MH</text></g>
</svg>"""

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def main():
    with tempfile.TemporaryDirectory() as d:
        p = osp.join(d, "model.svg")
        with open(p, "w") as f:
            f.write(SVG)
        out = C.convert(p, "cubicasa12", 0.01)
    sem, ins = out["semantic_ids"], out["instance_ids"]
    check(len(sem) == 5, f"one primitive per element: {len(sem)}")
    check(sem == [10, 1, 1, 4, 12], f"labels wall, door, door, toilet, background: {sem}")
    check(ins[0] == -1 and ins[4] == -1, "stuff and background have instance -1")
    check(ins[1] != ins[2] and ins[1] > 0 and ins[2] > 0, "two doors are two instances")
    check(out["texts"] and out["texts"][0]["text"] == "MH", "text kept")
    segs = out["coords"]
    check(max(math.dist(c[:2], c[2:]) for c in segs) <= 10.0 + 1e-6, "no segment longer than 1% of 1000")
    # The door's swing is an arc: its segments are not collinear with the chord.
    door = [c for c, pid in zip(segs, out["primitive_ids"]) if pid == 1]
    arc_pts = [(c[0], c[1]) for c in door]
    off_chord = max(abs((190 - 100) * (100 - y) - (x - 100) * (110 - 20)) / math.dist((100, 110), (190, 20))
                    for x, y in arc_pts)
    check(off_chord > 20, f"door swing sampled as an arc (max distance from chord {off_chord:.1f})")
    check(len(set(out["layer_ids"])) == len(sem), "no CAD layers: layer id = primitive id")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("ok - CubiCasa line converter: labels, instances, arcs, text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
