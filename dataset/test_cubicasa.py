#!/usr/bin/env python3
"""Self-check for the CubiCasa5K parser's two label spaces.

    python dataset/test_cubicasa.py

Runs against a synthetic SVG rather than the dataset, so it works without the
6 GB download and keeps meaning if the download moves. The structure mirrors
CubiCasa's real one: nested <g class="..."> where sub-parts (Panel, Glass,
Threshold) inherit the class of the object above them, which is what lets a
door's swing arc -- a <path> two groups deep -- come out as a door.
"""

import collections
import os
import os.path as osp
import shutil
import sys
import tempfile

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import taxonomy as TX  # noqa: E402
from parse_cubicasa import parse_svg  # noqa: E402

FAILURES = []

SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
 <g class="Wall"><polygon points="0,0 10,0 10,2 0,2"/></g>
 <g class="Door"><g class="Panel"><path d="M20,0 L30,0 L30,10"/></g></g>
 <g class="Window"><g class="Panel"><line x1="40" y1="0" x2="50" y2="0"/></g></g>
 <g class="Railing"><line x1="0" y1="20" x2="20" y2="20"/></g>
 <g class="Stairs"><line x1="0" y1="30" x2="20" y2="30"/></g>
 <g class="FixedFurniture Sink1"><polygon points="60,0 70,0 70,8 60,8"/></g>
 <g class="FixedFurniture Toilet2"><polygon points="60,20 68,20 68,28 60,28"/></g>
 <g class="FixedFurniture Bathtub"><polygon points="60,40 78,40 78,48 60,48"/></g>
 <g class="Space Bedroom"><polygon points="0,50 40,50 40,90 0,90"/></g>
</svg>"""


def check(cond, what):
    if not cond:
        FAILURES.append(what)


def main():
    d = tempfile.mkdtemp(prefix="archcad_cubi_")
    try:
        p = osp.join(d, "model.svg")
        with open(p, "w") as fh:
            fh.write(SVG)

        us4 = parse_svg(p, taxonomy="us4")
        arch = parse_svg(p, taxonomy="arch")

        check(len(us4["semanticIds"]) == len(arch["semanticIds"]),
              "the two taxonomies must see identical geometry")

        names = {**TX.ARCH_NAMES, **TX.COARSE_NAMES}
        got4 = {TX.CLASS_NAMES[k] for k in set(us4["semanticIds"])}
        gotA = {names[k] for k in set(arch["semanticIds"])}

        check(got4 == {"door", "window", "wall", "background"},
              f"us4 classes: {sorted(got4)}")

        # The point of the change: these were all background before.
        for want in ("sink", "toilet", "bath tub", "railing", "stairs"):
            check(want in gotA, f"arch should label {want!r}, got {sorted(gotA)}")

        # A door's subtype is not stated by CubiCasa, so it must be the coarse
        # id rather than a guess at "single door".
        check("door-any" in gotA, f"door should be coarse, got {sorted(gotA)}")

        # Room polygons stay background: they overlap everything inside them and
        # have no counterpart in FloorPlanCAD or a US layer set.
        check("bg" in gotA, "room polygons should remain background")

        # Background carries no instance; coarse ids above it do. The marginal
        # loss scores a coarse target as one object, so two doors sharing
        # instance -1 would be learned as a single door.
        for sem, ins in zip(arch["semanticIds"], arch["instanceIds"]):
            if sem == TX.ARCH_BG and ins != -1:
                FAILURES.append(f"background carries instance {ins}")
                break
            if sem in TX.COARSE_GROUPS and ins == -1:
                FAILURES.append(f"coarse class {sem} has no instance id")
                break

        # n_layers used to be hardcoded to 4, which was the class count at the
        # time; layerIds are a per-group counter and have nothing to do with it.
        check(arch["n_layers"] == max(arch["layerIds"]) + 1,
              f"n_layers {arch['n_layers']} should follow layerIds")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  {f}")
        return 1
    print("ok - CubiCasa parses into both label spaces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
