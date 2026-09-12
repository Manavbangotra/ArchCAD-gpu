#!/usr/bin/env python3
"""Check the three label-correction implementations agree.

    python dataset/test_merge_agreement.py

The same precedence -- parser < project layermap < tile layermap < brush -- is
implemented three times, and it has to be:

  * svgnet/data/svg.py        what the model trains on
  * tools/label_editor.py     what the tile list and the canvas show
  * tools/corpus_weights.py   what the class weights are computed from

They cannot share one implementation without the editor importing torch (it is
deliberately stdlib-only so it runs anywhere) or the loader importing the
editor. So they are separate, and the failure mode if they drift is nasty and
quiet: the editor shows a tile as corrected, the weights are computed as if it
were, and the model trains on something else.

This pins them together on a corpus built for the purpose.
"""

import json
import os
import os.path as osp
import shutil
import sys
import tempfile

HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, "tools"))
sys.path.insert(0, HERE)

FAILURES = []
STEM = "project_001_20260101_000000_deadbeef_p0001_t000_s2"


def build(root):
    """A tile with three layers, plus every kind of correction sidecar."""
    alld = osp.join(root, "all")
    os.makedirs(alld, exist_ok=True)
    n = 9
    tile = {
        "width": 100.0, "height": 100.0,
        "args": [[i, 0, i, 1, i, 2, i, 3] for i in range(n)],
        "lengths": [1.0] * n, "commands": [0] * n, "widths": [1.0] * n,
        "rgb": [[0, 0, 0]] * n,
        "semanticIds": [43] * n,
        "instanceIds": [-1] * n,
        "layerIds": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        "n_layers": 3,
        "layerNames": ["A-FLOR-PFIX", "A-DOOR", "A-ANNO-DIMS"],
    }
    with open(osp.join(alld, STEM + ".json"), "w") as fh:
        json.dump(tile, fh)
    # project rule, overridden for one layer by the tile rule, and one brush
    with open(osp.join(alld, "_project_layermap.json"), "w") as fh:
        json.dump({"version": 1, "rules": {"A-FLOR-PFIX": 25, "A-DOOR": 0}}, fh)
    with open(osp.join(alld, STEM + ".layermap.json"), "w") as fh:
        json.dump({"version": 1, "rules": {"A-FLOR-PFIX": 26}}, fh)
    with open(osp.join(alld, STEM + ".override.json"), "w") as fh:
        json.dump({"4": 18}, fh)
    # a split copy, to prove sidecars resolve through all/
    traind = osp.join(root, "train")
    os.makedirs(traind, exist_ok=True)
    shutil.copyfile(osp.join(alld, STEM + ".json"),
                    osp.join(traind, STEM + ".json"))
    with open(osp.join(root, ".taxonomy"), "w") as fh:
        fh.write("arch\n")
    return alld, traind


def main():
    root = tempfile.mkdtemp(prefix="archcad_merge_")
    try:
        alld, traind = build(root)
        path_all = osp.join(alld, STEM + ".json")
        path_train = osp.join(traind, STEM + ".json")
        with open(path_all) as fh:
            data = json.load(fh)

        # Expected, applying the precedence by hand:
        #   layer 0 (A-FLOR-PFIX): project says 25, tile says 26 -> 26
        #   layer 1 (A-DOOR):      project says 0                -> 0
        #   layer 2 (A-ANNO-DIMS): no rule                       -> 43 (bg)
        #   primitive 4:           brush says 18, beating layer 1's 0
        want = [26, 26, 26, 0, 18, 0, 43, 43, 43]

        # 1. the training loader
        from svgnet.data.svg import SVGDataset
        got_loader = list(SVGDataset.load(json_file=path_all, num_classes=43)[2][:9, 0])
        if got_loader != want:
            FAILURES.append(f"svgnet loader: {got_loader} != {want}")

        got_split = list(SVGDataset.load(json_file=path_train, num_classes=43)[2][:9, 0])
        if got_split != want:
            FAILURES.append(f"svgnet loader (split copy): {got_split} != {want}")

        # 2. the editor
        import label_editor as LE
        LE.ROOT = root
        LE.set_taxonomy("arch")
        got_editor = LE._corrected(path_all, data)
        if got_editor != want:
            FAILURES.append(f"label_editor: {got_editor} != {want}")

        got_editor_split = LE._corrected(path_train, data)
        if got_editor_split != want:
            FAILURES.append(f"label_editor (split copy): {got_editor_split} != {want}")

        # 3. the class-weight tool
        import corpus_weights as CW
        got_weights = CW._corrected(path_all, data, data["semanticIds"])
        if got_weights != want:
            FAILURES.append(f"corpus_weights: {got_weights} != {want}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  {f}")
        return 1
    print("ok - loader, editor and weight tool agree on correction precedence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
