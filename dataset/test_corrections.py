#!/usr/bin/env python3
"""Self-check for human label corrections reaching the training loader.

    python dataset/test_corrections.py

Built on a synthetic tile rather than the corpus, so it stays meaningful after a
re-parse. Covers the precedence that matters:

    parser labels  <  project layermap  <  tile layermap  <  per-primitive brush

The layer rules are the base ("everything on A-FLOR-PFIX is a toilet") and the
brush is the exception handler for them, which is the order a person expects
after correcting one stray primitive on an otherwise-correct layer.
"""

import json
import os
import os.path as osp
import shutil
import sys
import tempfile

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from svgnet.data.svg import SVGDataset  # noqa: E402

FAILURES = []
STEM = "project_001_20260101_000000_deadbeef_p0001_t000_s2"


def check(got, want, what):
    if list(got) != list(want):
        FAILURES.append(f"{what}: got {list(got)}, want {list(want)}")


def write_tile(d, sem, layer_ids, layer_names):
    n = len(sem)
    os.makedirs(d, exist_ok=True)
    tile = {
        "width": 100.0, "height": 100.0,
        "args": [[i, 0, i, 1, i, 2, i, 3] for i in range(n)],
        "lengths": [1.0] * n,
        "commands": [0] * n,
        "widths": [1.0] * n,
        "rgb": [[0, 0, 0]] * n,
        "semanticIds": list(sem),
        "instanceIds": [-1] * n,
        "layerIds": list(layer_ids),
        "n_layers": len(layer_names),
        "layerNames": list(layer_names),
    }
    with open(osp.join(d, STEM + ".json"), "w") as fh:
        json.dump(tile, fh)
    return osp.join(d, STEM + ".json")


def sem_of(json_file, **kw):
    # column 0 of `label` is the semantic id; only the real primitives matter,
    # the rest is background padding up to min_points.
    return SVGDataset.load(json_file=json_file, num_classes=43, **kw)[2][:6, 0]


def main():
    root = tempfile.mkdtemp(prefix="archcad_corr_")
    try:
        alld = osp.join(root, "all")
        # 6 primitives on 3 layers; parser called them all background (43).
        jf = write_tile(alld, [43] * 6, [0, 0, 1, 1, 2, 2],
                        ["A-FLOR-PFIX", "A-DOOR", "A-ANNO-DIMS"])

        check(sem_of(jf), [43] * 6, "no sidecars: parser labels unchanged")
        check(sem_of(jf, use_corrections=False), [43] * 6, "corrections disabled")

        # -- tile layermap: one decision covers every primitive on the layer ---
        with open(osp.join(alld, STEM + ".layermap.json"), "w") as fh:
            json.dump({"version": 1, "rules": {"A-FLOR-PFIX": 26}}, fh)  # toilet
        check(sem_of(jf), [26, 26, 43, 43, 43, 43], "tile layermap")
        check(sem_of(jf, use_corrections=False), [43] * 6,
              "layermap ignored when corrections are off")

        # -- per-primitive brush wins over the layer rule ---------------------
        with open(osp.join(alld, STEM + ".override.json"), "w") as fh:
            json.dump({"1": 18}, fh)                                   # sink
        check(sem_of(jf), [26, 18, 43, 43, 43, 43], "brush beats layermap")

        # -- project layermap applies, and the tile's own overrides it --------
        with open(osp.join(alld, "_project_layermap.json"), "w") as fh:
            json.dump({"version": 1,
                       "rules": {"A-DOOR": 0, "A-FLOR-PFIX": 25}}, fh)
        check(sem_of(jf), [26, 18, 0, 0, 43, 43],
              "project layermap applies; tile rule wins on A-FLOR-PFIX")

        # -- a split copy reads the sidecars that live under all/ -------------
        traind = osp.join(root, "train")
        os.makedirs(traind, exist_ok=True)
        shutil.copyfile(jf, osp.join(traind, STEM + ".json"))
        check(sem_of(osp.join(traind, STEM + ".json")), [26, 18, 0, 0, 43, 43],
              "split copy resolves sidecars to all/")

        # -- malformed sidecars must not take a training run down -------------
        with open(osp.join(alld, STEM + ".override.json"), "w") as fh:
            fh.write("{not json")
        check(sem_of(jf), [26, 26, 0, 0, 43, 43], "corrupt override is ignored")

        with open(osp.join(alld, STEM + ".override.json"), "w") as fh:
            json.dump({"9999": 5, "-1": 5, "x": 5}, fh)
        check(sem_of(jf), [26, 26, 0, 0, 43, 43],
              "out-of-range and non-numeric override keys are ignored")

        # -- coarse band-C ids collapse according to coarse_policy ----------
        # The classifier head has no column for a coarse id, so "bg" is the only
        # correct default until the marginal loss lands.
        cd = osp.join(root, "coarse", "all")
        cj = write_tile(cd, [51, 51, 43, 43, 43, 43], [0] * 6, ["A-DOOR"])
        check(sem_of(cj), [43, 43, 43, 43, 43, 43], "coarse_policy=bg (default)")
        check(sem_of(cj, coarse_policy="keep"), [51, 51, 43, 43, 43, 43],
              "coarse_policy=keep passes the coarse id through")
        check(sem_of(cj, coarse_policy="canonical"), [0, 0, 43, 43, 43, 43],
              "coarse_policy=canonical takes the group's modal member")

        # -- a tile whose layerIds outrun its layerNames is left alone --------
        bad = write_tile(osp.join(root, "bad", "all"), [43] * 6,
                         [0, 0, 1, 1, 7, 7], ["A-FLOR-PFIX", "A-DOOR"])
        with open(osp.join(root, "bad", "all", STEM + ".layermap.json"), "w") as fh:
            json.dump({"version": 1, "rules": {"A-FLOR-PFIX": 26}}, fh)
        check(sem_of(bad), [43] * 6, "malformed layerIds do not index out of bounds")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  {f}")
        return 1
    print("ok - corrections reach the loader, precedence and guards hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
