#!/usr/bin/env python3
"""Pick the US corpus tiles worth training on, as a file list the loader reads.

    python tools/select_tiles.py --root dataset/us_plans/json5

Writes <root>/<split>_selected.txt (one tile basename per line) and prints what
was dropped and why. SVGDataset reads it through `file_list:`.

Kept: tiles from a plan or untitled viewport (dataset/classify_viewports.py)
with at least `--min_fg` labelled primitives and at most `--max_prims`
primitives in total. Measured on the train split: 8,493 of 12,568 tiles are
plan viewports but only 2,248 of those carry any label, and a tile over 16k
primitives costs ~10x a median step (a 35,713-primitive tile took 22 s; the
neighbour search is quadratic) while making up 4.5% of tiles.

Empty tiles are not useless -- they teach "this is not a door" -- but at 68%
of the corpus they drown the signal; `--keep_empty` keeps a share of them.
"""

import argparse
import json
import os
import os.path as osp
import random
from collections import Counter

BG = 43


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset/us_plans/json5")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--min_fg", type=int, default=20)
    ap.add_argument("--max_prims", type=int, default=16000)
    ap.add_argument("--keep_empty", type=float, default=0.1,
                    help="fraction of label-free plan tiles to keep as negatives")
    ap.add_argument("--kinds", nargs="+", default=["plan", "untitled"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    with open(osp.join(a.root, "viewport_labels.json")) as f:
        kinds = json.load(f)
    rng = random.Random(a.seed)

    for split in a.splits:
        d = osp.join(a.root, split)
        names = sorted(n for n in os.listdir(d) if n.endswith("_s2.json"))
        why, keep = Counter(), []
        for name in names:
            kind = kinds.get(name, {}).get("kind", "missing")
            if kind not in a.kinds:
                why[f"viewport {kind}"] += 1
                continue
            with open(osp.join(d, name)) as f:
                sem = json.load(f)["semanticIds"]
            if len(sem) > a.max_prims:
                why["too many primitives"] += 1
                continue
            fg = sum(1 for s in sem if s != BG)
            if fg < a.min_fg:
                if rng.random() < a.keep_empty:
                    keep.append(name)
                    why["kept as negative"] += 1
                else:
                    why["no labels"] += 1
                continue
            keep.append(name)
        out = osp.join(a.root, f"{split}_selected.txt")
        with open(out, "w", newline="\n") as f:
            f.write("\n".join(keep) + "\n")
        print(f"{split}: kept {len(keep)} of {len(names)} -> {out}")
        for k, v in why.most_common():
            print(f"   {k:22s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
