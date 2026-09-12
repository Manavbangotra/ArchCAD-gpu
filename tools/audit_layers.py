#!/usr/bin/env python3
"""Rank the CAD layers a corpus failed to classify, by how much geometry they hold.

The layer rules in dataset/taxonomy.py label about half the layer vocabulary.
The other half is genuinely background -- annotation, dimensions, detail
linework -- but some of it is real geometry under a name the rules do not
recognise, and there is no way to tell which from the rules alone.

This counts actual primitives per layer name, so the output is a worklist
ordered by what it would be worth fixing rather than by how many distinct names
exist. A layer with 40,000 primitives and no class is worth a rule; one with 12
is not.

    python tools/audit_layers.py --root dataset/us_plans/json5
    python tools/audit_layers.py --root dataset/us_plans/json5 --class toilet

Requires a corpus parsed with layerNames (dataset/parse_pdf_plans.py after the
change that stops discarding them); older corpora carry only integer layer ids
and there is nothing to audit.
"""

import argparse
import json
import os
import os.path as osp
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "dataset"))
import taxonomy as TX  # noqa: E402


def tile_paths(root, split, sample, seed):
    d = osp.join(root, split)
    if not osp.isdir(d):
        d = osp.join(root, "all")
    if not osp.isdir(d):
        d = root
    files = sorted(osp.join(d, f) for f in os.listdir(d) if f.endswith("_s2.json"))
    if sample and sample < len(files):
        files = random.Random(seed).sample(files, sample)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--sample", type=int, default=400,
                    help="tiles to read (0 = all). Primitive counts scale, so a "
                         "sample is enough to rank")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--taxonomy", choices=("us4", "arch"), default="arch")
    ap.add_argument("--class", dest="cls", default=None,
                    help="show only layers assigned to this class name")
    ap.add_argument("--csv", help="write the full table here")
    a = ap.parse_args()

    if a.taxonomy == "arch":
        mapper, bg = TX.from_arch_layer, TX.ARCH_BG
        def name_of(i):
            return TX.COARSE_NAMES.get(i) or TX.ARCH_NAMES.get(i, f"?{i}")
    else:
        mapper, bg = TX.from_us_layer, TX.BACKGROUND
        def name_of(i):
            return TX.CLASS_NAMES.get(i, f"?{i}")

    files = tile_paths(a.root, a.split, a.sample, a.seed)
    if not files:
        raise SystemExit(f"no tiles under {a.root}")

    prims = Counter()           # layer name -> primitives
    tiles_with = Counter()      # layer name -> tiles it appears on
    no_names = 0
    read = 0
    for i, f in enumerate(files, 1):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        names = d.get("layerNames") or []
        if not names:
            no_names += 1
            continue
        read += 1
        c = Counter(d.get("layerIds") or [])
        for li, n in c.items():
            if li < len(names):
                prims[names[li]] += n
                tiles_with[names[li]] += 1
        if i % 200 == 0:
            print(f"  read {i}/{len(files)}", flush=True)

    if not prims:
        raise SystemExit(
            f"{no_names} tiles carry no layerNames -- this corpus predates the "
            f"parser change that keeps them. Re-parse to audit.")

    total = sum(prims.values())
    rows = []
    for layer, n in prims.items():
        cls = mapper(layer)
        rows.append((n, layer, cls, tiles_with[layer]))
    rows.sort(reverse=True)

    by_class = defaultdict(int)
    for n, _, cls, _ in rows:
        by_class[cls] += n

    print(f"\n{read} tiles read ({no_names} without layer names), "
          f"{total:,} primitives, {len(prims)} distinct layers\n")
    print(f"{'class':16s} {'share%':>8}  {'layers':>6}")
    print("-" * 36)
    for cls, n in sorted(by_class.items(), key=lambda kv: -kv[1]):
        k = sum(1 for _, _, c, _ in rows if c == cls)
        print(f"{name_of(cls):16s} {100 * n / total:8.2f}  {k:6d}")

    if a.cls:
        want = [i for i in list(TX.ARCH_NAMES) + list(TX.COARSE_NAMES)
                if name_of(i) == a.cls]
        if not want:
            raise SystemExit(f"no class named {a.cls!r}")
        sel = set(want)
        print(f"\nlayers assigned to {a.cls}:")
        for n, layer, cls, t in rows:
            if cls in sel:
                print(f"  {n:9,}  {t:5d} tiles  {layer}")
    else:
        print(f"\nunclassified layers by volume -- the worklist. A rule for the "
              f"top few is worth more\nthan a rule for the rest combined.\n")
        print(f"{'primitives':>11} {'share%':>7} {'tiles':>6}  layer")
        print("-" * 78)
        shown = 0
        for n, layer, cls, t in rows:
            if cls != bg:
                continue
            print(f"{n:11,} {100 * n / total:7.3f} {t:6d}  {layer[:44]}")
            shown += 1
            if shown >= a.top:
                break
        unl = sum(n for n, _, c, _ in rows if c == bg)
        print(f"\nbackground total {100 * unl / total:.1f}% of primitives across "
              f"{sum(1 for _, _, c, _ in rows if c == bg)} layers")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            fh.write("primitives,tiles,layer,class\n")
            for n, layer, cls, t in rows:
                safe = layer.replace('"', "'")
                fh.write(f'{n},{t},"{safe}",{name_of(cls)}\n')
        print(f"\nfull table -> {a.csv}")


if __name__ == "__main__":
    raise SystemExit(main())
