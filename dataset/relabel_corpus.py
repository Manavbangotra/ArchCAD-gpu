#!/usr/bin/env python3
"""Re-derive a corpus's labels from its stored layer names, without re-parsing.

This is what persisting `layerNames` bought. The expensive half of ingestion is
walking PDF content streams and rasterising pages; the labelling itself is a
regex over a layer name. Once the names are in the tile, improving the rules in
dataset/taxonomy.py no longer costs a re-parse -- it costs a pass over the JSON.

    python dataset/relabel_corpus.py --root dataset/us_plans/json5 --dry-run
    python dataset/relabel_corpus.py --root dataset/us_plans/json5

Instance ids are recomputed too. cluster_instances groups *same-class*
primitives, so a primitive that changes class has stale instance membership --
leaving it alone would put two different classes in one instance and quietly
corrupt panoptic evaluation.

Only semanticIds and instanceIds change. Geometry, layer ids, renders and every
sidecar are untouched, so a corpus can be relabelled repeatedly and a human's
corrections still apply on top at load time.
"""

import argparse
import json
import os
import os.path as osp
import shutil
import sys
from collections import Counter

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import numpy as np  # noqa: E402

import taxonomy as TX  # noqa: E402
from parse_pdf_plans import TAXONOMIES, cluster_instances  # noqa: E402


def relabel(path, mapper, bg_id, cluster_tol_frac=0.004):
    """Returns (changed, before_counter, after_counter) for one tile."""
    with open(path) as fh:
        d = json.load(fh)
    names = d.get("layerNames")
    if not names:
        return None
    lids = d.get("layerIds") or []
    old = d["semanticIds"]
    lut = [mapper(n) for n in names]
    new = [lut[li] if li < len(lut) else bg_id for li in lids]
    if len(new) != len(old):
        return None                      # malformed; leave it alone
    if new == old:
        return (False, Counter(old), Counter(new))

    # Re-cluster: instances are per class, so a class change invalidates them.
    args = d["args"]
    prims = [{"pts": a} for a in args]
    sem = np.array(new, dtype=np.int64)
    tol = cluster_tol_frac * max(d["width"], d["height"])
    inst = cluster_instances(prims, sem, tol, bg_id=bg_id)

    d["semanticIds"] = new
    d["instanceIds"] = inst.tolist()
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh)
    os.replace(tmp, path)
    return (True, Counter(old), Counter(new))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="corpus dir containing all/")
    ap.add_argument("--taxonomy", choices=sorted(TAXONOMIES), default=None,
                    help="default: the .taxonomy marker beside the corpus")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    root = a.root
    alld = osp.join(root, "all")
    if not osp.isdir(alld):
        alld = root

    tax = a.taxonomy
    if tax is None:
        for cand in (osp.join(root, ".taxonomy"), osp.join(alld, ".taxonomy")):
            try:
                with open(cand) as fh:
                    tax = fh.read().strip() or None
                if tax:
                    break
            except OSError:
                pass
        tax = tax or "us4"
    mapper, bg_id, _, _ = TAXONOMIES[tax]
    name_of = ((lambda i: TX.COARSE_NAMES.get(i) or TX.ARCH_NAMES.get(i, f"?{i}"))
               if tax == "arch" else (lambda i: TX.CLASS_NAMES.get(i, f"?{i}")))

    files = sorted(f for f in os.listdir(alld) if f.endswith("_s2.json"))
    if a.limit:
        files = files[:a.limit]
    print(f"taxonomy {tax}, background {bg_id}, {len(files)} tiles in {alld}")
    if a.dry_run:
        print("DRY RUN -- nothing will be written\n")

    changed = skipped = 0
    moves = Counter()
    before, after = Counter(), Counter()
    for i, f in enumerate(files, 1):
        p = osp.join(alld, f)
        try:
            if a.dry_run:
                with open(p) as fh:
                    d = json.load(fh)
                names = d.get("layerNames")
                if not names:
                    skipped += 1
                    continue
                lut = [mapper(n) for n in names]
                lids = d.get("layerIds") or []
                new = [lut[li] if li < len(lut) else bg_id for li in lids]
                old = d["semanticIds"]
                if len(new) != len(old):
                    skipped += 1
                    continue
                res = (new != old, Counter(old), Counter(new))
            else:
                res = relabel(p, mapper, bg_id)
                if res is None:
                    skipped += 1
                    continue
        except Exception as exc:
            print(f"  [fail] {f}: {str(exc)[:70]}")
            skipped += 1
            continue
        ch, b, aft = res
        before += b
        after += aft
        if ch:
            changed += 1
        if i % 500 == 0:
            print(f"  {i}/{len(files)}  ({changed} changed)", flush=True)

    total = sum(before.values()) or 1
    print(f"\n{changed} tiles changed, {skipped} skipped, "
          f"{total:,} primitives seen")
    print(f"\n{'class':16s} {'before%':>9} {'after%':>9} {'delta':>9}")
    print("-" * 47)
    for k in sorted(set(before) | set(after),
                    key=lambda x: -(after.get(x, 0) - before.get(x, 0))):
        b = 100 * before.get(k, 0) / total
        f_ = 100 * after.get(k, 0) / total
        if abs(f_ - b) < 0.005 and k != bg_id:
            continue
        print(f"{name_of(k):16s} {b:9.3f} {f_:9.3f} {f_ - b:+9.3f}")
    lb = 100 * (total - before.get(bg_id, 0)) / total
    la = 100 * (total - after.get(bg_id, 0)) / total
    print(f"\nlabelled {lb:.2f}% -> {la:.2f}%  ({la - lb:+.2f} points)")


if __name__ == "__main__":
    raise SystemExit(main())
