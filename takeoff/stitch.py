"""Overlapping tile predictions -> one class and one object per page primitive.

A sheet is cut into tiles that overlap by 15%, plus a `_full` tile repeating
the whole page when it is small enough. Summing per-tile counts would count
every door on a tile edge twice. Each tile carries `idxs`, the page index of
each of its primitives, so predictions are mapped back to the page and merged:

* an object predicted in two tiles is one object when the two predictions
  share primitives (same class, >= `merge_overlap` of the smaller one);
* a primitive claimed by several objects goes to the highest score;
* a primitive no object claims is background.

Objects are sets of page-primitive indices. No torch here: the model's output
is passed in as plain arrays, so this also runs on layer-derived labels.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PageObject:
    label: int
    score: float
    prims: np.ndarray                  # page primitive indices
    sources: list = field(default_factory=list)   # tile suffixes that saw it


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def stitch(num_prims, tile_predictions, merge_overlap=0.3, min_prims=1):
    """Merge per-tile instances into page objects.

    tile_predictions: iterable of (suffix, idxs, instances) where `idxs` maps
    tile primitive -> page primitive and `instances` is a list of
    {"masks": bool array over tile primitives (may be padded), "labels": int,
    "scores": float}, exactly what SVGNet.instance_inference returns.
    """
    cand = []                                          # (label, score, prim set, suffix)
    for suffix, idxs, instances in tile_predictions:
        idxs = np.asarray(idxs, dtype=np.int64)
        n = len(idxs)
        for inst in instances:
            m = np.asarray(inst["masks"], dtype=bool)[:n]
            if m.sum() < min_prims:
                continue
            cand.append((int(inst["labels"]), float(inst["scores"]),
                         np.unique(idxs[m]), suffix))
    if not cand:
        return [], np.full(num_prims, -1, dtype=np.int64)

    # Candidates sharing any primitive, by class.
    owners = {}
    for k, (lab, _, prims, _) in enumerate(cand):
        for p in prims.tolist():
            owners.setdefault((lab, p), []).append(k)
    dsu = _DSU(len(cand))
    checked = set()
    for ks in owners.values():
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = ks[i], ks[j]
                if (a, b) in checked or cand[a][3] == cand[b][3]:
                    continue                  # one tile never predicts an object twice
                checked.add((a, b))
                inter = np.intersect1d(cand[a][2], cand[b][2], assume_unique=True).size
                small = min(cand[a][2].size, cand[b][2].size)
                if inter >= merge_overlap * small:
                    dsu.union(a, b)

    groups = {}
    for k in range(len(cand)):
        groups.setdefault(dsu.find(k), []).append(k)

    objects = []
    for ks in groups.values():
        prims = np.unique(np.concatenate([cand[k][2] for k in ks]))
        score = max(cand[k][1] for k in ks)
        objects.append(PageObject(cand[ks[0]][0], score, prims, sorted({cand[k][3] for k in ks})))

    # Contested primitives go to the highest-scoring object; objects left with
    # nothing are dropped.
    owner = np.full(num_prims, -1, dtype=np.int64)
    best = np.full(num_prims, -np.inf)
    for i, ob in enumerate(objects):
        take = ob.score > best[ob.prims]
        owner[ob.prims[take]] = i
        best[ob.prims[take]] = ob.score
    kept, remap = [], {}
    for i, ob in enumerate(objects):
        mine = np.flatnonzero(owner == i)
        if mine.size >= min_prims:
            remap[i] = len(kept)
            kept.append(PageObject(ob.label, ob.score, mine, ob.sources))
    owner = np.array([remap.get(o, -1) if o >= 0 else -1 for o in owner.tolist()], dtype=np.int64)
    return kept, owner


def objects_from_labels(sem, ins, bg_id, score=1.0):
    """PageObjects straight from per-primitive (semantic, instance) labels.

    For running the takeoff on layer-derived labels -- the parser's own output
    -- which checks every later stage without a trained model, and gives the
    baseline a model has to beat.
    """
    sem = np.asarray(sem)
    ins = np.asarray(ins)
    objects = []
    fg = np.flatnonzero(sem != bg_id)
    keys = {}
    for i in fg.tolist():
        k = (int(sem[i]), int(ins[i]) if ins[i] >= 0 else -1)
        keys.setdefault(k, []).append(i)
    for (lab, _), prims in sorted(keys.items()):
        objects.append(PageObject(lab, score, np.array(prims, dtype=np.int64), ["labels"]))
    return objects


def split_connected(objects, args, labels, tol):
    """Split objects of the given classes into geometrically connected pieces.

    Walls and railings are "stuff": one mask per class per tile, which the
    stitcher then joins across tiles -- so every wall on the sheet, plan and
    elevations alike, became a single object whose centre sat in one viewport
    and took that viewport's scale. Splitting by endpoint connectivity (the
    same rule parse_pdf_plans.cluster_instances uses) restores one object per
    wall run, each in its own drawing.
    """
    import os.path as osp
    import sys
    sys.path.insert(0, osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "dataset"))
    from parse_pdf_plans import cluster_instances

    out = []
    for ob in objects:
        if ob.label not in labels or ob.prims.size < 2:
            out.append(ob)
            continue
        prims = [{"pts": args[i]} for i in ob.prims.tolist()]
        comp = cluster_instances(prims, np.zeros(len(prims), dtype=np.int64), tol, bg_id=-1)
        for c in np.unique(comp):
            out.append(PageObject(ob.label, ob.score, ob.prims[comp == c], ob.sources))
    return out
