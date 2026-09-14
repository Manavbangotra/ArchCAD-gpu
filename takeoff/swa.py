"""Sliding-window aggregation (CADSpotting SWA) of per-window predictions into page predictions.

A sheet is covered by windows of fixed real size stepped by half a window
(dataset/to_lines_us.py), so an interior primitive is seen by about four windows,
and near a window's edge a symbol is cut off and read without its context.

Semantics: each window's class probabilities are added to its primitives' page
entries with a weight that is 1 at the window centre and falls linearly to
`edge_weight` at the border (a square tent), then normalised; the page label is
the arg-max. A primitive is thus decided mostly by the windows it sits centrally in.

Instances: every predicted instance becomes a candidate with page primitives and
score = model score x mean weight of its primitives, so a door cut by a window
edge ranks below the same door seen whole in a neighbouring window. Candidates
are taken best-first; one that shares at least `merge_overlap` of the smaller
object's primitives with an accepted object of its class (from another window)
is merged into it -- a symbol larger than a window is assembled from its pieces
-- otherwise it becomes a new object. Primitives claimed by several objects go to
the best score; optionally (BFR "remask") primitives whose voted semantic label
disagrees with their object's class are removed.

Stuff classes (walls, railings, ...) are not instances: they come from the voted
semantics, one object per class, which the takeoff splits by connectivity.

Pure numpy; model outputs are passed in as arrays.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .stitch import PageObject


@dataclass
class WindowPrediction:
    suffix: str
    idxs: np.ndarray            # (n,) page primitive index of each window primitive
    positions: np.ndarray       # (n, 2) primitive centres in window-normalised coords, [-0.5, 0.5]
    sem_probs: np.ndarray       # (n, C+1) class probabilities, background last
    inst_masks: np.ndarray      # (m, n) bool
    inst_labels: np.ndarray     # (m,)
    inst_scores: np.ndarray     # (m,)


def tent_weight(positions, edge_weight=0.05):
    """1 at the window centre, `edge_weight` at the border (Chebyshev distance)."""
    d = np.abs(np.asarray(positions, dtype=np.float64)).max(axis=1) * 2.0     # 0 centre .. 1 border
    return np.clip(1.0 - d, 0.0, 1.0) * (1.0 - edge_weight) + edge_weight


def aggregate(num_prims: int, windows: Iterable[WindowPrediction], num_classes: int,
              stuff_classes: Sequence[int] = (), merge_overlap: float = 0.5, edge_weight: float = 0.05,
              remask: bool = True, min_prims: int = 1):
    """-> (objects: list[PageObject], sem_labels (num_prims,), sem_scores (num_prims,))

    Primitives no window saw get background (num_classes) with score 0.
    """
    C = num_classes
    votes = np.zeros((num_prims, C + 1), dtype=np.float64)
    weight_sum = np.zeros(num_prims, dtype=np.float64)
    stuff = set(int(c) for c in stuff_classes)
    cand = []                                   # (adjusted score, label, prims, suffix, raw score)
    for w in windows:
        idxs = np.asarray(w.idxs, dtype=np.int64)
        if idxs.size == 0:
            continue
        wt = tent_weight(w.positions, edge_weight)
        np.add.at(votes, idxs, np.asarray(w.sem_probs, dtype=np.float64) * wt[:, None])
        np.add.at(weight_sum, idxs, wt)
        masks = np.asarray(w.inst_masks, dtype=bool).reshape(-1, idxs.size)
        for m, lab, sc in zip(masks, np.asarray(w.inst_labels).tolist(), np.asarray(w.inst_scores).tolist()):
            if int(lab) in stuff or int(lab) >= C or m.sum() < min_prims:
                continue
            cand.append((float(sc) * float(wt[m].mean()), int(lab), np.unique(idxs[m]), w.suffix, float(sc)))

    seen = weight_sum > 0
    probs = np.zeros_like(votes)
    probs[seen] = votes[seen] / weight_sum[seen, None]
    sem_labels = np.full(num_prims, C, dtype=np.int64)
    sem_scores = np.zeros(num_prims, dtype=np.float64)
    sem_labels[seen] = probs[seen].argmax(1)
    sem_scores[seen] = probs[seen].max(1)

    # best-first merge per class
    cand.sort(key=lambda c: -c[0])
    objects: List[dict] = []
    owner = {}                                   # label -> (num_prims,) object index or -1
    for adj, lab, prims, suffix, raw in cand:
        own = owner.setdefault(lab, np.full(num_prims, -1, dtype=np.int64))
        hit = own[prims]
        hit = hit[hit >= 0]
        target = None
        if hit.size:
            ks, counts = np.unique(hit, return_counts=True)
            for k, cnt in sorted(zip(ks.tolist(), counts.tolist()), key=lambda t: -t[1]):
                ob = objects[k]
                if suffix in ob["sources"]:
                    continue                     # one window never predicts an object twice
                if cnt >= merge_overlap * min(prims.size, ob["prims"].size):
                    target = k
                    break
            if target is None and hit.size >= merge_overlap * prims.size:
                continue                         # mostly inside objects already accepted: a duplicate
        if target is None:
            objects.append(dict(label=lab, score=adj, prims=prims, sources={suffix}))
            own[prims[own[prims] < 0]] = len(objects) - 1
        else:
            ob = objects[target]
            ob["prims"] = np.union1d(ob["prims"], prims)
            ob["sources"].add(suffix)
            own[prims[own[prims] < 0]] = target

    # contested primitives across classes -> best score; remask by voted semantics
    best = np.full(num_prims, -np.inf)
    final_owner = np.full(num_prims, -1, dtype=np.int64)
    for i, ob in enumerate(objects):
        p = ob["prims"]
        if remask:
            p = p[sem_labels[p] == ob["label"]]
        take = ob["score"] > best[p]
        final_owner[p[take]] = i
        best[p[take]] = ob["score"]
    out = []
    for i, ob in enumerate(objects):
        mine = np.flatnonzero(final_owner == i)
        if mine.size >= min_prims:
            out.append(PageObject(ob["label"], ob["score"], mine, sorted(ob["sources"])))
    for c in sorted(stuff):
        prims = np.flatnonzero(sem_labels == c)
        if prims.size >= min_prims:
            out.append(PageObject(c, float(sem_scores[prims].mean()), prims, ["semantic"]))
    return out, sem_labels, sem_scores
