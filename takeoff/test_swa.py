#!/usr/bin/env python3
"""Sliding-window aggregation: voting weights, duplicates across windows, truncated
copies, objects larger than a window, stuff from semantics, remask.

    python takeoff/test_swa.py
"""
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from takeoff.swa import WindowPrediction, aggregate, tent_weight  # noqa: E402

FAILURES = []
C = 5          # classes 0..4, background 5; class 4 is stuff


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def onehot(labels, conf=0.9):
    p = np.full((len(labels), C + 1), (1 - conf) / C)
    p[np.arange(len(labels)), labels] = conf
    return p


def win(suffix, idxs, pos, labels, insts):
    idxs = np.asarray(idxs)
    masks = np.zeros((len(insts), len(idxs)), dtype=bool)
    for k, (members, _, _) in enumerate(insts):
        masks[k, [list(idxs).index(m) for m in members]] = True
    return WindowPrediction(suffix, idxs, np.asarray(pos, dtype=float), onehot(labels), masks,
                            np.array([l for _, l, _ in insts]), np.array([s for _, _, s in insts]))


def main():
    w = tent_weight(np.array([[0, 0], [0.5, 0], [0.25, -0.25]]), edge_weight=0.1)
    check(np.allclose(w, [1.0, 0.1, 0.55]), f"tent weight: {w}")

    # page primitives 0..9. A door (class 0) = prims 0,1,2 seen whole in window A
    # (centre) and truncated (0,1) at the edge of window B. Prims 5..9 are a wall (stuff 4).
    a = win("A", [0, 1, 2, 5, 6], [[0, 0], [0.05, 0], [0.1, 0], [0.2, 0.2], [0.3, 0.3]],
            [0, 0, 0, 4, 4], [([0, 1, 2], 0, 0.9)])
    b = win("B", [0, 1, 7, 8, 9], [[0.48, 0], [0.49, 0], [0, 0], [0.1, 0], [0.2, 0]],
            [0, 1, 4, 4, 4], [([0, 1], 0, 0.95), ([1], 1, 0.3)])
    objs, sem, score = aggregate(10, [a, b], C, stuff_classes=[4])
    doors = [o for o in objs if o.label == 0]
    check(len(doors) == 1 and doors[0].prims.tolist() == [0, 1, 2], f"door seen twice is one object: {[(o.label, o.prims.tolist()) for o in objs]}")
    check(sem[1] == 0, "edge vote for another class loses to the central window")
    walls = [o for o in objs if o.label == 4]
    check(len(walls) == 1 and walls[0].prims.tolist() == [5, 6, 7, 8, 9], "stuff from voted semantics, one object")
    check(sem[3] == C and sem[4] == C and score[3] == 0, "unseen primitives are background")
    check(not [o for o in objs if o.label == 1], "weak window-edge instance of the wrong class removed by remask")

    # a long counter (class 2) larger than a window: prims 0..5 seen as 0..3 in A and 2..5 in B
    a = win("A", [0, 1, 2, 3], [[-0.3, 0], [-0.1, 0], [0.1, 0], [0.3, 0]], [2, 2, 2, 2], [([0, 1, 2, 3], 2, 0.8)])
    b = win("B", [2, 3, 4, 5], [[-0.3, 0], [-0.1, 0], [0.1, 0], [0.3, 0]], [2, 2, 2, 2], [([2, 3, 4, 5], 2, 0.8)])
    objs, _, _ = aggregate(6, [a, b], C)
    check(len(objs) == 1 and objs[0].prims.tolist() == [0, 1, 2, 3, 4, 5] and objs[0].sources == ["A", "B"],
          f"object larger than a window assembled: {[(o.label, o.prims.tolist()) for o in objs]}")

    # two different doors side by side in one window stay two objects
    a = win("A", [0, 1, 2, 3], [[0, 0]] * 4, [0, 0, 0, 0], [([0, 1], 0, 0.9), ([2, 3], 0, 0.8)])
    objs, _, _ = aggregate(4, [a], C)
    check(len(objs) == 2, "neighbouring objects in one window are not merged")

    # same window predicting a duplicate that overlaps: suppressed, not merged
    a = win("A", [0, 1, 2], [[0, 0]] * 3, [0, 0, 0], [([0, 1, 2], 0, 0.9), ([0, 1], 0, 0.5)])
    objs, _, _ = aggregate(3, [a], C)
    check(len(objs) == 1 and objs[0].prims.tolist() == [0, 1, 2], "in-window duplicate suppressed")

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("ok - SWA: tent votes, cross-window duplicates, large objects, stuff, remask")
    return 0


if __name__ == "__main__":
    sys.exit(main())
