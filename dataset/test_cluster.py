#!/usr/bin/env python3
"""Self-check for cluster_instances: known components, and linear cost on a
dense cell (the all-pairs version took 81 s on one hatched sheet).

    python dataset/test_cluster.py
"""

import os.path as osp
import sys
import time

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import numpy as np  # noqa: E402

from parse_pdf_plans import cluster_instances  # noqa: E402

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def seg(x0, y0, x1, y1):
    return {"pts": [x0, y0, x0, y0, x1, y1, x1, y1]}


def main():
    BG = 43
    prims = [seg(0, 0, 1, 0), seg(1, 0, 2, 0), seg(2, 0, 3, 0),      # a chain: one door
             seg(50, 50, 51, 50), seg(51, 50, 52, 50),                 # far away: another door
             seg(3, 0, 4, 0),                                          # touches the chain, but a wall
             seg(0, 1, 1, 1)]                                          # background
    sem = np.array([0, 0, 0, 0, 0, 32, BG])
    inst = cluster_instances(prims, sem, tol=0.5, bg_id=BG)
    check(inst[0] == inst[1] == inst[2], "a connected chain is one instance")
    check(inst[3] == inst[4] and inst[3] != inst[0], "a separate symbol is another instance")
    check(inst[5] not in (inst[0], inst[3]), "classes never merge")
    check(inst[6] == -1, "background gets no instance")

    # 20,000 segments all ending in one cell: must be linear, not 4e8 unions.
    dense = [seg(10.0 + k * 1e-6, 10.0, 10.1, 10.1) for k in range(20000)]
    t = time.time()
    out = cluster_instances(dense, np.zeros(len(dense), dtype=np.int64), tol=0.5, bg_id=BG)
    dt = time.time() - t
    check(len(set(out.tolist())) == 1, "a dense cell is one component")
    check(dt < 5.0, f"dense cell clustered in {dt:.1f} s (quadratic would take minutes)")

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  {f}")
        return 1
    print("ok - cluster_instances components and cost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
