#!/usr/bin/env python3
"""US PDF -> line-window converter on a synthetic parsed page (no PDF needed).

    python dataset/test_to_lines_us.py
"""

import math
import os.path as osp
import sys

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.dirname(HERE))

import taxonomy as tx  # noqa: E402
import to_lines_us as U  # noqa: E402
from takeoff.scale import Scale  # noqa: E402

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def line(x0, y0, x1, y1):
    return [x0, y0, x0 + (x1 - x0) / 3, y0 + (y1 - y0) / 3, x0 + 2 * (x1 - x0) / 3, y0 + 2 * (y1 - y0) / 3, x1, y1]


def main():
    # 1/4" = 1'-0": 16.93 mm per point, so a 10 m window is ~590 pt. Page 900 pt wide.
    sc = Scale(48.0, "scale_string")
    args = [line(10, 10, 800, 10),                         # a long wall
            line(100, 100, 130, 100), line(100, 100, 100, 130),   # a door (two lines)
            line(700, 300, 720, 300)]                     # background tick
    args += [line(200 + i, 200, 201 + i, 200) for i in range(30)]   # 30 tiny fixture lines
    n = len(args)
    data = dict(args=args, commands=[0] * n,
                semanticIds=[tx.A_WALL, tx.DOOR_ANY, tx.DOOR_ANY, tx.ARCH_BG] + [tx.FIXTURE_ANY] * 30,
                instanceIds=[0, 1, 1, -1] + [2] * 30, layerIds=[0, 1, 1, 2] + [3] * 30,
                layerNames=["A-WALL", "A-DOOR", "0", "A-PLMB"])
    texts = [("BATH", (205, 205, 215, 212))]
    wins = U.convert_page(data, texts, lambda x, y: sc, lambda x, y: ("plan", "UNIT A"), 10.0, 0.01, 20, 60000,
                          dict(source="us", doc="t", page=1))
    check("colors" not in wins[0][1] and "widths" not in wins[0][1], "compact: no per-segment colours/widths")
    check(len(wins) >= 1, f"at least one window: {len(wins)}")
    size = 10000.0 / sc.mm_per_pt
    for _, rec in wins:
        check(abs(rec["viewBox"][2] - size) / size < 1e-5, "window side is 10 m at the page scale")
        segs = rec["coords"]
        check(max(math.dist(c[:2], c[2:]) for c in segs) <= size * 0.01 + 0.02, "segments densified to 1% (coords rounded to 0.01)")
        check(len(rec["layer_names"]) == len(set(rec["layer_ids"])), "a name per local layer id")
        check(all(p < len(rec["semantic_ids"]) for p in rec["primitive_ids"]), "primitive ids index labels")
        sem, ins = rec["semantic_ids"], rec["instance_ids"]
        for s, i in zip(sem, ins):
            if s in (tx.A_WALL, tx.ARCH_BG):
                check(i == -1, "stuff/background have no instance")
        doors = {i for s, i in zip(sem, ins) if s == tx.DOOR_ANY}
        check(len(doors) <= 1, "both door lines share one instance")
        check(rec["meta"]["mm_per_pt"] > 16 and rec["meta"]["viewport_kind"] == "plan", "meta recorded")
    first = wins[0][1]
    check(any(t["text"] == "BATH" for t in first["texts"]), "text inside the window is kept")
    t = next(t for t in first["texts"] if t["text"] == "BATH")
    # window starts at the drawing's lowest y (10), so its top edge is 10 + size
    check(abs(t["y"] - (10 + first["viewBox"][3] - 208.5)) < 0.01, "text y flipped to window y-down like the geometry")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("ok - US line windows: scale, densify, layers, instances, text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
