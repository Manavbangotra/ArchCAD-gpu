#!/usr/bin/env python3
"""Self-check for the takeoff package on synthetic inputs with known answers.

    python takeoff/test_takeoff.py

  1. scale strings and dimension chains read as the right factor;
  2. an object split across two overlapping tiles is counted once, and two
     neighbouring objects are not merged;
  3. a drawn 12' x 10' room with a door gap comes out as one room of 120 sq ft,
     named from the tag inside it;
  4. a door picks up the schedule mark printed next to it, not a farther one.
"""

import os.path as osp
import sys

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from takeoff import openings, rooms, scale, stitch  # noqa: E402

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def test_scale():
    for text, want in [('SCALE: 1/4" = 1\'-0"', 48.0), ('1/8"=1\'', 96.0), ('3/16" = 1\'-0"', 64.0),
                       ('1 1/2" = 1\'-0"', 8.0), ('1" = 1\'-0"', 12.0), ("SCALE 1:100", 100.0),
                       ("NTS", None), ("SEE 1/A-501", None)]:
        check(scale.scale_from_text(text) == want, f"scale_from_text({text!r}) != {want}")
    check(abs(scale.parse_dim_text("12'-6\"") - 3810.0) < 1e-6, "12'-6\" is 3810 mm")
    check(abs(scale.parse_dim_text('5 1/2"') - 139.7) < 1e-6, '5 1/2" is 139.7 mm')

    # A dimension chain at 1/4" = 1'-0": 10' segments are 2.5 paper inches = 180 pt.
    words = [("10'-0\"", 90.0 + 180.0 * k, 500.0) for k in range(5)]
    check(scale.scale_from_dimensions(words) == 48.0, "dimension chain measures 1/4\" = 1'-0\"")

    sm = scale.page_scales([('SCALE: 1/4" = 1\'-0"', 100.0, 90.0),
                            ('SCALE: 1-1/2" = 1\'-0"', 900.0, 90.0)], [])
    check(sm.at(110.0, 300.0).factor == 48.0, "a drawing takes the scale printed under it")
    check(sm.at(880.0, 300.0).factor == 8.0, "and its neighbour takes its own")
    check(abs(scale.Scale(48.0, "x").mm_per_pt - 25.4 / 72 * 48) < 1e-9, "mm per point")


def test_stitch():
    # Page primitives 0..9. Tile A holds 0-6, tile B holds 4-9 (overlap 4-6).
    a_idx = list(range(0, 7))
    b_idx = list(range(4, 10))

    def mask(tile_idx, page_prims):
        return np.isin(tile_idx, page_prims)

    door_a = {"masks": mask(a_idx, [4, 5, 6]), "labels": 0, "scores": 0.9}
    door_b = {"masks": mask(b_idx, [4, 5, 6, 7]), "labels": 0, "scores": 0.8}   # same door, seen again
    sink_b = {"masks": mask(b_idx, [8, 9]), "labels": 18, "scores": 0.7}
    wall_a = {"masks": mask(a_idx, [0, 1]), "labels": 32, "scores": 0.95}
    objs, owner = stitch.stitch(10, [("_t000", a_idx, [door_a, wall_a]),
                                     ("_t001", b_idx, [door_b, sink_b])])
    labels = sorted(o.label for o in objs)
    check(labels == [0, 18, 32], f"one door, one sink, one wall after stitching: {labels}")
    door = [o for o in objs if o.label == 0][0]
    check(door.prims.tolist() == [4, 5, 6, 7], f"door primitives are the union: {door.prims.tolist()}")
    check(owner[2] == -1 and owner[3] == -1, "unclaimed primitives are background")

    # Two doors side by side in one tile stay two doors.
    d1 = {"masks": mask(a_idx, [0, 1]), "labels": 0, "scores": 0.9}
    d2 = {"masks": mask(a_idx, [2, 3]), "labels": 0, "scores": 0.9}
    objs, _ = stitch.stitch(10, [("_t000", a_idx, [d1, d2])])
    check(len(objs) == 2, "neighbouring doors in one tile are not merged")


def _seg(x0, y0, x1, y1):
    return [x0, y0, x0, y0, x1, y1, x1, y1]


def test_rooms():
    # 1/4" = 1'-0": 1 ft = 18 pt. A 12' x 10' room = 216 x 180 pt, with a 3' door
    # gap in the bottom wall, inside a larger 30' x 20' outline.
    ft = 18.0
    W, H = 12 * ft, 10 * ft
    args = [
        _seg(0, 0, 4 * ft, 0), _seg(7 * ft, 0, W, 0),          # bottom wall with door gap
        _seg(W, 0, W, H), _seg(W, H, 0, H), _seg(0, H, 0, 0),
        _seg(W, 0, 30 * ft, 0), _seg(30 * ft, 0, 30 * ft, 20 * ft),   # the rest of the flat
        _seg(30 * ft, 20 * ft, 0, 20 * ft), _seg(0, 20 * ft, 0, H),
    ]
    mm_per_pt = scale.Scale(48.0, "x").mm_per_pt
    lines = [("BEDROOM #2", (4 * ft, 5 * ft - 4, 8 * ft, 5 * ft + 4)),
             ("LIVING", (15 * ft, 15 * ft - 4, 19 * ft, 15 * ft + 4))]
    found = rooms.build_rooms(args, list(range(len(args))), mm_per_pt, lines)
    bed = [r for r in found if r.name == "BEDROOM #2"]
    check(len(bed) == 1, f"the bedroom is found once: {[(r.name, r.area_m2) for r in found]}")
    if bed:
        sqft = bed[0].area_m2 * 10.7639
        check(abs(sqft - 120.0) / 120.0 < 0.08, f"bedroom is ~120 sq ft, got {sqft:.1f}")
        check(bed[0].is_sleeping, "a bedroom is a sleeping room")
    check(any(r.name == "LIVING" for r in found), "the L-shaped living room is found and named")

    # With the door detected: a 3' deep closet off the bedroom must come out
    # as its own room. A door-width closing (the fallback) fills it solid.
    closet = [_seg(W, 0, W + 3 * ft, 0), _seg(W + 3 * ft, 0, W + 3 * ft, 6 * ft),
              _seg(W + 3 * ft, 6 * ft, W, 6 * ft)]
    args2 = args[:2] + [_seg(W, 6 * ft, W, H), _seg(W, 0, W, 1 * ft)] + args[3:] + closet
    door_bedroom = (4 * ft, -1.0, 7 * ft, 3 * ft)          # swing into the room
    door_closet = (W - 2.0, 1 * ft, W + 0.5 * ft, 6 * ft)    # bifold, 6" deep, across the front
    lines2 = lines + [("CLOSET", (W + 0.8 * ft, 3 * ft - 3, W + 2.8 * ft, 3 * ft + 3))]
    found = rooms.build_rooms(args2, list(range(len(args2))), mm_per_pt, lines2,
                              door_boxes=[door_bedroom, door_closet])
    names = {r.name: round(r.area_m2 * 10.7639) for r in found}
    check("CLOSET" in names, f"a narrow closet is a room when doors seal it: {names}")
    if "BEDROOM #2" in names:
        check(abs(names["BEDROOM #2"] - 120) <= 10,
              f"door boxes are given back to the rooms: bedroom {names['BEDROOM #2']} sq ft")


def test_opening_tags():
    class Ob:
        def __init__(self, label, prims):
            self.label, self.prims, self.score = label, np.array(prims), 0.9
    ft = 18.0
    # A 3' door at x = 100..154 pt.
    args = [[100.0, 0, 100.0, 0, 154.0, 0, 154.0, 0], [100.0, 0, 100.0, 0, 100.0, 54.0, 100.0, 54.0]]
    objs = [Ob(0, [0, 1])]
    sched = {"door": {"05": {"item_id": "x.05", "mark": "05", "width_in": 36.0}}, "window": {}}
    words = [("05", 127.0, 60.0), ("05", 127.0, 400.0), ("12", 127.0, 58.0)]
    sm = scale.ScaleMap([], scale.Scale(48.0, "x"))
    ops = openings.build_openings(objs, args, {0: "door"}, sm.at, words, sched)
    check(len(ops) == 1 and ops[0].tag == "05", f"door takes the nearby schedule mark: {[o.tag for o in ops]}")
    check(ops and ops[0].schedule.get("width_in") == 36.0, "schedule row attached")
    check(ops and abs(ops[0].width_mm - 3 * 304.8) < 26, f"door measures ~3 ft: {ops and ops[0].width_mm}")
    ops = openings.build_openings(objs, args, {0: "door"}, sm.at, [("05", 127.0, 400.0)], sched)
    check(ops[0].tag == "", "a mark 19 ft away is not this door's")
    del ft


def main():
    tests = [test_scale, test_stitch, test_rooms, test_opening_tags]
    for fn in tests:
        fn()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  {f}")
        return 1
    print(f"ok - {len(tests)} groups, takeoff geometry, scale, stitching and tags")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
