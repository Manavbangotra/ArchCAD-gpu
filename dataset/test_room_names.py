#!/usr/bin/env python3
"""Self-check for room-tag parsing. Plain script, like test_taxonomy.py:

    python dataset/test_room_names.py

Every tag below was printed on a plan in this corpus.
"""

import doctest
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import room_names as r  # noqa: E402

FAILURES = []


def check(got, want, what):
    if got != want:
        FAILURES.append(f"{what}: got {got!r}, want {want!r}")


def group_of(text):
    p = r.parse(text)
    return None if p is None else r.GROUP_NAMES[p[1]]


def test_rooms():
    cases = [
        ("BEDROOM #1", "bedroom"),
        ("BEDROOM 2", "bedroom"),
        ("MASTER BEDROOM", "bedroom"),
        ("BDRM", "bedroom"),
        ("KITCHEN", "kitchen"),
        ("PANTRY", "kitchen"),
        ("PANTRY/LINEN", "kitchen"),       # pantry wins over linen by order
        ("LIVING", "living"),
        ("LIVING ROOM", "living"),
        ("GREAT ROOM", "living"),
        ("DINING", "living"),
        ("DEN", "living"),
        ("BATH", "bath"),
        ("BATHROOM", "bath"),
        ("MASTER BATH", "bath"),           # bath beats bedroom
        ("POWDER", "bath"),
        ("WC", "bath"),                    # alone it is the room
        ("CLOSET", "storage"),
        ("W.I.C.", "storage"),
        ("WALK-IN CLOSET", "storage"),
        ("MECH. CLOSET", "storage"),
        ("LAUNDRY", "storage"),
        ("LAUN.", "storage"),
        ("W/D", "storage"),
        ("STORAGE", "storage"),
        ("MECH.", "storage"),
        ("CLO", "storage"),
        ("ENTRY", "entry/hallway"),
        ("FOYER", "entry/hallway"),
        ("CORRIDOR", "entry/hallway"),
        ("GARAGE", "garage"),
        ("BALCONY", "outdoor"),
        ("PATIO", "outdoor"),
        ("OFFICE", "undefined"),
        ("STAIR", "undefined"),
        ("SERVER ROOM", "undefined"),
    ]
    for text, want in cases:
        check(group_of(text), want, f"parse({text!r})")


def test_not_rooms():
    for text in ["WC-1", "WC-2", "W-3", "D12A", "GD-2", "12'-0\" X 11'-6\"", "142 SF",
                 "DOOR SCHEDULE", "BATH ACCESSORIES", "KITCHEN ELEVATION",
                 "SEE NOTES", "FLOOR PLAN", "", None, "CL 2X4",
                 "THIS IS A VERY LONG NOTE ABOUT THE BEDROOM CLOSET FINISHES",
                 '01/A3.0 24" G.B. BATH #1',       # a callout merged onto a tag
                 "MECH. GRILL", "EXHAUST FAN ABOVE", "CLOSET ABOVE"]:
        check(r.parse(text), None, f"parse({text!r}) should reject")


def test_name_kept():
    check(r.parse("  BEDROOM   #1. "), ("BEDROOM #1", r.BEDROOM), "name is cleaned")
    check(r.parse("BATHROOM WC-1A"), ("BATHROOM", r.BATH), "a fixture tag on the line is dropped")
    check(r.parse("L-1 BATHROOM"), ("BATHROOM", r.BATH), "also before the name")


def test_merge_stacked():
    lines = [("WALK-IN", (10, 10, 40, 14)), ("CLOSET", (12, 15, 38, 19)),
             ("KITCHEN", (200, 10, 240, 14))]
    got = sorted(t for t, _ in r.merge_stacked(lines))
    check(got, ["KITCHEN", "WALK-IN CLOSET"], "stacked tag merge")
    far = [("WALK-IN", (10, 10, 40, 14)), ("CLOSET", (12, 40, 38, 44))]
    check(len(r.merge_stacked(far)), 2, "distant lines stay apart")


def main():
    tests = [test_rooms, test_not_rooms, test_name_kept, test_merge_stacked]
    for fn in tests:
        fn()
    fails, _ = doctest.testmod(r, verbose=False)
    if fails:
        FAILURES.append(f"{fails} doctest failure(s) in room_names")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  {f}")
        return 1
    print(f"ok - {len(tests)} groups, room tags parse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
