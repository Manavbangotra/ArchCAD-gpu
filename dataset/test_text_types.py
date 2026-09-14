#!/usr/bin/env python3
"""Text type/attribute parser on strings from the three corpora.

    python dataset/test_text_types.py
"""

import math
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import text_types as T  # noqa: E402

CASES = [
    # FloorPlanCAD-V2
    ("M1021", "door_code", (1000, 2100)), ("FM乙1021", "fire_door_code", (1000, 2100)),
    ("C1518", "window_code", (1500, 1800)), ("1000", "dimension", (1000,)), ("200x400", "dimension_pair", (200, 400)),
    ("-0.050", "level", None), ("12.5m²", "area", (12.5,)), ("1%", "slope", (1,)),
    ("上", "stair_up", None), ("下", "stair_down", None), ("阳台", "room_outdoor", None),
    ("厨房", "room_kitchen", None), ("飘窗", "hint_bay_window", None), ("乙", "grade_mark", None),
    # CubiCasa5K
    ("MH", "room_bedroom", None), ("TERASSI", "room_outdoor", None), ("KHH", "room_storage", None),
    ("WC", "room_bath", None),
    # US plan sets
    ("D-101", "door_tag", (101,)), ("W-3", "window_tag", (3,)), ("WB-4", "fixture_tag", (4,)),
    ("2'-0\"", "dimension", (609.6,)), ("BEDRM.", "room_bedroom", None), ("W.I.C.", "room_storage", None),
    ("SCALE: 1/4\" = 1'-0\"", "scale", (48,)), ("68 REQUIRED A2 - ONE BEDROOM", "required_count", (68,)),
    ("A7.0", "sheet_ref", None), ("5/A-501", "detail_ref", None), ("EQ.", "equal_mark", None),
    ("GFCI", "hint_electrical", None), ("SINK", "hint_fixture", None), ("2X6 WALL", "hint_structure", None),
    ("", "pad", None),
]


def main():
    bad = []
    for text, want, values in CASES:
        tok = T.parse(text)
        if tok.name != want:
            bad.append(f"{text!r}: type {tok.name}, want {want}")
            continue
        if values:
            for i, v in enumerate(values):
                got = math.copysign(math.expm1(abs(tok.attrs[i])), tok.attrs[i])
                if not tok.mask[i] or abs(got - v) > 1e-3 * max(1, abs(v)):
                    bad.append(f"{text!r}: attr {i} = {got:.3f}, want {v}")
    if T.parse("FM乙1021").grade != T.GRADE_ID["乙"] or not T.parse("FM乙1021").mask[3]:
        bad.append("fire door grade not recorded")
    if len(set(T.TYPE_NAMES)) != len(T.TYPE_NAMES):
        bad.append("duplicate type names")
    if bad:
        print(f"FAILED ({len(bad)}):")
        for b in bad:
            print("  " + b)
        return 1
    print(f"ok - {len(CASES)} strings typed across FloorPlanCAD-V2, CubiCasa and US plans ({len(T.TYPE_NAMES)} types)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
