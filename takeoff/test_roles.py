#!/usr/bin/env python3
"""Which drawings a takeoff counts, on real titles from the corpus.

    python takeoff/test_roles.py
"""

import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from takeoff import document as D  # noqa: E402

CASES = [
    ("A4.10 BUILDING TYPE 'B' FIRST FLOOR PLAN", "floor_plan"),
    ("1/4\" = 1'-0\" 01 TYPICAL GARAGE FLOOR PLAN", "floor_plan"),
    ("A3.7 UNIT B1 ALT. 01 FLOOR PLAN & INTERIOR ELEVATIONS", "enlarged_plan"),
    ("68 REQUIRED A2 - ONE BEDROOM, ONE BATH     672 S.F. 01 SCALE: 1/4\" = 1'-0\"", "enlarged_plan"),
    ("A2 UNIT REFLECTED CEILING PLAN 03", "reflected_ceiling"),
    ("TYPE D UNIT ENLARGED PLAN - DOMESTIC WATER SCALE: 1/4\"", "mep"),
    ("FIRST FLOOR PLAN - SYSTEMS", "mep"),
    ("BUILDING TYPE 'C' RADON MITIGATION PLAN 01", "mep"),
    ("FIRST FLOOR SHEAR WALL PLAN", "structural"),
    ("S202", "structural"),
    ("P401", "mep"),
    ("3/8\" = 1'-0\" TYPICAL CARPORT SIDE ELEV. 05", "not_a_plan"),
    ("KITCHEN ELEVATION", "not_a_plan"),
    ("CROSS ARCHITECTS, PLLC", "not_a_plan"),
    ("1-HR. ATTIC ACCESS PANEL 03 SCALE: 1 1/2\" = 1'-0\"", "not_a_plan"),
]


def main():
    bad = [(t, want, D.viewport_role("plan", t)) for t, want in CASES
           if D.viewport_role("plan", t) != want]
    if D.required_count("68 REQUIRED A2 - ONE BEDROOM") != 68 or \
            D.required_count("3 REQ'D @BLDGs #1, #6 & #7 BUILDING TYPE 'A'") != 3:
        bad.append(("required_count", "68 / 3", "wrong"))
    if bad:
        print(f"FAILED ({len(bad)}):")
        for t, want, got in bad:
            print(f"  {t!r}: want {want}, got {got}")
        return 1
    print(f"ok - {len(CASES)} drawing titles classified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
