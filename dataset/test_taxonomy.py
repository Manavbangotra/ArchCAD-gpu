#!/usr/bin/env python3
"""Self-check for the Arch-43 taxonomy and its CAD-layer rules.

No pytest in this repo, so this is a plain script: it exits non-zero on the
first failure and prints a summary otherwise.

    python dataset/test_taxonomy.py

The layer cases below are not invented. Every one of them is a real layer name
from the 90 layered plan sets in the corpus, and most are here because an
earlier version of the rules got them wrong.
"""

import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import taxonomy as t  # noqa: E402

FAILURES = []


def check(got, want, what):
    if got != want:
        FAILURES.append(f"{what}: got {got!r}, want {want!r}")


def name_of(cls):
    return t.COARSE_NAMES.get(cls) or t.ARCH_NAMES.get(cls, f"<unknown {cls}>")


def test_table():
    check(len(t.ARCH_CATEGORIES), 44, "ARCH_CATEGORIES length")
    check(t.ARCH_NUM_CLASSES, 43, "ARCH_NUM_CLASSES")
    check(t.ARCH_BG, 43, "ARCH_BG")
    check(t.ARCH_NAMES[t.ARCH_BG], "bg", "background is the last entry")

    # Band A must keep FloorPlanCAD's order, or a pretrained classifier head
    # transfers its rows onto the wrong classes.
    for idx, want in [(0, "single door"), (6, "window"), (10, "sofa"),
                      (26, "toilet"), (27, "stairs"), (32, "wall"),
                      (33, "curtain wall"), (34, "railing")]:
        check(t.ARCH_NAMES[idx], want, f"band A index {idx}")

    # Band B is appended after it, 0-based, and must not collide with bg.
    for idx, want in [(35, "column"), (36, "framing"), (37, "roof"),
                      (38, "electrical"), (39, "mechanical"), (40, "pipe"),
                      (41, "site"), (42, "equipment")]:
        check(t.ARCH_NAMES[idx], want, f"band B index {idx}")

    ids = [c["id"] for c in t.ARCH_CATEGORIES]
    check(ids, list(range(1, 45)), "ids are 1..44 with no gaps")


def test_coarse():
    # Every coarse id must sit above background, so the guards of the form
    # `id >= num_classes` that already exist treat them as background until the
    # marginal loss is switched on.
    for cid in t.COARSE_GROUPS:
        if cid <= t.ARCH_BG:
            FAILURES.append(f"coarse id {cid} ({name_of(cid)}) is not above bg")

    # Members must be real fine classes, never coarse ids or background.
    for cid, members in t.COARSE_GROUPS.items():
        for m in members:
            if not (0 <= m < t.ARCH_BG):
                FAILURES.append(f"{name_of(cid)} has out-of-range member {m}")

    check(t.canonical(t.DOOR_ANY), 0, "canonical(door-any) is single door")
    check(t.canonical(t.STAIRS), t.STAIRS, "canonical() passes fine ids through")
    check(t.COARSE_GROUPS[t.IGNORE], [], "ignore has no members")


def test_families():
    seen = [i for _, ix in t.FAMILIES for i in ix]
    check(sorted(seen), list(range(44)), "families cover 0..43 exactly once")


def test_layers():
    cases = [
        # -- the four ordering traps, each a real bug the ordering prevents ---
        ("A-WALL-PATT", "bg"),            # hatching, not a wall
        ("A-ELEV-LLGT", "bg"),            # AIA "ELEV" is ELEVATION, not elevator
        ("M-DIFF-WALL", "mechanical"),    # a wall diffuser, not a wall
        ("P-SANR-VENT", "pipe"),          # vent pipe, not a sanitary fixture

        # -- substring traps that over-matched before -----------------------
        ("A-03-CONC-DETECTABLE-SURFACE", "bg"),   # "detecTABLE"
        ("UGD - WOVEN SCOUR", "site"),            # "wOVENe"
        ("P-ORANGE FENCE (NEW)", "site"),         # "oRANGEe"
        ("A-REFRIGERANT LINES", "mechanical"),    # refrigerant, not a fridge
        ("L-PLNT-BEDS", "site"),                  # planting beds
        ("LINE_OVERHEAD WIRE", "electrical"),     # not a rolling door

        # -- doors ------------------------------------------------------------
        ("A-DOOR", "door-any"),
        ("A-DOOR-FRAM", "door-any"),
        ("A-DOOR-GLAZ", "door-any"),      # door beats window
        ("A-08-DOOR-OVERHEAD", "door-any"),
        ("XREF - 1st Floor|A-DOOR", "door-any"),  # XREF prefix is stripped

        # -- glazing ----------------------------------------------------------
        ("A-GLAZ", "window"),
        ("A-08-WINDOWS", "window"),       # plural, no word boundary
        ("CURTAIN WALL", "curtain wall"),
        ("A-GLAZ-CWMG", "curtain wall"),

        # -- walls, including the compounds \b would have dropped -------------
        ("A-WALL", "wall"),
        ("A-FIREWALL", "wall"),
        ("LINE_WALL", "wall"),
        ("B-CMUB-BS", "wall"),

        # -- the classes that did not exist before ---------------------------
        ("S-STRS", "stairs"),
        ("A-FLOR-STRS", "stairs"),
        ("A-FLOR-HRAL", "railing"),
        ("S-COLS", "column"),
        ("S-COLUMNS-SB", "column"),
        ("B-STUDB-BS", "framing"),
        ("S-TRUSSES-S", "framing"),
        ("A-ROOF-OTLN", "roof"),
        ("E-LITE-STD", "electrical"),
        ("M-DUCT-EXHS", "mechanical"),
        ("P-DOMW-CPIP", "pipe"),
        ("C-PROPERTY", "site"),
        ("C-PRKG-STRP-MRKG", "parking spot"),
        ("Q-SPCQ", "equipment"),
        ("A-ELEVATOR", "elevator"),

        # -- families the layer names cannot resolve: coarse, never a guess ---
        ("P-SANR-FIXT", "fixture-any"),
        ("A-FLOR-PFIX", "fixture-any"),
        ("I-FURN", "furniture-any"),
        ("A-FURN", "furniture-any"),
        ("Q-CASE", "cabinet"),
        ("A-MILLWORK", "cabinet"),

        # -- annotation stays background --------------------------------------
        ("A-ANNO-DIMS", "bg"),
        ("A-DETL-GENF", "bg"),
        ("WALLKEY", "bg"),                # a legend entry, not a wall
        ("0", "bg"),                      # AutoCAD's default layer
        ("", "bg"),
        (None, "bg"),
    ]
    for layer, want in cases:
        check(name_of(t.from_arch_layer(layer)), want, f"layer {layer!r}")


def test_us4_untouched():
    """The 4-class space must keep working: the 3-class configs still use it."""
    check(t.NUM_CLASSES, 3, "us4 NUM_CLASSES")
    check((t.DOOR, t.WINDOW, t.WALL, t.BACKGROUND), (0, 1, 2, 3), "us4 indices")
    check(t.from_us_layer("A-DOOR"), t.DOOR, "us4 from_us_layer")
    check(t.from_us_layer("A-GLAZ"), t.WINDOW, "us4 glazing")
    check(len(t.CATEGORIES), 4, "us4 CATEGORIES")


def main():
    tests = [test_table, test_coarse, test_families, test_layers,
             test_us4_untouched]
    for fn in tests:
        fn()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  {f}")
        return 1
    print(f"ok - {len(tests)} groups, Arch-43 taxonomy and layer rules agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
