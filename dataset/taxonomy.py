"""
Shared 4-class taxonomy for door/window spotting.

Every dataset in the pipeline maps onto the same indices so a model pretrained on
one can be fine-tuned on another without discarding the classifier head:

    0 door        1 window        2 wall (context)        3 background

Wall is a context class rather than a deliverable: doors and windows are defined
by the openings they occupy in walls, so labelling walls helps the model locate
them. Background absorbs everything else — on a US construction sheet that is the
large majority (title block, dimensions, hatching, notes, civil linework).

Rationale for collapsing to four classes: measured on labelled Chinese drawings
versus layer-tagged US sheets, door and window geometry is nearly identical
(door arc fraction 13% vs 8.6%; window 2% vs 0%). The China/US mismatch lives in
sheet *context*, not in the symbols, so a narrow taxonomy transfers well.
"""

import re

DOOR, WINDOW, WALL, BACKGROUND = 0, 1, 2, 3

NUM_CLASSES = 3          # real classes; BACKGROUND is the ignore/no-object id
BG_SEMANTIC_ID = BACKGROUND

CATEGORIES = [
    {"color": [224, 62, 155], "isthing": 1, "id": 1, "name": "door"},
    {"color": [96, 78, 245], "isthing": 1, "id": 2, "name": "window"},
    {"color": [167, 92, 32], "isthing": 0, "id": 3, "name": "wall"},
    {"color": [0, 0, 0], "isthing": 0, "id": 4, "name": "bg"},
]


# --------------------------------------------------------------------------- #
# FloorPlanCAD (1-based ids from dataset/parse_FpCAD_svg.py::RAW_TO_CLASS_ID)
# --------------------------------------------------------------------------- #
# 1-6 doors, 7-10 windows, 33 wall, 34 curtain wall. Everything else背景.
_FPCAD_TO_US4 = {}
for _i in range(1, 7):
    _FPCAD_TO_US4[_i] = DOOR
for _i in range(7, 11):
    _FPCAD_TO_US4[_i] = WINDOW
_FPCAD_TO_US4[33] = WALL
_FPCAD_TO_US4[34] = WALL


def from_floorplancad(class_id_1based):
    """FloorPlanCAD 1-based class id -> 4-class index."""
    return _FPCAD_TO_US4.get(class_id_1based, BACKGROUND)


# --------------------------------------------------------------------------- #
# CubiCasa5K (the `class` attribute on each <g>)
# --------------------------------------------------------------------------- #
def from_cubicasa(class_attr):
    """CubiCasa <g class="..."> -> 4-class index.

    Classes look like "Wall", "Door", "Window", "Space Bedroom",
    "FixedFurniture Sink". Only the first token matters for this taxonomy.
    """
    if not class_attr:
        return BACKGROUND
    head = class_attr.split()[0].strip()
    if head == "Door":
        return DOOR
    if head == "Window":
        return WINDOW
    if head == "Wall":
        return WALL
    return BACKGROUND


# --------------------------------------------------------------------------- #
# US construction documents (AIA CAD layer names)
# --------------------------------------------------------------------------- #
# Ordered: first match wins, so the more specific patterns come first.
# Conventions vary between firms — `A-Accessibility`, `A-ACCESSIBILITY` and
# `A- THIN BRICK` all appear in the corpus — so matching is case-insensitive and
# tolerant of separators. Adding a new firm should be one line here.
_US_LAYER_RULES = [
    # Exclude before include: these contain "DOOR"/"WALL" but are annotation,
    # not the object itself, and would otherwise poison the labels.
    #
    # KEY/LEGEND/TITLE/MATCHLINE were added when the include patterns below were
    # loosened -- "WALLKEY" is a legend entry, not a wall, and only the exclusion
    # keeps it out now that a bare "WALL" matches.
    (re.compile(r"(ANNO|DIM|NOTE|TEXT|TAG|SCHED|KEYN|IDEN|PATT|HATCH"
                r"|KEY|LEGEND|TITLE|MATCHLINE|MATCH[\s_-]*LINE)", re.I), BACKGROUND),

    # No \b around the keyword. Word boundaries looked tidy but silently dropped
    # three whole families of real layers, measured across this corpus:
    #   plurals     A-08-WINDOWS, A-WINDOWS, ARCHICAD DOORS, WALLS
    #   compounds   A-FIREWALL, A-RTWALL, A-03-CONC-TILTWALL, INTWALL
    #   underscores LINE_WALL, A-SECTION_WALL, MAIN EXTERIOR WALL_PEN_NO__149
    # ("_" is a word character, so \bWALL\b never matches "LINE_WALL".)
    # 72 distinct layer names were being sent to background this way, which is
    # why some buildings reported zero walls or zero windows.
    #
    # Order matters: DOOR before WINDOW so a combined "DOORWIN" layer lands on
    # door, and WINDOW before WALL so "CURTAIN WALL" stays a window.
    (re.compile(r"DOORS?|DR[\s_-]*OPNG", re.I), DOOR),
    (re.compile(r"GLAZ|WINDOWS?|WIN[\s_-]*WELL|CURTAIN[\s_-]*WALL|CW[\s_-]*SYS", re.I), WINDOW),
    (re.compile(r"WALLS?|PARTITION|PRTN", re.I), WALL),
]


def from_us_layer(layer_name):
    """AIA-style CAD layer name -> 4-class index."""
    if not layer_name:
        return BACKGROUND
    name = layer_name.split("|")[-1]  # drop XREF prefix: "XREF - 1st Floor|A-DOOR"
    for pattern, cls in _US_LAYER_RULES:
        if pattern.search(name):
            return cls
    return BACKGROUND


CLASS_NAMES = {DOOR: "door", WINDOW: "window", WALL: "wall", BACKGROUND: "background"}
