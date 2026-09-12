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


# =========================================================================== #
# The Arch-43 taxonomy
# =========================================================================== #
# The 4-class space above is kept verbatim so the existing 3-class configs stay
# reproducible. Everything below is the wider space the corpus is moving to.
#
# Three bands, and the numbering between them is load-bearing:
#
#   Band A, ids 1-35   FloorPlanCAD's classes, unchanged and in their original
#                      order, so a FpCAD-pretrained classifier head transfers
#                      row for row instead of being dropped as a size mismatch.
#   Band B, ids 36-43  US layer families FloorPlanCAD has no class for at all:
#                      structure, MEP, roof, site. Appended after band A so
#                      band A's meaning never shifts.
#   Band C, ids 51-55  Coarse labels. Stored in the tile JSON, never a column in
#                      the classifier head -- see COARSE_GROUPS for why.
#
# Band C starts at 51 rather than 44 deliberately: it leaves 44-50 free for
# future fine classes, so adding one later does not renumber the coarse ids and
# therefore does not invalidate every corpus already on disk.

_BAND_A = [
    # 1-6: doors
    {"color": [224, 62, 155], "isthing": 1, "id": 1, "name": "single door"},
    {"color": [157, 34, 101], "isthing": 1, "id": 2, "name": "double door"},
    {"color": [232, 116, 91], "isthing": 1, "id": 3, "name": "sliding door"},
    {"color": [101, 54, 72], "isthing": 1, "id": 4, "name": "folding door"},
    {"color": [172, 107, 133], "isthing": 1, "id": 5, "name": "revolving door"},
    {"color": [142, 76, 101], "isthing": 1, "id": 6, "name": "rolling door"},
    # 7-10: windows
    {"color": [96, 78, 245], "isthing": 1, "id": 7, "name": "window"},
    {"color": [26, 2, 219], "isthing": 1, "id": 8, "name": "bay window"},
    {"color": [63, 140, 221], "isthing": 1, "id": 9, "name": "blind window"},
    {"color": [233, 59, 217], "isthing": 1, "id": 10, "name": "opening symbol"},
    # 11-27: furniture, appliances, sanitary
    {"color": [122, 181, 145], "isthing": 1, "id": 11, "name": "sofa"},
    {"color": [94, 150, 113], "isthing": 1, "id": 12, "name": "bed"},
    {"color": [66, 107, 81], "isthing": 1, "id": 13, "name": "chair"},
    {"color": [123, 181, 114], "isthing": 1, "id": 14, "name": "table"},
    {"color": [94, 150, 83], "isthing": 1, "id": 15, "name": "TV cabinet"},
    {"color": [66, 107, 59], "isthing": 1, "id": 16, "name": "Wardrobe"},
    {"color": [145, 182, 112], "isthing": 1, "id": 17, "name": "cabinet"},
    {"color": [152, 147, 200], "isthing": 1, "id": 18, "name": "gas stove"},
    {"color": [113, 151, 82], "isthing": 1, "id": 19, "name": "sink"},
    {"color": [112, 103, 178], "isthing": 1, "id": 20, "name": "refrigerator"},
    {"color": [81, 107, 58], "isthing": 1, "id": 21, "name": "airconditioner"},
    {"color": [172, 183, 113], "isthing": 1, "id": 22, "name": "bath"},
    {"color": [141, 152, 83], "isthing": 1, "id": 23, "name": "bath tub"},
    {"color": [80, 72, 147], "isthing": 1, "id": 24, "name": "washing machine"},
    {"color": [100, 108, 59], "isthing": 1, "id": 25, "name": "squat toilet"},
    {"color": [182, 170, 112], "isthing": 1, "id": 26, "name": "urinal"},
    {"color": [238, 124, 162], "isthing": 1, "id": 27, "name": "toilet"},
    # 28-30: circulation
    {"color": [247, 206, 75], "isthing": 1, "id": 28, "name": "stairs"},
    {"color": [237, 112, 45], "isthing": 1, "id": 29, "name": "elevator"},
    {"color": [233, 59, 46], "isthing": 1, "id": 30, "name": "escalator"},
    # 31-35: uncountable ("stuff") classes
    {"color": [172, 107, 151], "isthing": 0, "id": 31, "name": "row chairs"},
    {"color": [102, 67, 62], "isthing": 0, "id": 32, "name": "parking spot"},
    {"color": [167, 92, 32], "isthing": 0, "id": 33, "name": "wall"},
    {"color": [121, 104, 178], "isthing": 0, "id": 34, "name": "curtain wall"},
    {"color": [64, 52, 105], "isthing": 0, "id": 35, "name": "railing"},
]

_BAND_B = [
    # 36-43: US construction-document families with no FloorPlanCAD counterpart.
    # A US plan set carries the whole discipline stack on one sheet, and calling
    # ductwork "background" is what taught the 4-class model that every MEP run
    # was noise.
    {"color": [90, 140, 190], "isthing": 1, "id": 36, "name": "column"},
    {"color": [130, 110, 70], "isthing": 0, "id": 37, "name": "framing"},
    {"color": [190, 120, 60], "isthing": 0, "id": 38, "name": "roof"},
    {"color": [240, 180, 40], "isthing": 0, "id": 39, "name": "electrical"},
    {"color": [70, 170, 170], "isthing": 0, "id": 40, "name": "mechanical"},
    {"color": [60, 130, 220], "isthing": 0, "id": 41, "name": "pipe"},
    {"color": [120, 160, 90], "isthing": 0, "id": 42, "name": "site"},
    {"color": [200, 90, 140], "isthing": 1, "id": 43, "name": "equipment"},
]

ARCH_CATEGORIES = _BAND_A + _BAND_B + [
    # Background sentinel. Excluded from the class list used for evaluation, and
    # its index doubles as the ignore/no-object id the model never predicts.
    {"color": [0, 0, 0], "isthing": 0, "id": 44, "name": "bg"},
]

ARCH_NUM_CLASSES = len(ARCH_CATEGORIES) - 1     # 43
ARCH_BG = ARCH_NUM_CLASSES                      # 43

ARCH_NAMES = {i: c["name"] for i, c in enumerate(ARCH_CATEGORIES)}
ARCH_HEX = {i: "#%02X%02X%02X" % tuple(c["color"])
            for i, c in enumerate(ARCH_CATEGORIES)}

# Named indices. Everything downstream indexes 0-based (id - 1), and writing the
# raw ids into the rule table below silently shifts every band-B class by one --
# it put "equipment" on the background id and filed S-COLS as framing. Names are
# cheap; that bug is not.
SINGLE_DOOR, DOUBLE_DOOR, SLIDING_DOOR = 0, 1, 2
FOLDING_DOOR, REVOLVING_DOOR, ROLLING_DOOR = 3, 4, 5
A_WINDOW, BAY_WINDOW, BLIND_WINDOW, OPENING_SYMBOL = 6, 7, 8, 9
SOFA, A_BED, CHAIR, A_TABLE, TV_CABINET, WARDROBE, CABINET = 10, 11, 12, 13, 14, 15, 16
GAS_STOVE, SINK, REFRIGERATOR, AIRCON = 17, 18, 19, 20
A_BATH, BATH_TUB, WASHING_MACHINE = 21, 22, 23
SQUAT_TOILET, URINAL, TOILET = 24, 25, 26
STAIRS, ELEVATOR, ESCALATOR = 27, 28, 29
ROW_CHAIRS, PARKING_SPOT, A_WALL, CURTAIN_WALL, RAILING = 30, 31, 32, 33, 34
COLUMN, FRAMING, ROOF = 35, 36, 37
ELECTRICAL, MECHANICAL, PIPE, SITE, EQUIPMENT = 38, 39, 40, 41, 42


# --------------------------------------------------------------------------- #
# Band C: coarse labels
# --------------------------------------------------------------------------- #
# A CAD layer name is coarser than this taxonomy. "A-DOOR" says the primitive is
# a door but not which of the six kinds; "P-SANR-FIXT" covers sink, bath, tub,
# urinal and toilet at once; "I-FURN" covers sofa, bed, chair and table. There is
# no honest fine label to write, and the two obvious answers are both wrong:
# guessing a member teaches the model something false, and writing background --
# which is what the 4-class pipeline does today -- teaches it that furniture IS
# background, which is worse.
#
# So the parser writes a coarse id and the loss marginalises over the group:
#
#     loss = -log sum(p_c for c in group)      instead of      -log p_c
#
# The mask target is exact either way (we know precisely which primitives form
# the door, only not its subtype), so only the classification term changes, and
# a group of one is ordinary cross-entropy.
#
# These ids are a data-side vocabulary only. They are always > ARCH_BG, so every
# existing guard of the form `id >= num_classes` treats them as background until
# the marginal path is switched on.
DOOR_ANY = 51
FURNITURE_ANY = 52
FIXTURE_ANY = 53
APPLIANCE_ANY = 54
IGNORE = 55

COARSE_GROUPS = {
    DOOR_ANY:      [0, 1, 2, 3, 4, 5],              # single..rolling door
    FURNITURE_ANY: [10, 11, 12, 13, 14, 15, 16],    # sofa..cabinet
    FIXTURE_ANY:   [18, 21, 22, 24, 25, 26],        # sink, bath, tub, toilets
    APPLIANCE_ANY: [17, 19, 20, 23],                # stove, fridge, a/c, washer
    IGNORE:        [],                              # scored by nothing
}

COARSE_NAMES = {DOOR_ANY: "door-any", FURNITURE_ANY: "furniture-any",
                FIXTURE_ANY: "fixture-any", APPLIANCE_ANY: "appliance-any",
                IGNORE: "ignore"}


def canonical(class_id):
    """Modal member of a coarse group, for the "just pick one" fallback.

    Only for configs that cannot run the marginal loss. It is a lie for every
    non-modal member, which is why it is not the default anywhere.
    """
    g = COARSE_GROUPS.get(class_id)
    return g[0] if g else class_id


# Grouping for the editor's class picker -- a flat list of 44 buttons is
# unusable. Every index 0..ARCH_BG appears in exactly one family; the assertion
# at the bottom of this module keeps it that way.
FAMILIES = [
    ("doors",       [0, 1, 2, 3, 4, 5]),
    ("windows",     [6, 7, 8, 9]),
    ("furniture",   [10, 11, 12, 13, 14, 15, 16]),
    ("appliances",  [17, 19, 20, 23]),
    ("sanitary",    [18, 21, 22, 24, 25, 26]),
    ("circulation", [27, 28, 29, 30]),
    ("structure",   [32, 33, 34, 35, 36]),
    ("envelope",    [37]),
    ("mep",         [38, 39, 40, 42]),
    ("site",        [31, 41]),
    ("background",  [43]),
]

assert sorted(i for _, ix in FAMILIES for i in ix) == list(range(len(ARCH_CATEGORIES))), \
    "FAMILIES must cover every Arch-43 index exactly once"


# --------------------------------------------------------------------------- #
# US construction documents -> Arch-43
# --------------------------------------------------------------------------- #
# Ordered, first match wins, matched against the layer name with any XREF prefix
# stripped. Same conventions as _US_LAYER_RULES above: case-insensitive, and no
# \b around keywords, because word boundaries silently dropped whole families of
# real layers (A-08-WINDOWS, A-FIREWALL, LINE_WALL -- see the note at the 4-class
# table).
#
# Three tiers, and the third is a deliberate refusal:
#
#   1. the name determines the class          -> the fine id
#   2. the name names the member              -> the fine id
#   3. the name gives the family only         -> a band-C coarse id, never a guess
#
# Ordering carries real weight here. Four traps, all present in the measured
# corpus of 2,323 distinct layer names:
#
#   A-WALL-PATT   is hatching, not wall  -> annotation must be tested first.
#   A-ELEV-LLGT   "ELEV" in AIA is ELEVATION, not elevator. A bare ELEV rule
#                 labels thousands of elevation primitives as elevators, so the
#                 elevator rule demands ELEVATOR / ELVTR / ELEV-CAB.
#   M-DIFF-WALL   is a mechanical wall diffuser -> the MEP rules must precede the
#                 wall rule.
#   P-SANR-VENT   is a vent pipe, not a fixture -> the sanitary rule demands FIXT,
#                 so the generic "^P-" rule below catches the pipework.
_ARCH_LAYER_RULES = [
    # -- tier 0: annotation, dimensions, detail linework, title block --------
    (re.compile(r"ANNO|DIM|NOTE|TEXT|TAG|SCHED|KEYN|IDEN|PATT|HATCH|KEY"
                r"|LEGEND|TITLE|MATCHLINE|MATCH[\s_-]*LINE|REVISION|VIEWPORT"
                r"|TTLB|WIPEOUT|GENM|IMPT|DETL|NORTH[\s_-]*ARROW", re.I), ARCH_BG),

    # -- civil/landscape first: "L-PLNT-BEDS" is a planting bed, not a bed, and
    #    "P-ORANGE FENCE" is site protection, not a kitchen range ------------
    (re.compile(r"^L[\s_-]|LANDSCAPE|PLNT|PLANTING|^UGD|PROPERTY|PAVE|CURB"
                r"|TOPO|\bSITE\b|SWALE|SCOUR|FENCE", re.I), SITE),

    # -- doors: subtype where stated, else the coarse id ----------------------
    (re.compile(r"SLIDING|SLDG|SLD[\s_-]*DR", re.I), SLIDING_DOOR),
    (re.compile(r"BIFOLD|BI[\s_-]*FOLD|FOLDING", re.I), FOLDING_DOOR),
    (re.compile(r"REVOLV", re.I), REVOLVING_DOOR),
    (re.compile(r"OVERHEAD[\s_-]*(DR|DOOR)|OVHD[\s_-]*(DR|DOOR)"
                r"|ROLL[\s_-]*UP|COILING|SECTIONAL[\s_-]*DOOR", re.I), ROLLING_DOOR),
    (re.compile(r"DBL[\s_-]*DR|DOUBLE[\s_-]*DOOR|DR[\s_-]*PAIR", re.I), DOUBLE_DOOR),
    (re.compile(r"DOORS?|DR[\s_-]*OPNG", re.I), DOOR_ANY),

    # -- glazing. Curtain wall first: it contains "WALL" and is its own class --
    (re.compile(r"CURTAIN[\s_-]*WALL|CW[\s_-]*SYS|GLAZ[\s_-]*CW", re.I), CURTAIN_WALL),
    (re.compile(r"GLAZ|WINDOWS?|WIN[\s_-]*WELL", re.I), A_WINDOW),

    # -- circulation ----------------------------------------------------------
    (re.compile(r"ESCAL|ESCL", re.I), ESCALATOR),
    (re.compile(r"ELEVATOR|ELVTR|ELEV[\s_-]*CAB|VERT[\s_-]*CIRC", re.I), ELEVATOR),
    (re.compile(r"STRS|STAIRS?", re.I), STAIRS),
    (re.compile(r"HRAL|HAND[\s_-]*RAIL|GUARD[\s_-]*RAIL|RAILING", re.I), RAILING),

    # -- sanitary: member where stated, else coarse ---------------------------
    (re.compile(r"WATER[\s_-]*CLOSET|TOILET", re.I), TOILET),
    (re.compile(r"URINAL", re.I), URINAL),
    (re.compile(r"BATH[\s_-]*TUB|\bTUB\b", re.I), BATH_TUB),
    (re.compile(r"LAVT|LAVATORY|\bSINKS?\b", re.I), SINK),
    (re.compile(r"SANR[\s_-]*FIXT|PFIX|PLMB[\s_-]*FIXT|FLOR[\s_-]*FIXT"
                r"|PLUMB[\s_-]*FIXT", re.I), FIXTURE_ANY),

    # -- casework/millwork. FloorPlanCAD calls this "cabinet". ----------------
    (re.compile(r"CASEWORK|Q[\s_-]*CASE|CSWK|MILLWORK|MILL[\s_-]*WK"
                r"|COUNTERTOP|CABINET|VANIT", re.I), CABINET),

    # -- furniture: member where stated, else coarse --------------------------
    (re.compile(r"SOFA|COUCH", re.I), SOFA),
    (re.compile(r"\bBEDS?\b(?!ROOM)", re.I), A_BED),
    (re.compile(r"WARDROBE|CLOSET[\s_-]*ROD", re.I), WARDROBE),
    (re.compile(r"\bTABLES?\b|\bDESKS?\b", re.I), A_TABLE),
    (re.compile(r"ROW[\s_-]*CHAIR|AUDITORIUM|THEATER[\s_-]*SEAT", re.I), ROW_CHAIRS),
    (re.compile(r"\bCHAIRS?\b|SEATING", re.I), CHAIR),
    (re.compile(r"FURN", re.I), FURNITURE_ANY),

    # -- appliances: member where stated, else coarse -------------------------
    (re.compile(r"REFRIGERATOR|FRIDGE", re.I), REFRIGERATOR),
    (re.compile(r"\bRANGES?\b|COOKTOP|\bSTOVE\b|\bOVENS?\b", re.I), GAS_STOVE),
    (re.compile(r"WASHER|DRYER|LAUNDRY|WASHING[\s_-]*MACH", re.I), WASHING_MACHINE),
    (re.compile(r"AIR[\s_-]*COND|MINI[\s_-]*SPLIT|\bCONDENSER\b", re.I), AIRCON),
    (re.compile(r"APPL", re.I), APPLIANCE_ANY),

    # -- structure ------------------------------------------------------------
    (re.compile(r"PRKG|PARKING", re.I), PARKING_SPOT),
    (re.compile(r"COLUMN|COLS?[\s_-]|COLS$|PILASTER", re.I), COLUMN),
    (re.compile(r"BEAM|JOIST|LINTEL|TRUSS|GIRDER|STUD|FRAMING|RAFTER",
                re.I), FRAMING),
    (re.compile(r"ROOF|RIDGE|SHINGLE", re.I), ROOF),

    # -- MEP, before the wall rule so M-DIFF-WALL lands on mechanical ---------
    (re.compile(r"^E[\s_-]|LITE|RCPT|WIRE|ELEC|POWR|PANL|SWCH|LIGHTING",
                re.I), ELECTRICAL),
    (re.compile(r"^M[\s_-]|DUCT|DIFF|HVAC|MECH|\bVAV\b|\bRTU\b|REFG|REFRIG",
                re.I), MECHANICAL),
    (re.compile(r"^P[\s_-]|^F[\s_-]|DOMW|NGAS|STRM|PIPE|SPKL|SANR|PROT",
                re.I), PIPE),
    (re.compile(r"^C[\s_-]", re.I), SITE),

    # -- walls ----------------------------------------------------------------
    # No \b around CMU, for the reason the 4-class table documents above: the
    # real layers are B-CMUB-BS, B-CMUHDR-S, S-CMUFND-F, and a word boundary
    # drops all three.
    (re.compile(r"WALLS?|PARTITION|PRTN|CMU|MASONRY|BRICK", re.I), A_WALL),

    # -- anything else equipment-shaped --------------------------------------
    (re.compile(r"EQPM|SPCQ|EQUIP", re.I), EQUIPMENT),
]

assert all(c == ARCH_BG or c in ARCH_NAMES or c in COARSE_GROUPS
           for _, c in _ARCH_LAYER_RULES), "a layer rule targets an unknown class"


def from_arch_layer(layer_name):
    """AIA-style CAD layer name -> Arch-43 index (or a band-C coarse id)."""
    if not layer_name:
        return ARCH_BG
    name = layer_name.split("|")[-1]  # drop XREF prefix: "XREF - 1st Floor|A-DOOR"
    for pattern, cls in _ARCH_LAYER_RULES:
        if pattern.search(name):
            return cls
    return ARCH_BG
