#!/usr/bin/env python3
"""Room tags printed on a plan -> a room name and one of CubiCasa's 12 room groups.

Standard library only, so the label editor can import it without numpy or torch.

US construction plans print a tag inside each room -- "BEDROOM #1", "W.I.C.",
"LAUN.", "PANTRY/LINEN" -- as real, positioned text. Measured over the 1,318 plan
pages of this corpus: CLOSET 1269, KITCHEN 1188, BEDROOM 1047, LIVING 853, WC 666,
CORRIDOR 602, BATHROOM 484, BATH 376, GARAGE 350, ENTRY 317, STORAGE 255,
BALCONY 245, PANTRY 156, W/D 151, DINING 150, LAUNDRY 144, W.I.C. 104, POWDER 44.
Firms abbreviate differently (`LAUN.`, `CLO.`, `DEN`), so matching is on
normalised tokens and an abbreviation table, not on exact words.

The groups follow CubiCasa5K so its plans can be used alongside these:

    0 Background  1 Outdoor  2 Wall  3 Kitchen  4 Living  5 Bedroom  6 Bath
    7 Entry/Hallway  8 Railing  9 Storage  10 Garage  11 Undefined

with the mapping agreed for this project: Kitchen <- kitchen, pantry; Storage <-
closet, walk-in closet, storage, laundry, utility, mechanical. CubiCasa itself files
laundry and pantry under Undefined; the raw name is always returned alongside the
group, so that choice stays reversible and "LAUNDRY" can still be told from "CLOSET".

    >>> parse("BEDROOM #1")
    ('BEDROOM #1', 5)
    >>> parse("WC-1") is None          # a plumbing fixture tag, not a room
    True
"""

import re

BACKGROUND, OUTDOOR, WALL, KITCHEN, LIVING, BEDROOM, BATH = 0, 1, 2, 3, 4, 5, 6
ENTRY, RAILING, STORAGE, GARAGE, UNDEFINED = 7, 8, 9, 10, 11

GROUP_NAMES = {
    BACKGROUND: "background", OUTDOOR: "outdoor", WALL: "wall", KITCHEN: "kitchen",
    LIVING: "living", BEDROOM: "bedroom", BATH: "bath", ENTRY: "entry/hallway",
    RAILING: "railing", STORAGE: "storage", GARAGE: "garage", UNDEFINED: "undefined",
}

# Fill colours for the annotator's room overlay. Chosen to stay readable at low
# opacity under dark linework and to keep the rooms a person confuses most
# (bath / kitchen / storage) visibly apart.
GROUP_HEX = {
    BACKGROUND: "#000000", OUTDOOR: "#7CB342", WALL: "#5D4037", KITCHEN: "#F9A825",
    LIVING: "#EF6C00", BEDROOM: "#3949AB", BATH: "#00897B", ENTRY: "#8E24AA",
    RAILING: "#6D4C41", STORAGE: "#78909C", GARAGE: "#546E7A", UNDEFINED: "#BDBDBD",
}

# Ordered: the first group with a matching token wins. The order is load-bearing
# for compound tags -- "MASTER BATH" is a bath, not a bedroom; "MECH. CLOSET" is
# storage; "PANTRY/LINEN" is kitchen -- so the more specific rooms are tried first.
_GROUP_TOKENS = [
    (BATH, {"BATH", "BATHROOM", "BATHRM", "BTH", "BA", "TOILET", "WC", "POWDER",
            "PWDR", "RESTROOM", "RESTROOMS", "LAV", "LAVATORY", "SHOWER", "SHWR",
            "WASHROOM", "ENSUITE"}),
    (KITCHEN, {"KITCHEN", "KIT", "KITCH", "KITCHENETTE", "PANTRY", "PAN"}),
    (STORAGE, {"CLOSET", "CLOSETS", "CLO", "CLOS", "CL", "WIC", "WALKIN", "STORAGE",
               "STOR", "STO", "LAUNDRY", "LAUN", "LNDRY", "WD", "UTILITY", "UTIL",
               "MECH", "MECHANICAL", "LINEN", "LIN"}),
    (BEDROOM, {"BEDROOM", "BEDROOMS", "BDRM", "BDR", "BR", "BED", "MASTER", "MSTR",
               "SUITE", "NURSERY", "GUEST"}),
    (LIVING, {"LIVING", "LIV", "FAMILY", "FAM", "GREAT", "DINING", "DIN", "DEN",
              "LOUNGE", "SITTING", "PARLOR", "BREAKFAST", "NOOK"}),
    (ENTRY, {"ENTRY", "ENTRANCE", "FOYER", "HALL", "HALLWAY", "CORRIDOR", "CORR",
             "VESTIBULE", "VEST", "LOBBY", "MUDROOM", "GALLERY"}),
    (GARAGE, {"GARAGE", "GAR", "CARPORT"}),
    (OUTDOOR, {"PATIO", "BALCONY", "BALC", "DECK", "PORCH", "TERRACE", "YARD",
               "COURTYARD", "LANAI", "VERANDA"}),
    (UNDEFINED, {"OFFICE", "STUDY", "STAIR", "STAIRS", "STAIRWELL", "ELEVATOR",
                 "TRASH", "AMENITY", "FITNESS", "CLUBHOUSE", "CLUB", "LEASING",
                 "MAIL", "ELECTRICAL", "ELEC", "IDF", "MDF", "JANITOR", "JAN", "ROOM",
                 "BONUS", "LOFT", "MEDIA", "GAME", "FLEX", "LIBRARY", "GYM", "POOL",
                 "SPA", "SAUNA", "WORKSHOP", "ATTIC", "BASEMENT", "SERVER"}),
]

# Tokens that are room words only when they stand for the room on their own. "BA"
# and "CL" are common elsewhere; demanding they be the whole tag (bar a number)
# keeps "CL" in "CL 2X4" out.
_STANDALONE_ONLY = {"BA", "CL", "BR", "PAN", "LIN", "GAR", "JAN", "FAM", "DIN", "BED",
                    "WC", "WD", "ROOM"}

# Not rooms, each seen in this corpus next to a room word.
_NOT_A_ROOM = re.compile(
    r"LEGEND|SCHEDULE|\bNOTES?\b|ACCESSOR|RECEPT|PANEL|SEALANT|PENETRAT|\bTYP\b|"
    r"DETAIL|SECTION|ELEVATION|\bELEV\b\.?|\bSCALE\b|\bPLAN\b|KEYNOTE|\bSEE\b|"
    r"\bEXISTING\b|\bDEMO\b|\bN\.?T\.?S\b|\bSIM\b|\bREF\b|CEILING|\bHDR\b|\bSOFFIT\b",
    re.I)

# Fixture and opening tags: WC-1, W-3, D12A, GD-2. These are the most common false
# positive -- 423 "WC-1"/"WC-2" on one document alone.
_TAG = re.compile(r"^(WC|W|D|DR|WD|GD|SD|P|L|T)[-.\s]?\d+[A-Z]?$", re.I)

# Dimensions and areas are not names: 12'-0" X 11'-6", 142 SF, 24" G.B.
_DIMENSION = re.compile(r"\d+\s*['′\"”″]|\d+\s*(SF|SQ\.?\s*FT)\b|\d+\s*[xX]\s*\d+", re.I)

# Detail and section callouts: 01/A3.0, 5/A-501. A room tag stacked under one
# must not absorb it.
_CALLOUT = re.compile(r"\b\d{1,2}\s*/\s*[A-Z]{1,2}[-.]?\d", re.I)


def _tokens(text):
    """Uppercase tokens with punctuation folded: 'W.I.C.' -> 'WIC', 'W/D' -> 'WD',
    'WALK-IN' -> 'WALKIN', 'BATH#1' -> 'BATH', '1'."""
    t = text.upper()
    # Dotted initialisms, 'W.I.C.' / 'W.I.C' -> 'WIC'.
    t = re.sub(r"\b(?:[A-Z]\.)+[A-Z]\b", lambda m: m.group().replace(".", ""), t)
    t = t.replace("W/D", "WD").replace("WALK-IN", "WALKIN")
    t = t.replace("WALK IN", "WALKIN")
    return [w for w in re.split(r"[^A-Z0-9]+", t) if w]


def clean(text):
    """Collapse whitespace and trim stray punctuation, keeping '#1', '/', '&'."""
    s = re.sub(r"\s+", " ", text or "").strip()
    return s.strip(" .:,;-")


def parse(text):
    """(clean room name, group id) for a room tag, or None if it is not one."""
    if not text:
        return None
    name = clean(text)
    if not name or len(name) > 40:
        return None
    if _TAG.match(name) or _NOT_A_ROOM.search(name) or _DIMENSION.search(name) \
            or _CALLOUT.search(name):
        return None
    toks = _tokens(name)
    words = [w for w in toks if not w.isdigit() and not re.fullmatch(r"[A-Z]?\d+[A-Z]?", w)]
    if not words or len(words) > 5:
        return None
    for group, vocab in _GROUP_TOKENS:
        for w in words:
            if w not in vocab:
                continue
            if w in _STANDALONE_ONLY and len(words) > 1 and not _paired_ok(w, words):
                continue
            return name, group
    return None


def _paired_ok(token, words):
    """Standalone-only tokens still count in a few well-known pairs."""
    pairs = {"BED": {"ROOM"}, "ROOM": set(), "WC": set(), "BR": {"MASTER"},
             "DIN": {"LIV"}, "FAM": {"RM", "ROOM"}}
    others = set(words) - {token}
    if token == "ROOM":
        # "SERVER ROOM", "STORAGE ROOM": the other word decides; only fall back to
        # ROOM when no other word is a room word at all.
        return not any(w in v for _, v in _GROUP_TOKENS for w in others)
    return bool(others & pairs.get(token, set()))


def merge_stacked(lines, gap=0.7):
    """Join tag lines stacked in one room: 'WALK-IN' over 'CLOSET'.

    `lines` is a list of (text, (x0, y0, x1, y1)) in any y-down frame. A short line
    directly above another, horizontally overlapping and within `gap` line-heights,
    is joined to it. Returns the same shape. Only short lines (<= 3 words) merge, so
    a paragraph of notes never swallows a tag.
    """
    items = sorted(lines, key=lambda l: (l[1][1], l[1][0]))
    used = [False] * len(items)
    out = []
    for i, (ti, bi) in enumerate(items):
        if used[i]:
            continue
        text, (x0, y0, x1, y1) = ti, bi
        if len(ti.split()) <= 3:
            h = max(y1 - y0, 1e-6)
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                tj, (a0, b0, a1, b1) = items[j]
                if b0 - y1 > gap * h:
                    break
                if len(tj.split()) > 3 or b0 < y1 - 0.5 * h:
                    continue
                overlap = min(x1, a1) - max(x0, a0)
                if overlap <= 0.3 * min(x1 - x0, a1 - a0):
                    continue
                text = f"{text} {tj}"
                x0, y0, x1, y1 = min(x0, a0), min(y0, b0), max(x1, a1), max(y1, b1)
                used[j] = True
                h = max(b1 - b0, 1e-6)
        out.append((text, (x0, y0, x1, y1)))
    return out
