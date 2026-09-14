#!/usr/bin/env python3
"""Drawing text -> (type, attributes): the input of TextCAD's text encoder (TACE).

TextCAD (arXiv 2607.12678, App. A) parses every text annotation with regex
patterns into a type and a small attribute vector -- "M1021" is a door 1000 wide
and 2100 high, "FM乙1021" a grade-B fire door -- and embeds that instead of the
characters (a BERT encoder was worse, and 100M parameters heavier). Their pattern
lists are dataset-specific and unpublished; this module is ours, written from the
text that actually occurs in the three training sources (measured on samples):

  FloorPlanCAD-V2  dimensions ("1000", "200x400"), Chinese room names (阳台, 厨房),
                   door/window codes (M1021, C1518, FM乙1021), stair arrows (上/下),
                   levels (-0.050), areas, slopes (1%).
  CubiCasa5K       Finnish room codes and names (MH, OH, K, WC, VH, TERASSI, PARVEKE).
  US plan sets     feet-inch dimensions, door/window/fixture tags (D-101, W-3, WB-4),
                   English room names (dataset/room_names.py), sheet and detail
                   references (A7.0, 5/A-501), scale strings, "N REQUIRED", notes.

Output of `parse(text)`: a Token with
  type     int, index into TYPE_NAMES
  attrs    three floats (signed log1p of the value in natural units: mm, m, m2, %)
  grade    int for a letter/character grade (0 = none), embedded separately
  mask     four bools: which of attrs[0..2] and grade are present

Standard library only.
"""

import math
import os.path as osp
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import room_names  # noqa: E402

MM_PER_INCH = 25.4
MM_PER_FOOT = 304.8

TYPE_NAMES = [
    "pad",                 # 0: reserved for padding
    "other",
    "dimension", "dimension_pair", "level", "area", "slope",
    "door_code", "window_code", "fire_door_code",
    "door_tag", "window_tag", "fixture_tag", "equipment_tag",
    "stair_up", "stair_down",
    "scale", "required_count",
    "sheet_ref", "detail_ref", "grid_label", "short_code", "number",
    "note", "equal_mark", "bullet", "grade_mark", "unit_mark",
    # Words that name a nearby object rather than a room: the strongest text cue
    # for the symbol classes (a bay-window note, "SINK", "GFCI").
    "hint_bay_window", "hint_void", "hint_hvac", "hint_plumbing", "hint_electrical",
    "hint_appliance", "hint_fixture", "hint_finish", "hint_structure",
] + [f"room_{g}" for g in ("outdoor", "kitchen", "living", "bedroom", "bath", "entry", "storage",
                           "garage", "undefined")]
TYPE_ID = {n: i for i, n in enumerate(TYPE_NAMES)}

GRADES = ["", "甲", "乙", "丙", "A", "B", "C"]
GRADE_ID = {g: i for i, g in enumerate(GRADES)}

_GROUP_TO_TYPE = {
    room_names.OUTDOOR: "room_outdoor", room_names.KITCHEN: "room_kitchen", room_names.LIVING: "room_living",
    room_names.BEDROOM: "room_bedroom", room_names.BATH: "room_bath", room_names.ENTRY: "room_entry",
    room_names.STORAGE: "room_storage", room_names.GARAGE: "room_garage", room_names.UNDEFINED: "room_undefined",
}

# Chinese room words (FloorPlanCAD-V2).
_ZH_ROOMS = [
    ("阳台", "room_outdoor"), ("露台", "room_outdoor"), ("花园", "room_outdoor"),
    ("厨房", "room_kitchen"), ("餐厅", "room_living"), ("客厅", "room_living"), ("起居", "room_living"),
    ("卧室", "room_bedroom"), ("宿舍", "room_bedroom"), ("主卧", "room_bedroom"), ("次卧", "room_bedroom"), ("客房", "room_bedroom"),
    ("卫生间", "room_bath"), ("卫", "room_bath"), ("浴室", "room_bath"), ("厕所", "room_bath"), ("洗手间", "room_bath"),
    ("走廊", "room_entry"), ("过道", "room_entry"), ("走道", "room_entry"), ("前室", "room_entry"), ("玄关", "room_entry"), ("门厅", "room_entry"), ("大堂", "room_entry"),
    ("储藏", "room_storage"), ("储物", "room_storage"), ("衣帽", "room_storage"), ("设备", "room_storage"),
    ("车库", "room_garage"), ("停车", "room_garage"),
    ("书房", "room_undefined"), ("办公", "room_undefined"), ("会议", "room_undefined"), ("楼梯", "room_undefined"),
    ("电梯", "room_undefined"), ("机房", "room_undefined"),
]

# Finnish room codes and words (CubiCasa5K).
_FI_ROOMS = {
    "MH": "room_bedroom", "H": "room_bedroom", "MAKUUHUONE": "room_bedroom", "ALKOVI": "room_bedroom",
    "OH": "room_living", "TUPA": "room_living", "RT": "room_living", "RUOK": "room_living", "TH": "room_living",
    "K": "room_kitchen", "KK": "room_kitchen", "KEITTIÖ": "room_kitchen", "KT": "room_kitchen",
    "WC": "room_bath", "KH": "room_bath", "PH": "room_bath", "KPH": "room_bath", "PESUH": "room_bath",
    "PESU": "room_bath", "PSH": "room_bath", "SH": "room_bath", "S": "room_bath", "SAUNA": "room_bath",
    "KHH": "room_storage", "VH": "room_storage", "VAR": "room_storage", "VARASTO": "room_storage",
    "PKH": "room_storage", "PUKUH": "room_storage", "TEKN": "room_storage", "TK": "room_storage",
    "KATT.H": "room_storage", "TEKN.TILA": "room_storage",
    "ET": "room_entry", "AULA": "room_entry", "AH": "room_entry", "KÄYTÄVÄ": "room_entry",
    "TERASSI": "room_outdoor", "PARVEKE": "room_outdoor", "ULKOTILA": "room_outdoor", "KUISTI": "room_outdoor",
    "LASITETTU PARVEKE": "room_outdoor", "KATOS": "room_outdoor",
    "AUTOTALLI": "room_garage", "AUTOKATOS": "room_garage", "AUTOVAJA": "room_garage", "AT": "room_garage",
    "TYÖH": "room_undefined", "PARVI": "room_undefined", "UNDEFINED": "room_undefined",
}

_FT_IN = re.compile(r"^(\d{1,3})'\s*[-–]?\s*(\d{1,2})?(?:\s+(\d{1,2})/(\d{1,2}))?\s*\"?$")
_IN = re.compile(r"^(\d{1,3})(?:\s+(\d{1,2})/(\d{1,2}))?\s*\"$")
_NUMBER = re.compile(r"^\d{1,6}$")
_DIM_PAIR = re.compile(r"^(\d{2,5})\s*[xX×*]\s*(\d{2,5})$")
_LEVEL = re.compile(r"^[±+-]?\d{1,3}\.\d{3}$")
_AREA = re.compile(r"^(\d+(?:\.\d+)?)\s*(m²|m2|㎡|SF|S\.F\.|SQ\.?\s*FT\.?)$", re.I)
_SLOPE = re.compile(r"^(?:[iI]\s*=\s*)?(\d+(?:\.\d+)?)\s*%$")
_CN_DOOR = re.compile(r"^(FM|M|MLC|TLM|JM)\s*([甲乙丙ABC])?\s*-?(\d{2})(\d{2})([a-zA-Z])?$")
_CN_WINDOW = re.compile(r"^(C|MC|LC|GC|TC|BYC)\s*-?(\d{2})(\d{2})([a-zA-Z])?$")
_DOOR_TAG = re.compile(r"^(?:D|DR|GD|SD)[-.]?(\d{1,4})[A-Z]?$|^(\d{3})[A-Z]?$", re.I)
_WINDOW_TAG = re.compile(r"^(?:W|WD)[-.]?(\d{1,3})[A-Z]?$", re.I)
_FIXTURE_TAG = re.compile(r"^(?:WC|WB|L|LAV|P|T|S|SH|UR|DF|FD|EF|WH|DW|MW)[-.]?(\d{1,3})[A-Z]?$", re.I)
_EQUIP_TAG = re.compile(r"^(?:AHU|RTU|FCU|CU|EF|WH|HP|MAU|VAV|B|P)-\d{1,3}[A-Z]?$", re.I)
_SHEET_REF = re.compile(r"^[A-Z]{1,2}-?\d{1,2}(?:\.\d{1,2})?[a-z]?$|^[A-Z]\d{3}$")
_DETAIL_REF = re.compile(r"^\d{1,2}\s*/\s*[A-Z]{1,2}-?\d")
_GRID = re.compile(r"^[A-Z]$|^[A-Z]\.\d$")
_SHORT = re.compile(r"^[A-Z]{1,3}\d{0,2}\.?$")
_REQUIRED = re.compile(r"\b(\d{1,4})\s*(?:REQ(?:'?D|UIRED))\b", re.I)
_STAIR_UP = re.compile(r"^(上|UP)$", re.I)
_STAIR_DN = re.compile(r"^(下|DN|DOWN)$", re.I)
_EQUAL = re.compile(r"^E\.?\s*Q\.?$", re.I)
_UNIT_MARK = re.compile(r"^(m²|m2|㎡|mm|cm|m|%|\(|\))$", re.I)

# Object hints, checked on whole words (EN) or substrings (ZH).
_HINTS_ZH = [("飘窗", "hint_bay_window"), ("上空", "hint_void"), ("风", "hint_hvac"), ("正压", "hint_hvac"),
             ("空调", "hint_hvac"), ("水", "hint_plumbing"), ("管井", "hint_plumbing"), ("电", "hint_electrical"),
             ("冰箱", "hint_appliance"), ("洗衣机", "hint_appliance"), ("灶", "hint_appliance"),
             ("洗手盆", "hint_fixture"), ("马桶", "hint_fixture"), ("淋浴", "hint_fixture"), ("柱", "hint_structure")]
_HINTS_EN = {
    "hint_hvac": {"HVAC", "AHU", "FURNACE", "DUCT", "RETURN", "SUPPLY", "E.F.", "EF", "EXHAUST", "DIFFUSER",
                  "GRILLE", "THERMOSTAT", "CONDENSER", "MINI-SPLIT"},
    "hint_plumbing": {"W.H.", "WH", "WATER", "HOSE", "BIB", "DRAIN", "FD", "CLEANOUT", "GAS"},
    "hint_electrical": {"GFCI", "USB", "USB-'C'", "OUTLET", "SWITCH", "PANEL", "J-BOX", "RECEPT", "220V",
                        "SMOKE", "CO", "DATA", "TV", "CATV", "DOORBELL", "LIGHT", "FIXTURE"},
    "hint_appliance": {"REF", "REF.", "FRIDGE", "REFRIGERATOR", "RANGE", "OVEN", "COOKTOP", "MICRO", "MW", "DW",
                       "DW.", "DISHWASHER", "WASHER", "DRYER", "W/D", "HOOD", "DISPOSAL"},
    "hint_fixture": {"SINK", "LAV", "LAV.", "TOILET", "TUB", "SHOWER", "SHWR", "URINAL", "BIDET", "VANITY",
                     "MIRROR", "GRAB", "BARS", "TOWEL", "ACCESSORIES"},
    "hint_finish": {"VINYL", "CARPET", "TILE", "LVT", "HARDWOOD", "CONCRETE", "SEALED", "PAINT", "BASE"},
    "hint_structure": {"COLUMN", "BEAM", "HEADER", "POST", "JOIST", "SHEAR", "2X4", "2X6", "2X8", "2X10", "2X12"},
}
_HINT_WORD = {w: name for name, words in _HINTS_EN.items() for w in words}


def _slog(v):
    return math.copysign(math.log1p(abs(v)), v)


@dataclass
class Token:
    type: int
    attrs: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    grade: int = 0
    mask: list = field(default_factory=lambda: [False, False, False, False])

    @property
    def name(self):
        return TYPE_NAMES[self.type]


def _tok(name, values=(), grade=None):
    t = Token(TYPE_ID[name])
    for i, v in enumerate(values[:3]):
        if v is not None:
            t.attrs[i] = _slog(float(v))
            t.mask[i] = True
    if grade:
        t.grade = GRADE_ID.get(grade, 0)
        t.mask[3] = t.grade != 0
    return t


def _ft_in(s):
    m = _FT_IN.match(s)
    if m:
        inch = int(m.group(2) or 0) + (int(m.group(3)) / int(m.group(4)) if m.group(3) else 0)
        return int(m.group(1)) * MM_PER_FOOT + inch * MM_PER_INCH
    m = _IN.match(s)
    if m:
        inch = int(m.group(1)) + (int(m.group(2)) / int(m.group(3)) if m.group(2) else 0)
        return inch * MM_PER_INCH
    return None


def parse(text):
    """Token for one annotation string. Never raises; unknown text is 'other'."""
    s = " ".join((text or "").replace("”", '"').replace("’", "'").split())
    if not s:
        return Token(TYPE_ID["pad"])
    u = s.upper()

    if s in ("●", "•", "·", "*"):
        return _tok("bullet")
    if _EQUAL.match(u):
        return _tok("equal_mark")
    if s in GRADE_ID and s:
        return _tok("grade_mark", grade=s)
    if _UNIT_MARK.match(s):
        return _tok("unit_mark")
    if _STAIR_UP.match(u):
        return _tok("stair_up")
    if _STAIR_DN.match(u):
        return _tok("stair_down")

    m = _CN_DOOR.match(u.replace(" ", "")) or _CN_DOOR.match(s.replace(" ", ""))
    if m:
        name = "fire_door_code" if m.group(1) == "FM" else "door_code"
        return _tok(name, (int(m.group(3)) * 100, int(m.group(4)) * 100), grade=m.group(2))
    m = _CN_WINDOW.match(u.replace(" ", ""))
    if m:
        return _tok("window_code", (int(m.group(2)) * 100, int(m.group(3)) * 100))

    mm = _ft_in(s)
    if mm is not None:
        return _tok("dimension", (mm,))
    if _NUMBER.match(s):
        v = int(s)
        # Chinese drawings dimension in mm; bare small integers are counts, bubbles, marks.
        return _tok("dimension", (v,)) if v >= 50 else _tok("number", (v,))
    m = _DIM_PAIR.match(s)
    if m:
        return _tok("dimension_pair", (int(m.group(1)), int(m.group(2))))
    if _LEVEL.match(s):
        return _tok("level", (float(s.replace("±", "")),))
    m = _AREA.match(s)
    if m:
        v = float(m.group(1))
        return _tok("area", (v if m.group(2).lower().startswith(("m", "㎡")) else v * 0.092903,))
    m = _SLOPE.match(s)
    if m:
        return _tok("slope", (float(m.group(1)),))

    # scale strings and required counts before tags: they contain digits too
    from_scale = _scale_factor(s)
    if from_scale:
        return _tok("scale", (from_scale,))
    m = _REQUIRED.search(u)
    if m:
        return _tok("required_count", (int(m.group(1)),))

    m = _DOOR_TAG.match(u)
    if m:
        return _tok("door_tag", (int(m.group(1) or m.group(2)),))
    m = _WINDOW_TAG.match(u)
    if m:
        return _tok("window_tag", (int(m.group(1)),))
    m = _FIXTURE_TAG.match(u)
    if m:
        return _tok("fixture_tag", (int(m.group(1)),))
    if _EQUIP_TAG.match(u):
        return _tok("equipment_tag")
    if _DETAIL_REF.match(u):
        return _tok("detail_ref")

    # rooms: Chinese words, Finnish codes, English names
    for word, name in _ZH_ROOMS:
        if word in s:
            return _tok(name)
    if u in _FI_ROOMS:
        return _tok(_FI_ROOMS[u])
    parsed = room_names.parse(s)
    if parsed:
        number = re.search(r"\d{1,4}", parsed[0])
        return _tok(_GROUP_TO_TYPE[parsed[1]], (int(number.group()) if number else None,))

    for word, name in _HINTS_ZH:
        if word in s:
            return _tok(name)
    for w in re.split(r"[\s,;:()]+", u):
        if w in _HINT_WORD:
            return _tok(_HINT_WORD[w])
    if _SHEET_REF.match(u):
        return _tok("sheet_ref")
    if _GRID.match(u):
        return _tok("grid_label")
    if _SHORT.match(u):
        return _tok("short_code")
    if len(s.split()) >= 2 or len(s) > 12:
        return _tok("note")
    return _tok("other")


def _scale_factor(s):
    try:
        from takeoff.scale import scale_from_text
    except ImportError:
        sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
        from takeoff.scale import scale_from_text
    return scale_from_text(s) if ("=" in s or ":" in s) else None


if __name__ == "__main__":
    for t in sys.argv[1:]:
        tok = parse(t)
        print(f"{t!r:30} {tok.name:16} attrs={[round(a, 3) for a in tok.attrs]} grade={tok.grade} mask={tok.mask}")
