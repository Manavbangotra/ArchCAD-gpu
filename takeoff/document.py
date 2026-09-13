"""The takeoff JSON.

Field names follow bim-ai's building schema (bim_ai/schema/building.py) where
the concept exists -- `id`, `tag`, `position`, `width_mm`, `schedule_ref`,
`polygon`, `area_m2`, `is_sleeping`, `mm_per_px`-style scale provenance -- so a
bim-ai reader recognises it; everything is millimetres in a floor-local, Y-up
frame. bim-ai has no model for fixtures, furniture or appliances; those are
this document's additions.

Coordinates are per drawing: each object is converted with the scale of the
viewport it sits in, so two viewports on one sheet are not in a common frame.

Totals count plan viewports only (including untitled drawings, which are
nearly always plans on these sets). Elevations and details draw the same
doors and fixtures again, and counting them is the classic takeoff double count.
"""

import os.path as osp
import re
import sys

import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "dataset"))
import room_names  # noqa: E402
import taxonomy as tx  # noqa: E402

SCHEMA_VERSION = "takeoff-0.1"

DOOR_CLASSES = set(tx.COARSE_GROUPS[tx.DOOR_ANY]) | {tx.DOOR_ANY}
WINDOW_CLASSES = {tx.A_WINDOW, tx.BAY_WINDOW, tx.BLIND_WINDOW}
WALL_CLASSES = {tx.A_WALL, tx.CURTAIN_WALL}
BARRIER_CLASSES = WALL_CLASSES | {tx.RAILING} | WINDOW_CLASSES
FIXTURE_CLASSES = set(tx.COARSE_GROUPS[tx.FIXTURE_ANY]) | {tx.FIXTURE_ANY}
APPLIANCE_CLASSES = set(tx.COARSE_GROUPS[tx.APPLIANCE_ANY]) | {tx.APPLIANCE_ANY}
FURNITURE_CLASSES = set(tx.COARSE_GROUPS[tx.FURNITURE_ANY]) | {tx.FURNITURE_ANY}
VERTICAL_CLASSES = {tx.STAIRS, tx.ELEVATOR, tx.ESCALATOR}

OPENING_KINDS = {c: "door" for c in DOOR_CLASSES}
OPENING_KINDS.update({c: "window" for c in WINDOW_CLASSES})


def class_name(cid):
    return tx.COARSE_NAMES.get(cid) or tx.ARCH_NAMES.get(cid, f"class-{cid}")


def category(cid):
    if cid in DOOR_CLASSES:
        return "doors"
    if cid in WINDOW_CLASSES:
        return "windows"
    if cid in WALL_CLASSES:
        return "walls"
    if cid in FIXTURE_CLASSES:
        return "fixtures"
    if cid in APPLIANCE_CLASSES:
        return "appliances"
    if cid in FURNITURE_CLASSES:
        return "furniture"
    if cid in VERTICAL_CLASSES:
        return "vertical_circulation"
    return "other"


def _mm(pt, origin_pt, mm_per_pt):
    return [round((pt[0] - origin_pt[0]) * mm_per_pt, 1), round((pt[1] - origin_pt[1]) * mm_per_pt, 1)]


_ROLES = [
    # "... FLOOR PLAN & INTERIOR ELEVATIONS" is a plan sheet, hence the lookahead.
    ("not_a_plan", re.compile(r"\bELEV\b\.?|^(?!.*\bPLAN\b).*ELEVATIONS?\s*$|\bDETAILS?\b|"
                              r"ACCESS\s+(PANEL|DOOR)|"
                              r"\bKEY\b|\bLEGEND\b|\bNOTES\b|\bSCHEDULE\b|ARCHITECTS?\b|\bPLLC\b|"
                              r"\bINC\b\.?|\bLLC\b", re.I)),
    ("reflected_ceiling", re.compile(r"REFLECTED|CEILING|\bRCP\b", re.I)),
    ("electrical", re.compile(r"LIGHTING|ELECTRICAL|POWER|\bLOW\s+VOLTAGE", re.I)),
    ("mep", re.compile(r"MECHANICAL|HVAC|PLUMBING|FIRE\s*(ALARM|SPRINKLER|PROTECTION)|"
                       r"DOMESTIC\s+WATER|SANITARY|\bWASTE\b|\bVENT\b|GAS\s+PIPING|"
                       r"SPRINKLER|DUCTWORK|\bPIPING\b|\bSYST?EMS\b|RADON", re.I)),
    ("structural", re.compile(r"FRAMING|FOUNDATION|SLAB|STRUCTURAL|SHEAR\s*WALL|FOOTING|"
                              r"\bCOLUMN\b", re.I)),
    ("roof", re.compile(r"\bROOF\b|\bATTIC\b|DRAFTSTOP", re.I)),
    ("site", re.compile(r"\bSITE\b|GRADING|LANDSCAPE|UTILITY\s+PLAN", re.I)),
    ("demolition", re.compile(r"\bDEMO(LITION)?\b", re.I)),
    # A title that is only a sheet number names its discipline by the prefix.
    ("structural", re.compile(r"^\s*S-?\d{1,3}(\.\d+)?\s*$", re.I)),
    ("mep", re.compile(r"^\s*[MP]-?\d{1,3}(\.\d+)?\s*$", re.I)),
    ("electrical", re.compile(r"^\s*E-?\d{1,3}(\.\d+)?\s*$", re.I)),
    ("enlarged_plan", re.compile(r"ENLARGED|\bUNIT\b|TYPICAL\s+(UNIT|APARTMENT)|"
                                 r"\bBEDROOM\b|\bEFFICIENCY\b|\d+\s*S\.?\s*F\.?\b", re.I)),
    ("floor_plan", re.compile(r"FLOOR\s*PLAN|\bLEVEL\b|DIMENSION\s*PLAN|OVERALL", re.I)),
]
COUNTED_ROLES = ("floor_plan", "enlarged_plan", "plan", "untitled")

# "68 REQUIRED A2 - ONE BEDROOM", "3 REQ'D @BLDGs #1, #6 & #7": how many times
# the drawn unit or building is built. Reported, never multiplied into the
# totals -- an overall floor plan may already draw the same units.
_REQUIRED = re.compile(r"\b(\d{1,4})\s*(?:REQ(?:'?D|UIRED)|TYP(?:ICAL)?\s+OF)\b", re.I)


def required_count(title):
    m = _REQUIRED.search(title or "")
    return int(m.group(1)) if m else None


def viewport_role(kind, title):
    """What a titled drawing is for, as far as a takeoff is concerned.

    A reflected ceiling plan, a lighting plan and a framing plan all draw the
    walls and doors again, and classify_viewports keeps them as "plan" because
    their geometry is good training data. Counted, they are the second, third
    and fourth copy of every door on the floor. Order matters: "ENLARGED UNIT
    PLAN - DOMESTIC WATER" is plumbing, so disciplines are tested before
    "unit" / "floor plan".
    """
    if kind == "untitled":
        return "untitled"
    if kind == "drop":
        return "not_a_plan"
    for role, rx in _ROLES:
        if rx.search(title or ""):
            return role
    return "plan"


def _counts_as_plan(role):
    return role in COUNTED_ROLES


def page_entry(page_no, sheet, objects, args, lengths, openings, rooms, viewport_of, scale_at,
               origin_pt=(0.0, 0.0), pdf_origin=(0.0, 0.0)):
    """One page of the document.

    viewport_of(x, y) -> (kind, title); scale_at(x, y) -> takeoff.scale.Scale.
    `pdf_origin` shifts page-local points back to PDF user space for the
    `pdf_bbox` / `pdf_polygon` fields (points, y up), which locate every item
    on the sheet itself.
    """
    ox, oy = pdf_origin
    p = page_no
    entry = {"page_index": p, "sheet_no": sheet.get("sheet_no", ""),
             "sheet_title": sheet.get("sheet_title", ""), "viewports": {},
             "doors": [], "windows": [], "walls": [], "rooms": [], "fixtures": [],
             "appliances": [], "furniture": [], "vertical_circulation": [], "other": []}

    def viewport(x, y):
        kind, title = viewport_of(x, y)
        key = title or "(untitled)"
        role = viewport_role(kind, title)
        if key not in entry["viewports"]:
            sc = scale_at(x, y)
            entry["viewports"][key] = {"kind": kind, "role": role,
                                       "required_count": required_count(title),
                                       "scale": sc.text or f"x{sc.factor:g}",
                                       "scale_factor": sc.factor, "scale_source": sc.source}
        return role, key

    by_obj = {op.obj: op for op in openings}
    n_d = n_w = 0
    for k, ob in enumerate(objects):
        cat = category(ob.label)
        pts = np.asarray([args[i] for i in ob.prims], dtype=np.float64).reshape(-1, 8)
        x0, y0, x1, y1 = pts[:, 0::2].min(), pts[:, 1::2].min(), pts[:, 0::2].max(), pts[:, 1::2].max()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        kind, vp = viewport(cx, cy)
        sc = scale_at(cx, cy)
        base = {"class": class_name(ob.label), "score": round(float(ob.score), 3),
                "viewport": vp, "viewport_role": kind, "counted": _counts_as_plan(kind),
                "position": _mm((cx, cy), origin_pt, sc.mm_per_pt),
                "bbox_mm": [round((x1 - x0) * sc.mm_per_pt), round((y1 - y0) * sc.mm_per_pt)],
                "pdf_bbox": [round(x0 + ox, 1), round(y0 + oy, 1), round(x1 + ox, 1), round(y1 + oy, 1)],
                "source": "model" if ob.sources != ["labels"] else "layers"}
        if cat in ("doors", "windows"):
            op = by_obj.get(k)
            if cat == "doors":
                n_d += 1
                ident = f"D-P{p}-{n_d:03d}"
            else:
                n_w += 1
                ident = f"WIN-P{p}-{n_w:03d}"
            item = {"id": ident, "tag": op.tag if op else "", **base,
                    "width_mm": op.width_mm if op else base["bbox_mm"][0],
                    "width_source": "schedule" if op and op.schedule.get("width_in") else "measured",
                    "schedule_ref": op.schedule_ref if op else None}
            if op and op.schedule:
                item["schedule"] = op.schedule
                if op.schedule.get("width_in"):
                    item["width_mm"] = round(op.schedule["width_in"] * 25.4, 1)
                if op.schedule.get("height_in"):
                    item["height_mm"] = round(op.schedule["height_in"] * 25.4, 1)
                    item["height_source"] = "schedule"
            entry[cat].append(item)
        elif cat == "walls":
            item = {"id": f"W-P{p}-{len(entry['walls']) + 1:03d}", **base,
                    # Sum of drawn linework. A wall drawn as two face lines
                    # measures about twice its run; not halved, because single
                    # line walls exist on the same sheets.
                    "linework_length_mm": round(float(np.sum(np.asarray(lengths)[ob.prims])) * sc.mm_per_pt),
                    "primitives": int(ob.prims.size)}
            entry["walls"].append(item)
        else:
            entry[cat].append({"id": f"{cat[:3].upper()}-P{p}-{len(entry[cat]) + 1:03d}", **base})

    for i, rm in enumerate(rooms):
        c = np.mean(np.asarray(rm.polygon_pt), axis=0)
        kind, vp = viewport(*c)
        sc = scale_at(*c)
        entry["rooms"].append({
            "id": f"R-P{p}-{i + 1:03d}", "name": rm.name, "number": rm.number,
            "group": room_names.GROUP_NAMES[rm.group], "is_sleeping": rm.is_sleeping,
            "polygon": [_mm(q, origin_pt, sc.mm_per_pt) for q in rm.polygon_pt],
            "pdf_polygon": [[round(q[0] + ox, 1), round(q[1] + oy, 1)] for q in rm.polygon_pt],
            "area_m2": rm.area_m2, "area_sqft": round(rm.area_m2 * 10.7639, 1),
            "viewport": vp, "viewport_role": kind, "counted": _counts_as_plan(kind),
            "source": "walls+text"})
    return entry


def totals(pages):
    out = {"doors": {"count": 0, "by_class": {}, "by_tag": {}},
           "windows": {"count": 0, "by_class": {}, "by_tag": {}},
           "fixtures": {}, "appliances": {}, "furniture": {}, "vertical_circulation": {},
           "rooms": {"count": 0, "area_m2": 0.0, "by_group": {}},
           "walls": {"linework_length_mm": 0},
           # Everything, counted or not, by what the drawing it sits on is for --
           # so an estimator can see what the plan-only totals left out.
           "by_viewport_role": {}}
    for pg in pages:
        for cat in ("doors", "windows", "rooms", "fixtures", "appliances", "furniture"):
            for it in pg[cat]:
                r = out["by_viewport_role"].setdefault(it["viewport_role"], {})
                r[cat] = r.get(cat, 0) + 1
        for cat in ("doors", "windows"):
            for it in pg[cat]:
                if not it["counted"]:
                    continue
                t = out[cat]
                t["count"] += 1
                t["by_class"][it["class"]] = t["by_class"].get(it["class"], 0) + 1
                tag = it["tag"] or "(untagged)"
                t["by_tag"][tag] = t["by_tag"].get(tag, 0) + 1
        for cat in ("fixtures", "appliances", "furniture", "vertical_circulation"):
            for it in pg[cat]:
                if it["counted"]:
                    out[cat][it["class"]] = out[cat].get(it["class"], 0) + 1
        for rm in pg["rooms"]:
            if not rm["counted"]:
                continue
            r = out["rooms"]
            r["count"] += 1
            r["area_m2"] = round(r["area_m2"] + rm["area_m2"], 2)
            g = r["by_group"].setdefault(rm["group"], {"count": 0, "area_m2": 0.0})
            g["count"] += 1
            g["area_m2"] = round(g["area_m2"] + rm["area_m2"], 2)
        for w in pg["walls"]:
            if w["counted"]:
                out["walls"]["linework_length_mm"] += w["linework_length_mm"]
    return out


def document(source_pdf, page_count, pages, model_info, warnings=()):
    warnings = list(warnings)
    roles = {it["viewport_role"] for pg in pages for it in pg["doors"] + pg["rooms"]}
    if {"floor_plan", "enlarged_plan"} <= roles:
        warnings.append("both overall floor plans and enlarged unit plans are counted; "
                        "units drawn on both appear twice in the totals -- see by_viewport_role")
    return {
        "project": {"name": osp.splitext(osp.basename(source_pdf))[0], "source_pdf": source_pdf,
                    "page_count": page_count, "units": "mm", "generated_by": "ArchCAD-gpu takeoff",
                    "schema_version": SCHEMA_VERSION},
        "model": model_info,
        "totals": totals(pages),
        "pages": pages,
        "warnings": warnings,
    }
