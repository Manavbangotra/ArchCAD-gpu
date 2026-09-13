"""Doors and windows: where, how wide, which type mark, which schedule row.

A plan tags each opening with its type mark ("W3", "D-101", or a bare "01" in a
hexagon) next to it, and the schedule gives that mark's size and operation.
With a schedule available, only its marks are accepted as tags -- far stricter
than any pattern, because "01" is also a room number, a grid bubble and a
keynote. Without one, bim-ai's tag patterns (ported from
extraction/opening_detector.py, widened for leading-zero and letter marks) are
used.

Each tag is claimed by at most one opening, nearest first, within a radius set
in real millimetres so the rule means the same at every drawing scale.
"""

import json
import re
from dataclasses import dataclass, field

import numpy as np

# W1, W3B, W-12 -> window; D-101, GD-01, 150A -> door (bim-ai); plus 01, 2A, A1.
_WINDOW_TAG = re.compile(r"^W-?\d{1,3}[A-Z]?$")
_DOOR_TAG = re.compile(r"^(?:G?D-?\d{1,3}[A-Z]?|\d{3}[A-Z]?)$")
_GENERIC_TAG = re.compile(r"^(?:[A-Z]?\d{1,3}[A-Z]?|[A-Z]{1,2})$")

# A token that can be a mark: "D-107A", "01", "W3", "B" -- not "FLOOR" or "1ST".
_MARKISH = re.compile(r"^(?:[A-Z]{0,3}-?\d{1,4}[A-Z]?|[A-Z]{1,2})$", re.I)

TAG_RADIUS_MM = 1500.0        # a tag sits within ~5 ft of its opening
MM_PER_INCH = 25.4


@dataclass
class Opening:
    kind: str                     # door | window
    cls: int
    obj: int                      # index into the page's objects
    position_pt: tuple            # (x, y) page PDF points, y up
    width_mm: float
    score: float
    tag: str = ""
    schedule_ref: str = None
    schedule: dict = field(default_factory=dict)


def load_schedule(path):
    """{category: {normalised mark: item}} from a Schedule-detection planset JSON."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    out = {"door": {}, "window": {}}
    for it in doc.get("items", []):
        cat = str(it.get("category", "")).lower()
        if cat in out and it.get("mark"):
            # A schedule row can carry several marks ("D-107A D-107B D-106").
            for mark in str(it["mark"]).split():
                if _MARKISH.match(mark):
                    out[cat].setdefault(norm_mark(mark), it)
    return out


def norm_mark(mark):
    return re.sub(r"[\s\-_.]", "", str(mark).upper())


def _tag_ok(text, kind, schedule):
    t = norm_mark(text)
    if not t or len(t) > 6:
        return None
    if schedule and schedule.get(kind):
        return t if t in schedule[kind] else None
    if kind == "window" and _WINDOW_TAG.match(t):
        return t
    if kind == "door" and _DOOR_TAG.match(t):
        return t
    return None


def bbox_of(args, prims):
    pts = np.asarray([args[i] for i in prims], dtype=np.float64).reshape(-1, 8)
    xs, ys = pts[:, 0::2], pts[:, 1::2]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def build_openings(objects, args, kinds, scale_at, words, schedule=None):
    """Openings for every door/window object.

    objects   PageObjects (takeoff.stitch)
    args      page primitive control points, PDF points (y up, page-local)
    kinds     {class id: "door" | "window"}
    scale_at  (x, y) -> takeoff.scale.Scale
    words     (text, x, y) in the same frame as args
    """
    openings = []
    for k, ob in enumerate(objects):
        kind = kinds.get(ob.label)
        if not kind:
            continue
        x0, y0, x1, y1 = bbox_of(args, ob.prims)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        sc = scale_at(cx, cy)
        width_mm = max(x1 - x0, y1 - y0) * sc.mm_per_pt
        openings.append(Opening(kind, ob.label, k, (cx, cy), round(width_mm / MM_PER_INCH) * MM_PER_INCH,
                                ob.score))

    # Tags: nearest-first global assignment, one tag per opening and vice versa.
    cands = {}
    for kind in {op.kind for op in openings}:
        cands[kind] = [(t, x, y, wi) for wi, (text, x, y) in enumerate(words)
                       for t in [_tag_ok(text, kind, schedule)] if t]
    pairs = []
    for oi, op in enumerate(openings):
        radius_pt = TAG_RADIUS_MM / scale_at(*op.position_pt).mm_per_pt
        for t, x, y, wi in cands[op.kind]:
            d = ((x - op.position_pt[0]) ** 2 + (y - op.position_pt[1]) ** 2) ** 0.5
            if d <= radius_pt:
                pairs.append((d, oi, wi, t))
    pairs.sort()
    used_o, used_w = set(), set()
    for d, oi, wi, t in pairs:
        if oi in used_o or wi in used_w:
            continue
        used_o.add(oi)
        used_w.add(wi)
        op = openings[oi]
        op.tag = t
        if schedule and t in schedule.get(op.kind, {}):
            item = schedule[op.kind][t]
            op.schedule_ref = item.get("item_id") or t
            op.schedule = {k: item.get(k) for k in ("mark", "width_in", "height_in", "type_text")
                           if item.get(k) is not None}
    return openings
