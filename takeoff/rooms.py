"""Rooms: closed spaces bounded by walls, named from the tags printed inside.

Method ported from bim-ai geometry/room_builder.py (raster flood), run on the
model's wall and window primitives instead of detected wall segments:

1. draw wall, curtain-wall, railing and window primitives onto a mask;
2. seal doorways with the detected doors' boxes and close only small drafting
   gaps (bim-ai closes with a door-width kernel, which also fills closets and
   corridors narrower than a door is wide -- see build_rooms);
3. connected components of what is left are candidate rooms; anything touching
   the drawing's border is the outside;
4. contour -> simplified polygon, area from pixels x scale^2;
5. name and group from room tags inside the polygon (dataset/room_names.py).

Room tags matter as much as walls here: a closet with no tag is still a room,
but a component with no tag and no reasonable size is title-block furniture.
"""

import os.path as osp
import re
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "dataset"))
import room_names  # noqa: E402

DOOR_SEAL_MM = 1150.0         # fallback seal when no door boxes are given
SMALL_GAP_MM = 150.0          # drafting gaps between wall lines, not openings
ROOM_MIN_M2 = 1.2
ROOM_MAX_M2 = 2500.0
UNNAMED_MIN_M2 = 3.0
MAX_RASTER = 7000             # pixels on the long side


@dataclass
class Room:
    polygon_pt: list              # [(x, y)] page PDF points, y up
    area_m2: float
    name: str = ""
    group: int = room_names.UNDEFINED
    number: str = ""
    tags: list = field(default_factory=list)

    @property
    def is_sleeping(self):
        return self.group == room_names.BEDROOM


def _raster_frame(args, prims):
    pts = np.asarray([args[i] for i in prims], dtype=np.float64).reshape(-1, 8)
    x0, y0 = pts[:, 0::2].min(), pts[:, 1::2].min()
    x1, y1 = pts[:, 0::2].max(), pts[:, 1::2].max()
    return x0, y0, x1, y1


def build_rooms(args, barrier_prims, mm_per_pt, lines, door_boxes=()):
    """Rooms from barrier primitives on one page (or one viewport).

    args           page primitive control points (8 numbers each), PDF points, y up
    barrier_prims  indices of wall / curtain wall / railing / window primitives
    mm_per_pt      real millimetres per PDF point for this drawing
    lines          (text, (x0, y0, x1, y1)) text lines, PDF points, y up
    door_boxes     (x0, y0, x1, y1) of detected doors, PDF points

    Doorways are sealed with the doors themselves, not with a door-width
    closing. bim-ai closes the wall mask with a ~1.15 m kernel, which also
    fills every space narrower than that -- closets, corridors, a WIC -- so
    they never came out as rooms. Here the closing only bridges drafting gaps
    (SMALL_GAP_MM), each door's box is a barrier, and afterwards the door-box
    pixels are handed back to the nearest room so no area is lost.
    """
    from scipy import ndimage

    barrier_prims = np.asarray(barrier_prims, dtype=np.int64)
    if barrier_prims.size < 4 or mm_per_pt <= 0:
        return []
    bx0, by0, bx1, by1 = _raster_frame(args, barrier_prims)
    pad = DOOR_SEAL_MM / mm_per_pt                      # page points
    bx0, by0, bx1, by1 = bx0 - pad, by0 - pad, bx1 + pad, by1 + pad
    r = min(4.0, MAX_RASTER / max(bx1 - bx0, by1 - by0, 1e-6))   # pixels per point
    W = int(np.ceil((bx1 - bx0) * r)) + 1
    H = int(np.ceil((by1 - by0) * r)) + 1

    def to_px(x, y):
        return (x - bx0) * r, (by1 - y) * r               # y down in the raster

    mask = np.zeros((H, W), np.uint8)
    for i in barrier_prims.tolist():
        a = args[i]
        pts = np.array([to_px(a[j], a[j + 1]) for j in range(0, 8, 2)], dtype=np.float64)
        cv2.polylines(mask, [np.round(pts).astype(np.int32)], False, 255, 2)

    mm_per_px = mm_per_pt / r
    doors = np.zeros((H, W), np.uint8)
    for (dx0, dy0, dx1, dy1) in door_boxes:
        u0, v0 = to_px(dx0, dy1)
        u1, v1 = to_px(dx1, dy0)
        cv2.rectangle(doors, (int(u0), int(v0)), (int(np.ceil(u1)), int(np.ceil(v1))), 255, -1)

    gap = max(3, int(round(SMALL_GAP_MM / mm_per_px)) | 1)
    sealed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap)))
    sealed = cv2.bitwise_or(sealed, doors)
    if not door_boxes:
        # No door geometry to seal with: fall back to the door-width closing.
        seal = max(3, int(round(DOOR_SEAL_MM / mm_per_px)) | 1)
        sealed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (seal, seal)))
    free = cv2.bitwise_not(sealed)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)

    px_m2 = (mm_per_px ** 2) / 1e6
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x <= 1 or y <= 1 or x + bw >= W - 1 or y + bh >= H - 1:
            continue                                     # the outside world
        if area * px_m2 <= ROOM_MAX_M2:
            keep[i] = True
    labels = np.where(keep[labels], labels, 0).astype(np.int32)

    # Door boxes back to the nearest kept room: a doorway's floor belongs to
    # the rooms on either side of it, split down the middle.
    if door_boxes and keep.any():
        give = (doors > 0) & (mask == 0)
        _, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
        near = labels[iy, ix]
        labels = np.where(give & (labels == 0), near, labels)

    rooms = []
    found = ndimage.find_objects(labels)
    for i, sl in enumerate(found, start=1):
        if sl is None or not keep[i]:
            continue
        comp = (labels[sl] == i).astype(np.uint8) * 255
        area = float(cv2.countNonZero(comp)) * px_m2
        if not (ROOM_MIN_M2 <= area <= ROOM_MAX_M2):
            continue
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        poly = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True).reshape(-1, 2)
        if len(poly) < 3:
            continue
        ys, xs = sl[0].start, sl[1].start
        poly_pt = [((px + xs) / r + bx0, by1 - (py + ys) / r) for px, py in poly.tolist()]
        rooms.append(Room(poly_pt, round(area, 2)))

    _name_rooms(rooms, lines)
    rooms = [rm for rm in rooms if rm.name or rm.area_m2 >= UNNAMED_MIN_M2]
    return _drop_containers(rooms)


def _inside(poly, x, y):
    return cv2.pointPolygonTest(np.asarray(poly, dtype=np.float32), (float(x), float(y)), False) >= 0


def _name_rooms(rooms, lines):
    if not rooms or not lines:
        return
    # Stacked tags ("WALK-IN" over "CLOSET") join in a y-down frame.
    # Both the merged lines and the raw ones are candidates: a merge can pull in
    # a callout above the tag, and then the raw tag line is the one that parses.
    flipped = [(t, (x0, -y1, x1, -y0)) for t, (x0, y0, x1, y1) in lines]
    merged = [(t, (x0, -y1, x1, -y0)) for t, (x0, y0, x1, y1) in room_names.merge_stacked(flipped)]
    seen = set(merged)
    cands = merged + [ln for ln in lines if ln not in seen]
    centres = np.array([((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for _, b in cands])
    for rm in rooms:
        poly = np.asarray(rm.polygon_pt)
        (px0, py0), (px1, py1) = poly.min(0), poly.max(0)
        near = np.flatnonzero((centres[:, 0] >= px0) & (centres[:, 0] <= px1)
                              & (centres[:, 1] >= py0) & (centres[:, 1] <= py1))
        hits = []
        for k in near.tolist():
            text, (x0, y0, x1, y1) = cands[k]
            if not _inside(rm.polygon_pt, *centres[k]):
                continue
            rm.tags.append(text)
            parsed = room_names.parse(text)
            if parsed:
                hits.append((y1 - y0, parsed))           # taller text first
            elif not rm.number and re.fullmatch(r"[A-Z]?\d{2,4}[A-Z]?", text.strip()):
                rm.number = text.strip()
        if hits:
            hits.sort(key=lambda h: -h[0])
            rm.name, rm.group = hits[0][1]


def _drop_containers(rooms):
    """A page border or legend box read as walls becomes one unnamed 'room'
    around real ones; drop any unnamed room holding two others' centroids."""
    cents = [tuple(np.mean(np.asarray(r.polygon_pt), axis=0)) for r in rooms]
    keep = []
    for i, r in enumerate(rooms):
        inside = sum(1 for j, c in enumerate(cents) if j != i and _inside(r.polygon_pt, *c))
        if not (inside >= 2 and not r.name):
            keep.append(r)
    return keep
