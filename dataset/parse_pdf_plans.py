#!/usr/bin/env python3
"""
Convert US construction-document PDFs into the JSON form svgnet/data/svg.py reads.

Architectural PDFs exported from Revit/AutoCAD keep their drawing as real vector
geometry and tag it with the original CAD layer names via PDF optional-content
groups (OCGs). That gives a per-primitive label for free — no manual annotation:

    <</OCProperties ...>>            layer registry
    /OC /oc7 BDC ... EMC             content belonging to that layer

This walks each page's content stream, tracks the graphics state and the
marked-content stack, and assigns every line/curve segment the layer it was drawn
on. Layer names then map to classes via dataset/taxonomy.py.

Only pikepdf (MPL-2.0) is used for PDF access — deliberately not PyMuPDF, which is
AGPL and would encumber downstream use.

Usage:
    python dataset/parse_pdf_plans.py --pdf_dir <dir> --output_dir <dir> [--render]
"""

import argparse
import glob
import json
import math
import os
import os.path as osp
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from taxonomy import (BACKGROUND, CATEGORIES, CLASS_NAMES,  # noqa: E402
                      NUM_CLASSES, from_us_layer)

# Command-type one-hot slots, matching dataset/parse_FpCAD_svg.py.
CMD_LINE, CMD_ARC = 0, 1


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def mat_mul(m, n):
    """Concatenate two PDF matrices [a b c d e f]."""
    a1, b1, c1, d1, e1, f1 = m
    a2, b2, c2, d2, e2, f2 = n
    return (a1 * a2 + b1 * c2,
            a1 * b2 + b1 * d2,
            c1 * a2 + d1 * c2,
            c1 * b2 + d1 * d2,
            e1 * a2 + f1 * c2 + e2,
            e1 * b2 + f1 * d2 + f2)


def apply(m, x, y):
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def bezier_points(p0, p1, p2, p3):
    """Sample a cubic at t = 0, 1/3, 2/3, 1 — the four-point form the loader wants."""
    out = []
    for t in (0.0, 1 / 3, 2 / 3, 1.0):
        u = 1 - t
        x = u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]
        out.extend((x, y))
    return out


def line_points(p0, p1):
    """A straight segment, interpolated to four control points."""
    (x1, y1), (x2, y2) = p0, p1
    return [x1, y1,
            x1 + (x2 - x1) / 3, y1 + (y2 - y1) / 3,
            x1 + 2 * (x2 - x1) / 3, y1 + 2 * (y2 - y1) / 3,
            x2, y2]


# --------------------------------------------------------------------------- #
# content stream walker
# --------------------------------------------------------------------------- #
class PlanExtractor:
    """Collect layer-tagged vector primitives from one PDF page."""

    MAX_XOBJECT_DEPTH = 12   # guards against pathological or cyclic Form nesting

    def __init__(self, pdf):
        self.pdf = pdf
        self.prims = []

    # -- layer resolution -------------------------------------------------- #
    @staticmethod
    def _ocg_name(obj):
        try:
            if "/Name" in obj:
                return str(obj.Name)
        except Exception:
            pass
        return None

    def _layer_for(self, props, key):
        """Resolve a /Properties key (e.g. /oc7) to its CAD layer name.

        pikepdf dictionary keys keep their leading slash — props["oc7"] raises
        while props["/oc7"] resolves, so normalise rather than strip.
        """
        if not key.startswith("/"):
            key = "/" + key
        try:
            entry = props[key]
        except Exception:
            return None
        name = self._ocg_name(entry)
        if name:
            return name
        # Membership dicts (/OCMD) point at one or more OCGs via /OCGs.
        try:
            ocgs = entry.OCGs
            if isinstance(ocgs, list) and ocgs:
                return self._ocg_name(ocgs[0])
            return self._ocg_name(ocgs)
        except Exception:
            return None

    # -- main walk --------------------------------------------------------- #
    def run(self, stream_owner, resources, ctm, layer, depth=0):
        import pikepdf

        try:
            ops = pikepdf.parse_content_stream(stream_owner)
        except Exception:
            return

        try:
            props = resources.Properties
        except Exception:
            props = None
        try:
            xobjects = resources.XObject
        except Exception:
            xobjects = None

        gs = ctm                 # current transform
        gs_stack = []
        mc_stack = []            # marked-content: layer name or None per BDC
        cur_layer = layer
        width = 1.0
        width_stack = []
        rgb = (0, 0, 0)
        rgb_stack = []

        start = None             # subpath start, for `h`
        cur = None               # current point
        pending = []             # (points, kind, length) awaiting a paint op

        for instr in ops:
            op = str(instr.operator)
            args = instr.operands

            try:
                if op == "q":
                    gs_stack.append(gs); width_stack.append(width); rgb_stack.append(rgb)
                elif op == "Q":
                    if gs_stack:
                        gs = gs_stack.pop(); width = width_stack.pop(); rgb = rgb_stack.pop()
                elif op == "cm":
                    gs = mat_mul(tuple(float(a) for a in args), gs)
                elif op == "w":
                    width = float(args[0])
                elif op in ("RG", "SC", "SCN") and len(args) >= 3:
                    rgb = tuple(int(max(0.0, min(1.0, float(a))) * 255) for a in args[:3])
                elif op == "G" and len(args) == 1:
                    v = int(max(0.0, min(1.0, float(args[0]))) * 255); rgb = (v, v, v)

                # ---- marked content: this is where layers come from -------- #
                elif op == "BDC":
                    name = None
                    if len(args) >= 2 and str(args[0]) == "/OC" and props is not None:
                        name = self._layer_for(props, str(args[1]))
                    mc_stack.append(cur_layer)
                    if name:
                        cur_layer = name
                elif op == "BMC":
                    mc_stack.append(cur_layer)
                elif op == "EMC":
                    if mc_stack:
                        cur_layer = mc_stack.pop()

                # ---- path construction ------------------------------------ #
                elif op == "m":
                    cur = apply(gs, float(args[0]), float(args[1])); start = cur
                elif op == "l":
                    if cur is not None:
                        nxt = apply(gs, float(args[0]), float(args[1]))
                        pending.append((line_points(cur, nxt), CMD_LINE,
                                        math.dist(cur, nxt)))
                        cur = nxt
                elif op == "c" and len(args) >= 6:
                    if cur is not None:
                        p1 = apply(gs, float(args[0]), float(args[1]))
                        p2 = apply(gs, float(args[2]), float(args[3]))
                        p3 = apply(gs, float(args[4]), float(args[5]))
                        pts = bezier_points(cur, p1, p2, p3)
                        pending.append((pts, CMD_ARC, self._polyline_len(pts)))
                        cur = p3
                elif op in ("v", "y") and len(args) >= 4:
                    if cur is not None:
                        a = apply(gs, float(args[0]), float(args[1]))
                        b = apply(gs, float(args[2]), float(args[3]))
                        p1, p2 = (cur, a) if op == "v" else (a, b)
                        pts = bezier_points(cur, p1, p2, b)
                        pending.append((pts, CMD_ARC, self._polyline_len(pts)))
                        cur = b
                elif op == "re" and len(args) >= 4:
                    x, y, w_, h_ = (float(a) for a in args[:4])
                    corners = [apply(gs, x, y), apply(gs, x + w_, y),
                               apply(gs, x + w_, y + h_), apply(gs, x, y + h_)]
                    for i in range(4):
                        p, q = corners[i], corners[(i + 1) % 4]
                        pending.append((line_points(p, q), CMD_LINE, math.dist(p, q)))
                    cur = start = corners[0]
                elif op == "h":
                    if cur is not None and start is not None and cur != start:
                        pending.append((line_points(cur, start), CMD_LINE,
                                        math.dist(cur, start)))
                        cur = start

                # ---- paint: commit the path -------------------------------- #
                elif op in ("S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n"):
                    if op != "n":  # `n` ends the path without painting it
                        for pts, kind, length in pending:
                            self.prims.append({
                                "pts": pts, "cmd": kind, "len": length,
                                "layer": cur_layer, "w": abs(width) or 1.0, "rgb": rgb,
                            })
                    pending = []
                    cur = start = None

                # ---- form xobjects ----------------------------------------- #
                elif op == "Do" and xobjects is not None and depth < self.MAX_XOBJECT_DEPTH:
                    key = str(args[0]).lstrip("/")
                    try:
                        xo = xobjects[key]
                    except Exception:
                        continue
                    if str(getattr(xo, "Subtype", "")) != "/Form":
                        continue
                    sub_ctm = gs
                    if "/Matrix" in xo:
                        sub_ctm = mat_mul(tuple(float(v) for v in xo.Matrix), gs)
                    sub_res = xo.Resources if "/Resources" in xo else resources
                    self.run(xo, sub_res, sub_ctm, cur_layer, depth + 1)

            except Exception:
                # A malformed operator should not abandon the whole sheet.
                continue

    @staticmethod
    def _polyline_len(pts):
        total = 0.0
        for i in range(0, len(pts) - 2, 2):
            total += math.dist((pts[i], pts[i + 1]), (pts[i + 2], pts[i + 3]))
        return total


# --------------------------------------------------------------------------- #
# instances
# --------------------------------------------------------------------------- #
def cluster_instances(prims, sem, tol):
    """Group same-class primitives into instances by endpoint proximity.

    Layers say "these are doors"; they do not say which segments form *one* door.
    Union-find over a spatial grid gives connected components cheaply, which is a
    good proxy: the linework of a single door symbol is contiguous, and separate
    doors are separated by wall.
    """
    n = len(prims)
    inst = np.full(n, -1, dtype=np.int64)
    if n == 0 or tol <= 0:
        return inst

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Bucket endpoints by class so only same-class primitives can merge.
    buckets = {}
    for i, p in enumerate(prims):
        if sem[i] == BACKGROUND:
            continue
        for (x, y) in ((p["pts"][0], p["pts"][1]), (p["pts"][6], p["pts"][7])):
            key = (sem[i], int(x // tol), int(y // tol))
            buckets.setdefault(key, []).append(i)

    for key, members in buckets.items():
        cls, gx, gy = key
        # Also look at neighbouring cells so a join across a cell edge still merges.
        near = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                near.extend(buckets.get((cls, gx + dx, gy + dy), ()))
        for a in members:
            for b in near:
                union(a, b)

    remap, nxt = {}, 0
    for i in range(n):
        if sem[i] == BACKGROUND:
            continue
        r = find(i)
        if r not in remap:
            remap[r] = nxt
            nxt += 1
        inst[i] = remap[r]
    return inst


# --------------------------------------------------------------------------- #
# page -> json
# --------------------------------------------------------------------------- #
def parse_page(pdf, page, page_index, render_png=None, cluster_tol_frac=0.004):
    ex = PlanExtractor(pdf)
    try:
        res = page.Resources
    except Exception:
        res = None
    ex.run(page, res, (1, 0, 0, 1, 0, 0), None)

    prims = ex.prims
    if not prims:
        raise ValueError("no vector primitives found")

    xs = [c for p in prims for c in p["pts"][0::2]]
    ys = [c for p in prims for c in p["pts"][1::2]]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)

    sem = np.array([from_us_layer(p["layer"]) for p in prims], dtype=np.int64)
    tol = cluster_tol_frac * max(width, height)
    inst = cluster_instances(prims, sem, tol)

    # Layer id per distinct layer name — the model takes this as a separate input.
    layer_ids, seen = [], {}
    for p in prims:
        nm = (p["layer"] or "0").split("|")[-1]
        if nm not in seen:
            seen[nm] = len(seen)
        layer_ids.append(seen[nm])

    # Page box and content origin are kept so a tile can be cropped out of the
    # page render. Without them the image branch would look at the whole sheet
    # while the point branch sees one tile, and the two would disagree.
    # Renderers use CropBox when present, and apply /Rotate. Both have to be
    # carried through or the cropped image will not match the geometry.
    box = None
    for attr in ("CropBox", "MediaBox"):
        try:
            box = [float(v) for v in getattr(page, attr)]
            break
        except Exception:
            continue
    if not box:
        box = [0.0, 0.0, width, height]
    try:
        rotate = int(page.Rotate) % 360
    except Exception:
        rotate = 0

    data = {
        "origin": [minx, miny],
        "page_box": box,
        "rotate": rotate,
        "width": width,
        "height": height,
        # Shift to a local origin so downstream normalisation behaves.
        "args": [[(v - minx) if i % 2 == 0 else (v - miny)
                  for i, v in enumerate(p["pts"])] for p in prims],
        "lengths": [p["len"] for p in prims],
        "commands": [p["cmd"] for p in prims],
        "widths": [p["w"] for p in prims],
        "rgb": [list(p["rgb"]) for p in prims],
        "semanticIds": sem.tolist(),
        "instanceIds": inst.tolist(),
        "layerIds": layer_ids,
        "n_layers": len(seen),
    }
    if render_png:
        data["image"] = osp.basename(render_png)
    return data


TILE_KEYS = ("lengths", "commands", "widths", "rgb",
             "semanticIds", "instanceIds", "layerIds")


def _bbox(args, idxs):
    """Bounding box of the chosen primitives' control points."""
    xs = [v for i in idxs for v in args[i][0::2]]
    ys = [v for i in idxs for v in args[i][1::2]]
    return (min(xs), min(ys), max(xs), max(ys)) if xs else (0.0, 0.0, 0.0, 0.0)


def _emit_tile(data, idxs, ox, oy, tw, th):
    """Build one tile dict from a subset of a sheet's primitives.

    Coordinates come out **y-down**, matching the tile's PNG and the SVG
    convention the CubiCasa and FloorPlanCAD converters already use.

    PDF space has y growing upwards; a raster has it growing down. Emitting the
    raw PDF y left the point branch and the image branch vertically mirrored
    with respect to each other -- the model was being shown a drawing and a set
    of labels that disagreed about which end was the top. Measured over 10 tiles,
    labels landed on the drawing's ink 44% of the time under a y-flip against
    17% as stored, so the flip is not a matter of taste.
    """
    tile = {k: [data[k][i] for i in idxs] for k in TILE_KEYS}
    tile["args"] = [[(v - ox) if j % 2 == 0 else (th - (v - oy))
                     for j, v in enumerate(data["args"][i])] for i in idxs]
    tile["width"], tile["height"] = tw, th
    tile["n_layers"] = data.get("n_layers", 1)
    # Where this tile sits on the page, so its image can be cropped to match.
    tile["origin"] = [data["origin"][0] + ox, data["origin"][1] + oy]
    tile["page_box"] = data["page_box"]
    tile["rotate"] = data.get("rotate", 0)
    return tile


def _split_region(data, idxs, ox, oy, w, h, max_prims, min_prims, depth, max_depth):
    """Recursively halve a region until every part is under the cap.

    A single fixed grid only works if geometry is spread evenly, and on
    construction sheets it is not — schedules and dense plan areas concentrate
    primitives. Sizing one grid by the sheet total left tiles far over the cap
    (408 above 10k against a 6k cap, the worst at 126k), which the quadratic
    neighbourhood search cannot absorb. Halving the longer axis adapts to
    wherever the density actually is.
    """
    if len(idxs) <= max_prims or depth >= max_depth:
        if len(idxs) >= min_prims:
            yield (ox, oy), _emit_tile(data, idxs, ox, oy, w, h)
        return

    args = data["args"]
    if w >= h:                       # split the longer axis
        mid = ox + w / 2.0
        lo = [i for i in idxs if sum(args[i][0::2]) / 4.0 < mid]
        hi = [i for i in idxs if sum(args[i][0::2]) / 4.0 >= mid]
        parts = ((lo, ox, oy, w / 2.0, h), (hi, mid, oy, w / 2.0, h))
    else:
        mid = oy + h / 2.0
        lo = [i for i in idxs if sum(args[i][1::2]) / 4.0 < mid]
        hi = [i for i in idxs if sum(args[i][1::2]) / 4.0 >= mid]
        parts = ((lo, ox, oy, w, h / 2.0), (hi, ox, mid, w, h / 2.0))

    # A split that separates nothing would recurse forever; emit as-is instead.
    if not lo or not hi:
        if len(idxs) >= min_prims:
            yield (ox, oy), _emit_tile(data, idxs, ox, oy, w, h)
        return

    for sub, sx, sy, sw, sh in parts:
        yield from _split_region(data, sub, sx, sy, sw, sh,
                                 max_prims, min_prims, depth + 1, max_depth)


def tile_sheet(data, max_prims, min_prims=200, max_depth=8, overlap=0.0):
    """Split a sheet into tiles, each small enough to train on.

    `overlap` (0-0.5) grows every tile outward by that fraction of its size, so
    neighbouring tiles share a margin. Hard block boundaries cut objects in half
    -- a door on a tile edge appears as two partial doors and is learnable as
    neither. CADSpotting reports 16.7 PQ from overlapping windows over plain
    block partitioning. Tiles are still assigned by centroid for the split, then
    widened, so a primitive can appear in more than one tile.

    A construction sheet carries 45k-550k primitives; FloorPlanCAD drawings, which
    this architecture was designed around, carry 2k-7k. The neighbourhood search
    in the point branch is quadratic in primitive count, so a full sheet is both
    far slower and a different scale of scene than the model expects.

    Instances are clustered on the whole sheet before tiling, so ids stay
    consistent across tile boundaries. Yields (suffix, tile_dict).
    """
    n = len(data["args"])
    if n <= max_prims:
        # Route the whole sheet through _emit_tile too, rather than yielding the
        # raw dict: it is what applies the y-up -> y-down flip, and a sheet small
        # enough to skip tiling must not end up in a different convention from
        # every other tile.
        whole = _emit_tile(data, list(range(n)), 0.0, 0.0, data["width"], data["height"])
        if data.get("image"):
            whole["image"] = data["image"]
        yield "", whole
        return

    regions = list(_split_region(data, list(range(n)), 0.0, 0.0,
                                 data["width"], data["height"],
                                 max_prims, min_prims, 0, max_depth))
    if overlap <= 0:
        for k, (_, tile) in enumerate(regions):
            yield f"_t{k:03d}", tile
        return

    # Re-collect each region with a margin, from the full primitive list.
    args = data["args"]
    cx = [sum(a[0::2]) / 4.0 for a in args]
    cy = [sum(a[1::2]) / 4.0 for a in args]
    for k, ((ox, oy), tile) in enumerate(regions):
        tw, th = tile["width"], tile["height"]
        mx, my = tw * overlap, th * overlap
        x0, x1 = ox - mx, ox + tw + mx
        y0, y1 = oy - my, oy + th + my
        idxs = [i for i in range(n) if x0 <= cx[i] < x1 and y0 <= cy[i] < y1]
        if len(idxs) < min_prims:
            continue
        # Guard: a margin that swallows the sheet defeats the point of tiling.
        if len(idxs) > max_prims * 3:
            idxs = [i for i in range(n) if ox <= cx[i] < ox + tw and oy <= cy[i] < oy + th]
            x0, y0, tw2, th2 = ox, oy, tw, th
        else:
            tw2, th2 = x1 - x0, y1 - y0

        # Fit the tile to the geometry it actually contains, not to the nominal
        # window. Primitives are chosen by centroid, so a long wall whose midpoint
        # falls inside can reach well outside; the image is cropped to the tile
        # rectangle, so anything beyond it had no pixels underneath and the point
        # branch and image branch disagreed about the extent of the scene.
        bx0, by0, bx1, by1 = _bbox(args, idxs)
        x0, y0 = min(x0, bx0), min(y0, by0)
        tw2 = max(x1, bx1) - x0
        th2 = max(y1, by1) - y0
        yield f"_t{k:03d}", _emit_tile(data, idxs, x0, y0, tw2, th2)


def crop_tile_image(page_png, tile, out_png, size=980):
    """Cut this tile's area out of the rendered page.

    The point branch sees one tile; the image branch must see the same patch or
    the two disagree about where things are. PDF y grows upwards and image y
    grows downwards, so the vertical axis is flipped here.
    """
    from PIL import Image

    try:
        img = Image.open(page_png).convert("RGB")
    except Exception:
        return None

    x0, y0, x1, y1 = tile["page_box"]
    pw, ph = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    rot = int(tile.get("rotate", 0)) % 360

    # After /Rotate 90 or 270 the rendered page is transposed, so the scale
    # factors come from the swapped dimensions.
    if rot in (90, 270):
        s = min(img.width / ph, img.height / pw)
    else:
        s = min(img.width / pw, img.height / ph)

    def to_pixel(x, y):
        """PDF point (y up, origin bottom-left) -> image pixel (y down)."""
        u, v = x - x0, y - y0            # offset within the page box
        if rot == 90:
            return v * s, u * s
        if rot == 180:
            return (pw - u) * s, v * s
        if rot == 270:
            return (ph - v) * s, (pw - u) * s
        return u * s, (ph - v) * s

    ox, oy = tile["origin"]
    corners = [to_pixel(ox, oy),
               to_pixel(ox + tile["width"], oy),
               to_pixel(ox + tile["width"], oy + tile["height"]),
               to_pixel(ox, oy + tile["height"])]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]

    box = (max(0, int(min(xs))), max(0, int(min(ys))),
           min(img.width, int(max(xs))), min(img.height, int(max(ys))))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return None

    patch = img.crop(box)
    # The render is the page turned clockwise by /Rotate; turn the patch back
    # the other way so it lines up with the primitive coordinates.
    if rot:
        patch = patch.rotate(rot, expand=True)
    patch.resize((size, size)).save(out_png)
    return out_png


def render_page(pdf_path, page_no, out_png, size=980):
    """Rasterise one page with poppler's pdftoppm (GPL tool, invoked, not linked)."""
    stem = osp.splitext(out_png)[0]
    try:
        subprocess.run(["pdftoppm", "-png", "-r", "100", "-f", str(page_no),
                        "-l", str(page_no), "-scale-to", str(size),
                        pdf_path, stem],
                       check=True, capture_output=True, timeout=180)
    except Exception:
        return None
    for cand in (f"{stem}-{page_no}.png", f"{stem}-{page_no:02d}.png",
                 f"{stem}-{page_no:03d}.png", f"{stem}.png"):
        if osp.isfile(cand):
            if cand != out_png:
                os.replace(cand, out_png)
            return out_png
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf_dir", help="directory of plan-set PDFs")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip a planset whose tiles are already in --output_dir")
    ap.add_argument("--pdf", help="a single PDF instead of --pdf_dir")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--render", action="store_true", help="rasterise a PNG per sheet")
    ap.add_argument("--img_size", type=int, default=980)
    ap.add_argument("--min_prims", type=int, default=500,
                    help="skip sheets with fewer primitives (covers, index pages)")
    ap.add_argument("--require_labels", action="store_true",
                    help="skip sheets with no door/window primitives (for training data)")
    ap.add_argument("--max_pages", type=int, default=None)
    ap.add_argument("--max_page_prims", type=int, default=120000,
                    help="skip sheets denser than this. Tiling is superlinear in "
                         "primitive count, and a handful of pathological sheets "
                         "(one MicroStation set reaches 269k paths on a page) can "
                         "stall a run for hours while contributing little.")
    ap.add_argument("--overlap", type=float, default=0.0,
                    help="grow each tile by this fraction so neighbours share a "
                         "margin; avoids cutting objects at tile edges")
    ap.add_argument("--max_prims", type=int, default=6000,
                    help="tile sheets above this many primitives (FloorPlanCAD "
                         "drawings are 2k-7k; the point branch scales quadratically)")
    args = ap.parse_args()

    import pikepdf

    os.makedirs(args.output_dir, exist_ok=True)
    pdfs = ([args.pdf] if args.pdf else
            sorted(osp.join(args.pdf_dir, f) for f in os.listdir(args.pdf_dir)
                   if f.lower().endswith(".pdf")))

    kept = skipped = failed = 0
    for path in pdfs:
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", osp.splitext(osp.basename(path))[0])
        try:
            pdf = pikepdf.open(path)
        except Exception as exc:
            print(f"[fail] {osp.basename(path)}: {exc}")
            failed += 1
            continue

        if args.skip_existing and glob.glob(osp.join(args.output_dir, f"{stem}_p*_s2.json")):
            print(f"[have] {stem}")
            continue

        pages = list(pdf.pages)[:args.max_pages] if args.max_pages else list(pdf.pages)
        for i, page in enumerate(pages, start=1):
            name = f"{stem}_p{i:04d}"
            try:
                png = osp.join(args.output_dir, f"{name}_s2.png") if args.render else None
                data = parse_page(pdf, page, i, render_png=png)
            except Exception:
                skipped += 1
                continue

            if len(data["args"]) < args.min_prims:
                skipped += 1
                continue

            if args.max_page_prims and len(data["args"]) > args.max_page_prims:
                print(f"[skip] {name}: {len(data['args'])} primitives "
                      f"exceeds --max_page_prims {args.max_page_prims}")
                skipped += 1
                continue

            page_png = None
            if args.render:
                tmp_png = osp.join(args.output_dir, f".{name}_page.png")
                page_png = render_page(path, i, tmp_png, args.img_size * 3)

            for suffix, tile in tile_sheet(data, args.max_prims, overlap=args.overlap):
                if len(tile["args"]) < args.min_prims:
                    skipped += 1
                    continue
                sem = np.array(tile["semanticIds"])
                if args.require_labels and not ((sem == 0) | (sem == 1)).any():
                    skipped += 1
                    continue

                if page_png:
                    tile_png = osp.join(args.output_dir, f"{name}{suffix}_s2.png")
                    if crop_tile_image(page_png, tile, tile_png, args.img_size):
                        tile["image"] = osp.basename(tile_png)
                for k in ("origin", "page_box"):
                    tile.pop(k, None)   # bookkeeping only; not part of the schema

                with open(osp.join(args.output_dir, f"{name}{suffix}_s2.json"), "w") as f:
                    json.dump(tile, f)
                kept += 1
                if kept % 50 == 0:
                    print(f"  ... {kept} tiles written", flush=True)

            if page_png and osp.isfile(page_png):
                os.remove(page_png)   # the per-tile crops are what we keep

        pdf.close()

    print(f"[done] {kept} sheets -> {args.output_dir} (skipped {skipped}, failed {failed})")


if __name__ == "__main__":
    main()
