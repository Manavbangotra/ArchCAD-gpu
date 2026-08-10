#!/usr/bin/env python3
"""
Convert FloorPlanCAD annotated SVGs into the JSON form svgnet/data/svg.py reads.

The README references this script but it was never published, so this is a
reimplementation against the loader's documented schema.

Each SVG geometry element (path / line / polyline / circle / ellipse) becomes
one "primitive", represented by four control points. Straight segments get
their interior points by interpolation, so every primitive has the same shape
and `angles_with_horizontal` can take the chord from point 0 to point 3.

Output, one file per drawing, named "<stem>_s2.json":

    width, height   drawing extent in SVG user units
    args            (N, 8) four (x, y) control points per primitive
    lengths         (N,)   arc length
    commands        (N,)   primitive type index, see COMMAND_TYPES
    widths          (N,)   stroke width
    rgb             (N, 3) stroke colour 0-255
    semanticIds     (N,)   0-based class index, background = 35
    instanceIds     (N,)   instance index, -1 for stuff/background
    layerIds        (N,)   CAD layer index
    image           companion PNG filename, if one was found

Usage:
    python dataset/parse_FpCAD_svg.py --data_dir <svg_dir> --output_dir <json_dir>
"""

import argparse
import json
import os
import os.path as osp
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np

SVG_NS = "{http://www.w3.org/2000/svg}"
INKSCAPE_NS = "{http://www.inkscape.org/namespaces/inkscape}"

# One-hot slot for each primitive type (svg.py allocates exactly 4).
COMMAND_TYPES = {"line": 0, "arc": 1, "circle": 2, "ellipse": 3}

# Background class index; must match svgnet.data.svg.BG_SEMANTIC_ID.
BG_SEMANTIC_ID = 35

# FloorPlanCAD's SVG files number their classes differently from the label
# space the model trains on. This is the official remap from the dataset's
# reference baseline (CADTransformer, config/anno_config.py :: RemapDict),
# taking a raw `semantic-id` to the 1-based class id used by SVG_CATEGORIES.
#
# Spot-checked against CAD layer names in the released test set:
#   raw 1  -> 33 wall        (layers "WALL", "A-WALL-BLOK")
#   raw 33 -> 35 railing     (layers "栏杆", "扶手")
#   raw 9  -> 7  window      (layer  "A-WINDOW")
#   raw 3  -> 1  single door (layer  "A-DOOR")
#   raw 30 -> 28 stairs      (layer  "STAIR")
#   raw 22 -> 18 gas stove   (layer  "洁具厨具")
RAW_TO_CLASS_ID = {
    0: 0,
    1: 33, 2: 34, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
    11: 9, 12: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
    20: 20, 21: 21, 22: 18, 23: 19, 24: 22, 25: 23, 26: 24, 27: 25,
    28: 26, 29: 27, 30: 28, 31: 29, 32: 30, 33: 35, 34: 31, 35: 32,
}


def remap_semantic(raw_id, taxonomy="full"):
    """Raw SVG semantic-id -> 0-based training class index.

    taxonomy="full" gives FloorPlanCAD's 35 classes (background 35).
    taxonomy="us4" collapses to door / window / wall / background, matching
    dataset/taxonomy.py, so FloorPlanCAD, CubiCasa and US drawings can be mixed
    in one training set.

    Anything unlabelled, out of range, or explicitly 0 becomes background.
    """
    class_id = RAW_TO_CLASS_ID.get(raw_id, 0)
    if taxonomy == "us4":
        from taxonomy import from_floorplancad
        return from_floorplancad(class_id)
    if class_id <= 0:
        return BG_SEMANTIC_ID
    return class_id - 1  # 1-based id -> 0-based index


# --------------------------------------------------------------------------- #
# attribute helpers
# --------------------------------------------------------------------------- #
def _strip_ns(tag):
    return tag.split("}")[-1] if "}" in tag else tag


def _float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(str(value).strip().replace("px", ""))
    except ValueError:
        return default


def _parse_color(value, default=(0, 0, 0)):
    """Parse a stroke colour: #rgb, #rrggbb, or rgb(r,g,b)."""
    if not value or value in ("none", "transparent"):
        return default
    value = value.strip()

    if value.startswith("#"):
        h = value[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) == 6:
            try:
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                return default
        return default

    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", value)
    if m:
        return tuple(int(g) for g in m.groups())
    return default


def _get_attr(elem, names, inherited=None, default=None):
    """First present attribute among `names`, falling back to inherited style."""
    for name in names:
        if name in elem.attrib:
            return elem.attrib[name]
    if inherited:
        for name in names:
            if name in inherited:
                return inherited[name]
    return default


def _viewbox_size(root):
    """Drawing extent, preferring viewBox over width/height attributes."""
    vb = root.get("viewBox")
    if vb:
        parts = [float(p) for p in re.split(r"[,\s]+", vb.strip()) if p]
        if len(parts) == 4:
            return parts[2], parts[3]
    return _float(root.get("width"), 1.0), _float(root.get("height"), 1.0)


# --------------------------------------------------------------------------- #
# geometry -> 4 control points
# --------------------------------------------------------------------------- #
def _four_points_from_segment(seg):
    """Sample an svgpathtools segment at t = 0, 1/3, 2/3, 1."""
    pts = [seg.point(t) for t in (0.0, 1 / 3, 2 / 3, 1.0)]
    return [coord for p in pts for coord in (p.real, p.imag)]


def _segment_kind(seg):
    name = type(seg).__name__
    if name == "Line":
        return "line"
    if name == "Arc":
        return "arc"
    return "arc"  # bezier segments are curved; group them with arcs


def _primitives_from_path(d_attr):
    """Split a path's `d` into segments, each as (points, kind, length)."""
    from svgpathtools import parse_path

    try:
        path = parse_path(d_attr)
    except Exception:
        return []

    out = []
    for seg in path:
        try:
            pts = _four_points_from_segment(seg)
            length = float(seg.length())
        except Exception:
            continue
        if not np.all(np.isfinite(pts)):
            continue
        out.append((pts, _segment_kind(seg), length))
    return out


def _primitives_from_shape(elem, tag):
    """Handle the primitive shape elements that are not <path>."""
    if tag == "line":
        x1, y1 = _float(elem.get("x1")), _float(elem.get("y1"))
        x2, y2 = _float(elem.get("x2")), _float(elem.get("y2"))
        pts = [x1, y1,
               x1 + (x2 - x1) / 3, y1 + (y2 - y1) / 3,
               x1 + 2 * (x2 - x1) / 3, y1 + 2 * (y2 - y1) / 3,
               x2, y2]
        return [(pts, "line", float(np.hypot(x2 - x1, y2 - y1)))]

    if tag in ("circle", "ellipse"):
        cx, cy = _float(elem.get("cx")), _float(elem.get("cy"))
        if tag == "circle":
            rx = ry = _float(elem.get("r"))
        else:
            rx, ry = _float(elem.get("rx")), _float(elem.get("ry"))
        # Four points around the ellipse; chord 0->3 spans the horizontal axis.
        pts = [cx - rx, cy,
               cx, cy - ry,
               cx + rx, cy,
               cx, cy + ry]
        perimeter = float(np.pi * (3 * (rx + ry) - np.sqrt((3 * rx + ry) * (rx + 3 * ry))))
        return [(pts, tag, perimeter)]

    if tag in ("polyline", "polygon"):
        raw = elem.get("points", "")
        nums = [float(v) for v in re.split(r"[,\s]+", raw.strip()) if v]
        coords = list(zip(nums[0::2], nums[1::2]))
        if tag == "polygon" and len(coords) > 2:
            coords.append(coords[0])
        out = []
        for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
            pts = [x1, y1,
                   x1 + (x2 - x1) / 3, y1 + (y2 - y1) / 3,
                   x1 + 2 * (x2 - x1) / 3, y1 + 2 * (y2 - y1) / 3,
                   x2, y2]
            out.append((pts, "line", float(np.hypot(x2 - x1, y2 - y1))))
        return out

    return []


# --------------------------------------------------------------------------- #
# main conversion
# --------------------------------------------------------------------------- #
def parse_svg(svg_path, taxonomy="full"):
    """Read one annotated SVG into the loader's dict schema."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    width, height = _viewbox_size(root)

    args, lengths, commands, widths, rgbs = [], [], [], [], []
    semantic_ids, instance_ids, layer_ids = [], [], []
    layer_lookup = OrderedDict()

    def walk(elem, inherited):
        # Group elements carry stroke styling and the CAD layer name.
        style = dict(inherited)
        for key in ("stroke", "stroke-width", "semantic-id", "instance-id",
                    "layer", "data-layer"):
            if key in elem.attrib:
                style[key] = elem.attrib[key]

        tag = _strip_ns(elem.tag)

        # FloorPlanCAD writes the CAD layer on <g> as an Inkscape layer:
        #   <g id="layerWALL" inkscape:groupmode="layer" inkscape:label="WALL">
        if tag == "g":
            label = elem.get(INKSCAPE_NS + "label")
            if label is None:
                gid = elem.get("id")
                if gid:
                    label = gid[5:] if gid.startswith("layer") else gid
            if label:
                style["layer"] = label

        prims = []
        if tag == "path":
            prims = _primitives_from_path(elem.get("d", ""))
        elif tag in ("line", "circle", "ellipse", "polyline", "polygon"):
            prims = _primitives_from_shape(elem, tag)

        if prims:
            # Primitives with no semantic-id at all are unannotated background;
            # that is the common case in these files.
            raw_sem = _get_attr(elem, ["semantic-id", "semanticId"], style, None)
            bg = BG_SEMANTIC_ID if taxonomy == "full" else 3
            sem = bg if raw_sem is None else remap_semantic(int(_float(raw_sem, 0)), taxonomy)

            ins = int(_float(_get_attr(elem, ["instance-id", "instanceId"], style, -1), -1))
            if sem == bg:
                ins = -1

            layer_name = _get_attr(elem, ["layer", "data-layer"], style, "0")
            if layer_name not in layer_lookup:
                layer_lookup[layer_name] = len(layer_lookup)
            layer_id = layer_lookup[layer_name]

            stroke_w = _float(_get_attr(elem, ["stroke-width"], style, 1.0), 1.0)
            color = _parse_color(_get_attr(elem, ["stroke"], style, "#000000"))

            for pts, kind, length in prims:
                args.append(pts)
                lengths.append(length)
                commands.append(COMMAND_TYPES.get(kind, 0))
                widths.append(stroke_w if stroke_w > 0 else 1.0)
                rgbs.append(list(color))
                semantic_ids.append(sem)
                instance_ids.append(ins)
                layer_ids.append(layer_id)

        for child in elem:
            walk(child, style)

    walk(root, {})

    if not args:
        raise ValueError(f"no drawable primitives found in {svg_path}")

    return {
        "width": width if width > 0 else 1.0,
        "height": height if height > 0 else 1.0,
        "args": args,
        "lengths": lengths,
        "commands": commands,
        "widths": widths,
        "rgb": rgbs,
        "semanticIds": semantic_ids,
        "instanceIds": instance_ids,
        "layerIds": layer_ids,
    }


def find_companion_png(svg_path, png_dirs):
    """Locate the rendered PNG that goes with this drawing, if present."""
    stem = osp.splitext(osp.basename(svg_path))[0]
    for d in png_dirs:
        if not d:
            continue
        candidate = osp.join(d, stem + ".png")
        if osp.isfile(candidate):
            return candidate
    sibling = osp.splitext(svg_path)[0] + ".png"
    return sibling if osp.isfile(sibling) else None


def render_png(data, out_path, size=980):
    """Rasterise the primitives so the image pathway has something to look at.

    A faithful stand-in for the dataset's own renders: strokes are drawn in
    their annotated colour on white, scaled to fit `size`.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    sx = size / data["width"]
    sy = size / data["height"]

    for pts, color, w in zip(data["args"], data["rgb"], data["widths"]):
        xy = [(pts[i] * sx, pts[i + 1] * sy) for i in range(0, 8, 2)]
        draw.line(xy, fill=tuple(int(c) for c in color),
                  width=max(1, int(round(w * min(sx, sy)))))

    img.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="directory of annotated SVGs")
    ap.add_argument("--output_dir", required=True, help="where to write *_s2.json")
    ap.add_argument("--png_dir", default=None,
                    help="directory of pre-rendered PNGs (optional)")
    ap.add_argument("--render", action="store_true",
                    help="rasterise a PNG when no pre-rendered one is found")
    ap.add_argument("--img_size", type=int, default=980)
    ap.add_argument("--taxonomy", choices=["full", "us4"], default="full",
                    help="full = FloorPlanCAD's 35 classes; us4 = door/window/wall")
    ap.add_argument("--limit", type=int, default=None,
                    help="only convert the first N drawings (useful for smoke tests)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    svgs = sorted(f for f in os.listdir(args.data_dir) if f.lower().endswith(".svg"))
    if args.limit:
        svgs = svgs[:args.limit]
    print(f"[parse] {len(svgs)} svg files from {args.data_dir}")

    ok, failed = 0, []
    for name in svgs:
        svg_path = osp.join(args.data_dir, name)
        stem = osp.splitext(name)[0]
        try:
            data = parse_svg(svg_path, args.taxonomy)
        except Exception as exc:
            failed.append((name, str(exc)))
            continue

        png = find_companion_png(svg_path, [args.png_dir])
        if png:
            # Copy alongside the json so the loader finds it by stem.
            import shutil
            dst = osp.join(args.output_dir, f"{stem}_s2.png")
            if osp.abspath(png) != osp.abspath(dst):
                shutil.copyfile(png, dst)
            data["image"] = osp.basename(dst)
        elif args.render:
            dst = osp.join(args.output_dir, f"{stem}_s2.png")
            render_png(data, dst, size=args.img_size)
            data["image"] = osp.basename(dst)

        with open(osp.join(args.output_dir, f"{stem}_s2.json"), "w") as f:
            json.dump(data, f)
        ok += 1

        if ok % 200 == 0:
            print(f"  ... {ok}/{len(svgs)}")

    print(f"[done] converted {ok}, failed {len(failed)} -> {args.output_dir}")
    for name, err in failed[:10]:
        print(f"  FAIL {name}: {err}")


if __name__ == "__main__":
    main()
