#!/usr/bin/env python3
"""
Convert CubiCasa5K into the JSON form svgnet/data/svg.py reads, for pretraining.

CubiCasa5K ships each floor plan as a raster image plus an SVG whose annotations
are *vector polygons* carrying a class attribute:

    <g class="Wall"><polygon points="x,y x,y ..."/></g>
    <g class="Door"><polygon points="..."/></g>
    <g class="FixedFurniture Sink"><polygon points="..."/></g>

That matters: because the annotations are vector, both halves of the model can be
pretrained — polygon edges feed the point branch, the raster image feeds the image
branch — rather than only the image half. Instance ids come free, one per polygon,
which is the part that has to be inferred by clustering on CAD data.

Known limitation, accepted deliberately: these polygons are object *outlines*, not
real CAD linework, so there is none of the dimension-line/hatching/text clutter a
construction sheet carries. Pretraining on this is cleaner than the real task;
stage 2 (dataset/parse_pdf_plans.py) is what adapts the model to that.

LICENCE: CubiCasa5K is CC BY-NC 4.0. Anything trained on it is research-only and
must not ship in a commercial product. Keep its checkpoints separate.

Usage:
    python dataset/parse_cubicasa.py --data_dir dataset/cubicasa/cubicasa5k \
        --output_dir dataset/cubicasa/json/train --split_file train.txt
"""

import argparse
import json
import math
import os
import os.path as osp
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from taxonomy import BACKGROUND, from_cubicasa  # noqa: E402

SVG_NS = "{http://www.w3.org/2000/svg}"
CMD_LINE = 0


def _strip_ns(tag):
    return tag.split("}")[-1] if "}" in tag else tag


def _float(v, default=0.0):
    try:
        return float(str(v).strip().replace("px", ""))
    except (TypeError, ValueError):
        return default


def _svg_size(root):
    vb = root.get("viewBox")
    if vb:
        parts = [float(p) for p in re.split(r"[,\s]+", vb.strip()) if p]
        if len(parts) == 4:
            return parts[2], parts[3]
    return _float(root.get("width"), 1.0), _float(root.get("height"), 1.0)


def _polygon_points(g):
    """Points of the first <polygon> child, as [(x, y), ...]."""
    for child in g:
        if _strip_ns(child.tag) == "polygon":
            raw = child.get("points", "")
            pts = []
            for token in raw.replace(",", " ").split():
                pts.append(float(token))
            return list(zip(pts[0::2], pts[1::2]))
    return []


def _line_points(p0, p1):
    (x1, y1), (x2, y2) = p0, p1
    return [x1, y1,
            x1 + (x2 - x1) / 3, y1 + (y2 - y1) / 3,
            x1 + 2 * (x2 - x1) / 3, y1 + 2 * (y2 - y1) / 3,
            x2, y2]


def parse_svg(svg_path, with_text=False):
    """One CubiCasa model.svg -> the loader's dict schema."""
    root = ET.parse(svg_path).getroot()
    width, height = _svg_size(root)

    args, lengths, commands, widths, rgbs = [], [], [], [], []
    sem_ids, ins_ids, layer_ids = [], [], []
    instance = 0
    xs, ys = [], []

    for g in root.iter(SVG_NS + "g"):
        cls_attr = g.get("class")
        if not cls_attr:
            continue
        cls = from_cubicasa(cls_attr)
        if cls == BACKGROUND:
            # Rooms and furniture become background under this taxonomy; keeping
            # their outlines gives the model realistic non-target geometry to
            # reject rather than an unnaturally empty scene.
            pass

        pts = _polygon_points(g)
        if len(pts) < 2:
            continue

        # Close the polygon: an outline is a loop.
        ring = pts + [pts[0]] if pts[0] != pts[-1] else pts
        emitted = False
        for a, b in zip(ring, ring[1:]):
            if a == b:
                continue
            args.append(_line_points(a, b))
            lengths.append(math.dist(a, b))
            commands.append(CMD_LINE)
            widths.append(1.0)
            rgbs.append([0, 0, 0])
            sem_ids.append(cls)
            # Instance per polygon; background stays -1 ("stuff").
            ins_ids.append(instance if cls != BACKGROUND else -1)
            # NOT the class. CubiCasa has no CAD layers, and setting layerId to
            # the class hands the answer straight to the model as an input --
            # layerId == semanticId on every sample. Group by polygon instead,
            # which is structural information a real drawing could also provide.
            layer_ids.append(instance)
            xs.extend((a[0], b[0])); ys.extend((a[1], b[1]))
            emitted = True
        if emitted and cls != BACKGROUND:
            instance += 1

    if not args:
        raise ValueError(f"no polygons found in {svg_path}")

    # Some CubiCasa SVGs lack a usable viewBox; fall back to the drawn extent.
    if width <= 1 or height <= 1:
        width = max(max(xs) - min(xs), 1e-6)
        height = max(max(ys) - min(ys), 1e-6)

    result = {
        "width": width,
        "height": height,
        "args": args,
        "lengths": lengths,
        "commands": commands,
        "widths": widths,
        "rgb": rgbs,
        "semanticIds": sem_ids,
        "instanceIds": ins_ids,
        "layerIds": layer_ids,
        "n_layers": 4,
    }
    if with_text:
        result["texts"] = extract_texts(root)
    return result


def extract_texts(root):
    """Collect the words printed on the drawing, with their positions.

    Stored alongside the geometry but NOT fed to the model. TextCAD
    (arXiv:2607.12678) shows text helps, and this is the data it needs — but
    measured on this corpus the words are room names, dimensions and fixture
    codes (CL, CB, SINK, WC). Across 5,778 sampled items, "door" and "window"
    appear zero times, so for a door/window/wall taxonomy there is nothing here
    to gain. Kept because it is nearly free, and because drawings that *do*
    carry door tags (US construction sets with D1/W1 marks and door schedules)
    would benefit.
    """
    out = []
    for t in root.iter(SVG_NS + "text"):
        content = "".join(t.itertext()).strip()
        if not content:
            continue
        out.append([_float(t.get("x")), _float(t.get("y")), content[:64]])
    return out


def find_image(sample_dir):
    for candidate in ("F1_scaled.png", "F1_original.png", "F1_scaled.jpg"):
        p = osp.join(sample_dir, candidate)
        if osp.isfile(p):
            return p
    return None


def sample_dirs(data_dir, split_file=None):
    """Sample folders, from the dataset's split list when one is given."""
    if split_file:
        path = split_file if osp.isfile(split_file) else osp.join(data_dir, split_file)
        if osp.isfile(path):
            out = []
            with open(path) as f:
                for line in f:
                    rel = line.strip().strip("/")
                    if rel:
                        out.append(osp.join(data_dir, rel))
            return out
        print(f"[warn] split file {split_file} not found; walking {data_dir}")

    out = []
    for dirpath, _, files in os.walk(data_dir):
        if "model.svg" in files:
            out.append(dirpath)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="extracted cubicasa5k root")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split_file", default=None,
                    help="train.txt / val.txt / test.txt from the dataset")
    ap.add_argument("--extract_text", action="store_true",
                    help="also store the drawing's text elements (not used by "
                         "the model; see extract_texts)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--require_labels", action="store_true",
                    help="skip plans with no door or window polygons")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dirs = sample_dirs(args.data_dir, args.split_file)
    if args.limit:
        dirs = dirs[:args.limit]
    print(f"[parse] {len(dirs)} CubiCasa samples")

    import shutil
    kept, skipped, failed = 0, 0, []
    for d in dirs:
        svg = osp.join(d, "model.svg")
        if not osp.isfile(svg):
            skipped += 1
            continue
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", osp.relpath(d, args.data_dir).strip("/"))
        try:
            data = parse_svg(svg, args.extract_text)
        except Exception as exc:
            failed.append((stem, str(exc)))
            continue

        if args.require_labels and not any(s in (0, 1) for s in data["semanticIds"]):
            skipped += 1
            continue

        img = find_image(d)
        if img:
            dst = osp.join(args.output_dir, f"{stem}_s2.png")
            shutil.copyfile(img, dst)
            data["image"] = osp.basename(dst)

        with open(osp.join(args.output_dir, f"{stem}_s2.json"), "w") as f:
            json.dump(data, f)
        kept += 1
        if kept % 250 == 0:
            print(f"  ... {kept}/{len(dirs)}")

    print(f"[done] converted {kept}, skipped {skipped}, failed {len(failed)} -> {args.output_dir}")
    for stem, err in failed[:5]:
        print(f"  FAIL {stem}: {err}")


if __name__ == "__main__":
    main()
