#!/usr/bin/env python3
"""CubiCasa5K -> VecFormer line JSONs (vecformer/data/floorplancad format).

    python dataset/to_lines_cubicasa.py --data_dir dataset/cubicasa5k/cubicasa5k \
        --output_dir dataset/cubicasa5k/lines --labels cubicasa12 --workers 16

Writes <output_dir>/{train,val,test}/<sample>.json from the dataset's split files.

Why a new converter rather than dataset/parse_cubicasa.py: that one serves the
SymPointV2-style loader and flattens geometry the way it wanted -- SVG elliptical
arcs become their chord, circles become 16-gons, and runs are refitted into
"line"/"arc" records whose four stored points are a straight segment. For a
line-based model that erases door swings. Here every SVG element is one
primitive and every curve is sampled the way VecFormer samples FloorPlanCAD
(vecformer/data/floorplancad/preprocess.py): 2 points for a straight segment, 9
for a curve, more until no interval exceeds `dynamic_sampling_ratio` of the
drawing's shorter side; consecutive samples become line segments.

Labels follow the object hierarchy exactly as parse_cubicasa.py does (a root
object -- Door, Window, Wall, Railing, Stairs, FixedFurniture -- sets the class and
a new instance; its parts inherit both). Two label spaces:

  cubicasa12  TextCAD's CubiCasa protocol: 10 things (window, door, closet,
              electrical appliance, toilet, sink, sauna bench, fireplace,
              bathtub, chimney) and 2 stuff (wall, railing); background 12.
  arch43      this repository's Arch-43 ids via dataset/taxonomy.py (coarse ids
              51-54 where CubiCasa names only the family); background 43.

Stuff classes get instance -1 (VecFormer's convention). CubiCasa has no CAD
layers, so layer_ids = primitive_ids: the "no prior" setting. Text elements
(room names, dimensions) are kept in `texts` for the text-aware stage.
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
from parse_cubicasa import _EDITOR, _float, _pairs, _strip_ns, sample_dirs  # noqa: E402
from svg_geom import IDENTITY, apply, mat_mul, parse_transform  # noqa: E402
import taxonomy as tx  # noqa: E402

# ----------------------------------------------------------------- labels --
CUBI12_NAMES = ["window", "door", "closet", "electrical appliance", "toilet", "sink",
                "sauna bench", "fireplace", "bathtub", "chimney", "wall", "railing"]
CUBI12_STUFF = {10, 11}
CUBI12_BG = 12

# Checked in order on the class attribute with non-letters removed, lower case.
_CUBI12_RULES = [
    ("window", 0), ("door", 1), ("closet", 2), ("electricalappli", 3), ("toilet", 4),
    ("sink", 5), ("sauna", 6), ("fireplace", 7), ("bathtub", 8), ("chimney", 9),
    ("wall", 10), ("railing", 11),
]
ROOT_TOKENS = tx.CUBI_ROOT_TOKENS


def cubicasa12(class_attr):
    flat = re.sub(r"[^a-z]", "", (class_attr or "").lower())
    if not flat or flat.startswith("space"):
        return CUBI12_BG
    for token, cid in _CUBI12_RULES:
        if token in flat:
            return cid
    return CUBI12_BG


LABEL_SPACES = {
    # name: (mapper, background id, stuff ids)
    "cubicasa12": (cubicasa12, CUBI12_BG, CUBI12_STUFF),
    "arch43": (tx.from_cubicasa_arch, tx.ARCH_BG, set(tx.STUFF_CLASSES)),
}


# --------------------------------------------------------------- geometry --
def _segments_of_path(d):
    """svgpathtools segments of a `d` string (lines, arcs, beziers), local coords."""
    from svgpathtools import parse_path
    try:
        return list(parse_path(d))
    except Exception:
        return []


def _is_straight(seg):
    return seg.__class__.__name__ == "Line"


def _sample(point_at, straight, ctm, max_len):
    """Points along one segment in root coordinates, densified to max_len."""
    k = 2 if straight else 9
    while True:
        ts = [i / (k - 1) for i in range(k)]
        pts = [apply(ctm, *point_at(t)) for t in ts]
        gap = max(math.dist(a, b) for a, b in zip(pts, pts[1:]))
        if max_len <= 0 or gap <= max_len or k >= 4096:
            return pts
        k = max(k + 1, int(math.ceil((k - 1) * gap / max_len)) + 1)


def element_polylines(el, ctm, max_len):
    """One SVG element's geometry as a list of point runs in root coordinates."""
    tag = _strip_ns(el.tag)
    runs = []
    if tag == "path":
        cur = []
        for seg in _segments_of_path(el.get("d")):
            pts = _sample(lambda t, s=seg: (s.point(t).real, s.point(t).imag), _is_straight(seg), ctm, max_len)
            if cur and math.dist(cur[-1], pts[0]) < 1e-9:
                cur.extend(pts[1:])
            else:
                if len(cur) > 1:
                    runs.append(cur)
                cur = pts
        if len(cur) > 1:
            runs.append(cur)
    elif tag in ("polygon", "polyline", "line", "rect"):
        if tag == "line":
            corners = [(_float(el.get("x1")), _float(el.get("y1"))), (_float(el.get("x2")), _float(el.get("y2")))]
        elif tag == "rect":
            x, y, w, h = (_float(el.get(a)) for a in ("x", "y", "width", "height"))
            if w <= 0 or h <= 0:
                return []
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
        else:
            corners = _pairs(el.get("points"))
            if tag == "polygon" and len(corners) > 2:
                corners = corners + [corners[0]]
        run = []
        for a, b in zip(corners, corners[1:]):
            pts = _sample(lambda t, a=a, b=b: (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), True, ctm, max_len)
            run.extend(pts if not run else pts[1:])
        if len(run) > 1:
            runs.append(run)
    elif tag in ("circle", "ellipse"):
        cx, cy = _float(el.get("cx")), _float(el.get("cy"))
        rx = ry = _float(el.get("r")) if tag == "circle" else None
        if tag == "ellipse":
            rx, ry = _float(el.get("rx")), _float(el.get("ry"))
        if rx > 0 and ry > 0:
            pts = _sample(lambda t: (cx + rx * math.cos(2 * math.pi * t), cy + ry * math.sin(2 * math.pi * t)),
                          False, ctm, max_len)
            runs.append(pts)
    return runs


def _svg_box(root):
    vb = root.get("viewBox")
    if vb:
        parts = [float(p) for p in re.split(r"[,\s]+", vb.strip()) if p]
        if len(parts) == 4 and parts[2] > 1 and parts[3] > 1:
            return parts
    w, h = _float(root.get("width"), 0.0), _float(root.get("height"), 0.0)
    return [0.0, 0.0, w, h] if w > 1 and h > 1 else None


# ---------------------------------------------------------------- convert --
def convert(svg_path, label_space="cubicasa12", dynamic_sampling_ratio=0.01):
    mapper, bg, stuff = LABEL_SPACES[label_space]
    root = ET.parse(svg_path).getroot()
    box = _svg_box(root)
    max_len = min(box[2], box[3]) * dynamic_sampling_ratio if box else 0.0

    out = dict(viewBox=None, coords=[], colors=[], widths=[], primitive_ids=[], layer_ids=[],
               semantic_ids=[], instance_ids=[], primitive_lengths=[], texts=[])
    seen = set()
    counters = {"instance": 0, "primitive": 0}
    xs, ys = [], []

    def walk(el, ctm, cls, instance):
        tag = _strip_ns(el.tag)
        if tag in ("defs", "desc", "marker", "symbol", "use"):
            return
        cls_attr = el.get("class") or ""
        if _EDITOR.search(cls_attr):
            return
        if "display:none" in (el.get("style") or "").replace(" ", ""):
            return
        ctm = mat_mul(ctm, parse_transform(el.get("transform")))
        if tag == "text":
            content = " ".join("".join(el.itertext()).split())
            if content:
                x, y = apply(ctm, _float(el.get("x")), _float(el.get("y")))
                out["texts"].append(dict(text=content[:64], x=x, y=y, size=_float(el.get("font-size")),
                                         angle=0.0, layer_id=-1))
            return
        head = cls_attr.split()[0].strip() if cls_attr else ""
        if head in ROOT_TOKENS:
            cls = mapper(cls_attr)
            counters["instance"] += 1
            instance = counters["instance"]
        runs = element_polylines(el, ctm, max_len)
        if runs:
            key = tuple(round(v, 2) for run in runs for p in run for v in p)
            if key not in seen:
                seen.add(key)
                pid = counters["primitive"]
                length = 0.0
                for run in runs:
                    for a, b in zip(run, run[1:]):
                        if math.dist(a, b) < 1e-9:
                            continue
                        out["coords"].append([a[0], a[1], b[0], b[1]])
                        out["colors"].append([0, 0, 0])
                        out["widths"].append(1.0)
                        out["primitive_ids"].append(pid)
                        out["layer_ids"].append(pid)          # no CAD layers: "no prior"
                        length += math.dist(a, b)
                        xs.extend((a[0], b[0]))
                        ys.extend((a[1], b[1]))
                if length > 0:
                    out["semantic_ids"].append(cls)
                    out["instance_ids"].append(instance if cls != bg and cls not in stuff else -1)
                    out["primitive_lengths"].append(length)
                    counters["primitive"] += 1
                else:
                    # every segment was degenerate: drop the rows appended for it
                    while out["primitive_ids"] and out["primitive_ids"][-1] == pid:
                        for k in ("coords", "colors", "widths", "primitive_ids", "layer_ids"):
                            out[k].pop()
        for child in el:
            walk(child, ctm, cls, instance)

    walk(root, IDENTITY, bg, -1)
    if not out["coords"]:
        raise ValueError(f"no geometry in {svg_path}")
    if box is None:
        box = [min(xs), min(ys), max(max(xs) - min(xs), 1e-6), max(max(ys) - min(ys), 1e-6)]
    out["viewBox"] = box
    return out


def _job(job):
    src, dst, label_space, ratio = job
    try:
        data = convert(src, label_space, ratio)
    except Exception as exc:          # a bad SVG must not stop the corpus
        return src, str(exc)
    os.makedirs(osp.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(data, f)
    return src, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="extracted cubicasa5k root (holds train.txt)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--labels", choices=sorted(LABEL_SPACES), default="cubicasa12")
    ap.add_argument("--dynamic_sampling_ratio", type=float, default=0.01)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    jobs = []
    for split in a.splits:
        dirs = sample_dirs(a.data_dir, f"{split}.txt")
        if a.limit:
            dirs = dirs[:a.limit]
        for d in dirs:
            svg = osp.join(d, "model.svg")
            if osp.isfile(svg):
                name = re.sub(r"[^A-Za-z0-9_.-]", "_", osp.relpath(d, a.data_dir).strip("/\\"))
                jobs.append((svg, osp.join(a.output_dir, split, name + ".json"), a.labels, a.dynamic_sampling_ratio))
    print(f"{len(jobs)} samples -> {a.output_dir} ({a.labels})", flush=True)

    failed = []
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for i, (src, err) in enumerate(pool.imap_unordered(_job, jobs, chunksize=8), 1):
                if err:
                    failed.append((src, err))
                if i % 500 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)
    else:
        for src, err in map(_job, jobs):
            if err:
                failed.append((src, err))
    print(f"done: {len(jobs) - len(failed)} converted, {len(failed)} failed")
    for src, err in failed[:10]:
        print(f"  FAIL {src}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
