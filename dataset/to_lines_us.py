#!/usr/bin/env python3
"""US plan-set PDFs -> VecFormer line JSONs, in fixed real-world windows.

    python dataset/to_lines_us.py --pdf_dir "C:/Users/me/Downloads/plansets" \
        --output_dir dataset/us_plans/lines --split_from dataset/us_plans/json5 --workers 8

Writes <output_dir>/{train,test}/<doc>_p<page>_w<k>.json in the format of
vecformer/data/floorplancad, plus fields the product model uses:
  layer_names   the CAD layer name behind each layer id (layer-name embedding)
  texts         PDF text inside the window: {text, x, y, size, angle, layer_id}
  meta          {source, doc, page, window_m, mm_per_pt, scale_source, viewport_kind,
                 viewport_title}

Why windows in metres, not the primitive-count tiles of dataset/parse_pdf_plans.py:
tiles sized by primitive count make a door 5-10x larger in one tile than in its
neighbour, and the model normalises every tile to [-0.5, 0.5]. A window of fixed
real size (default 10 m, stepped by half -- CADSpotting's sliding windows) keeps
symbols at a constant scale; the drawing scale of each viewport comes from its
printed scale string or dimension chains (takeoff/scale.py). Overlapping windows
are recombined at inference by overlap-weighted voting (CADSpotting SWA).

Labels are the CAD-layer labels of dataset/taxonomy.py (Arch-43 ids plus coarse
51-54 where a layer names only the family), with instances from the parser's
endpoint clustering; stuff classes get instance -1 as in FloorPlanCAD. Geometry is
the parser's primitives: straight segments stay one segment, curves use their
four samples, and every segment is split until no piece exceeds
`--dynamic_sampling_ratio` of the window side (VecFormer uses 1% of the drawing's
shorter side). Coordinates are window-local with y down, like an SVG.

The split follows an existing corpus directory (train/ and test/ tile names ->
document -> side), so no building appears on both sides.
"""

import argparse
import glob
import json
import math
import os
import os.path as osp
import re
import sys
from collections import defaultdict

HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import taxonomy as tx  # noqa: E402

STUFF = set(tx.STUFF_CLASSES)
DOC_RE = re.compile(r"^(.*)_p\d{4}")


def split_of_docs(corpus_dir):
    side = {}
    for split in ("train", "test"):
        d = osp.join(corpus_dir, split)
        if not osp.isdir(d):
            continue
        for name in os.listdir(d):
            m = DOC_RE.match(name)
            if m:
                side[m.group(1)] = split
    return side


def _segments(pts, cmd):
    """Parser primitive (4 samples) -> list of (x0, y0, x1, y1) in page space."""
    p = [(pts[i], pts[i + 1]) for i in range(0, 8, 2)]
    if cmd == 0 or all(math.dist(p[0], q) < 1e-9 for q in p[1:]):   # a line: first to last sample
        return [(p[0][0], p[0][1], p[3][0], p[3][1])]
    return [(a[0], a[1], b[0], b[1]) for a, b in zip(p, p[1:])]


def _densify(seg, max_len):
    x0, y0, x1, y1 = seg
    n = max(1, int(math.ceil(math.dist((x0, y0), (x1, y1)) / max_len))) if max_len > 0 else 1
    return [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n,
             x0 + (x1 - x0) * (i + 1) / n, y0 + (y1 - y0) * (i + 1) / n) for i in range(n)]


def windows(x0, y0, x1, y1, size, step):
    xs = [x0] if x1 - x0 <= size else [x0 + k * step for k in range(int(math.ceil((x1 - x0 - size) / step)) + 1)]
    ys = [y0] if y1 - y0 <= size else [y0 + k * step for k in range(int(math.ceil((y1 - y0 - size) / step)) + 1)]
    return [(wx, wy, wx + size, wy + size) for wy in ys for wx in xs]


def convert_page(data, text_lines, scale_at, viewport_of, window_m, ratio, min_fg, max_segments, meta_base):
    """Windows of one parsed page -> list of (suffix, json dict)."""
    import numpy as np
    args = data["args"]
    n = len(args)
    if n == 0:
        return []
    arr = np.asarray(args, dtype=np.float64).reshape(-1, 8)
    cx, cy = arr[:, 0::2].mean(1), arr[:, 1::2].mean(1)
    sem = np.asarray(data["semanticIds"])
    ins = np.asarray(data["instanceIds"])
    layer = np.asarray(data["layerIds"])
    names = data.get("layerNames", [])
    cmds = data.get("commands", [0] * n)

    # One window size per drawing scale on the page (viewports differ).
    by_scale = defaultdict(list)
    for i in range(n):
        by_scale[round(scale_at(cx[i], cy[i]).mm_per_pt, 6)].append(i)

    out = []
    k = 0
    for mm_per_pt, idx in by_scale.items():
        idx = np.asarray(idx)
        size = window_m * 1000.0 / mm_per_pt                  # window side in page points
        gx0, gy0 = arr[idx, 0::2].min(), arr[idx, 1::2].min()
        gx1, gy1 = arr[idx, 0::2].max(), arr[idx, 1::2].max()
        for wx0, wy0, wx1, wy1 in windows(gx0, gy0, gx1, gy1, size, size / 2):
            inside = idx[(cx[idx] >= wx0) & (cx[idx] < wx1) & (cy[idx] >= wy0) & (cy[idx] < wy1)]
            if inside.size == 0 or int((sem[inside] != tx.ARCH_BG).sum()) < min_fg:
                continue
            max_len = size * ratio
            rec = dict(viewBox=[0.0, 0.0, size, size], coords=[], colors=[], widths=[], primitive_ids=[],
                       layer_ids=[], semantic_ids=[], instance_ids=[], primitive_lengths=[], texts=[])
            local_layers = {}
            local_ins = {}
            for pid, i in enumerate(inside.tolist()):
                segs = [s for seg in _segments(args[i], cmds[i]) for s in _densify(seg, max_len)]
                if not segs:
                    continue
                lid = local_layers.setdefault(int(layer[i]), len(local_layers))
                for x0, y0, x1, y1 in segs:
                    rec["coords"].append([x0 - wx0, wy1 - y0, x1 - wx0, wy1 - y1])   # y down
                    rec["colors"].append([0, 0, 0])
                    rec["widths"].append(1.0)
                    rec["primitive_ids"].append(len(rec["semantic_ids"]))
                    rec["layer_ids"].append(lid)
                c = int(sem[i])
                rec["semantic_ids"].append(c)
                if c == tx.ARCH_BG or c in STUFF or ins[i] < 0:
                    rec["instance_ids"].append(-1)
                else:
                    rec["instance_ids"].append(local_ins.setdefault(int(ins[i]), len(local_ins) + 1))
                rec["primitive_lengths"].append(sum(math.dist((s[0], s[1]), (s[2], s[3])) for s in segs))
            if len(rec["coords"]) > max_segments:
                continue
            rec["layer_names"] = [None] * len(local_layers)
            for g, lid in local_layers.items():
                rec["layer_names"][lid] = names[g] if g < len(names) else None
            for text, (tx0, ty0, tx1, ty1) in text_lines:
                mx, my = (tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0
                if wx0 <= mx < wx1 and wy0 <= my < wy1:
                    rec["texts"].append(dict(text=text[:64], x=mx - wx0, y=wy1 - my, size=abs(ty1 - ty0),
                                             angle=0.0, layer_id=-1))
            wcx, wcy = (wx0 + wx1) / 2.0, (wy0 + wy1) / 2.0
            sc = scale_at(wcx, wcy)
            kind, title = viewport_of(wcx, wcy)
            rec["meta"] = dict(meta_base, window_m=window_m, mm_per_pt=mm_per_pt, scale_source=sc.source,
                               viewport_kind=kind, viewport_title=title)
            out.append((f"_w{k:03d}", rec))
            k += 1
    return out


def convert_document(job):
    pdf_path, out_dir, window_m, ratio, min_fg, max_segments, budget, max_page_prims, pages = job
    import pikepdf
    import pymupdf
    from classify_viewports import boilerplate, nearest_title, page_box, page_titles
    from parse_pdf_plans import parse_page
    from takeoff.scale import page_scales
    import importlib.util
    spec = importlib.util.spec_from_file_location("takeoff_cli", osp.join(ROOT, "tools", "takeoff.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", osp.splitext(osp.basename(pdf_path))[0])
    written, skipped = 0, 0
    pdf = pikepdf.open(pdf_path)
    doc = pymupdf.open(pdf_path)
    skip = boilerplate(doc)
    try:
        for p, page in enumerate(list(pdf.pages), start=1):
            if pages and p not in pages:
                continue
            try:
                data = parse_page(pdf, page, p, taxonomy="arch", time_budget=budget)
            except Exception:
                skipped += 1
                continue
            if max_page_prims and len(data["args"]) > max_page_prims:
                skipped += 1
                continue
            box = page_box(page)
            origin = data["origin"]
            words, lines = cli.text_layer(doc[p - 1], box, origin)
            titles = [(k, x - origin[0], y - origin[1], t) for k, x, y, t in page_titles(doc[p - 1], box, skip)]

            def viewport_of(x, y, titles=titles):
                hit = nearest_title(x, y, titles) if titles else None
                return hit if hit else ("untitled", "")

            scales = page_scales([(t, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for t, b in lines], words)
            for suffix, rec in convert_page(data, lines, scales.at, viewport_of, window_m, ratio, min_fg,
                                            max_segments, dict(source="us", doc=stem, page=p)):
                with open(osp.join(out_dir, f"{stem}_p{p:04d}{suffix}.json"), "w") as f:
                    json.dump(rec, f)
                written += 1
    finally:
        pdf.close()
        doc.close()
    return stem, written, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split_from", required=True, help="corpus dir whose train/ and test/ define the split")
    ap.add_argument("--window_m", type=float, default=10.0)
    ap.add_argument("--dynamic_sampling_ratio", type=float, default=0.01)
    ap.add_argument("--min_fg", type=int, default=20, help="labelled primitives a window needs")
    ap.add_argument("--max_segments", type=int, default=60000)
    ap.add_argument("--page_time_budget", type=float, default=120.0)
    ap.add_argument("--max_page_prims", type=int, default=800000)
    ap.add_argument("--docs", nargs="*", help="only these document stems")
    ap.add_argument("--pages", type=int, nargs="*", help="only these 1-based pages (for checks)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()

    side = split_of_docs(a.split_from)
    jobs = []
    for pdf in sorted(glob.glob(osp.join(a.pdf_dir, "*.pdf"))):
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", osp.splitext(osp.basename(pdf))[0])
        if stem not in side or (a.docs and stem not in a.docs):
            continue
        out = osp.join(a.output_dir, side[stem])
        os.makedirs(out, exist_ok=True)
        jobs.append((pdf, out, a.window_m, a.dynamic_sampling_ratio, a.min_fg, a.max_segments,
                     a.page_time_budget, a.max_page_prims, set(a.pages or [])))
    print(f"{len(jobs)} documents ({sum(1 for j in jobs if j[1].endswith('train'))} train)", flush=True)
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for stem, w, s in pool.imap_unordered(convert_document, jobs):
                print(f"  {stem}: {w} windows ({s} pages skipped)", flush=True)
    else:
        for stem, w, s in map(convert_document, jobs):
            print(f"  {stem}: {w} windows ({s} pages skipped)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
