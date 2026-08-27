#!/usr/bin/env python3
"""Label every tile by the kind of drawing it was cut from.

A construction sheet is not one drawing. A single frame carries a floor plan
next to ten interior elevations, kitchen sections and details, so filtering by
sheet is far too coarse -- measured over three plan sets, 31% of pages are plan
only, 34% elevation only and 21% carry both. Keeping only whole "plan" sheets
throws away the plans that share a frame with elevations; keeping the mixed
sheets imports the elevations.

Each drawing on a sheet is titled underneath it ("UNIT B1 P01 FLOOR PLAN",
"KITCHEN ELEVATION"), and those titles carry coordinates. This walks the same
tiling the parser produced -- verified to reproduce its tile names exactly --
works out which title each tile sits above, and writes a label per tile. The
geometry parse is re-run without rendering, which is the cheap half, so an
existing corpus can be filtered without being rebuilt.

    python dataset/classify_viewports.py \
        --pdf_dir /home/sunday/Music/all_plansets \
        --tiles_dir dataset/us_plans/json5/all \
        --out dataset/us_plans/json5/viewport_labels.json

Then filter with `--apply`, which moves everything that is not a plan into a
`_rejected/` folder beside the tiles rather than deleting it.
"""

import argparse
import glob
import json
import os
import os.path as osp
import re
import shutil
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import pikepdf
import pymupdf

from parse_pdf_plans import parse_page, tile_sheet

# Order matters: the first pattern that hits decides, so the exclusions have to
# be tested before the generic "PLAN".
DROP = re.compile(
    r"\b(ELEVATION|SECTION|DETAIL|SCHEDULE|DIAGRAM|LEGEND|NOTES?|KEY\s*PLAN"
    r"|BUILDING\s*KEY|KEY\s*MAP|VICINITY"
    r"|ROOF\s*PLAN|SITE\s*PLAN)\b", re.I)

# Kept on request even though they are annotated for fixtures rather than
# dimensions: a reflected ceiling or lighting plan still draws the walls and
# doors, so the geometry is a floor plan and worth training on.
UNIT_PLAN = re.compile(r"\bBEDROOM\b|SQ\.?\s*FT|\bUNIT\s+[A-Z]?\d", re.I)

# "A4.2a", "A 3.02", "CS-4" -- the sheet's own number, printed large in the
# title block. Not a drawing title, but unique per sheet so it survives the
# boilerplate filter, and it anchored tiles on a lighting sheet to "plan".
SHEET_NO = re.compile(r"^[A-Z]{1,3}\s?[-.]?\s?\d{1,2}(\.\d{1,2})?[a-z]?$", re.I)
KEEP = re.compile(
    r"\b(FLOOR\s*PLAN|DIMENSION\s*PLAN|FOUNDATION\s*PLAN|SLAB\s*PLAN"
    r"|FRAMING\s*PLAN|FINISH(?:ES)?\s*PLAN|FURNITURE\s*PLAN)\b", re.I)


def title_kind(text):
    """plan / drop / plan-by-default for a drawing title.

    Titles do not reliably say "floor plan". On these sheets the plan is headed
    "B1 P01 - TWO BEDROOM / TWO BATH - 962 SQ. FT." and the word plan appears
    nowhere, while every elevation, section and detail says so outright. So the
    elevations are what gets recognised, and anything else titled is assumed to
    be a plan -- matching on plan keywords instead silently dropped real floor
    plans.
    """
    if UNIT_PLAN.search(text) and not DROP.search(text):
        return "plan"
    if DROP.search(text):
        return "drop"
    return "plan"


def boilerplate(doc, sample=40, share=0.35):
    """Text the title block repeats on most sheets.

    Every sheet carries the firm's name, address and project title in large
    type, which outnumbers the real drawing titles 1739 to 357 and would anchor
    tiles to nonsense. A drawing title appears on one sheet; a title block
    appears on all of them, which is the difference used here.
    """
    from collections import Counter
    seen, pages = Counter(), 0
    step = max(1, len(doc) // sample)
    for i in range(0, len(doc), step):
        pages += 1
        for t in {x[1] for x in _big_lines(doc[i])}:
            seen[t] += 1
    if pages < 3:
        return set()
    return {t for t, n in seen.items() if n >= share * pages}


def _big_lines(doc_page):
    """(size, text, bbox) per text block, for blocks larger than body copy.

    Per *block*, not per line: a title wraps, and "BUILDING TYPE 'A'" on one
    line with "- ENLARGED FIRST FLOOR LIGHTING PLAN" on the next reads as a
    plan if the halves are judged apart, which kept a whole lighting sheet.
    """
    blocks, sizes = [], []
    for block in doc_page.get_text("dict")["blocks"]:
        parts, size = [], 0.0
        x0 = y0 = 1e18
        x1 = y1 = -1e18
        for line in block.get("lines", []):
            spans = line.get("spans") or []
            if not spans:
                continue
            size = max(size, max(s["size"] for s in spans))
            sizes.append(max(s["size"] for s in spans))
            parts.append("".join(s["text"] for s in spans).strip())
            bx0, by0, bx1, by1 = line["bbox"]
            x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
        text = " ".join(p for p in parts if p).strip()
        if 3 < len(text) <= 90:
            blocks.append((size, text, (x0, y0, x1, y1)))
    if not sizes:
        return []
    sizes.sort()
    floor = max(1.5 * sizes[len(sizes) // 2], 11.0)
    return [b for b in blocks
            if b[0] >= floor
            and not re.fullmatch(r"[\d\W]+", b[1])   # drawing bubble numbers
            and not SHEET_NO.match(b[1])]


def page_titles(doc_page, media_y1, skip=frozenset()):
    """Titled drawings on this sheet as (kind, x, y, text) in PDF space, y-up.

    PyMuPDF reports text boxes in the unrotated page frame with y growing down,
    which is the frame `parse_page` measures geometry in once y is flipped --
    checked on a rotated sheet, where all 11 titles then landed inside the
    geometry bounding box and the derotated variants did not.
    """
    out = []
    for _size, text, (x0, y0, x1, y1) in _big_lines(doc_page):
        if text in skip:
            continue
        out.append((title_kind(text), (x0 + x1) / 2.0,
                    media_y1 - (y0 + y1) / 2.0, text))
    return out


def nearest_title(cx, cy, titles):
    """The title this tile belongs to.

    A drawing's title is printed below it, so a title beneath the tile is the
    likely owner and one above it usually belongs to the drawing overhead --
    hence the asymmetric penalty rather than plain nearest-neighbour.
    """
    best, best_d = None, None
    for kind, tx, ty, text in titles:
        dy = cy - ty                      # positive when the tile sits above
        d = abs(cx - tx) + (dy if dy >= 0 else 3.0 * -dy)
        if best_d is None or d < best_d:
            best, best_d = (kind, text), d
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf_dir", required=True)
    ap.add_argument("--tiles_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="move non-plan tiles into _rejected/ beside the tiles")
    # These must match how the corpus was parsed or the tiling will not line up.
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--max_prims", type=int, default=6000)
    ap.add_argument("--min_prims", type=int, default=500)
    ap.add_argument("--emit_page_max", type=int, default=40000)
    ap.add_argument("--max_page_prims", type=int, default=120000)
    ap.add_argument("--tile_units", type=float, default=0.0)
    ap.add_argument("--tile_doors", type=float, default=0.0)
    a = ap.parse_args()

    have = {osp.basename(f) for f in glob.glob(osp.join(a.tiles_dir, "*_s2.json"))}
    labels = {}
    counts = {"plan": 0, "drop": 0, "untitled": 0, "not_in_corpus": 0}

    for path in sorted(glob.glob(osp.join(a.pdf_dir, "*.pdf"))):
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", osp.splitext(osp.basename(path))[0])
        if not any(h.startswith(stem + "_p") for h in have):
            continue
        try:
            pdf = pikepdf.open(path)
            doc = pymupdf.open(path)
        except Exception as exc:
            print(f"[fail] {stem}: {exc}", flush=True)
            continue

        skip = boilerplate(doc)
        for i, page in enumerate(list(pdf.pages), start=1):
            try:
                data = parse_page(pdf, page, i)
            except Exception:
                continue
            n = len(data["args"])
            if n < a.min_prims or (a.max_page_prims and n > a.max_page_prims):
                continue

            try:
                dp = doc[i - 1]
                titles = page_titles(dp, dp.mediabox.y1, skip)
            except Exception:
                titles = []

            ox0, oy0 = data["origin"]
            page_plan_only = (titles and all(t[0] == "plan" for t in titles))

            for suffix, tile in tile_sheet(
                    data, a.max_prims, min_prims=200, overlap=a.overlap,
                    emit_page_max=a.emit_page_max, tile_units=a.tile_units,
                    tile_doors=a.tile_doors):
                if len(tile["args"]) < a.min_prims:
                    continue
                name = f"{stem}_p{i:04d}{suffix}_s2.json"
                if name not in have:
                    counts["not_in_corpus"] += 1
                    continue

                if suffix == "_full":
                    # A whole-page sample spans every drawing on the sheet, so it
                    # is only a plan if nothing else shares the frame.
                    kind, why = ("plan", "page is plan only") if page_plan_only \
                        else ("drop", "whole page, mixed frame")
                elif not titles:
                    kind, why = "untitled", ""
                else:
                    tx, ty = tile["origin"]
                    hit = nearest_title(tx + tile["width"] / 2.0,
                                        ty + tile["height"] / 2.0, titles)
                    kind, why = hit[0], hit[1]

                labels[name] = {"kind": kind, "title": why}
                counts[kind] = counts.get(kind, 0) + 1

        pdf.close()
        doc.close()
        print(f"[ok] {stem}: {counts['plan']} plan / {counts['drop']} drop so far",
              flush=True)

    with open(a.out, "w") as f:
        json.dump(labels, f)
    print(f"\n[done] {len(labels)} tiles labelled -> {a.out}")
    for k, v in sorted(counts.items()):
        print(f"   {k:14s} {v}")

    if a.apply:
        rej = osp.join(a.tiles_dir, "_rejected")
        os.makedirs(rej, exist_ok=True)
        moved = 0
        for name, lab in labels.items():
            if lab["kind"] == "plan":
                continue
            for ext in (".json", ".png"):
                src = osp.join(a.tiles_dir, name.replace(".json", ext))
                if osp.exists(src):
                    shutil.move(src, osp.join(rej, osp.basename(src)))
                    moved += 1
        print(f"[apply] moved {moved} files into {rej}")


if __name__ == "__main__":
    main()
