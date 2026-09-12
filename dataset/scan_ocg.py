#!/usr/bin/env python3
"""Find which plan sets actually carry CAD layers, and drop the duplicates.

Two facts about a directory of client plan sets, both measured on this corpus
and both expensive to learn the hard way:

  * Only some PDFs carry optional-content groups. The rest were flattened on
    export, so every primitive comes back with no layer and the whole sheet
    labels as background. Of 230 local plan sets, 90 carry layers and 140 do not.

  * The same document is uploaded many times under different project ids. Those
    90 layered PDFs are only 26 distinct documents -- one appears 42 times,
    another 8, another 6. Ingesting them all would put 42 copies of one building
    in the corpus, and because dataset/split_us_plans.py splits on the numeric
    project id rather than on content, the same building would land in train and
    in test at once.

So this emits one stem per distinct document, layered only, oldest upload first.

    python dataset/scan_ocg.py --pdf_dir <dir> --out layered_stems.txt
    python dataset/scan_ocg.py --pdf_dir <dir> --out new.txt --exclude_ingested dataset/us_plans/json4/all

Deliberately uses pypdf, not pikepdf: this only reads the document catalogue, it
runs before the parser's dependencies matter, and it is the cheap half.
"""

import argparse
import glob
import hashlib
import json
import os
import os.path as osp
import re
import warnings

warnings.filterwarnings("ignore")

STEM = re.compile(r"(project_\d+_\d+_\d+_[0-9a-f]+)_p\d+")


def layer_names(path):
    """Distinct OCG layer names in a PDF, or [] when it was flattened."""
    from pypdf import PdfReader
    from pypdf.generic import IndirectObject

    def deref(o):
        return o.get_object() if isinstance(o, IndirectObject) else o

    reader = PdfReader(path)
    root = deref(reader.trailer["/Root"])
    if "/OCProperties" not in root:
        return [], len(reader.pages)
    ocp = deref(root["/OCProperties"])
    names = []
    for g in deref(ocp.get("/OCGs", [])) or []:
        n = deref(g).get("/Name")
        if n:
            names.append(str(n))
    return names, len(reader.pages)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ingested_stems(tiles_dir):
    out = set()
    for t in glob.glob(osp.join(tiles_dir, "*_s2.json")):
        m = STEM.match(osp.basename(t))
        if m:
            out.add(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf_dir", required=True)
    ap.add_argument("--out", required=True, help="stem list, one per line")
    ap.add_argument("--report", help="also write the full scan as JSON")
    ap.add_argument("--exclude_ingested", metavar="TILES_DIR",
                    help="omit documents already represented in this tile dir")
    ap.add_argument("--keep_duplicates", action="store_true",
                    help="do not collapse identical documents (you do not want this)")
    a = ap.parse_args()

    pdfs = sorted(glob.glob(osp.join(a.pdf_dir, "*.pdf")))
    rows, failed = {}, 0
    for i, path in enumerate(pdfs, 1):
        stem = osp.splitext(osp.basename(path))[0]
        try:
            names, pages = layer_names(path)
        except Exception as exc:
            print(f"[fail] {stem}: {str(exc)[:60]}")
            failed += 1
            continue
        rows[stem] = {"layers": len(names), "pages": pages,
                      "sha256": sha256(path) if names else None}
        if i % 25 == 0:
            print(f"  scanned {i}/{len(pdfs)}", flush=True)

    layered = {k: v for k, v in rows.items() if v["layers"] > 0}
    print(f"\n{len(rows)} PDFs scanned ({failed} unreadable)")
    print(f"  layered   {len(layered)}")
    print(f"  flattened {len(rows) - len(layered)}  (no OCGs; nothing to label)")

    # One stem per distinct document. Oldest upload wins, so the chosen stem is
    # stable across re-runs as new copies arrive.
    by_hash = {}
    for stem in sorted(layered):
        by_hash.setdefault(layered[stem]["sha256"], []).append(stem)
    chosen = ([s for group in by_hash.values() for s in group]
              if a.keep_duplicates else [g[0] for g in by_hash.values()])
    dupes = sum(len(g) - 1 for g in by_hash.values())
    print(f"  distinct documents {len(by_hash)}  ({dupes} duplicate uploads collapsed)")

    if a.exclude_ingested:
        have = ingested_stems(a.exclude_ingested)
        have_hashes = {layered[s]["sha256"] for s in have if s in layered}
        before = len(chosen)
        chosen = [s for s in chosen if layered[s]["sha256"] not in have_hashes]
        print(f"  already ingested   {before - len(chosen)}")

    chosen.sort()
    # newline="" so Windows does not translate to CRLF: this list is consumed by
    # shell loops and by --pdf_list, and a trailing \r turns every stem into a
    # filename that does not exist.
    with open(a.out, "w", newline="") as fh:
        fh.write("\n".join(chosen) + "\n")
    pages = sum(layered[s]["pages"] for s in chosen)
    print(f"\nwrote {len(chosen)} stems ({pages} pages) -> {a.out}")

    if a.report:
        with open(a.report, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"wrote full scan -> {a.report}")


if __name__ == "__main__":
    raise SystemExit(main())
