#!/usr/bin/env python3
"""Build a US plan-set corpus end to end: scan, convert, split, weigh.

    python tools/build_corpus.py --pdf_dir "C:/Users/me/Downloads/plansets" \
                                 --root dataset/us_plans/json5

Replaces dataset/finish_us_corpus.sh, which cannot run here: it shells out to
pgrep, hardcodes /home/sunday/venv/bin/python, and carries a four-class weight
table in a bash heredoc.

Four stages, each skippable, each safe to re-run:

  1. scan     which PDFs carry CAD layers, deduplicated by content hash
  2. convert  one PDF at a time, so an interrupt costs one document
  3. split    train/test by building, stratified on the opening mix
  4. weigh    measure class balance and print the config line

Stage 2 is driven per-PDF on purpose. parse_pdf_plans.py's --skip_existing keys
on the PDF stem rather than the page, so a single process interrupted at page
200 of 357 would skip that whole document on resume and silently lose the rest.
One document per process means an interrupt lands on a boundary, and the log
says exactly where it stopped.
"""

import argparse
import os
import os.path as osp
import shutil
import subprocess
import sys
import time

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
PY = sys.executable


def run(args, **kw):
    print("    $ " + " ".join(osp.basename(a) if a.endswith(".py") else a
                              for a in args[1:3]) + " ...", flush=True)
    return subprocess.run(args, cwd=ROOT, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf_dir", required=True)
    ap.add_argument("--root", required=True, help="corpus dir; tiles land in <root>/all")
    ap.add_argument("--taxonomy", choices=("us4", "arch"), default="arch")
    ap.add_argument("--stems", default=None,
                    help="reuse an existing stem list instead of scanning")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--test-frac", type=float, default=0.22)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["scan", "convert", "split", "weigh"])
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--max_prims", type=int, default=6000)
    ap.add_argument("--min_prims", type=int, default=500)
    ap.add_argument("--img_size", type=int, default=980)
    a = ap.parse_args()

    alld = osp.join(a.root, "all")
    os.makedirs(alld, exist_ok=True)
    stems_file = a.stems or osp.join(a.root, "layered_stems.txt")
    t_start = time.time()

    if "convert" not in a.skip and not a.no_render and shutil.which("pdftoppm") is None:
        ap.error("--render needs poppler's pdftoppm on PATH. Install poppler, or "
                 "pass --no-render and accept that the image branch trains on "
                 "blank canvases.")

    # -- 1. scan -------------------------------------------------------------
    if "scan" not in a.skip and not a.stems:
        print("[1/4] scanning for usable plan sets")
        r = run([PY, "dataset/scan_ocg.py", "--pdf_dir", a.pdf_dir,
                 "--out", stems_file])
        if r.returncode:
            return r.returncode
    else:
        print(f"[1/4] scan skipped, using {stems_file}")

    with open(stems_file) as fh:
        stems = [ln.strip() for ln in fh if ln.strip()]
    print(f"      {len(stems)} documents\n")

    # -- 2. convert ----------------------------------------------------------
    if "convert" not in a.skip:
        print(f"[2/4] converting into {alld}")
        failed = []
        for i, stem in enumerate(stems, 1):
            pdf = osp.join(a.pdf_dir, stem + ".pdf")
            if not osp.exists(pdf):
                print(f"  [{i}/{len(stems)}] {stem}: no such PDF")
                failed.append(stem)
                continue
            print(f"  [{i}/{len(stems)}] {stem}", flush=True)
            cmd = [PY, "dataset/parse_pdf_plans.py", "--pdf", pdf,
                   "--output_dir", alld, "--taxonomy", a.taxonomy,
                   "--img_size", str(a.img_size), "--min_prims", str(a.min_prims),
                   "--max_prims", str(a.max_prims), "--overlap", str(a.overlap),
                   "--emit_page_max", "40000", "--max_page_prims", "120000",
                   "--render_timeout", "600", "--skip_existing"]
            if not a.no_render:
                cmd.append("--render")
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
            print(f"        {tail[0][:100]}")
            if r.returncode:
                failed.append(stem)
                print(f"        FAILED: {(r.stderr or '').strip()[-200:]}")
        if failed:
            print(f"\n      {len(failed)} document(s) failed: {failed[:5]}")
        print()

    n = len([f for f in os.listdir(alld) if f.endswith("_s2.json")])
    print(f"      {n} tiles in {alld}\n")

    # -- 3. split ------------------------------------------------------------
    if "split" not in a.skip:
        print("[3/4] splitting by building")
        r = run([PY, "dataset/split_us_plans.py", "--root", a.root,
                 "--test-frac", str(a.test_frac)])
        if r.returncode:
            return r.returncode
        print()

    # -- 4. weigh ------------------------------------------------------------
    if "weigh" not in a.skip:
        print("[4/4] measuring class balance")
        run([PY, "tools/corpus_weights.py", "--root", a.root, "--split", "train"])

    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
