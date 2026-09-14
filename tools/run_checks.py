#!/usr/bin/env python3
"""Run every self-check in the repo.

    python tools/run_checks.py

There is no pytest here and no CI, so the checks are standalone scripts that
exit non-zero on failure. This runs them all and reports which fell over, so
"did I break anything" is one command rather than four.

The taxonomy check is stdlib-only and always runs. The others import
svgnet.data.svg, which pulls in torch; they are skipped with a clear message
rather than a stack trace if it is not installed, so this stays useful on a
machine that only does data preparation.
"""

import os.path as osp
import subprocess
import sys
import time

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))

CHECKS = [
    ("taxonomy and CAD-layer rules", "dataset/test_taxonomy.py", False),
    ("CubiCasa label spaces", "dataset/test_cubicasa.py", False),
    ("room tags parse", "dataset/test_room_names.py", False),
    ("instance clustering", "dataset/test_cluster.py", False),
    ("takeoff geometry", "takeoff/test_takeoff.py", False),
    ("takeoff drawing roles", "takeoff/test_roles.py", False),
    ("vecformer strict PQ", "vecformer/checks/test_strict_pq.py", True),
    ("vecformer preprocessor (V1+V2 SVG)", "vecformer/checks/test_preprocess_v2.py", True),
    ("corrections reach the loader", "dataset/test_corrections.py", True),
    ("loader/editor/weights agree", "dataset/test_merge_agreement.py", True),
    ("set loss over target rows", "tools/test_set_loss.py", True),
    ("multi-source batches", "tools/test_multi_source.py", True),
    ("fast paths match loops", "tools/test_fast_paths.py", True),
]


def needs_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    have_torch = needs_torch()
    width = max(len(n) for n, _, _ in CHECKS)
    failed = skipped = 0
    for name, script, torchy in CHECKS:
        if torchy and not have_torch:
            print(f"{name:{width}s}  SKIP  (needs torch)")
            skipped += 1
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, osp.join(ROOT, script)],
                           capture_output=True, text=True, cwd=ROOT)
        dt = time.time() - t0
        if r.returncode == 0:
            print(f"{name:{width}s}  ok    {dt:5.1f}s")
        else:
            failed += 1
            print(f"{name:{width}s}  FAIL  {dt:5.1f}s")
            for line in (r.stdout + r.stderr).strip().splitlines()[-12:]:
                print(f"    {line}")
    total = len(CHECKS)
    print(f"\n{total - failed - skipped}/{total} passed"
          + (f", {skipped} skipped" if skipped else "")
          + (f", {failed} FAILED" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
