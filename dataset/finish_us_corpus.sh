#!/usr/bin/env bash
# Wait for parse_pdf_plans.py to finish, then split by project and recompute the
# class weights the US config needs. Safe to run while the parser is still going.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/sunday/venv/bin/python}
ROOT=dataset/us_plans/json4

while pgrep -f parse_pdf_plans >/dev/null; do sleep 60; done
echo "=== parser finished: $(ls $ROOT/all/*_s2.json 2>/dev/null | wc -l) tiles ==="

PYTHONPATH=dataset $PY dataset/split_us_plans.py --root $ROOT --test-frac 0.22

echo
echo "=== class weights from the train split ==="
PYTHONPATH=dataset $PY - <<'EOF'
import glob, json, math, random
from collections import Counter
files = sorted(glob.glob("dataset/us_plans/json4/train/*_s2.json"))
random.Random(0).shuffle(files)
c = Counter()
for f in files[:600]:
    try: c.update(json.load(open(f))["semanticIds"])
    except Exception: pass
N = {0: "door", 1: "window", 2: "wall", 3: "bg"}
tot = sum(c.values()) or 1
print(f"sampled {min(600, len(files))} of {len(files)} train tiles, {tot} primitives")
for k in (0, 1, 2, 3):
    print(f"  {N[k]:7s} {c[k]:9d}  {100*c[k]/tot:5.2f}%")
real = sum(c[k] for k in (0, 1, 2)) or 1
w = [1/math.sqrt(c[k]/real) for k in (0, 1, 2)]
m = sum(w)/3
print("\n  class_weights: [" + ", ".join(f"{x/m:.3f}" for x in w) + "]   # door, window, wall")
EOF
