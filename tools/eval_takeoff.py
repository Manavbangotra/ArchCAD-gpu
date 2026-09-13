#!/usr/bin/env python3
"""Score a takeoff JSON against a human-verified takeoff.

    python tools/eval_takeoff.py --pred plans.takeoff.json --truth verified.csv [--project 461]

This is the number that says whether the output is good enough to ship; the
model's mIoU/PQ are for iterating on the model. The verified takeoff is read
from a plain CSV export, one row per counted line item:

    project,category,item,count,area_sqft
    461,door,01,24,
    461,door,05,96,
    461,window,B,120,
    461,fixture,toilet,48,
    461,room,bedroom,72,11520

category   door | window | fixture | appliance | furniture | room
item       door/window: the type mark as printed ("01", "W3");
           fixture/appliance/furniture: a class name (toilet, sink, bath tub,
           refrigerator, ...) or "*" for the category total;
           room: a room group (bedroom, bath, kitchen, living, storage,
           entry/hallway, garage, outdoor, undefined) or "*"
count      the verified quantity
area_sqft  rooms only, optional: total area of that group

Reports, per category, each item's verified vs predicted count and the
category's count accuracy, 1 - sum|pred - true| / sum true (1.0 is perfect;
it goes negative when predictions are badly inflated). Room area error is a
percentage of verified area.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict

CAT_KEYS = {"door": "doors", "window": "windows", "fixture": "fixtures",
            "appliance": "appliances", "furniture": "furniture", "room": "rooms"}


def norm_mark(mark):
    return re.sub(r"[\s\-_.]", "", str(mark).upper())


def predicted(doc):
    """{category: {item: count}}, plus room areas, from counted items only."""
    counts = defaultdict(lambda: defaultdict(int))
    areas = defaultdict(float)
    for pg in doc["pages"]:
        for cat, key in CAT_KEYS.items():
            for it in pg.get(key, []):
                if not it.get("counted"):
                    continue
                if cat in ("door", "window"):
                    item = norm_mark(it["tag"]) if it.get("tag") else "(untagged)"
                elif cat == "room":
                    item = it["group"]
                    areas[item] += it.get("area_sqft", 0.0)
                    areas["*"] += it.get("area_sqft", 0.0)
                else:
                    item = it["class"]
                counts[cat][item] += 1
                counts[cat]["*"] += 1
    return counts, areas


def read_truth(path, project=None):
    counts = defaultdict(lambda: defaultdict(int))
    areas = defaultdict(float)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if project and str(row.get("project", "")).strip() != str(project):
                continue
            cat = row["category"].strip().lower()
            if cat not in CAT_KEYS:
                continue
            item = row["item"].strip()
            item = norm_mark(item) if cat in ("door", "window") else item.lower()
            counts[cat][item] += int(float(row["count"] or 0))
            if item != "*":
                counts[cat]["*"] += int(float(row["count"] or 0))
            if cat == "room" and row.get("area_sqft"):
                areas[item] += float(row["area_sqft"])
                if item != "*":
                    areas["*"] += float(row["area_sqft"])
    return counts, areas


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--project")
    ap.add_argument("--json", help="also write the scores here")
    a = ap.parse_args()

    with open(a.pred, encoding="utf-8") as f:
        doc = json.load(f)
    p_counts, p_areas = predicted(doc)
    t_counts, t_areas = read_truth(a.truth, a.project)
    if not t_counts:
        print("no verified rows matched", file=sys.stderr)
        return 2

    report = {}
    for cat in CAT_KEYS:
        truth = t_counts.get(cat)
        if not truth:
            continue
        items = sorted(k for k in truth if k != "*")
        rows, err, tot = [], 0, 0
        for item in items:
            t, p = truth[item], p_counts[cat].get(item, 0)
            rows.append((item, t, p))
            err += abs(p - t)
            tot += t
        t_all, p_all = truth.get("*", tot), p_counts[cat].get("*", 0)
        acc_items = (1 - err / tot) if tot else None
        acc_total = (1 - abs(p_all - t_all) / t_all) if t_all else None
        print(f"\n{cat}: verified {t_all}, predicted {p_all}"
              + (f", item accuracy {acc_items:.2f}" if acc_items is not None else "")
              + (f", total accuracy {acc_total:.2f}" if acc_total is not None else ""))
        for item, t, p in rows:
            flag = "" if t == p else ("  over" if p > t else "  under")
            print(f"   {item:18s} {t:6d} {p:6d}{flag}")
        entry = {"verified": t_all, "predicted": p_all, "item_accuracy": acc_items,
                 "total_accuracy": acc_total, "items": {i: {"verified": t, "predicted": p} for i, t, p in rows}}
        if cat == "room" and t_areas:
            entry["area_error_pct"] = {}
            for g, ta in sorted(t_areas.items()):
                pa = p_areas.get(g, 0.0)
                pct = 100.0 * (pa - ta) / ta if ta else None
                entry["area_error_pct"][g] = pct
                print(f"   area {g:13s} {ta:9.0f} {pa:9.0f} sq ft  ({pct:+.1f}%)")
        report[cat] = entry

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
