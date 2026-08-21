#!/usr/bin/env python3
"""Split converted US plan tiles into train/ and test/ by project.

Splitting by tile would put tiles of the *same building* on both sides, and
because a plan set repeats its unit types across sheets, the model could score
well by recognising the building rather than by recognising doors. So whole
projects go to one side or the other.

Two subtleties this handles:

  * The same project can appear as more than one upload
    (project_723_20260710_... and project_723_20260715_... are one building).
    Grouping is on the numeric project id, not the filename, or those two would
    straddle the split.
  * Projects vary enormously in size, and a test set with almost no windows
    cannot measure window IoU. Projects are picked for the test side to hit the
    target fraction while keeping door and window share close to the corpus.

Files are hard-linked, not copied, so re-splitting is free and costs no disk.

    python dataset/split_us_plans.py --root dataset/us_plans/json4 --test-frac 0.22
"""

import argparse
import json
import os
import os.path as osp
import random
import re
import shutil
from collections import Counter, defaultdict

PROJECT = re.compile(r"^(project_\d+)_")
NAMES = {0: "door", 1: "window", 2: "wall", 3: "bg"}


def project_of(filename):
    m = PROJECT.match(filename)
    return m.group(1) if m else filename.split("_p")[0]


def scan(all_dir):
    """project -> {tiles, door, window, wall, files}"""
    stats = defaultdict(lambda: {"tiles": 0, "files": [],
                                 "door": 0, "window": 0, "wall": 0, "bg": 0})
    for f in sorted(os.listdir(all_dir)):
        if not f.endswith("_s2.json"):
            continue
        key = project_of(f)
        s = stats[key]
        s["tiles"] += 1
        s["files"].append(f)
        try:
            c = Counter(json.load(open(osp.join(all_dir, f)))["semanticIds"])
        except Exception:
            continue
        for k, n in NAMES.items():
            s[n] += c.get(k, 0)
    return stats


def choose_test(stats, test_frac, min_projects=4, overshoot=1.25):
    """Pick whole projects for the test side.

    Greedy: repeatedly take the project that moves the door/window mix closest
    to the corpus mix, until the tile target is met.

    Two constraints, both learned the hard way. The overshoot cap applies to the
    *first* pick as well, or one oversized project swallows the whole budget in a
    single step. And the test side needs several buildings, not one: a single
    project measures how well the model learned that architect's drafting
    conventions, which is not the question.
    """
    total_tiles = sum(s["tiles"] for s in stats.values())
    target = total_tiles * test_frac
    corpus = {k: sum(s[k] for s in stats.values()) for k in ("door", "window", "wall")}
    denom = sum(corpus.values()) or 1
    corpus_mix = {k: v / denom for k, v in corpus.items()}

    chosen, remaining = [], dict(stats)
    cur = {"door": 0, "window": 0, "wall": 0}
    tiles = 0
    while remaining and (tiles < target * 0.85 or len(chosen) < min_projects):
        cap = target * overshoot
        best, best_score = None, None
        for key, s in remaining.items():
            if s["door"] == 0 and s["window"] == 0:
                continue                      # cannot measure openings
            if tiles + s["tiles"] > cap and len(chosen) >= min_projects:
                continue
            trial = {k: cur[k] + s[k] for k in cur}
            d = sum(trial.values()) or 1
            score = sum(abs(trial[k] / d - corpus_mix[k]) for k in cur)
            # Prefer projects that leave us near the target rather than far over.
            score += abs(tiles + s["tiles"] - target) / max(target, 1) * 0.5
            if best_score is None or score < best_score:
                best, best_score = key, score
        if best is None:
            break
        s = remaining.pop(best)
        chosen.append(best)
        tiles += s["tiles"]
        for k in cur:
            cur[k] += s[k]
    return chosen


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dir containing all/")
    ap.add_argument("--test-frac", type=float, default=0.22)
    ap.add_argument("--test-projects", nargs="*", help="override the automatic choice")
    ap.add_argument("--cap", type=int, default=0,
                    help="max tiles per project (0 = no cap). Plan sets differ "
                         "enormously in density -- one set can be a quarter of the "
                         "corpus -- and without a cap the model mostly learns that "
                         "firm's drafting conventions rather than what a door is.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_dir = osp.join(args.root, "all")
    stats = scan(all_dir)

    if args.cap:
        rng = random.Random(args.seed)
        for key, s in stats.items():
            if s["tiles"] <= args.cap:
                continue
            keep = set(rng.sample(s["files"], args.cap))
            scale = args.cap / s["tiles"]
            dropped = s["tiles"] - args.cap
            s["files"] = [f for f in s["files"] if f in keep]
            s["tiles"] = args.cap
            for k in ("door", "window", "wall", "bg"):
                s[k] = int(s[k] * scale)      # counts are only used for balancing
            print(f"[cap] {key}: dropped {dropped} tiles, kept {args.cap}")
    print(f"{len(stats)} projects, {sum(s['tiles'] for s in stats.values())} tiles\n")
    print(f"{'project':16s} {'tiles':>6} {'door':>8} {'window':>8} {'wall':>8}")
    for k in sorted(stats, key=lambda x: -stats[x]["tiles"]):
        s = stats[k]
        print(f"{k:16s} {s['tiles']:6d} {s['door']:8d} {s['window']:8d} {s['wall']:8d}")

    test = args.test_projects or choose_test(stats, args.test_frac)
    print(f"\ntest projects: {', '.join(sorted(test))}")

    for split in ("train", "test"):
        d = osp.join(args.root, split)
        if osp.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    counts = {"train": Counter(), "test": Counter()}
    tiles = Counter()
    for key, s in stats.items():
        split = "test" if key in test else "train"
        for f in s["files"]:
            for name in (f, f.replace("_s2.json", "_s2.png")):
                src = osp.join(all_dir, name)
                if osp.exists(src):
                    link(src, osp.join(args.root, split, name))
            tiles[split] += 1
        for k in ("door", "window", "wall", "bg"):
            counts[split][k] += s[k]

    print()
    for split in ("train", "test"):
        c = counts[split]
        tot = sum(c.values()) or 1
        share = " ".join(f"{k}={100*c[k]/tot:.2f}%" for k in ("door", "window", "wall"))
        print(f"{split:5s} {tiles[split]:5d} tiles  {sum(c.values()):9d} primitives  {share}")
    frac = tiles["test"] / max(sum(tiles.values()), 1)
    print(f"\ntest fraction {frac:.1%} of tiles, split by project (no building in both)")


if __name__ == "__main__":
    main()
