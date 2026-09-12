#!/usr/bin/env python3
"""Measure a corpus's class balance and emit the class_weights line for a config.

Replaces the class-weight block in dataset/finish_us_corpus.sh, which is a bash
heredoc hardcoded to four classes and to a Linux venv path.

    python tools/corpus_weights.py --root dataset/us_plans/json5 --split train

Counts on the train split only -- weighting on the test split would leak it --
and prints both a readable share table and a paste-ready YAML line.

Two decisions worth knowing about:

  * A coarse label (door-any, fixture-any) is evidence for its whole group but
    for no single member. Its count is spread uniformly over the members before
    weighting. Without this, every door subtype reads as zero-support on a
    US-only corpus while the marginal loss is actively training them.

  * Weights are clipped. A class at 0.01% otherwise earns a weight ~30x the mean
    and destabilises the run long before it helps that class.
"""

import argparse
import json
import os
import os.path as osp
import random
import sys
from collections import Counter

sys.path.insert(0, osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "dataset"))
import taxonomy as TX  # noqa: E402


def count(root, split, sample, seed, corrections=True):
    d = osp.join(root, split)
    if not osp.isdir(d):
        d = osp.join(root, "all")
    files = sorted(f for f in os.listdir(d) if f.endswith("_s2.json"))
    if sample and sample < len(files):
        files = random.Random(seed).sample(files, sample)
    c = Counter()
    tiles = 0
    for i, f in enumerate(files, 1):
        try:
            with open(osp.join(d, f)) as fh:
                data = json.load(fh)
        except Exception:
            continue
        sem = data["semanticIds"]
        if corrections:
            sem = _corrected(osp.join(d, f), data, sem)
        c.update(sem)
        tiles += 1
        if i % 500 == 0:
            print(f"  counted {i}/{len(files)}", flush=True)
    return c, tiles, len(files)


def _corrected(path, data, sem):
    """Apply layer rules and brush corrections, as svgnet/data/svg.py does."""
    root_all = osp.join(osp.dirname(osp.dirname(path)), "all")
    canon = root_all if osp.isdir(root_all) else osp.dirname(path)
    base = osp.basename(path).replace("_s2.json", "")

    def side(kind):
        p = osp.join(canon, f"{base}_s2.{kind}.json")
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception:
            return None

    sem = list(sem)
    names = data.get("layerNames") or []
    if names:
        rules = {}
        try:
            with open(osp.join(canon, "_project_layermap.json")) as fh:
                rules.update(json.load(fh).get("rules", {}))
        except Exception:
            pass
        rules.update((side("layermap") or {}).get("rules", {}))
        if rules:
            lut = {i: rules[n] for i, n in enumerate(names) if n in rules}
            for i, li in enumerate(data.get("layerIds") or []):
                if i < len(sem) and li in lut:
                    sem[i] = lut[li]
    for k, v in (side("override") or {}).items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(sem):
            sem[i] = int(v)
    return sem


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--taxonomy", choices=("us4", "arch"), default=None,
                    help="default: read the .taxonomy marker beside the corpus")
    ap.add_argument("--sample", type=int, default=0,
                    help="count this many tiles instead of all (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip", type=float, nargs=2, default=(0.3, 5.0),
                    metavar=("LO", "HI"))
    ap.add_argument("--no-corrections", action="store_true")
    a = ap.parse_args()

    tax = a.taxonomy
    if tax is None:
        for cand in (osp.join(a.root, ".taxonomy"), osp.join(a.root, "all", ".taxonomy")):
            try:
                with open(cand) as fh:
                    tax = fh.read().strip() or None
                if tax:
                    break
            except OSError:
                pass
        tax = tax or "us4"

    if tax == "arch":
        names = dict(TX.ARCH_NAMES)
        n_classes, bg = TX.ARCH_NUM_CLASSES, TX.ARCH_BG
        groups = TX.COARSE_GROUPS
    else:
        names = dict(TX.CLASS_NAMES)
        n_classes, bg = TX.NUM_CLASSES, TX.BACKGROUND
        groups = {}

    print(f"taxonomy {tax}: {n_classes} classes, background {bg}")
    counts, tiles, total_files = count(a.root, a.split, a.sample, a.seed,
                                       corrections=not a.no_corrections)

    # Spread each coarse id over its members before weighting.
    eff = Counter({k: v for k, v in counts.items() if k < bg})
    coarse_total = 0
    for cid, members in groups.items():
        n = counts.get(cid, 0)
        if not n or not members:
            continue
        coarse_total += n
        for m in members:
            eff[m] += n / len(members)

    total = sum(counts.values()) or 1
    fg = sum(eff.values()) or 1
    print(f"\n{tiles} tiles of {total_files}, {total:,} primitives")
    print(f"background {100 * counts.get(bg, 0) / total:.2f}%  "
          f"labelled {100 * (total - counts.get(bg, 0)) / total:.2f}%")
    if coarse_total:
        print(f"coarse labels {100 * coarse_total / total:.2f}% "
              f"(spread over their group members for weighting)")

    print(f"\n{'id':>3} {'class':18s} {'share%':>8} {'weight':>7}")
    weights = []
    for i in range(n_classes):
        f = eff.get(i, 0.0) / fg
        w = (1.0 / max(f, 1e-9) ** 0.5) if f > 0 else 0.0
        weights.append(w)
    live = [w for w in weights if w > 0]
    mean = sum(live) / len(live) if live else 1.0
    out = []
    for i in range(n_classes):
        w = weights[i] / mean if weights[i] > 0 else 1.0
        # Zero-support classes get 1.0 and it provably does not matter:
        # F.cross_entropy weights by the *target* class, and a class that never
        # appears as a target never contributes to the loss.
        w = min(max(w, a.clip[0]), a.clip[1])
        out.append(round(w, 3))
        share = 100 * eff.get(i, 0.0) / fg
        flag = "" if eff.get(i, 0) else "   (no support)"
        print(f"{i:3d} {names.get(i, '?'):18s} {share:8.3f} {w:7.3f}{flag}")

    print("\n# paste into the config:")
    print(f"    class_weights: {out}")
    print(f"    # measured on {a.root} {a.split}: {tiles} tiles, "
          f"{total:,} primitives, clip {a.clip[0]}-{a.clip[1]}")


if __name__ == "__main__":
    raise SystemExit(main())
