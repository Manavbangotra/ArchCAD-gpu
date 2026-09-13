#!/usr/bin/env python3
"""Self-check for multi-source training data (svgnet/data/multi.py).

    python tools/test_multi_source.py [--fpcad DIR] [--us DIR]

Each DIR holds a `train/` folder of *_s2.json tiles. Defaults are the full
corpora; the check reads only a few drawings from each. Pins:
  1. every batch the sampler yields comes from one source, in weight proportion;
  2. instance ids never collide across sources, even in a mixed batch;
  3. the annotated mask reaches the batch, ANDed across drawings;
  4. stuff classes arrive as one mask per class (instanceId -1).
"""

import argparse
import collections
import os.path as osp
import sys

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, "dataset"))

import torch  # noqa: E402

import taxonomy as tx  # noqa: E402
from svgnet.data.multi import MultiSourceDataset  # noqa: E402

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def build(fpcad, us):
    common = dict(data_norm="mean", aug=False, img_size=128, num_classes=43)
    return MultiSourceDataset(
        sources=[dict(name="fpcad", data_root=fpcad, weight=0.25, annotated="fpcad",
                      stuff_classes="default", max_samples=6),
                 dict(name="us", data_root=us, weight=0.75, annotated="us",
                      coarse_policy="keep", stuff_classes="default", max_samples=6)],
        split="train", samples_per_epoch=400, **common)


def test_sampler(ds):
    counts = collections.Counter()
    for batch in ds.batch_sampler(2, seed=3):
        srcs = {ds.names[ds.locate(i)[0]] for i in batch}
        check(len(srcs) == 1, f"batch mixes sources: {srcs}")
        counts[srcs.pop()] += 1
    share = counts["us"] / max(1, sum(counts.values()))
    check(0.65 < share < 0.85, f"US share of batches {share:.2f}, weight 0.75")
    a = list(ds.batch_sampler(2, seed=3))
    s = ds.batch_sampler(2, seed=3)
    s.set_epoch(0)
    check(a == list(s), "the same seed and epoch give the same batches")


def test_ids_and_meta(ds):
    lo_f, _ = ds.ranges()[0]
    lo_u, _ = ds.ranges()[1]
    items = [ds[lo_f], ds[lo_u]]           # local index 0 of each source
    batch = ds.collate_fn(items)
    check(len(batch) == 10, f"collate returns 10 fields, got {len(batch)}")
    label, offsets, meta = batch[2], batch[3], batch[8]
    ins = label[:, 1]
    first = ins[:offsets[0]]
    second = ins[offsets[0]:]
    a = set(first[first >= 0].tolist())
    b = set(second[second >= 0].tolist())
    check(not (a & b), f"{len(a & b)} instance ids shared across sources")

    check(meta["source"] == ["fpcad", "us"], f"sources in meta: {meta['source']}")
    ann = meta["annotated"]
    want = set(tx.ANNOTATED["fpcad"]) & set(tx.ANNOTATED["us"])
    got = set(torch.nonzero(ann[:43]).flatten().tolist())
    check(got == want, "annotated mask is the AND of both sources")
    check(bool(ann[43]), "background is always annotated")

    for (coord, feat, lab, *_rest) in items:
        sem, ins_ = lab[:, 0], lab[:, 1]
        stuff = torch.isin(sem, torch.tensor(tx.STUFF_CLASSES))
        check(bool((ins_[stuff] == -1).all()), "stuff classes carry instanceId -1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpcad", default=osp.join(ROOT, "dataset/FloorplanCAD/json"))
    ap.add_argument("--us", default=osp.join(ROOT, "dataset/us_plans/json5"))
    a = ap.parse_args()
    for d in (a.fpcad, a.us):
        if not osp.isdir(osp.join(d, "train")):
            print(f"SKIP - no train/ under {d}")
            return 0
    ds = build(a.fpcad, a.us)
    tests = [test_sampler, test_ids_and_meta]
    for fn in tests:
        fn(ds)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  {f}")
        return 1
    print(f"ok - {len(tests)} groups, multi-source batches are clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
