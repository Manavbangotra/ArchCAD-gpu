#!/usr/bin/env python3
"""Joint-training data: per-source datasets, sampling weights, source ids, split
mapping, background remap, no augmentation on evaluation splits, weighted sampler.

    python vecformer/checks/test_multisource.py
"""
import json
import os
import os.path as osp
import sys
import tempfile
import types

import torch

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(HERE, ".."))
import cpu_kernels  # noqa: E402

cpu_kernels.install()

from data.multisource import build  # noqa: E402
from model.vecformer.vecformer_trainer import VecFormerTrainer  # noqa: E402

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def write(root, split, n, bg):
    os.makedirs(osp.join(root, split), exist_ok=True)
    for i in range(n):
        d = dict(viewBox=[0, 0, 100, 100], coords=[[10, 10, 30, 10], [60, 60, 90, 60]],
                 primitive_ids=[0, 1], layer_ids=[0, 1], semantic_ids=[0, bg], instance_ids=[1, -1],
                 primitive_lengths=[20.0, 30.0])
        with open(osp.join(root, split, f"{i}.json"), "w") as f:
            json.dump(d, f)


def main():
    aug = dict(random_vertical_flip=1.0, random_horizontal_flip=1.0, random_rotate=False,
               random_scale=[2.0, 2.0], random_translation=[0.0, 0.0])
    no_aug = dict(random_vertical_flip=0.0, random_horizontal_flip=0.0, random_rotate=False,
                  random_scale=[1.0, 1.0], random_translation=[0.0, 0.0])
    with tempfile.TemporaryDirectory() as d:
        fp, us = osp.join(d, "fp"), osp.join(d, "us")
        for split, n in (("train", 4), ("val", 2), ("test", 2)):
            write(fp, split, n, 35)
        for split, n in (("train", 20), ("test", 3)):       # US has no val split
            write(us, split, n, 43)
        args = dict(common=dict(train_transform_args=aug, eval_transform_args=no_aug),
                    sources=[dict(name="fpcad", root_dir=fp, weight=0.4, background_remap=[35, 43]),
                             dict(name="us", root_dir=us, weight=0.6, splits=dict(val="test"))])
        splits, collate = build(args)
        train = splits.train
        check(len(train) == 24 and train.names == ["fpcad", "us"], f"train concat: {len(train)} {train.names}")
        w = train.sample_weights
        check(abs(w[:4].sum().item() - 0.4) < 1e-9 and abs(w[4:].sum().item() - 0.6) < 1e-9,
              f"sampling shares follow weights, not corpus size: {w[:4].sum().item():.3f}, {w[4:].sum().item():.3f}")
        check(sorted(splits.val) == ["fpcad", "us"] and len(splits.val["us"]) == 3, "per-source val, US val from test")
        fp_item, us_item = train[0], train[4]
        check(fp_item.source_id == 0 and us_item.source_id == 1, "source ids follow taxonomy.ANNOTATED order")
        check(fp_item.sem_ids.tolist() == [0, 43], f"FloorPlanCAD background remapped to 43: {fp_item.sem_ids.tolist()}")
        check(us_item.sem_ids.tolist() == [0, 43], "US labels untouched")
        tr, ev = splits.train.datasets[1][0], splits.val["us"][0]
        check(not torch.allclose(tr.coords, ev.coords), "train split is augmented")
        raw = splits.val["us"]
        check(raw.split == "val" and raw.data_dir.endswith("test"), "US val reads test files with eval transforms")
        e2 = splits.test["us"][0]
        check(torch.allclose(ev.coords, e2.coords), "evaluation items are not augmented")
        batch = collate([fp_item, us_item])
        check(batch["source_ids"].tolist() == [0, 1], "mixed batch keeps per-drawing source ids")

        from data.multisource import holdout_docs
        files = [f"doc{d}_p{p:04d}_w000.json" for d in range(5) for p in range(1, 4)]
        tr_f, va_f = holdout_docs(files, 2)
        docs = lambda fs: {f.split("_p")[0] for f in fs}   # noqa: E731
        check(len(docs(va_f)) == 2 and not docs(tr_f) & docs(va_f) and len(tr_f) + len(va_f) == 15,
              "holdout by whole documents")
        us_dir = us
        args2 = dict(args, sources=[dict(name="us", root_dir=us_dir, weight=1.0, val_from_train=dict(docs=1))])
        s2, _ = build(args2)
        check(len(s2.train) + len(s2.val["us"]) == 20 and len(s2.val["us"]) > 0 and s2.val["us"].split == "val"
              and len(s2.test["us"]) == 3, f"val_from_train: {len(s2.train)} train, {len(s2.val['us'])} val")

        args3 = dict(args, sources=[dict(name="fpcad", root_dir=fp, weight=0.4, evaluate=False),
                                    dict(name="us", root_dir=us, weight=0.6, val_from_train=dict(docs=1))])
        s3, _ = build(args3)
        check(sorted(s3.val) == ["us"] and sorted(s3.test) == ["us"] and len(s3.train) == 4 + len(s2.train),
              "evaluate: false trains on a source without val/test sets")
        check(s3.val["us"].data_paths == s2.val["us"].data_paths,
              "the US validation files do not depend on which other sources are mixed in")

        fake = types.SimpleNamespace(train_dataset=train, args=types.SimpleNamespace(seed=3))
        sampler = VecFormerTrainer._get_train_sampler(fake)
        draws = []
        for _ in range(200):
            draws += list(iter(sampler))
        share_us = sum(1 for i in draws if i >= 4) / len(draws)
        check(abs(share_us - 0.6) < 0.02, f"weighted sampler draws US at its weight: {share_us:.3f}")
        again = VecFormerTrainer._get_train_sampler(fake)
        check(list(iter(again)) == list(iter(VecFormerTrainer._get_train_sampler(fake))), "same seed, same order on every rank")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("ok - multisource: weights, source ids, split mapping, remap, eval without augmentation, sampler")
    return 0


if __name__ == "__main__":
    sys.exit(main())
