#!/usr/bin/env python3
"""Model-in-the-takeoff path: sheet -> real-size windows -> model -> SWA -> page objects.

An oracle "model" that returns each window's own layer labels must give back every
page object exactly once, including objects cut by window borders; the real
VecFormer (small, random weights, CPU kernel stand-ins) must run the same path.

    python takeoff/test_vecformer_takeoff.py
"""
import os.path as osp
import sys

import numpy as np

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, "dataset"))
import taxonomy as tx  # noqa: E402
from takeoff.scale import Scale  # noqa: E402
from takeoff.vecformer_model import page_objects_swa  # noqa: E402

FAILURES = []
C = tx.ARCH_NUM_CLASSES


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def line(x0, y0, x1, y1):
    return [x0, y0, x0 + (x1 - x0) / 3, y0 + (y1 - y0) / 3, x0 + 2 * (x1 - x0) / 3, y0 + 2 * (y1 - y0) / 3, x1, y1]


def page():
    """1/4" scale: a 10 m window is ~590 pt, stepped by ~295 pt. Page ~1500 x 700 pt."""
    args, sem, ins = [], [], []
    for k, (x, y) in enumerate([(100, 100), (580, 300), (1200, 500), (890, 120)]):   # 4 single doors
        for seg in (line(x, y, x + 30, y), line(x, y, x, y + 30), line(x + 30, y, x + 21, y + 21)):
            args.append(seg), sem.append(tx.SINGLE_DOOR), ins.append(k)
    for x in range(0, 1500, 50):                                     # a wall run across the sheet
        args.append(line(x, 50, x + 50, 50)), sem.append(32), ins.append(-1)
    for x in range(20, 1480, 70):                                    # background ticks
        args.append(line(x, 650, x + 5, 650)), sem.append(C), ins.append(-1)
    n = len(args)
    return dict(args=args, commands=[0] * n, semanticIds=sem, instanceIds=ins,
                layerIds=[0 if s == 0 else 1 if s == 32 else 2 for s in sem], layerNames=["A-DOOR", "A-WALL", "0"])


class Oracle:
    num_classes = C
    stuff_classes = list(tx.STUFF_CLASSES)

    def predict(self, records):
        out = []
        for rec in records:
            sem = np.asarray(rec["semantic_ids"])
            ins = np.asarray(rec["instance_ids"])
            probs = np.zeros((len(sem), C + 1))
            probs[np.arange(len(sem)), np.clip(sem, 0, C)] = 1.0
            keys = sorted({i for i in ins.tolist() if i >= 0})
            masks = np.stack([ins == k for k in keys]) if keys else np.zeros((0, len(sem)), bool)
            labels = np.array([sem[ins == k][0] for k in keys], dtype=np.int64)
            out.append(dict(sem_probs=probs, inst_masks=masks, inst_labels=labels, inst_scores=np.ones(len(keys))))
        return out


def main():
    data = page()
    sc = Scale(48.0, "scale_string")
    view = lambda x, y: ("plan", "")                                  # noqa: E731
    objects, n_win = page_objects_swa(data, [], lambda x, y: sc, view, Oracle())
    doors = [o for o in objects if o.label == 0]
    check(n_win > 2, f"sheet covered by several windows: {n_win}")
    check(len(doors) == 4, f"each door once, also across window borders: {len(doors)}")
    check(all(o.prims.size == 3 for o in doors), f"doors complete: {[o.prims.tolist() for o in doors]}")
    walls = [o for o in objects if o.label == 32]
    check(len(walls) == 1 and walls[0].prims.size == 30, "wall primitives from voted semantics")
    check(not [o for o in objects if o.label == C], "no background objects")

    cropped, n_crop = page_objects_swa(data, [], lambda x, y: sc, view, Oracle(), max_prims=12)
    doors_c = [o for o in cropped if o.label == 0]
    check(n_crop > n_win and len(doors_c) == 4 and all(o.prims.size == 3 for o in doors_c),
          f"dense windows as crops: {n_crop} records, doors {[o.prims.tolist() for o in doors_c]}")

    from takeoff.vecformer_model import WindowModel
    sys.path.insert(0, osp.join(ROOT, "vecformer", "checks"))
    model = WindowModel(osp.join(ROOT, "vecformer", "configs", "model", "product_arch43.yaml"), cpu_kernels=True,
                        device="cpu", model_args_override=_small())
    objects, n_win2 = page_objects_swa(data, [("BATH", (100, 100, 120, 110))], lambda x, y: sc, view, model)
    check(n_win2 == n_win, "real model sees the same windows")
    check(all(0 <= o.label < C and o.prims.max() < len(data["args"]) for o in objects),
          f"real model output is page objects: {len(objects)}")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print(f"ok - takeoff with the window model: {n_win} windows, oracle recovers every object once; VecFormer runs")
    return 0


def _small():
    sys.path.insert(0, osp.join(ROOT, "vecformer"))
    sys.path.insert(0, osp.join(ROOT, "vecformer", "checks"))
    import cpu_kernels
    cpu_kernels.install()
    from test_model_text_cpu import small_config
    s = small_config(True)
    return dict(backbone_config=s.backbone_config, cad_decoder_config=s.cad_decoder_config)


if __name__ == "__main__":
    sys.exit(main())
