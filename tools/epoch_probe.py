#!/usr/bin/env python3
"""
Score a checkpoint on a few fixed held-out sheets, and draw what it predicts.

Reports two views, because they answer different questions:

  masked  - the project's own metric. It ignores background primitives
            (ignore_label = class count), so it asks "of the real doors,
            windows and walls, how many were classified correctly?"
  honest  - the same comparison with background included, which is what using
            the model on a whole sheet actually looks like. A model that paints
            every line as "door" scores well on the first and badly here.

The gap between them is the point: early in training the masked score can look
respectable while the model never predicts background at all.

Usage:
    python tools/epoch_probe.py <config> <checkpoint> <out_dir> [n_tiles]
"""

import json
import logging
import os
import os.path as osp
import sys

import numpy as np
import torch
import yaml
from munch import Munch
from PIL import Image, ImageDraw

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from svgnet.data import build_dataset          # noqa: E402
from svgnet.model.svgnet import SVGNet         # noqa: E402
from svgnet.util import load_checkpoint        # noqa: E402

NAMES = {0: "door", 1: "window", 2: "wall", 3: "bg"}
COL = {0: (255, 0, 150), 1: (0, 90, 255), 2: (190, 90, 20), 3: (233, 233, 233)}
SIZE = 460


def pick_tiles(ds, n):
    """The same tiles every epoch, chosen for door/window content."""
    scored = []
    for i, f in enumerate(ds.data_list):
        s = np.array(json.load(open(f))["semanticIds"])
        scored.append((int(((s == 0) | (s == 1)).sum()), i))
    scored.sort(reverse=True)
    step = max(1, len(scored) // (n * 4))
    return [i for _, i in scored[::step][:n]]


def draw(d, labels, stripe):
    img = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    sx, sy = SIZE / d["width"], SIZE / d["height"]
    for target in (3, 2, 1, 0):                    # background first
        for pts, c in zip(d["args"], labels):
            if c != target:
                continue
            xy = [(pts[k] * sx, SIZE - pts[k + 1] * sy) for k in range(0, 8, 2)]
            dr.line(xy, fill=COL[c], width=1 if c == 3 else (2 if c == 2 else 3))
    dr.rectangle([0, 0, SIZE - 1, 5], fill=stripe)
    return img


def main():
    cfg_path, ckpt, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    os.makedirs(out_dir, exist_ok=True)

    lg = logging.getLogger("probe")
    lg.info = lambda *a, **k: None
    lg.warning = lambda *a, **k: None

    cfg = Munch.fromDict(yaml.safe_load(open(cfg_path).read()))
    model = SVGNet(cfg.model)
    load_checkpoint(ckpt, lg, model)
    model.eval()

    ds = build_dataset(cfg.data.test, lg)
    idxs = pick_tiles(ds, n)

    C = cfg.model.semantic_classes + 1
    tp = np.zeros(C); fp = np.zeros(C); fn = np.zeros(C)
    gt_n = np.zeros(C); pr_n = np.zeros(C)
    m_tp = np.zeros(C); m_fp = np.zeros(C); m_fn = np.zeros(C)
    panels = []
    all_scores = []

    for i in idxs:
        d = json.load(open(ds.data_list[i]))
        gt = np.array(d["semanticIds"])
        with torch.no_grad():
            out = model(ds.collate_fn([ds[i]]), return_loss=False)
        scores = out["semantic_scores"].cpu().numpy()[:len(gt)]
        pred = scores.argmax(1)
        conf = scores.max(1)
        all_scores.append((scores, gt))

        for c in range(C):
            tp[c] += ((pred == c) & (gt == c)).sum()
            fp[c] += ((pred == c) & (gt != c)).sum()
            fn[c] += ((pred != c) & (gt == c)).sum()
            gt_n[c] += (gt == c).sum()
            pr_n[c] += (pred == c).sum()
        keep = gt != cfg.model.semantic_classes      # masked view: drop background
        for c in range(C):
            m_tp[c] += ((pred[keep] == c) & (gt[keep] == c)).sum()
            m_fp[c] += ((pred[keep] == c) & (gt[keep] != c)).sum()
            m_fn[c] += ((pred[keep] != c) & (gt[keep] == c)).sum()

        panels.append((draw(d, gt, (0, 150, 0)), draw(d, pred, (200, 0, 0))))

    tag = osp.splitext(osp.basename(ckpt))[0]
    canvas = Image.new("RGB", (SIZE * 2 + 10, len(panels) * (SIZE + 10)), (255, 255, 255))
    for r, (a, b) in enumerate(panels):
        canvas.paste(a, (0, r * (SIZE + 10)))
        canvas.paste(b, (SIZE + 10, r * (SIZE + 10)))
    img_path = osp.join(out_dir, f"{tag}.png")
    canvas.save(img_path)

    def iou(t, f_p, f_n):
        return 100 * t / max(1.0, t + f_p + f_n)

    total = gt_n.sum()
    parts = []
    for c in (0, 1, 2):
        prec = 100 * tp[c] / max(1.0, tp[c] + fp[c])
        rec = 100 * tp[c] / max(1.0, tp[c] + fn[c])
        parts.append(f"{NAMES[c]} masked_iou={iou(m_tp[c], m_fp[c], m_fn[c]):5.1f} "
                     f"iou={iou(tp[c], fp[c], fn[c]):5.1f} P={prec:5.1f} R={rec:5.1f}")
    # Background rejection by confidence.
    #
    # semantic_inference drops the no-object column (svgnet/model/svgnet.py:158),
    # so argmax over the semantic map can never return background — reporting
    # "how often does it predict background" from that map measures nothing.
    # What is meaningful is whether a confidence threshold separates real
    # objects from background: sweep one and report the best mean IoU it buys.
    bg = cfg.model.semantic_classes
    S = np.concatenate([s for s, _ in all_scores], axis=0)
    G = np.concatenate([g for _, g in all_scores], axis=0)
    conf = S.max(1)
    arg = S.argmax(1)

    best = (0.0, None, None)
    for t in np.quantile(conf, np.linspace(0.05, 0.95, 19)):
        p = np.where(conf >= t, arg, bg)
        ious = []
        for c in range(bg):
            i = ((p == c) & (G == c)).sum()
            u = ((p == c) | (G == c)).sum()
            ious.append(100.0 * i / max(1, u))
        if np.mean(ious) > best[0]:
            best = (float(np.mean(ious)), float(t), ious)

    sep = (f"best_thresholded_mIoU={best[0]:5.1f} at conf>={best[1]:.3f} "
           f"(door={best[2][0]:4.1f} window={best[2][1]:4.1f} wall={best[2][2]:4.1f})")
    print(f"{tag} | " + " | ".join(parts) + f" | {sep} | {img_path}", flush=True)


if __name__ == "__main__":
    main()
