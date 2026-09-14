#!/usr/bin/env python3
"""Strict vs upstream panoptic quality on hand-made cases.

    python vecformer/checks/test_strict_pq.py

Loads the evaluator module by path, so it runs without flash-attn / spconv.
"""

import importlib.util
import os.path as osp
import sys

import torch

HERE = osp.dirname(osp.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "evaluator", osp.join(HERE, "..", "model", "vecformer", "evaluator", "evaluator.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def run(pred_masks, pred_labels, tgt_masks, tgt_labels, n, num_classes=3, ignore=3):
    e = ev.Evaluator(ev.EvaluatorConfig(num_classes=num_classes, ignore_label=ignore,
                                        iou_threshold=0.5, output_dir=""))
    preds = {"pred_masks": [torch.tensor(pred_masks, dtype=torch.bool).reshape(-1, n)],
             "pred_labels": [torch.tensor(pred_labels, dtype=torch.long)]}
    tg = {"target_masks": [torch.tensor(tgt_masks, dtype=torch.bool).reshape(-1, n)],
          "target_labels": [torch.tensor(tgt_labels, dtype=torch.long)],
          "prim_lens": [torch.ones(n)]}
    states = e.eval_panoptic_quality(preds, tg)
    mc = ev.MetricsComputer(ev.MetricsComputerConfig(num_classes=num_classes,
                                                     thing_class_idxs=[0, 1], stuff_class_idxs=[2]))
    mc._update_metric_states({k: v.tolist() for k, v in states.items()})
    return states, mc._compute_panoptic_quality()


def main():
    # 1. A perfect match: both definitions give PQ 100.
    s, m = run([[1, 1, 0, 0]], [0], [[1, 1, 0, 0]], [0], 4)
    check(abs(m["PQ"] - 100) < 1e-3 and abs(m["strict_PQ"] - 100) < 1e-3, f"perfect match: {m['PQ']}, {m['strict_PQ']}")

    # 2. A correct object plus a hallucinated one overlapping nothing.
    #    Upstream ignores the hallucination (PQ 100); strict counts a FP: RQ = 1/(1+0.5).
    s, m = run([[1, 1, 0, 0], [0, 0, 1, 1]], [0, 1], [[1, 1, 0, 0]], [0], 4)
    check(abs(m["PQ"] - 100) < 1e-3, f"upstream ignores unmatched prediction: {m['PQ']}")
    check(abs(m["strict_PQ"] - 100 / 1.5) < 1e-3, f"strict counts it: {m['strict_PQ']}")

    # 3. Right mask, wrong class: upstream FP only (no FN); strict FP + FN.
    s, m = run([[1, 1, 0, 0]], [1], [[1, 1, 0, 0]], [0], 4)
    check(int(s["fn_per_class"].sum()) == 0 and int(s["fp_per_class"].sum()) == 1, "upstream: FP, no FN")
    check(int(s["strict_fn_per_class"].sum()) == 1 and int(s["strict_fp_per_class"].sum()) == 1, "strict: FP and FN")

    # 4. Ignore-label predictions and targets are excluded from both.
    s, m = run([[1, 1, 0, 0], [0, 0, 1, 1]], [0, 3], [[1, 1, 0, 0], [0, 0, 1, 1]], [0, 3], 4)
    check(abs(m["strict_PQ"] - 100) < 1e-3, f"ignore label excluded: {m['strict_PQ']}")

    # 5. IoU at exactly 0.5 is not a match (strictly greater, as upstream).
    s, m = run([[1, 1, 0, 0]], [0], [[1, 1, 1, 1]], [0], 4)
    check(int(s["strict_tp_per_class"].sum()) == 0, "IoU 0.5 is not a match")

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("ok - strict and upstream PQ behave as documented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
