#!/usr/bin/env python3
"""Ablation report on fake run directories shaped like the Trainer's output.

    python tools/test_ablation_report.py
"""
import json
import os
import os.path as osp
import sys
import tempfile

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import ablation_report  # noqa: E402

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def fake_run(root, name, pqs, test_pq, door_pq):
    ck = osp.join(root, name, "checkpoint-60000")
    os.makedirs(ck)
    hist = [{"loss": 3.0, "step": 100}] + [{"eval_us_PQ": v, "step": 7500 * (i + 1)} for i, v in enumerate(pqs)]
    json.dump({"log_history": hist, "best_model_checkpoint": ck}, open(osp.join(ck, "trainer_state.json"), "w"))
    os.makedirs(osp.join(root, name, "test"))
    json.dump({"eval_us_PQ": test_pq, "eval_us_strict_PQ": test_pq - 5, "eval_us_class_1_PQ": door_pq,
               "eval_us_class_2_PQ": 0.0}, open(osp.join(root, name, "test", "test_results.json"), "w"))


def main():
    with tempfile.TemporaryDirectory() as d:
        fake_run(d, "A", [20.0, 35.0, 33.0], 31.0, 40.0)
        fake_run(d, "B", [25.0, 41.0, 44.0], 42.5, 55.0)
        text = ablation_report.report(d)
    check("| A | US only | 35.0 | 15000 | 33.0 | 3 |" in text, "validation best, step and last for A")
    check("| B | US + FloorPlanCAD | 44.0 | 22500 | 44.0 | 3 |" in text, "validation for B")
    check("run B (US + FloorPlanCAD), PQ 42.5" in text, "winner on test")
    check("| single door | 40.0 | 55.0 |" in text and "double door" not in text, "per-class rows only where scored")
    check("C |" not in text, "missing runs are left out")
    if FAILURES:
        print(text)
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("ok - ablation report: validation best/last, test table, per-class, winner")
    return 0


if __name__ == "__main__":
    sys.exit(main())
