#!/usr/bin/env python3
"""Summarise the data ablation (vecformer/scripts/ablation_data.sh) as a Markdown table.

    python tools/ablation_report.py vecformer/outputs/ablation_data [--out report.md]

For each run directory (A, B, C) it reads:
  - the evaluation history on the held-out US projects (trainer_state.json log_history):
    best eval_us_PQ and the step it was reached, and the last value;
  - the one-off US test evaluation of the best checkpoint (<run>/test/test_results.json).
Per-class PQ is listed for every class some run scores above zero, by class name.
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(ROOT, "dataset"))

RUN_NAMES = {"A": "US only", "B": "US + FloorPlanCAD", "C": "US + FloorPlanCAD + CubiCasa"}
HEADLINE = [("PQ", "PQ"), ("strict_PQ", "strict PQ"), ("thing_PQ", "thing PQ"), ("stuff_PQ", "stuff PQ"),
            ("RQ", "RQ"), ("SQ", "SQ"), ("F1", "F1"), ("wF1", "wF1")]


def load_run(run_dir):
    states = sorted(glob.glob(osp.join(run_dir, "checkpoint-*", "trainer_state.json")), key=osp.getmtime)
    history = json.load(open(states[-1]))["log_history"] if states else []
    evals = [h for h in history if "eval_us_PQ" in h]
    best = max(evals, key=lambda h: h["eval_us_PQ"]) if evals else None
    test_path = osp.join(run_dir, "test", "test_results.json")
    test = json.load(open(test_path)) if osp.isfile(test_path) else None
    return dict(evals=evals, best=best, last=evals[-1] if evals else None, test=test)


def fmt(v):
    return "—" if v is None else f"{v:.1f}"


def report(out_dir):
    import taxonomy as tx
    runs = {r: load_run(osp.join(out_dir, r)) for r in ("A", "B", "C") if osp.isdir(osp.join(out_dir, r))}
    if not runs:
        return f"No runs found under {out_dir}.\n"
    lines = ["# Data ablation: which training mix gives the best US model?", ""]
    lines += ["All runs: same model from scratch, same number of steps, scored on the same held-out US "
              "projects (validation) and the US test split.", ""]

    lines += ["## Validation (held-out US projects)", "",
              "| Run | Mix | Best US PQ | at step | Last US PQ | Evaluations |", "|---|---|---|---|---|---|"]
    for r, d in runs.items():
        b, l = d["best"], d["last"]
        lines.append(f"| {r} | {RUN_NAMES[r]} | {fmt(b and b['eval_us_PQ'])} | {b['step'] if b else '—'} | "
                     f"{fmt(l and l['eval_us_PQ'])} | {len(d['evals'])} |")

    lines += ["", "## Test (best checkpoint, US test split)", "",
              "| Run | " + " | ".join(h for _, h in HEADLINE) + " |", "|---|" + "---|" * len(HEADLINE)]
    for r, d in runs.items():
        t = d["test"] or {}
        lines.append(f"| {r} | " + " | ".join(fmt(t.get(f"eval_us_{k}")) for k, _ in HEADLINE) + " |")

    classes = []
    for c in range(tx.ARCH_NUM_CLASSES):
        vals = [(d["test"] or {}).get(f"eval_us_class_{c + 1}_PQ") for d in runs.values()]
        if any(v for v in vals if v):
            classes.append((c, vals))
    if classes:
        lines += ["", "## Test PQ per class (classes any run scores)", "",
                  "| Class | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
        for c, vals in classes:
            lines.append(f"| {tx.ARCH_NAMES[c]} | " + " | ".join(fmt(v) for v in vals) + " |")

    scored = {r: d["test"].get("eval_us_PQ") for r, d in runs.items() if d["test"]}
    if len(scored) >= 2:
        winner = max(scored, key=scored.get)
        lines += ["", f"**Best on US test: run {winner} ({RUN_NAMES[winner]}), PQ {scored[winner]:.1f}.** "
                  "A gap under ~1 PQ is within run-to-run noise; repeat with another seed before deciding on it."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--out", help="also write the report here")
    a = ap.parse_args()
    text = report(a.out_dir)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
