#!/usr/bin/env bash
#
# CPU inference demo on FloorPlanCAD.
#
# Unlike tools/demo.sh (which targets an 8-GPU SLURM node), this runs on a
# single CPU. It is slow by design — the point is to exercise the full
# pipeline without a GPU.
#
# Prerequisites:
#   python dataset/download_data.py --split test
#   python dataset/parse_FpCAD_svg.py --data_dir <svg_dir> \
#          --output_dir dataset/FloorplanCAD/json/test --render
#
# NOTE: no trained DPSS checkpoint has been published. Without one the model
# runs with random weights, so predictions are meaningless — this verifies the
# plumbing, not the accuracy.

set -euo pipefail

export PYTHONPATH=./
export ARCHCAD_DEVICE=cpu

# Prefer a project-local virtualenv, then $PYTHON, then whatever is on PATH.
if [ -z "${PYTHON:-}" ]; then
    if [ -x .venv/bin/python ]; then
        PYTHON=.venv/bin/python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    else
        PYTHON=python
    fi
fi

CONFIG=${CONFIG:-configs/svg/svg_pointT_fpcad_cpu.yaml}
CHECKPOINT=${CHECKPOINT:-model.pth}
DATADIR=${DATADIR:-dataset/FloorplanCAD/json/test}
OUT=${OUT:-outputs}

mkdir -p "$OUT"

"$PYTHON" tools/inference.py \
    "$CONFIG" \
    "$CHECKPOINT" \
    --datadir "$DATADIR" \
    --out "$OUT"
