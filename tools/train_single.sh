#!/usr/bin/env bash
#
# Single-GPU training (e.g. one RTX 3060 12GB).
#
# tools/train_dist.sh targets 8 GPUs and, as shipped, is missing line
# continuations after `torchrun`. This script runs train.py directly — no
# torchrun, no --dist, no --sync_bn (which is meaningless on one device).
#
# Memory: measured peaks for one training step at batch size 1, fp32, on the
# largest drawing in the FloorPlanCAD test split (7,392 primitives):
#
#     img_size 980, 800 queries  ~7.8 GB
#     img_size 512, 400 queries  ~3.8 GB
#     img_size 384, 200 queries  ~2.0 GB
#
# Add ~1.6 GB for parameters, gradients and AdamW state. Enabling fp16 in the
# config roughly halves the activation share.

set -euo pipefail

if [ -z "${PYTHON:-}" ]; then
    if [ -x .venv/bin/python ]; then
        PYTHON=.venv/bin/python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    else
        PYTHON=python
    fi
fi

export PYTHONPATH=./

CONFIG=${CONFIG:-configs/svg/svg_pointT_fpcad_12gb.yaml}
EXP_NAME=${EXP_NAME:-fpcad_single}
EXTRA=${EXTRA:-}

# Helps with fragmentation when running close to the memory ceiling.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# DETERMINISTIC=1 makes repeat runs bit-identical, at a real cost in speed.
#
# Seeding alone is not enough: thread scheduling changes the order of float
# reductions, and over an epoch that compounded into a 35-point IoU swing here.
# Pinning the thread count removes it — two runs then matched exactly.
#
# Use it when a comparison must be attributable (e.g. measuring whether
# pretraining actually helped). Leave it off for production runs and average
# over 2-3 seeds instead; single-threaded training is far slower.
if [ "${DETERMINISTIC:-0}" = "1" ]; then
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    echo "[train_single] deterministic mode: threads pinned to 1"
fi

"$PYTHON" tools/train.py \
    "$CONFIG" \
    --exp_name "$EXP_NAME" \
    ${EXTRA}
