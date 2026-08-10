#!/usr/bin/env bash
#
# Unattended training that survives power cuts.
#
# Picks up from the newest checkpoint if one exists, otherwise starts fresh, and
# restarts itself if the process dies. Intended to be launched automatically at
# boot (see WINDOWS_SETUP.md), so that an outage costs only the time the machine
# was actually off.
#
#   CONFIG=configs/svg/svg_pointT_fpcad_3060.yaml ./tools/train_resume.sh
#
# Environment:
#   CONFIG        config yaml                  (default: 3060 config)
#   EXP_NAME      experiment name              (default: fpcad_3060)
#   WORK_DIR      checkpoint/log directory     (default: work_dirs/<EXP_NAME>)
#   MAX_RETRIES   restarts after a crash       (default: 5; 0 = never give up)
#   LOG           log file                     (default: <WORK_DIR>/resume.log)

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

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
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

CONFIG=${CONFIG:-configs/svg/svg_pointT_fpcad_3060.yaml}
EXP_NAME=${EXP_NAME:-fpcad_3060}
WORK_DIR=${WORK_DIR:-work_dirs/$EXP_NAME}
MAX_RETRIES=${MAX_RETRIES:-5}
LOG=${LOG:-$WORK_DIR/resume.log}

mkdir -p "$WORK_DIR"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Prefer latest.pth; fall back to the previous one if a crash mid-write left the
# newest file unreadable. checkpoint_save writes atomically, so this is belt and
# braces rather than an expected path.
pick_checkpoint() {
    for candidate in "$WORK_DIR/latest.pth" "$WORK_DIR/latest_prev.pth"; do
        if [ -s "$candidate" ]; then
            if "$PYTHON" - "$candidate" <<'EOF' >/dev/null 2>&1
import sys, torch
torch.load(sys.argv[1], map_location="cpu", weights_only=False)
EOF
            then
                echo "$candidate"
                return
            fi
            say "checkpoint $candidate is unreadable, trying older one"
        fi
    done
    echo ""
}

# A single instance only — a boot-triggered task must not stack on a run that is
# already going.
LOCK="$WORK_DIR/.training.lock"
if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK"
    if ! flock -n 9; then
        say "training already running (lock held); nothing to do"
        exit 0
    fi
fi

attempt=0
while : ; do
    attempt=$((attempt + 1))
    CKPT=$(pick_checkpoint)

    if [ -n "$CKPT" ]; then
        say "attempt $attempt: resuming from $CKPT"
        RESUME_ARG=(--resume "$CKPT")
    else
        say "attempt $attempt: no checkpoint found, starting from scratch"
        RESUME_ARG=()
    fi

    "$PYTHON" tools/train.py "$CONFIG" \
        --exp_name "$EXP_NAME" \
        --work_dir "$WORK_DIR" \
        "${RESUME_ARG[@]}" 2>&1 | tee -a "$LOG"

    status=${PIPESTATUS[0]}
    if [ "$status" -eq 0 ]; then
        say "training finished successfully"
        exit 0
    fi

    say "training exited with status $status"
    if [ "$MAX_RETRIES" -ne 0 ] && [ "$attempt" -ge "$MAX_RETRIES" ]; then
        say "giving up after $attempt attempts — see $LOG"
        exit "$status"
    fi

    say "restarting in 30s ..."
    sleep 30
done
