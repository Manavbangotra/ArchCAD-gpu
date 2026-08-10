#!/usr/bin/env bash
#
# Emit one line per finished epoch: the run's own metrics, plus a probe of the
# current checkpoint on fixed held-out sheets (tools/epoch_probe.py).
#
# Also emits on failure. A watcher that only reports good news is
# indistinguishable from a watcher whose job died.
#
#   LOG=<train log> WORK=<work dir> CONFIG=<cfg> EPOCHS=20 ./tools/watch_epochs.sh

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-.venv/bin/python}
WORK=${WORK:-work_dirs/us_cpu}
CONFIG=${CONFIG:-configs/svg/svg_pointT_us_cpu.yaml}
EPOCHS=${EPOCHS:-20}
LOG=${LOG:-$(ls -t $WORK/2*.log 2>/dev/null | head -1)}
PROBES=$WORK/probes
mkdir -p "$PROBES"

seen=0
quiet=0
while [ "$seen" -lt "$EPOCHS" ]; do
    sleep 60

    if ! pgrep -f "tools/train.py" >/dev/null 2>&1; then
        # Distinguish "finished" from "died" — both stop the process.
        if [ "$seen" -ge "$EPOCHS" ]; then
            echo "TRAINING COMPLETE after $seen epochs"
        else
            echo "TRAINING STOPPED at epoch $seen of $EPOCHS — process gone (check $WORK/resume.log)"
        fi
        exit 0
    fi

    # `grep -c` prints 0 and exits 1 when there are no matches, so a `|| echo 0`
    # fallback appends a second line and the later [ ] comparison sees "0\n0".
    n=$(grep -cE "mIoU / " "$LOG" 2>/dev/null | head -1)
    n=${n:-0}
    if [ "$n" -le "$seen" ]; then
        quiet=$((quiet + 1))
        # ~2h with no epoch boundary means something is wrong, not just slow.
        if [ "$quiet" -ge 120 ]; then
            echo "WARNING: no epoch completed in ~2h (still at $seen) — training may be stuck"
            quiet=0
        fi
        continue
    fi
    quiet=0
    seen=$n

    metrics=$(grep -E "Class_door  IoU|Class_window  IoU|Class_wall  IoU" "$LOG" | tail -3 \
              | sed 's/.*Class_//; s/  IoU: /=/' | tr '\n' ' ')
    pq=$(grep -E "^.*PQ / RQ / SQ" "$LOG" | tail -3 | head -1 | sed 's/.*: //')

    cp -f "$WORK/latest.pth" "$WORK/.probe_e${seen}.pth" 2>/dev/null || continue
    probe=$(OMP_NUM_THREADS=1 ARCHCAD_DEVICE=cpu PYTHONPATH=./ \
            $PYTHON tools/epoch_probe.py "$CONFIG" "$WORK/.probe_e${seen}.pth" "$PROBES" 3 2>/dev/null \
            | tail -1)
    rm -f "$WORK/.probe_e${seen}.pth"

    echo "EPOCH $seen/$EPOCHS | official: $metrics| PQ $pq | probe: ${probe#*| }"
done

echo "TRAINING COMPLETE after $seen epochs"
