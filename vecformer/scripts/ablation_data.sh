#!/usr/bin/env bash
# Data ablation: which training mix gives the best US model for the same GPU budget?
#   A  US plan windows only
#   B  US + FloorPlanCAD-V2
#   C  US + FloorPlanCAD-V2 + CubiCasa5K
# Each run: same model from scratch, same steps, best checkpoint on 3 held-out US
# projects, then one evaluation on the US test split. Summary: tools/ablation_report.py.
#
#   cd vecformer && bash scripts/ablation_data.sh                 # A, B, C in turn (8 GPUs)
#   RUNS="A B" bash scripts/ablation_data.sh                      # a subset
#   MAX_STEPS=20000 RUNS="A B C" bash scripts/ablation_data.sh    # a cheaper trend check
#   LOCAL=1 RUNS=A bash scripts/ablation_data.sh                  # one 12 GB GPU (Windows ok)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
OUT=${OUT:-outputs/ablation_data}
RUNS=${RUNS:-"A B C"}

BASE_CFG=configs/ablation_data.yaml
MODEL_CFG=configs/model/ablation_arch43.yaml
LAUNCH=(torchrun --nproc_per_node="$NPROC_PER_NODE")
if [ -n "${LOCAL:-}" ]; then
    BASE_CFG=configs/ablation_data_local.yaml
    MODEL_CFG=configs/model/ablation_arch43_local.yaml
    LAUNCH=(python)                                   # single process; launch.py sets up gloo
fi
TRAIN_CFG=$BASE_CFG
mkdir -p "$OUT"
if [ -n "${MAX_STEPS:-}" ]; then
    # a shorter budget keeps warmup 5%, decay over the last 20% and 8 evaluations
    TRAIN_CFG=$OUT/ablation_data_${MAX_STEPS}.yaml
    python - "$MAX_STEPS" "$TRAIN_CFG" "$BASE_CFG" <<'PY'
import sys, yaml
steps, out = int(sys.argv[1]), sys.argv[2]
cfg = yaml.safe_load(open(sys.argv[3]))
cfg.update(max_steps=steps, eval_steps=max(steps // 8, 1))
cfg["lr_scheduler_kwargs"]["num_decay_steps"] = steps // 5
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
PY
fi

declare -A DATA=([A]=ablation_A_us [B]=ablation_B_us_fpcad [C]=ablation_C_all)
need() { [ -d "$1/train" ] || { echo "missing corpus $1 (see cloud/RUNBOOK.md section 2)"; exit 1; }; }
need ../dataset/us_plans/lines
for r in $RUNS; do
    case $r in B|C) need datasets/FloorPlanCAD-V2-textcad ;; esac
    case $r in C) need ../dataset/cubicasa5k/lines_arch43 ;; esac
done

for r in $RUNS; do
    data=configs/data/${DATA[$r]}.yaml
    run=$OUT/$r
    echo "== run $r: $data -> $run"
    python ../cloud/dryrun_joint.py --data "$data" --model "$MODEL_CFG"
    MASTER_PORT=$((29500 + RANDOM % 100)) "${LAUNCH[@]}" launch.py --launch_mode train --config_path "$TRAIN_CFG" --model_args_path "$MODEL_CFG" --data_args_path "$data" --run_name "ablation_$r" --save_total_limit 2 --output_dir "$run" 2>&1 | tee "$OUT/$r.train.log"
    best=$(python - "$run" <<'PY'
import glob, json, os, sys
states = sorted(glob.glob(os.path.join(sys.argv[1], "checkpoint-*", "trainer_state.json")), key=os.path.getmtime)
print(json.load(open(states[-1]))["best_model_checkpoint"] if states else "")
PY
)
    [ -n "$best" ] || { echo "run $r: no best checkpoint found"; exit 1; }
    echo "== run $r: testing $best on the US test split"
    MASTER_PORT=$((29600 + RANDOM % 100)) "${LAUNCH[@]}" launch.py --launch_mode test --resume_from_checkpoint "$best" --config_path "$TRAIN_CFG" --model_args_path "$MODEL_CFG" --data_args_path "$data" --run_name "ablation_${r}_test" --output_dir "$run/test" 2>&1 | tee "$OUT/$r.test.log"
done
python ../tools/ablation_report.py "$OUT"
