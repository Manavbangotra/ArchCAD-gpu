#!/usr/bin/env bash
# First thing to run on a rented GPU machine, before any paid long run.
# Checks: CUDA kernels import; the repository's CPU checks; VecFormer trains on a
# small subset (loss falls) and evaluation reports upstream and strict PQ; the
# text-fusion model (real spconv/flash-attn under the stage hook) trains on the
# same subset; the joint product dry run on whichever corpora are present.
#
#   bash cloud/smoke_test.sh [LINE_JSON_ROOT]      # default vecformer/datasets/FloorPlanCAD-sampled-as-line-jsons
#
# Takes ~10 minutes on one A100. Writes to vecformer/outputs/smoke_<time>.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/vecformer"
DATA=${1:-datasets/FloorPlanCAD-sampled-as-line-jsons}
STAMP=$(date +%Y%m%d_%H%M%S)
SUB=outputs/smoke_$STAMP/data
mkdir -p "$SUB/train" "$SUB/val" "$SUB/test"

python - <<'PY'
import torch, flash_attn, spconv.pytorch, torch_scatter
assert torch.cuda.is_available(), "no CUDA device"
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpus", torch.cuda.device_count(),
      "|", torch.cuda.get_device_name(0), "| flash_attn", flash_attn.__version__)
PY

(cd "$ROOT" && python tools/run_checks.py)

for split in train val; do
    n=$([ "$split" = train ] && echo 64 || echo 16)
    ls "$DATA/$split" | head -n "$n" | while read -r f; do ln -sf "$(realpath "$DATA/$split/$f")" "$SUB/$split/$f"; done
done
cp -r "$SUB/val/." "$SUB/test/"

cat > "outputs/smoke_$STAMP/data.yaml" <<YAML
dataset_name: floorplancad
dataset_args:
    root_dir: $SUB
    use_text: ${USE_TEXT:-false}
    train_transform_args:
        random_vertical_flip: 0.5
        random_horizontal_flip: 0.5
        random_rotate: false
        random_scale: [0.8, 1.2]
        random_translation: [0.1, 0.1]
    eval_transform_args:
        random_vertical_flip: 0.0
        random_horizontal_flip: 0.0
        random_rotate: false
        random_scale: [1.0, 1.0]
        random_translation: [0.0, 0.0]
YAML

export PYTHONPATH=$(pwd):${PYTHONPATH:-}
torchrun --nproc_per_node=1 --master_port=11499 launch.py \
    --launch_mode train \
    --config_path configs/vecformer.yaml \
    --model_args_path configs/model/vecformer.yaml \
    --data_args_path "outputs/smoke_$STAMP/data.yaml" \
    --run_name smoke \
    --output_dir "outputs/smoke_$STAMP/run" \
    --max_steps 400 --eval_steps 200 --logging_steps 20 --warmup_ratio 0.0 \
    --dataloader_num_workers 4 --save_total_limit 1 \
    2>&1 | tee "outputs/smoke_$STAMP/train.log"

echo
echo "== text fusion model, 100 steps =="
sed -i "s/use_text: .*/use_text: true/" "outputs/smoke_$STAMP/data.yaml"
torchrun --nproc_per_node=1 --master_port=11498 launch.py     --launch_mode train     --config_path configs/textcad.yaml     --model_args_path configs/model/textcad.yaml     --data_args_path "outputs/smoke_$STAMP/data.yaml"     --run_name smoke_text     --output_dir "outputs/smoke_$STAMP/run_text"     --max_steps 100 --eval_steps 100 --logging_steps 20 --warmup_ratio 0.0     --dataloader_num_workers 4 --save_total_limit 1     2>&1 | tee "outputs/smoke_$STAMP/train_text.log"

echo
echo "== joint product dry run (sources present on disk) =="
python ../cloud/dryrun_joint.py || echo "dry run failed or no joint corpora present"

echo
echo "== check =="
grep -E "'loss'" "outputs/smoke_$STAMP/train.log" | head -2
grep -E "'loss'" "outputs/smoke_$STAMP/train.log" | tail -2
grep -Eo "'eval_(strict_)?PQ': [0-9.]+" "outputs/smoke_$STAMP/train.log" | tail -4
grep -Eo "'text_l0': [0-9.]+" "outputs/smoke_$STAMP/train_text.log" | tail -1
echo "Pass if: loss at the end is clearly below the start, eval_PQ / eval_strict_PQ are printed,"
echo "the text run logs text_l0 (FloorPlanCAD-V2 data; V1 has no text, then text_l0 is absent), and the dry run says ok."
