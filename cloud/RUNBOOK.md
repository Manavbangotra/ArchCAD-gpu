# Cloud runbook: training the line-based symbol spotter

Plan: `C:\Users\Manav Bangotra\.claude\plans\i-have-few-concerns-rosy-cook.md` (phases 0-6).
Code: `vecformer/` (VecFormer, NeurIPS 2025, with local changes listed by `git log -- vecformer/`).

## 0. Machine
- **Reproduction runs (Phase 2-3):** 8x A100 80GB (or 8x H100), the papers' setting. Batch 2 per GPU.
- **Development / smoke / ablations:** 1x A100 or L40S is enough.
- Linux, NVIDIA driver supporting CUDA 11.8+, >= 500 GB local disk, >= 64 GB shared memory for data loaders.

## 1. Environment (either)
```bash
git clone <this repo> archcad && cd archcad
docker build -t archcad-vecformer -f cloud/Dockerfile .        # or:
bash cloud/setup_vm.sh && conda activate vecformer
```

## 2. Data
FloorPlanCAD as used by VecFormer (official split 6,965 / 810 / 3,827, attribute names semanticId/instanceId):
```bash
cd vecformer && mkdir -p datasets && cd datasets
gdown "https://drive.google.com/uc?id=1wsOQxIXjsqYzMlUpPNRjyQiMnwgVbtJG" && unzip FloorPlanCAD.zip && cd ..
export PYTHONPATH=$(pwd)
python data/floorplancad/preprocess.py --input_dir=$(pwd)/datasets/FloorPlanCAD \
    --output_dir=$(pwd)/datasets/FloorPlanCAD-sampled-as-line-jsons --dynamic_sampling --connect_lines \
    --max_workers $(nproc) --use_progress_bar
for s in train val test; do echo $s $(ls datasets/FloorPlanCAD-sampled-as-line-jsons/$s | wc -l); done   # expect 6965 810 3827
```
FloorPlanCAD-V2 (with text, for the TextCAD stage) is the release under `dataset/FloorplanCAD/` on the
development machine; the preprocessor now reads its `semantic-id` spelling and keeps `<text>` as `texts`.

## 3. Smoke test (must pass before paying for a long run)
```bash
bash cloud/smoke_test.sh
```

## 4. Phase 2 - reproduce VecFormer
```bash
cd vecformer
OUTPUT_DIR=outputs NPROC_PER_NODE=8 bash scripts/train.sh                              # released augmentation
# second run, paper augmentation:
torchrun --nproc_per_node=8 launch.py --launch_mode train --config_path configs/vecformer.yaml \
    --model_args_path configs/model/vecformer.yaml --data_args_path configs/data/floorplancad_paper.yaml \
    --run_name vecformer_paperaug --save_total_limit 5 --output_dir outputs/paperaug
```
Watch in TensorBoard (`outputs/*/runs`): `eval_PQ`, `eval_strict_PQ`, `eval_F1`, thing/stuff PQ.
Health checks: semantic F1 rising within the first ~5k steps; PQ > 0 and rising by ~20k.
The default model config has the layer prior on (`use_layer_fusion`, layer id as z), i.e. the paper's
91.1 setting. **Gate:** test PQ >= 89.5 (paper 91.1); the no-prior ablation (paper 88.4) comes later. Test:
```bash
bash scripts/test.sh      # edit the checkpoint path first
```

## 5. Checkpoints off the machine
Credentials from the environment only (rotate the keys that were pasted in chat earlier):
```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=us-east-1
aws s3 sync vecformer/outputs s3://<bucket>/training/vecformer/ --exclude "*/data/*"
```
