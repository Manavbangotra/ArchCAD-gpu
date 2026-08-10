# Training on one RTX 3060 (12GB)

Doors, windows and walls on CubiCasa5K. This repo is the CPU-verified pipeline
with the GPU-only pieces added back.

Everything here has been run and checked on CPU first, so what remains is the
training itself.

---

## 1. Setup (WSL2 or Linux)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-cuda.txt
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Must print `True NVIDIA GeForce RTX 3060`. `WINDOWS_SETUP.md` covers WSL2 from
scratch, including the driver mistake that costs people an afternoon.

## 2. Data

```bash
# CubiCasa5K, 5.5GB, from Zenodo
python dataset/parse_cubicasa.py \
    --data_dir dataset/cubicasa/cubicasa5k \
    --split_file dataset/cubicasa/cubicasa5k/train.txt \
    --output_dir dataset/cubicasa/json3/train --require_labels
# ...and again with val.txt -> json3/test
```

4,199 train + 400 test convert cleanly.

## 3. Train

```bash
CONFIG=configs/svg/svg_pointT_cc_3060.yaml bash tools/train_single.sh
```

Expect roughly **10 minutes per epoch** over all 4,199 drawings, so a 60-epoch
run finishes overnight. `tools/train_resume.sh` instead if the machine may lose
power — it resumes from the newest good checkpoint.

**Check the first epoch before walking away:** `Running on cuda`,
`effective batch 16 = 2 x 1 gpu x 8 accum`, loss falling, and `nvidia-smi`
showing ~6GB. If seconds-per-step is far worse than ~0.15, see *pointops* below.

---

## What changed from the CPU repo, and why

### Class weighting — the fix for the failure we actually observed

On CPU the model drifted toward walls: door IoU fell **37.5 → 25.5** in one
epoch while wall rose **60.2 → 68.0**. Walls are 58.6% of labelled primitives
against 22.0% door and 19.4% window, so ignoring doors was simply the cheaper
option.

`criterion.class_weights: [1.121, 1.192, 0.687]` — sqrt-inverse frequency from
the training split, normalised to mean 1. Gentle enough not to destabilise the
loss, firm enough that dropping doors costs something.

**This is the change most likely to matter. Watch door IoU specifically.**

### Overlapping tiles

`parse_pdf_plans.py --overlap 0.15` grows each tile so neighbours share a
margin. Hard block edges cut objects in half — a door on a boundary becomes two
partial doors and is learnable as neither. Measured on one plan set, door
primitives went 5,933 → 11,448 because edge objects now appear whole somewhere.

CADSpotting reports **+16.7 PQ** for overlapping windows over block
partitioning. Costs ~1.7x larger tiles; affordable on a GPU, not on CPU.

Only affects PDF plan sets. CubiCasa drawings are small enough not to tile.

### Full settings restored

| | CPU | Here |
|---|---|---|
| Image size | 192 | **700** |
| Queries | 50 | **800** |
| Drawings | 2,000 | **all 4,199** |
| Epochs | 12 | **60** |
| Batch | 1 | **2 x 8 accum = 16** |
| fp16 | off | **on** |

Measured memory, fp32 batch 1: 700px/800q = **6.0GB**, 980px/800q = 7.8GB,
980px batch 2 ≈ 14GB (does not fit). fp16 roughly halves activations, so 700px
at batch 2 is comfortable in 12GB.

### Text extraction — prepared, deliberately not used

`parse_cubicasa.py --extract_text` stores the words printed on each drawing.

TextCAD (arXiv:2607.12678) reaches 96.53 PQ on CubiCasa partly through text, so
the data is worth having. **But measured on this corpus it will not help this
task:** across 5,778 sampled text items, "door" and "window" appear **zero
times**. CubiCasa's words are room names, dimensions and fixture codes — `CL`,
`CB`, `SINK`, `WC`. TextCAD's gain comes largely from room and fixture classes,
which this 3-class taxonomy does not have.

It is left off rather than wired in. On US construction drawings, which carry
`D1`/`W1` marks and door schedules, it would be worth revisiting.

### Layer fusion stays off

`model.use_layer_fusion: False`. Pooling features per CAD layer helps only when
layers are an independent hint available at inference. Neither held: labels
derived from layer names made the grouping a perfect giveaway (predicting each
layer group's majority class scored **100.0%**), and the flattened drawings this
targets carry exactly one layer. See `dataset/taxonomy.py`.

---

## What to expect

Published results on CubiCasa: TextCAD **96.5**, VecFormer **94.6** — both with
far more compute (TextCAD: 4x RTX A6000, 700 epochs).

Realistic here, one 3060, ~60 epochs, 3 classes: **70-85 PQ**. 90 is possible
but should not be assumed. The gap is architectural — this codebase is DPSS
(86.2 on FloorPlanCAD's 35 classes), a generation behind VecFormer (88.1).

If 90+ is a hard requirement, the honest route is
[VecFormer](https://github.com/WesKwong/VecFormer) — published 88.1, Apache-2.0,
line-based, and the only method in this area with released code. No method in
this field publishes trained weights; every route requires training yourself.

## Judge it on the right numbers

- **Door and window IoU**, not overall accuracy. Background is 74-89% of
  primitives, so predicting background everywhere already scores ~88%.
- **thing PQ separately from stuff PQ.** On CPU, overall PQ rose to 5.0 while
  thing PQ (doors and windows, counted individually) stayed at 0.1 — the gain
  was almost entirely walls. One number hid that.
- `tools/epoch_probe.py` reports both, plus truth-vs-prediction images on fixed
  held-out drawings so epochs are directly comparable.

## pointops

`modules/pointops/` is a pure-PyTorch stand-in for a CUDA extension. It runs on
GPU unmodified. Its farthest-point-sampling loop is Python-level and may
bottleneck — measure seconds-per-step first, and only compile the real extension
if the number justifies it.

## Licence

CubiCasa5K is **CC BY-NC 4.0**. Models trained from it are research-only and
must not ship commercially. Keep those checkpoints separate from anything
trained solely on your own data.
