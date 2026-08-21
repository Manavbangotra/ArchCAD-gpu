# Runbook: training door/window/wall on your own drawings

Everything up to the training itself has been prepared and checked on CPU. What
is left is two training runs on the 3060. This is the exact sequence, plus what
to look at while it runs and when to stop it.

Read `GPU_TRAINING.md` alongside this — it explains *why* the settings are what
they are. This file is the *what to type*.

---

## 0. Setup, once

```bash
cd /path/to/ArchCAD-gpu
python3 -m venv .venv
.venv/bin/pip install -r requirements-cuda.txt
.venv/bin/python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print `True NVIDIA GeForce RTX 3060`. `WINDOWS_SETUP.md` covers WSL2 from
scratch if that line prints `False`.

The data directories (`dataset/cubicasa`, `dataset/us_plans`) must come across
too — they are the converted JSON, not the raw PDFs.

---

## 1. Stage 1 — CubiCasa5K

```bash
CONFIG=configs/svg/svg_pointT_cc_3060.yaml bash tools/train_single.sh
```

Roughly 10 min/epoch over 4,194 drawings, 60 epochs, so about overnight.
Use `tools/train_resume.sh` instead if the machine may sleep or lose power.

**Check the first epoch before walking away.** You want to see:

- `Running on cuda`
- `effective batch 16 = 2 x 1 gpu x 8 accum`
- loss falling
- `nvidia-smi` showing ~6 GB of 12 GB
- **no** `N of M batches skipped` warnings (a new message; if batches are being
  skipped in bulk, something is wrong with the data, not the model)

If seconds-per-step is far above ~0.15, the Python-level farthest-point-sampling
loop in `modules/pointops/` is the bottleneck; compiling the real CUDA extension
is then worth the time.

## 2. Stage 2 — your US drawings

```bash
CONFIG=configs/svg/svg_pointT_us_3060.yaml bash tools/train_single.sh
```

This config already points `pretrain:` at `./work_dirs/cc_3060/best.pth`, so
stage 1 must finish first. `tools/train.py` loads it non-strict and skips
size-mismatched tensors, so the backbone transfers cleanly.

---

## 3. What to watch — and what to ignore

**Ignore overall accuracy.** Background is 87.6% of primitives in the US corpus,
so a model that predicts background everywhere already scores ~88%. It means
nothing.

Watch these, per epoch:

| Number | Why |
|---|---|
| **door IoU**, **window IoU** | separately. The failure mode is these falling while wall rises |
| **thing PQ** vs **stuff PQ** | thing = doors+windows counted individually; stuff = walls. The earlier run reached PQ 5.0 with thing PQ 0.1 — one number hid that completely |
| batches skipped | should be zero or near it |

```bash
bash tools/watch_epochs.sh            # live per-epoch table
python tools/epoch_probe.py --help    # truth-vs-prediction images on fixed sheets
```

**The collapse signature to abort on**: door IoU rising for 2-3 epochs and then
falling steadily while wall IoU climbs. That is the model discovering that
ignoring doors is cheaper. It is what happened before (door IoU 37.5 → 2.5 over
8 epochs). `class_weights` in both configs is the countermeasure; if it happens
anyway, raise the door and window weights and restart rather than waiting.

Realistic target, one 3060, 3 classes: **70-85 PQ**. Published work reaches
94.6 (VecFormer) and 96.5 (TextCAD) on CubiCasa with far more compute.

---

## 4. Then test it on a real sheet

```bash
python tools/inference.py configs/svg/svg_pointT_us_3060.yaml \
    work_dirs/us_3060/best.pth --datadir <a held-out project's tiles>
```

For a whole sheet rather than tiles, the tiling pipeline in the SymPointV2
checkout does the windowing and merging:

```bash
python /var/www/html/SymPointV2/tools/tile_plan.py sheet.svg \
    configs/svg/svg_pointT_us_3060.yaml work_dirs/us_3060/best.pth \
    --plan-crop <x0,y0,x1,y1> --scales 34:20 26:16 16:11 --out out/
```

The baseline to beat, measured with the original FloorPlanCAD weights on
Country Meadows A1.02: **~28% of primitives usefully labelled, ~52% door recall**.

---

## 5. Honest caveats to carry into any result

- **The labels are a heuristic.** They come from AIA CAD layer names in the PDF's
  optional content groups. The test split is by *project*, so no building is in
  both sets, but both sets are labelled the same way — the metrics measure
  agreement with the layer heuristic, not with ground truth. A systematic
  layer-naming error would be invisible.
- **Layer fusion stays off.** With labels derived from layers, predicting each
  layer group's majority class already scores 100%, so feeding layer identity to
  the model would be handing over the answer.
- **Licence.** CubiCasa5K is CC BY-NC 4.0. Any checkpoint warm-started from
  stage 1 is research-only. For something shippable, train stage 2 from scratch
  (`pretrain: ''`) and expect it to need more epochs.
- **Only 19 of 59 plansets are usable.** The rest have no optional-content
  groups, so there are no free labels in them — they are not bad drawings, just
  flattened on export.
