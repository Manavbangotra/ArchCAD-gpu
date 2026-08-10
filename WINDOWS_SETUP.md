# Training on a Windows PC with an RTX 3060

Step-by-step notes for training DPSS on FloorPlanCAD on a home Windows machine,
including how to survive scheduled power cuts.

We run everything inside **WSL2** (Ubuntu running inside Windows). Your RTX 3060
works there, and every script in this repo runs unchanged.

**Tick the boxes as you go.** If a step fails, stop and fix it — later steps
depend on it.

---

## Part 1 — Install WSL2 and check the GPU

> ⚠️ **The one mistake to avoid:** do **not** install an NVIDIA driver inside
> Ubuntu. The Windows driver already shares the GPU with WSL. Installing a Linux
> driver breaks it, and it is annoying to undo.

- [ ] **1.1** On Windows, install the latest **NVIDIA Game Ready** or **Studio**
  driver from nvidia.com. This is the only driver you need.

- [ ] **1.2** Open **PowerShell as Administrator** and run:
  ```powershell
  wsl --install -d Ubuntu-22.04
  ```
  Reboot when it asks. On first launch, pick a username and password (write them
  down — the password is needed for `sudo`).

- [ ] **1.3** Open Ubuntu and check the GPU is visible:
  ```bash
  nvidia-smi
  ```
  You must see **NVIDIA GeForce RTX 3060**.
  **If this fails, stop here.** Nothing below will work until it does.

- [ ] **1.4** Give WSL enough memory. In Windows, create the file
  `C:\Users\<YourName>\.wslconfig` containing:
  ```ini
  [wsl2]
  memory=24GB
  swap=8GB
  ```
  Then in PowerShell: `wsl --shutdown`, and reopen Ubuntu.

---

## Part 2 — Get the code and install it

The public GitHub repo is missing the data loader and several fixes. **Use your
own copy of this repository**, not a fresh clone of the upstream project.

- [ ] **2.1** Basic tools:
  ```bash
  sudo apt update && sudo apt install -y python3-venv python3-pip git xz-utils
  ```

- [ ] **2.2** Get the code (replace with your fork URL):
  ```bash
  cd ~ && git clone <your-fork-url> ArchCAD && cd ArchCAD
  ```

- [ ] **2.3** Create the environment and install **CUDA** PyTorch:
  ```bash
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements-cuda.txt
  ```
  > `requirements-cpu.txt` is for machines without a GPU — do not use it here.

- [ ] **2.4** Confirm PyTorch sees the GPU:
  ```bash
  .venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
  Must print `True NVIDIA GeForce RTX 3060`.

- [ ] **2.5** Put the pretrained image weights in place. Either copy
  `hrnet_ocr_cocostuff_3965_torch04.pth` (270MB) into `svgnet/pretrained/`, or:
  ```bash
  cd svgnet/pretrained && bash download.sh && cd ../..
  ```

> **About `modules/pointops`:** this repo ships a pure-PyTorch version that runs
> on the GPU with no compiling. Start with it. Only try building the real CUDA
> extension if training turns out to be too slow — note your seconds-per-iteration
> in Part 5 first, so it is a decision based on a number.

---

## Which task are you training?

Two separate jobs live in this repo. Pick one before Part 3.

**A. Doors and windows on US construction drawings** (the newer work). Labels come
free from the CAD layer names inside the PDFs — no manual annotation. Three
classes plus background. See "US door/window training" at the end of this file.

**B. Reproducing the DPSS paper on FloorPlanCAD** (35 classes, Chinese CAD).
That is what Parts 3-6 below describe.

---

## Part 3 — Get the data (~20GB free disk needed, ~40GB total with checkpoints)

- [ ] **3.1** Download all three archives:
  ```bash
  .venv/bin/python dataset/download_data.py --split all
  ```
  This checks each archive is not corrupt and re-downloads once if it is.
  (One download failed this way during testing — full size, but damaged.)

- [ ] **3.2** Convert the drawings into the format the model reads:
  ```bash
  V=.venv/bin/python
  $V dataset/parse_FpCAD_svg.py --data_dir dataset/FloorplanCAD/train_1 \
      --output_dir dataset/FloorplanCAD/json/train --png_dir dataset/FloorplanCAD/train_1
  $V dataset/parse_FpCAD_svg.py --data_dir dataset/FloorplanCAD/train_2 \
      --output_dir dataset/FloorplanCAD/json/train --png_dir dataset/FloorplanCAD/train_2
  $V dataset/parse_FpCAD_svg.py --data_dir dataset/FloorplanCAD/test \
      --output_dir dataset/FloorplanCAD/json/test --png_dir dataset/FloorplanCAD/test
  ```
  This takes a while. It prints how many drawings converted and how many failed.

---

## Part 4 — Make the PC survive power cuts

Three things, and **the first one is not software**.

- [ ] **4.1 — BIOS (most important).** Restart, press `Del` / `F2` during boot to
  enter BIOS/UEFI. Find a setting called **"Restore on AC Power Loss"** (also
  called "AC Power Recovery" or "After Power Failure") and set it to **Power On**.
  Save and exit.

  > Without this, your PC just stays off when electricity comes back, and no
  > script can do anything about it.

- [ ] **4.2 — Stop Windows interrupting.** In Windows:
  - Settings → System → Power: screen/sleep set to **Never**.
  - In an Administrator PowerShell: `powercfg /h off` (turns off hibernate).
  - Settings → Windows Update → **pause updates**, so it cannot reboot mid-run.

- [ ] **4.3 — Auto-restart training on boot.** Open **Task Scheduler** →
  *Create Task* (not "Basic Task"):
  - **General:** name it `ArchCAD training`. Tick **"Run whether user is logged
    on or not"** and **"Run with highest privileges"**.
  - **Triggers:** New → *At startup* → tick **Delay task for: 1 minute**.
  - **Actions:** New → *Start a program*
    - Program: `wsl.exe`
    - Arguments (replace `<linux-user>` with your Ubuntu username):
      ```
      -d Ubuntu-22.04 -u <linux-user> -e bash -lc "cd ~/ArchCAD && ./tools/train_resume.sh"
      ```
  - **Settings:** untick **"Stop the task if it runs longer than"**.

`tools/train_resume.sh` finds the newest checkpoint, resumes from it, and restarts
itself if the run crashes. A lock file stops two copies running at once.

---

## Part 5 — Test small before the long run

**Do not launch the 50-epoch run without doing this first.**

- [ ] **5.1** Make a tiny test set (50 drawings):
  ```bash
  mkdir -p /tmp/mini && ls dataset/FloorplanCAD/json/train/*_s2.json | head -50 \
    | while read f; do cp "${f%.json}".* /tmp/mini/; done
  ```

- [ ] **5.2** Run one short epoch and watch it. You want to see:
  - `Running on cuda`
  - `effective batch 16 = 1 x 1 gpu x 16 accum`
  - loss going **down**
  - `nvidia-smi` (in another window) showing ~6GB used
  - **write down the seconds per iteration** — you need it next

- [ ] **5.3** Set `save_interval_iters` in
  `configs/svg/svg_pointT_fpcad_3060.yaml` so it saves roughly every 10–15
  minutes. If one iteration takes 0.4s, then 15 min ≈ **2000** iterations
  (the current default).

- [ ] **5.4** Decide on speed vs precision. Setting `fp16: True` is roughly
  1.5–2× faster on a 3060 and halves memory, for a very small precision risk.
  Given how long this run takes, it is usually worth it.

---

## Part 6 — Start the real run

```bash
cd ~/ArchCAD
CONFIG=configs/svg/svg_pointT_fpcad_3060.yaml ./tools/train_resume.sh
```

- [ ] **6.1** **Test the auto-restart on purpose.** While it is training, reboot
  the PC. It should power back up and continue on its own, and
  `work_dirs/fpcad_3060/resume.log` should say `resuming from ...`.

  This is the single most important check. Do it now, not after a real outage.

- [ ] **6.2** Check progress any time:
  ```bash
  tail -f ~/ArchCAD/work_dirs/fpcad_3060/resume.log
  ```

---

## What to expect

- Roughly **1–2 hours per epoch**.
- 50 epochs ≈ **3–5 days of actual training**, but with only ~12 hours of power
  per day that is **1–2 weeks on the calendar**.
- Validation `mIoU` and `PQ` should climb over the first few epochs. They start at
  0.000 because the model begins knowing nothing.
- Checkpoints are 1.2GB each. `save_freq: 10` keeps about 13GB total instead of
  60GB.

### A limit worth knowing about

The paper starts the shape half of the model from weights pretrained on another
dataset (PointTransformerV2 on ScanNetV2). Those weights were never published in a
usable form, and this code uses an older version of that component. Your model
starts that part from zero.

So expect to land **below the paper's 86.2 PQ** — and that would be true on any
hardware, not just yours. It is worth opening an issue on the authors' GitHub
asking for their trained checkpoint and the missing `svgnet/data` package.

---

## If something breaks

| Problem | Fix |
|---|---|
| `nvidia-smi` fails in Ubuntu | Update the **Windows** driver. Never install one in Ubuntu. |
| `torch.cuda.is_available()` is `False` | You installed CPU PyTorch. Reinstall with `requirements-cuda.txt`. |
| `CUDA out of memory` | Lower `img_size` to 512, or set `fp16: True`. |
| Training didn't restart after a cut | BIOS "Restore on AC Power Loss" is off (4.1), or the scheduled task isn't set to run when logged out (4.3). |
| Disk filling up | Raise `save_freq` in the config. |
| Extraction fails | Re-run `download_data.py`; it verifies and retries automatically. |

---

# US door/window training

This is the two-stage job: practise on a public dataset, then train on your own
plan sets. Everything below has been run and checked on CPU already — what is
left is the training itself, which needs the GPU.

## Step 1 — Convert your plan sets (labels are free)

Architectural PDFs keep the drawing as real vector lines and tag each line with
the original CAD layer name (`A-DOOR`, `A-GLAZING`, `A-WALL`). The parser reads
those tags, so nothing has to be labelled by hand.

```bash
python dataset/parse_pdf_plans.py \
    --pdf_dir <folder of plan-set PDFs> \
    --output_dir dataset/us_plans/json/train \
    --require_labels --render --img_size 384
```

Measured on the 59-planset corpus: 53 sets are vector, 23 carry layer names, and
those produced **3,051 tiles containing 18,700 doors and 9,467 windows**.

Two things the parser handles that are easy to get wrong:

- **Tiling.** A construction sheet holds 45k-556k line segments; this model was
  built for 2k-7k, and its neighbour search is quadratic. Sheets are split into
  a grid of ~6k-segment tiles. Instances are grouped before tiling, so a door
  split across a tile edge keeps one id.
- **Tile images.** Each tile gets its own crop of the page render, adjusted for
  `/Rotate` and `CropBox`. Without that the image half of the model looks at a
  different part of the sheet than the geometry half.

### Split by building, never by tile

Hold out **whole plan sets** for validation. Tiles from one building share that
firm's drafting habits, so a random tile split reports generalization that is not
real. The prepared split is 16 buildings for training, 5 held out.

## Step 2 — Convert CubiCasa5K (optional pretraining)

```bash
python dataset/download_data.py --split cubicasa    # or fetch from Zenodo
python dataset/parse_cubicasa.py \
    --data_dir dataset/cubicasa/cubicasa5k \
    --split_file dataset/cubicasa/cubicasa5k/train.txt \
    --output_dir dataset/cubicasa/json/train --require_labels
```

4,199 train + 400 test plans convert cleanly. Its annotations are vector
polygons, so both halves of the model get pretrained, and instance ids come free.

> **Licence:** CubiCasa5K is CC BY-NC 4.0. Any model trained from it is
> research-only and must not ship commercially. Keep its checkpoints under
> `work_dirs/cubicasa/`. A licence-clean model is always one run away: do stage 2
> with no `pretrain:` line.

## Step 3 — Train

```bash
# stage 1 (optional): practise on CubiCasa
CONFIG=configs/svg/svg_pointT_cubicasa.yaml bash tools/train_single.sh

# stage 2: your plan sets, warm-started from stage 1
#   set  pretrain: 'work_dirs/cubicasa/best.pth'  in the config first
CONFIG=configs/svg/svg_pointT_us.yaml bash tools/train_single.sh
```

Both configs use the same 3 classes + background, so the classifier head carries
over instead of being thrown away. Verified: loading a stage-1 checkpoint into a
stage-2 model transfers everything except tensors whose shape genuinely differs.

## Step 4 — Judge it honestly

Run the same evaluation three ways:

1. **Held-out buildings** — the cross-firm question.
2. **Flattened plan sets** (vector, but no layer tags). These are the real target:
   layer reading cannot help there, and neither can rules tuned to one firm.
3. **Against the existing rule-based detector**, same sheets. If it does not beat
   the rules on firms they were not tuned for, it has not earned its place.

**Run the ablation:** stage 2 with and without the stage-1 warm start. It is the
only way to know whether CubiCasa pretraining helped at all. Use
`DETERMINISTIC=1` for that comparison — without it, thread scheduling alone moved
IoU by 35 points between identical runs here.

## Expect a hard class imbalance

Doors are **3.9%** of the line segments and windows **1.9%**; the rest is text,
dimensions and hatching. Early epochs will look like the model predicts
background for everything. That is normal for this task — judge it on door and
window IoU, not on overall accuracy, which is misleadingly high from background
alone.
