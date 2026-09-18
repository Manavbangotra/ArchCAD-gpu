#!/usr/bin/env python3
"""Convert a released SymPointV2 checkpoint into an Arch-43 starting point.

    python tools/finetune_spv2.py --checkpoint path/to/best.pth \
        --config configs/svg/svg_pointT_us43_spv2ft.yaml \
        --out work_dirs/us43_spv2ft/spv2_arch43_init.pth

The released checkpoint is point-only (no image branch), 35 FloorPlanCAD classes,
500 queries. Arch-43 ids 0-34 ARE FloorPlanCAD's classes, in order (dataset/taxonomy.py
band A), so the classifier head and the label embedding transfer row for row: rows
0-34 keep their meaning, the background row moves to the end, and the eight US-only
classes (column, framing, roof, electrical, mechanical, pipe, site, equipment) start
fresh. Everything else must match exactly, or the script says which tensors did not.

The result is a normal checkpoint, so training loads it through the config's
`pretrain` field with no special casing.
"""
import argparse
import io
import os
import os.path as osp
import sys

import torch
import yaml
from munch import Munch

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, "dataset"))


def convert(checkpoint, config, out):
    from svgnet.model.svgnet import SVGNet

    cfg = Munch.fromDict(yaml.safe_load(io.open(config, encoding="utf-8")))
    model = SVGNet(cfg.model, criterion=None)
    target = model.state_dict()
    src = {k: v for k, v in torch.load(checkpoint, map_location="cpu", weights_only=False)["net"].items()
           if not k.startswith("criterion.")}

    new, head_rows, mismatched = {}, [], []
    for k, v in src.items():
        if k not in target:
            mismatched.append((k, "not in this model"))
        elif target[k].shape == v.shape:
            new[k] = v
        elif k in ("decoder.class_embed_head.weight", "decoder.class_embed_head.bias", "decoder.label_enc.weight"):
            row = target[k].clone()
            n_src = v.shape[0] - 1                       # source classes, last row = background
            row[:n_src] = v[:n_src]
            row[-1] = v[-1]
            new[k], _ = row, head_rows.append(f"{k}: {n_src} class rows + background -> {row.shape[0]} rows")
        else:
            mismatched.append((k, f"{tuple(v.shape)} vs {tuple(target[k].shape)}"))

    missing = [k for k in target if k not in new]
    model.load_state_dict(new, strict=False)
    print(f"transferred {len(new)}/{len(target)} tensors")
    for line in head_rows:
        print("  head:", line)
    if mismatched:
        print("  NOT transferred:")
        for k, why in mismatched:
            print(f"    {k}: {why}")
    if missing:
        print(f"  freshly initialised ({len(missing)}): {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    os.makedirs(osp.dirname(osp.abspath(out)), exist_ok=True)
    torch.save({"net": model.state_dict(), "epoch": 0}, out)
    print("wrote", out)
    return dict(transferred=len(new), missing=missing, mismatched=mismatched)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/svg/svg_pointT_us43_spv2ft.yaml")
    ap.add_argument("--out", default="work_dirs/us43_spv2ft/spv2_arch43_init.pth")
    a = ap.parse_args()
    convert(a.checkpoint, a.config, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
