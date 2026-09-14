#!/usr/bin/env python3
"""Text through the VecFormer data path: typed, positioned, augmented with the lines, collated.

    python vecformer/checks/test_text_data.py
"""
import json
import os
import os.path as osp
import sys
import tempfile
import types

import torch


def _scatter(src, index, dim=0, reduce="mean"):
    n = int(index.max()) + 1
    out = torch.zeros(n, dtype=src.dtype).index_add_(0, index, src)
    if reduce == "mean":
        out = out / torch.bincount(index, minlength=n).clamp(min=1).to(src.dtype)
    return out


sys.modules.setdefault("torch_scatter", types.SimpleNamespace(scatter=_scatter))
HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, ".."))

from data.floorplancad.floorplancad import FloorPlanCAD  # noqa: E402

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


DRAWING = dict(
    viewBox=[0, 0, 100, 100],
    coords=[[10, 10, 30, 10], [30, 10, 30, 30], [60, 60, 90, 60]],
    primitive_ids=[0, 0, 1], layer_ids=[0, 0, 1], semantic_ids=[0, 35], instance_ids=[1, -1],
    primitive_lengths=[40.0, 30.0],
    texts=[dict(text="M1021", x=20, y=12, size=2, angle=0, layer_id=0),
           dict(text="BEDROOM", x=75, y=65, size=3, angle=-90, layer_id=1),
           dict(text="", x=0, y=0, size=1, angle=0, layer_id=0)],
)


def main():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(osp.join(d, "train"))
        with open(osp.join(d, "train", "a.json"), "w") as f:
            json.dump(DRAWING, f)
        no_aug = dict(random_vertical_flip=0.0, random_horizontal_flip=0.0, random_rotate=False,
                      random_scale=[1.0, 1.0], random_translation=[0.0, 0.0])
        ds = FloorPlanCAD(d, "train", no_aug, no_aug, use_text=True)
        item = ds[0]
        check(item.text is not None and len(item.text["text_types"]) == 2, "empty text dropped, two kept")
        check(item.coords.shape[0] == 3, f"lines unchanged in count: {item.coords.shape}")
        check(torch.allclose(item.text_pos[0], torch.tensor([20 / 100 - 0.5, 12 / 100 - 0.5])),
              f"text position normalised like lines: {item.text_pos[0].tolist()}")

        flip = dict(no_aug, random_horizontal_flip=1.0, random_scale=[1.5, 1.5])
        ds2 = FloorPlanCAD(d, "train", flip, flip, use_text=True)
        torch.manual_seed(0)
        it2 = ds2[0]
        # the text sits on its door line in both versions: same relative place
        # coords are segment centres after the transform; segment 0 is centred at (20, 10),
        # the text at (20, 12): same x, and 2 units (0.02 normalised) away in y, times scale 1.5
        c0 = it2.coords[0]
        check(abs(float(it2.text_pos[0, 0] - c0[0])) < 1e-4 and abs(abs(float(it2.text_pos[0, 1] - c0[1])) - 0.03) < 1e-4,
              f"text augmented together with lines: text {it2.text_pos[0].tolist()} centre {c0.tolist()}")
        check(abs(float(it2.text_pos[0, 0]) - 0.45) < 1e-4, f"flip and scale applied to text: {it2.text_pos[0, 0].item():.3f} (expect 0.45)")

        g0, g2 = item.text["text_geo"][1], it2.text["text_geo"][1]       # BEDROOM, angle -90, size 3
        check(torch.allclose(g0, torch.tensor([-1.0, 0.0, 3.0]), atol=1e-4), f"geometry without augmentation: {g0.tolist()}")
        # horizontal flip mirrors x: a text reading straight up keeps sin, cos stays 0; scale 1.5 grows size
        check(torch.allclose(g2, torch.tensor([-1.0, 0.0, 4.5]), atol=1e-4), f"geometry follows flip and scale: {g2.tolist()}")
        g_m = it2.text["text_geo"][0]                                      # M1021, angle 0 -> flipped reads leftwards
        check(torch.allclose(g_m[:2], torch.tensor([0.0, -1.0]), atol=1e-4), f"flip reverses reading direction: {g_m.tolist()}")
        check("text_vec" not in item.text and "text_size_ratio" not in item.text, "helper tensors not passed to the model")

        batch = FloorPlanCAD.collate_fn([item, it2])
        check(batch["text_types"].shape[0] == 4 and batch["text_cu_seqlens"].tolist() == [0, 2, 4], "collated with offsets")
        check(batch["text_masks"].shape == (4, 4) and batch["text_pos"].shape == (4, 2), "collated shapes")

        ds_off = FloorPlanCAD(d, "train", no_aug, no_aug)
        b_off = FloorPlanCAD.collate_fn([ds_off[0]])
        check("text_types" not in b_off, "text off: batch unchanged from upstream")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("ok - text is typed, normalised, augmented with the lines and collated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
