#!/usr/bin/env python3
"""Layer-name embedding: tokens, XREF prefixes, per-drawing dropout, collation offsets,
identity at initialisation, effect and gradients once trained.

    python vecformer/checks/test_layer_names.py
"""
import json
import os
import os.path as osp
import sys
import tempfile

import torch

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(HERE, ".."))
import cpu_kernels  # noqa: E402

cpu_kernels.install()

from data.floorplancad.floorplancad import FloorPlanCAD  # noqa: E402
from data.floorplancad.layer_names import layer_token_table, name_tokens  # noqa: E402
from model.vecformer.modeling_vecformer import VecFormer  # noqa: E402
from test_model_text_cpu import drawing, small_config  # noqa: E402

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def feats(model, batch):
    torch.manual_seed(7)
    dd = model._get_data_dict(batch["coords"], batch["feats"], batch["cu_seqlens"], prim_ids=batch["prim_ids"],
                              layer_ids=batch["layer_ids"])
    nf = model._layer_name_feats(batch["layer_ids"], batch["cu_seqlens"], batch["layer_tokens"], batch["layer_token_cu"])
    hook = lambda name, point: point.feat + nf if name == "embedding" else None   # noqa: E731
    return model.backbone(dd, batch["cu_seqlens"], batch["prim_ids"], stage_hook=hook)[0]


def main():
    a, b = name_tokens("A-Door"), name_tokens("A-DOOR")
    check(a == b and a[0] > 0 and a[1] > 0 and a[2] == 0, f"case-insensitive words: {a}")
    check(name_tokens("xref-plan|A-Door") == a and name_tokens("plan$0$A-Door") == a, "XREF binding removed")
    check(name_tokens(None) == [0, 0, 0, 0] and name_tokens("A-WALL-ABOVE")[2] > 0, "padding and multi-word names")
    t = layer_token_table(["A-Door", None], 3)
    check(t.shape == (3, 4) and t[1].sum() == 0 and t[2].sum() == 0, "unnamed and missing layers are padding")

    no_aug = dict(random_vertical_flip=0.0, random_horizontal_flip=0.0, random_rotate=False,
                  random_scale=[1.0, 1.0], random_translation=[0.0, 0.0])
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(osp.join(d, "train"))
        for i in range(2):
            dr = drawing(i)
            dr["layer_names"] = ["A-Door", "A-Window"] if i == 0 else []
            with open(osp.join(d, "train", f"{i}.json"), "w") as f:
                json.dump(dr, f)
        ds = FloorPlanCAD(d, "train", no_aug, no_aug, use_layer_names=True)
        items = [ds[0], ds[1]]
        dropped = FloorPlanCAD(d, "train", no_aug, no_aug, use_layer_names=True, layer_name_dropout=1.0)[0]
    check(items[0].layer_tokens is not None and items[0].layer_tokens[0].sum() > 0, "names become tokens")
    check(dropped.layer_tokens.sum() == 0, "dropout removes a drawing's names")
    batch = FloorPlanCAD.collate_fn(items)
    batch.pop("data_paths")
    n0 = int(items[0].layer_ids.max()) + 1
    check(batch["layer_token_cu"].tolist()[:2] == [0, n0] and batch["layer_tokens"].shape[1] == 4, "collated with offsets")

    cfg = small_config(False)
    cfg.layer_name_config = dict(cfg.layer_name_config, enabled=True)
    model = VecFormer(cfg).eval()
    with torch.no_grad():
        nf = model._layer_name_feats(batch["layer_ids"], batch["cu_seqlens"], batch["layer_tokens"], batch["layer_token_cu"])
    check(nf.shape == (batch["coords"].shape[0], 16) and nf.abs().sum() == 0, "zero-initialised: no effect at init")
    torch.nn.init.normal_(model.layer_name_proj.weight, std=0.1)
    with torch.no_grad():
        nf = model._layer_name_feats(batch["layer_ids"], batch["cu_seqlens"], batch["layer_tokens"], batch["layer_token_cu"])
    n_first = int(batch["cu_seqlens"][1])
    check(nf[:n_first].abs().sum() > 0 and nf[n_first:].abs().sum() == nf[n_first:].abs().sum() * 0 + (
        model.layer_name_proj.bias.abs().sum() * (nf.shape[0] - n_first)), "named drawing gets name features, unnamed only the bias")
    model.train()
    out = model(**batch)
    out.loss.backward()
    g = model.layer_name_emb.weight.grad
    check(bool(torch.isfinite(out.loss)) and g is not None and g.abs().sum() > 0 and g[0].abs().sum() == 0,
          "trains; gradient reaches word embeddings, not padding")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("ok - layer names: tokens, xref, dropout, offsets, identity at init, gradients")
    return 0


if __name__ == "__main__":
    sys.exit(main())
