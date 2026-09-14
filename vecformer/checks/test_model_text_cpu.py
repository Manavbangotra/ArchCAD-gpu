#!/usr/bin/env python3
"""Full VecFormer with TextCAD text fusion, on CPU with pure-torch kernel stand-ins.

Builds the real model (PTv3 backbone, LFE, CAD decoder, criterion) from synthetic
drawings loaded through the real dataset and collate code, and checks:
  - text off: the model has no text modules and ignores text inputs
  - text on, at initialisation: backbone features are exactly those without text
  - training step: finite loss with the L0 term logged, gradients reach MSF and TACE
  - once fusion weights are non-zero, text changes the features, and only through
    the configured backbone levels

    python vecformer/checks/test_model_text_cpu.py
"""
import json
import os
import os.path as osp
import random
import sys
import tempfile

import torch

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(HERE, ".."))
import cpu_kernels  # noqa: E402

cpu_kernels.install()

from data.floorplancad.floorplancad import FloorPlanCAD  # noqa: E402
from data.floorplancad.text_features import NUM_GRADES, NUM_TYPES  # noqa: E402
from model.vecformer.configuration_vecformer import VecFormerConfig  # noqa: E402
from model.vecformer.modeling_vecformer import VecFormer  # noqa: E402

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def drawing(seed):
    rng = random.Random(seed)
    coords, prim_ids, layer_ids, sem, inst, lengths, texts = [], [], [], [], [], [], []
    words = ["D1", "BEDROOM", "3600", "W2", "KITCHEN", "M1021", "UP"]
    for r in range(24):
        x, y = rng.uniform(20, 900), rng.uniform(20, 900)
        w, h = rng.uniform(10, 60), rng.uniform(10, 60)
        cls = rng.choice([0, 1, 30])                    # door, window, wall (stuff)
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        for i in range(4):
            (x1, y1), (x2, y2) = corners[i], corners[(i + 1) % 4]
            pid = len(sem)
            # two segments per primitive, as the sampler produces
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            coords += [[x1, y1, mx, my], [mx, my, x2, y2]]
            prim_ids += [pid, pid]
            layer_ids += [cls, cls]
            sem.append(cls)
            inst.append(-1 if cls == 30 else r)
            lengths.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        texts.append(dict(text=rng.choice(words), x=x + w / 2, y=y + h / 2, size=5, angle=0, layer_id=cls))
    return dict(viewBox=[0, 0, 1000, 1000], coords=coords, primitive_ids=prim_ids, layer_ids=layer_ids,
                semantic_ids=sem, instance_ids=inst, primitive_lengths=lengths, texts=texts)


def small_config(text_enabled, levels=("enc0", "enc1", "enc2", "enc3", "enc4")):
    cfg = VecFormerConfig()
    cfg.backbone_config = dict(cfg.backbone_config, enc_channels=(16, 32, 64, 64, 64),
                               enc_num_head=(2, 2, 4, 4, 4), dec_channels=(32, 32, 64, 64),
                               dec_num_head=(2, 2, 4, 4), enc_depths=(1, 1, 1, 1, 1),
                               dec_depths=(1, 1, 1, 1), drop_path=0.0)
    cfg.cad_decoder_config = dict(cfg.cad_decoder_config, input_dim=32, embed_dim=64, n_heads=4, n_blocks=2)
    cfg.text_config = dict(cfg.text_config, enabled=text_enabled, num_types=NUM_TYPES,
                           num_grades=NUM_GRADES, levels=levels)
    return cfg


def backbone_feats(model, batch, with_text):
    torch.manual_seed(7)                                 # PTv3 shuffles serialisation orders
    data_dict = model._get_data_dict(batch["coords"], batch["feats"], batch["cu_seqlens"],
                                     prim_ids=batch["prim_ids"], layer_ids=batch["layer_ids"])
    hook = None
    if with_text:
        text = model._text_context(*(batch[k] for k in ("text_types", "text_attrs", "text_grades", "text_masks",
                                                       "text_geo", "text_pos", "text_cu_seqlens")))

        def hook(name, point):
            if name not in model.msf:
                return None
            return model.msf[name](point.feat, point.coord[:, :2], point.batch, text)[0]
    feats, _ = model.backbone(data_dict, batch["cu_seqlens"], batch["prim_ids"], stage_hook=hook)
    return feats


def main():
    torch.manual_seed(0)
    no_aug = dict(random_vertical_flip=0.0, random_horizontal_flip=0.0, random_rotate=False,
                  random_scale=[1.0, 1.0], random_translation=[0.0, 0.0])
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(osp.join(d, "train"))
        for i in range(2):
            with open(osp.join(d, "train", f"{i}.json"), "w") as f:
                json.dump(drawing(i), f)
        ds = FloorPlanCAD(d, "train", no_aug, no_aug, use_text=True)
        batch = FloorPlanCAD.collate_fn([ds[0], ds[1]])
    batch.pop("data_paths", None)

    plain = VecFormer(small_config(False))
    check(not hasattr(plain, "tace") and not hasattr(plain, "msf"), "text off: no text modules")
    plain.train()
    out = plain(**batch)
    check(bool(torch.isfinite(out.loss)) and "text_l0" not in out.dict_sublosses,
          f"text off: text inputs ignored, loss {out.loss.item():.3f}")

    model = VecFormer(small_config(True))
    model.eval()
    with torch.no_grad():
        f0, f1 = backbone_feats(model, batch, False), backbone_feats(model, batch, True)
    check(f0.shape == f1.shape and torch.allclose(f0, f1, atol=1e-6), "text on, at init: features unchanged")

    model.train()
    out = model(**batch)
    check(bool(torch.isfinite(out.loss)) and "text_l0" in out.dict_sublosses and out.dict_sublosses["text_l0"] > 0,
          f"training loss finite with L0 logged: {out.loss.item():.3f}, {dict((k, round(float(v), 3)) for k, v in out.dict_sublosses.items())}")
    out.loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if n.startswith("msf.enc4.out.")}
    check(all(g is not None and g.abs().sum() > 0 for g in grads.values()) and grads, "gradient reaches MSF output")

    for level in model.msf.values():
        torch.nn.init.normal_(level.out.weight, std=0.05)
    model.zero_grad()
    model.train()
    out = model(**batch)
    out.loss.backward()
    g = model.tace.type_emb.weight.grad
    check(g is not None and g.abs().sum() > 0, "gradient reaches the text encoder (TACE)")
    check(model.msf["enc0"].rel_q.weight.grad is not None and model.msf["enc0"].rel_q.weight.grad.abs().sum() > 0,
          "gradient reaches MSF relevance filtering")
    model.eval()
    with torch.no_grad():
        f0, f1 = backbone_feats(model, batch, False), backbone_feats(model, batch, True)
    check(not torch.allclose(f0, f1, atol=1e-4), "trained fusion: text changes the features")

    dec_only = VecFormer(small_config(True, levels=("dec0",)))
    check(set(dec_only.msf.keys()) == {"dec0"} and dec_only.msf["dec0"].rel_k.in_features == 32,
          "decoder level uses decoder width")

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("ok - VecFormer with text fusion: off is upstream, identity at init, loss and gradients, text takes effect")
    return 0


if __name__ == "__main__":
    sys.exit(main())
