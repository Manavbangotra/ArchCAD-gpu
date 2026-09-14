#!/usr/bin/env python3
"""Arch-43 label space in VecFormer: coarse ids as marginals, per-source annotated
classes, ignore ids, evaluation resolution; and a full model train + eval pass.

    python vecformer/checks/test_label_space.py
"""
import json
import math
import os
import os.path as osp
import sys
import tempfile

import torch
import torch.nn.functional as F

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(HERE, ".."))
sys.path.insert(0, osp.join(HERE, "..", "..", "dataset"))
import cpu_kernels  # noqa: E402

cpu_kernels.install()

import taxonomy as tx  # noqa: E402
from data.floorplancad.floorplancad import FloorPlanCAD  # noqa: E402
from model.vecformer.configuration_vecformer import VecFormerConfig  # noqa: E402
from model.vecformer.criterion.instance_criterion import InstanceCriterion  # noqa: E402
from model.vecformer.criterion.label_space import LabelSpace  # noqa: E402
from model.vecformer.criterion.semantic_criterion import SemanticCriterion  # noqa: E402
from model.vecformer.modeling_vecformer import VecFormer  # noqa: E402
from test_model_text_cpu import drawing, small_config  # noqa: E402

FAILURES = []
C = tx.ARCH_NUM_CLASSES
FIX = tx.FIXTURE_ANY
US = list(tx.ANNOTATED).index("us")


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def unit():
    ls = LabelSpace.build("arch43", C)
    rows = ls.target_rows(torch.tensor([21, FIX, tx.IGNORE, 99, C]), source=-1)
    check(rows[0].sum() == 1 and rows[0, 21], "fine id is one-hot")
    check(rows[1].nonzero().flatten().tolist() == sorted(tx.COARSE_GROUPS[FIX]), "coarse id covers its members")
    check(not rows[2].any() and not rows[3].any(), "ignore and unknown ids are empty")
    check(rows[4].sum() == 1 and rows[4, C], "background on a fully annotated source is background only")
    bg_us = ls.target_rows(torch.tensor([C]), source=US)[0]
    check(bg_us[C] and bg_us[21] and not bg_us[32], "US background also covers toilet (unannotated), not wall")

    torch.manual_seed(0)
    logits = torch.randn(6, C + 1)
    target = torch.tensor([0, 5, 21, C, 32, 7])
    sem = SemanticCriterion(num_semantic_classes=C, ce_loss_weight=1.0)
    plain = sem._get_loss([logits], [target])["ce_loss"]
    sem.label_space = ls
    marg = sem._get_loss([logits], [target], [-1])["ce_loss"]
    check(torch.allclose(plain, marg, atol=1e-5), f"fine targets: marginal == cross-entropy ({plain.item():.4f} vs {marg.item():.4f})")
    coarse = sem._get_loss([logits[:1]], [torch.tensor([FIX])], [-1])["ce_loss"]
    p = logits[0].softmax(-1)[tx.COARSE_GROUPS[FIX]].sum()
    check(abs(coarse.item() + math.log(p.item())) < 1e-5, "coarse target: -log(sum of member probabilities)")
    ign = sem._get_loss([logits[:2]], [torch.tensor([tx.IGNORE, tx.IGNORE])], [-1])["ce_loss"]
    check(ign.item() == 0.0, "all-ignore drawing contributes zero loss")
    us_bg = sem._get_loss([logits[:1]], [torch.tensor([C])], [US])["ce_loss"]
    check(us_bg.item() < F.cross_entropy(logits[:1], torch.tensor([C])).item(), "US background is a softer target than plain background")

    inst = InstanceCriterion(num_instance_classes=C, label_smoothing=0.1)
    inst.label_space = ls
    q = torch.randn(4, C + 1)
    t_fine = torch.tensor([0, 21, 32, 7])
    check(torch.allclose(inst._get_class_loss(q, t_fine),
                         F.cross_entropy(q, t_fine, weight=inst._get_ce_weight(t_fine), label_smoothing=0.1)),
          "instance class loss: fine targets use upstream cross-entropy")
    lc = inst._get_class_loss(q, torch.tensor([0, FIX, 32, tx.DOOR_ANY]))
    check(bool(torch.isfinite(lc)) and lc > 0, f"instance class loss with coarse targets finite: {lc.item():.3f}")
    cost = inst.matcher._get_class_cost(q, torch.tensor([FIX, 21]))
    check(torch.allclose(cost[:, 0], 1 - q.softmax(-1)[:, tx.COARSE_GROUPS[FIX]].sum(-1), atol=1e-6)
          and torch.allclose(cost[:, 1], 1 - q.softmax(-1)[:, 21], atol=1e-6), "matcher class cost uses marginals")

    res = ls.resolve(torch.tensor([FIX, FIX, tx.IGNORE, 21, C, 70]), torch.tensor([25, 3, 5, 0, 2, 1]))
    check(res.tolist() == [25, tx.COARSE_GROUPS[FIX][0], C, 21, C, C],
          f"evaluation resolution: member kept, non-member -> first member, ignore/unknown -> bg: {res.tolist()}")
    check(ls.is_instance_target(FIX) and not ls.is_instance_target(tx.IGNORE) and not ls.is_instance_target(C),
          "instance targets: coarse yes, ignore and background no")
    check(LabelSpace.build(None, 35) is None, "no label space by default")


def arch_drawing(seed):
    d = drawing(seed)
    # relabel into Arch-43: doors stay 0, window 1 -> fixture-any, wall 30 -> 32 (stuff);
    # a few primitives ignored, some background
    remap = {0: 0, 1: FIX, 30: 32}
    d["semantic_ids"] = [remap[s] for s in d["semantic_ids"]]
    for i in range(0, len(d["semantic_ids"]), 11):
        d["semantic_ids"][i], d["instance_ids"][i] = tx.IGNORE, -1
    for i in range(3, len(d["semantic_ids"]), 13):
        d["semantic_ids"][i], d["instance_ids"][i] = C, -1
    return d


def model_pass():
    no_aug = dict(random_vertical_flip=0.0, random_horizontal_flip=0.0, random_rotate=False,
                  random_scale=[1.0, 1.0], random_translation=[0.0, 0.0])
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(osp.join(d, "train"))
        for i in range(2):
            with open(osp.join(d, "train", f"{i}.json"), "w") as f:
                json.dump(arch_drawing(i), f)
        ds = FloorPlanCAD(d, "train", no_aug, no_aug, use_text=True, source_id=US)
        batch = FloorPlanCAD.collate_fn([ds[0], ds[1]])
    check(batch["source_ids"].tolist() == [US, US], "collate carries source ids")
    batch.pop("data_paths")

    base = small_config(True)
    cfg = VecFormerConfig(num_semantic_classes=C, num_instance_classes=C,
                          thing_class_idxs=[c for c in range(C) if c not in tx.STUFF_CLASSES],
                          stuff_class_idxs=tx.STUFF_CLASSES, label_space="arch43",
                          backbone_config=base.backbone_config, cad_decoder_config=base.cad_decoder_config,
                          text_config=base.text_config)
    model = VecFormer(cfg)
    check(model.config.cad_decoder_config["num_semantic_classes"] == C, "43-class head")

    t = model.prepare_targets(batch["sem_ids"], batch["inst_ids"], batch["prim_lengths"], batch["cu_numprims"],
                              batch["source_ids"])
    labels = torch.cat(t["list_target_inst_labels"]).tolist()
    check(FIX in labels and tx.IGNORE not in labels and C not in labels,
          f"instance targets keep coarse ids, drop ignore and background: {sorted(set(labels))}")

    model.train()
    torch.manual_seed(1)
    out = model(**batch)
    check(bool(torch.isfinite(out.loss)), f"training loss finite: {out.loss.item():.3f}")
    out.loss.backward()
    head = [p.grad for n, p in model.named_parameters() if "cad_decoder" in n and p.grad is not None]
    check(head and all(bool(torch.isfinite(g).all()) for g in head), "finite gradients in the decoder")

    model.eval()
    with torch.no_grad():
        ev = model(**batch)
    ms = ev.metric_states
    check(ms is not None and all(v.shape[0] == C for k, v in ms.items() if k.endswith("per_class")),
          f"evaluation runs with coarse and ignore labels: {sorted(ms) if ms else ms}")


def main():
    unit()
    model_pass()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("ok - label space: marginals, annotated masks, ignore, eval resolution; Arch-43 model trains and evaluates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
