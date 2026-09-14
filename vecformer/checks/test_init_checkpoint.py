#!/usr/bin/env python3
"""Warm start across stages: a 35-class model without text initialises a 43-class
model with text; backbone and decoder trunk load, the class heads and text modules
do not.

    python vecformer/checks/test_init_checkpoint.py
"""
import os.path as osp
import sys
import tempfile

import torch

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(HERE, ".."))
import cpu_kernels  # noqa: E402

cpu_kernels.install()

from model.vecformer import build, load_matching  # noqa: E402
from model.vecformer.modeling_vecformer import VecFormer  # noqa: E402
from test_model_text_cpu import small_config  # noqa: E402

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def main():
    torch.manual_seed(0)
    src = VecFormer(small_config(False))
    base = small_config(True)
    with tempfile.TemporaryDirectory() as d:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in src.state_dict().items()}, osp.join(d, "model.safetensors"))
        args = dict(num_semantic_classes=43, num_instance_classes=43, label_space="arch43",
                    thing_class_idxs=[c for c in range(43) if c not in (30, 31, 32, 33, 34)],
                    stuff_class_idxs=[30, 31, 32, 33, 34], backbone_config=base.backbone_config,
                    cad_decoder_config=base.cad_decoder_config, text_config=base.text_config, init_checkpoint=d)
        dst, trainer_cls = build(args)
        report = load_matching(VecFormer(dst.config), d)
    s, t = src.state_dict(), dst.state_dict()
    backbone = [k for k in s if k.startswith("backbone.")]
    check(backbone and all(torch.equal(s[k], t[k]) for k in backbone), "backbone weights loaded")
    check(any("text" in k or k.startswith(("tace.", "msf.")) for k in report["missing"]), "text modules left at init")
    check(report["shape_mismatch"], "class heads reported as shape mismatches")
    check(all(t[k].shape != s[k].shape for k in report["shape_mismatch"]), "mismatched tensors keep the new shape")
    check("init_checkpoint" not in dst.config.to_dict(), "init_checkpoint is not stored in the model config")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print(f"ok - warm start: {report['loaded']} tensors loaded, {len(report['shape_mismatch'])} head tensors re-initialised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
