#!/usr/bin/env python3
"""TACE and MSF text fusion: shapes, masking, gates, identity at init, gradients,
per-drawing isolation.

    python vecformer/checks/test_textcad.py
"""
import importlib.util
import os.path as osp
import sys

import torch

HERE = osp.dirname(osp.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "textcad", osp.join(HERE, "..", "model", "vecformer", "text", "textcad.py"))
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def main():
    torch.manual_seed(0)
    n_text = 7
    tace = tc.TACE(num_types=46, num_grades=7)
    types = torch.randint(1, 46, (n_text,))
    attrs = torch.randn(n_text, 3)
    grades = torch.randint(0, 7, (n_text,))
    masks = torch.rand(n_text, 4) > 0.5
    masks[0] = False                                  # an annotation with no attributes
    geo = torch.randn(n_text, 3)
    x = tace(types, attrs, grades, masks, geo)
    check(x.shape == (n_text, 32) and bool(torch.isfinite(x).all()), f"TACE output {tuple(x.shape)} finite")
    attrs2 = attrs.clone()
    attrs2[~masks[:, :3]] += 100.0
    x2 = tace(types, attrs2, grades, masks, geo)
    check(torch.allclose(x, x2, atol=1e-5), "masked-out attributes have no effect")

    msf = tc.MSFTextFusion(line_dim=64)
    n = 50
    feat = torch.randn(n, 64, requires_grad=True)
    pos = torch.rand(n, 2) - 0.5
    batch = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(20, dtype=torch.long)])
    text = tc.TextContext(feats=x, pos=torch.rand(n_text, 2) - 0.5, batch=torch.tensor([0, 0, 0, 1, 1, 1, 1]))
    msf.train()
    out, l0 = msf(feat, pos, batch, text)
    check(torch.allclose(out, feat), "zero-initialised fusion is the identity")
    check(0 < l0.item() <= n_text, f"L0 complexity counts gates: {l0.item():.3f}")

    torch.nn.init.normal_(msf.out.weight, std=0.1)
    out, l0 = msf(feat, pos, batch, text)
    (out.pow(2).mean() + 1e-4 * l0 + x.sum() * 0).backward()
    check(feat.grad is not None and feat.grad.abs().sum() > 0, "gradient reaches line features")
    check(msf.rel_q.weight.grad is not None and msf.rel_q.weight.grad.abs().sum() > 0, "gradient reaches relevance")
    check(tace.type_emb.weight.grad is not None and tace.type_emb.weight.grad.abs().sum() > 0,
          "gradient reaches the text encoder")

    msf.eval()
    only_1 = tc.TextContext(feats=x[3:].detach(), pos=text.pos[3:], batch=text.batch[3:])
    o_a, _ = msf(feat.detach(), pos, batch, only_1)
    check(torch.allclose(o_a[:30], feat.detach()[:30]), "text never crosses drawings")
    o1, _ = msf(feat.detach(), pos, batch, text)
    o2, _ = msf(feat.detach(), pos, batch, text)
    check(torch.allclose(o1, o2), "eval is deterministic")
    empty = tc.TextContext(torch.zeros(0, 32), torch.zeros(0, 2), torch.zeros(0, dtype=torch.long))
    o3, l3 = msf(feat.detach(), pos, batch, empty)
    check(torch.allclose(o3, feat.detach()) and l3.item() == 0, "no text is a no-op")

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("ok - TACE and MSF: masking, gates, identity at init, gradients, per-drawing isolation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
