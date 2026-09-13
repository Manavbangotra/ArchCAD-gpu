#!/usr/bin/env python3
"""The sped-up paths must compute what the slow ones did. CPU only.

    python tools/test_fast_paths.py

  1. farthest point sampling in NumPy picks the same points as the torch loop;
  2. batched denoising queries equal the per-target loop, given the same random
     draws (the loop draws x then y per target, the batch draws them at once).
"""

import os.path as osp
import sys

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402
from munch import Munch  # noqa: E402

import svgnet.model.decoder as decoder_mod  # noqa: E402
from modules.pointops.functions import pointops  # noqa: E402
from svgnet.model.svgnet import SVGNet  # noqa: E402

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def fps_torch(pts, m):
    """The loop pointops used before, verbatim in behaviour."""
    sel = torch.zeros(m, dtype=torch.long)
    best = torch.full((pts.shape[0],), float("inf"))
    last = 0
    for i in range(1, m):
        best = torch.minimum(best, torch.sum((pts - pts[last]) ** 2, dim=1))
        last = int(torch.argmax(best).item())
        sel[i] = last
    return sel


def test_fps():
    g = torch.Generator().manual_seed(0)
    for n in (5, 60, 900, 4000):
        p = torch.rand(n, 3, generator=g)
        p[:, 2] = 0
        m = max(1, n // 4)
        check(torch.equal(fps_torch(p, m), torch.from_numpy(pointops._fps_numpy(p.numpy(), m))),
              f"FPS selection differs at n={n}")
    # Through the public entry point with two drawings in one batch.
    p = torch.rand(1000, 3, generator=g)
    off = torch.tensor([400, 1000], dtype=torch.int32)
    new = torch.tensor([100, 250], dtype=torch.int32)
    got = pointops.furthestsampling(p, off, new).long()
    want = torch.cat([fps_torch(p[:400], 100), fps_torch(p[400:], 150) + 400])
    check(torch.equal(got, want), "batched furthestsampling differs from per-drawing FPS")


def test_denoising_queries():
    cfg = yaml.safe_load(open(osp.join(ROOT, "configs/svg/svg_pointT_fpcad43_3060.yaml"),
                              encoding="utf-8"))
    torch.manual_seed(0)
    model = SVGNet(Munch.fromDict(cfg["model"]), criterion=None).eval()
    dec = model.decoder
    n = 900
    coords = torch.rand(n, 3)
    sem = torch.randint(0, 44, (n,))
    sem[sem > 40] = 53                                   # coarse ids get no query
    ins = torch.randint(0, 30, (n,))
    targets = model.prepare_targets(torch.stack([sem, ins], 1))
    stage = {"up": [{"p_out": coords}], "tgt": targets}
    q = torch.zeros(dec.num_queries, 1, 256)
    pos = torch.zeros(dec.num_queries, 1, 256)
    count = int((targets[0]["labels"] < dec.num_classes).sum())
    draws = torch.rand(count, 2)

    real_rand = decoder_mod.torch.rand
    try:
        seq = iter(draws.flatten().tolist())
        decoder_mod.torch.rand = lambda *a, **k: torch.tensor([next(seq)])
        dec._dn_loop = True
        loop = dec.query_for_dn2(stage, q, pos)
        decoder_mod.torch.rand = lambda *a, **k: draws
        dec._dn_loop = False
        fast = dec.query_for_dn2(stage, q, pos)
    finally:
        decoder_mod.torch.rand = real_rand
        dec._dn_loop = False

    check(torch.equal(loop[3], fast[3]), "denoising target indices differ")
    check(torch.allclose(loop[0], fast[0]), "denoising query features differ")
    check(torch.allclose(loop[1], fast[1], atol=1e-5), "denoising query positions differ")
    check(torch.equal(loop[2], fast[2]), "denoising attention masks differ")


def main():
    tests = [test_fps, test_denoising_queries]
    for fn in tests:
        fn()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  {f}")
        return 1
    print(f"ok - {len(tests)} groups, fast paths match the loops they replace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
