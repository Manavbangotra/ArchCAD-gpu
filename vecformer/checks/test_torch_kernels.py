#!/usr/bin/env python3
"""Pure-torch kernel replacements against direct reference computations.

    python vecformer/checks/test_torch_kernels.py
"""
import itertools
import os.path as osp
import sys

import torch

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, ".."))
from utils import torch_kernels as tk  # noqa: E402

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.allow_tf32 = False          # the dense reference must be exact
    torch.backends.cuda.matmul.allow_tf32 = False
    # attention: sequences of mixed lengths vs per-sequence softmax attention
    lens = [5, 7, 5, 3, 7]
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32, device=dev)
    qkv = torch.randn(sum(lens), 3, 2, 8, device=dev)
    out = tk.flash_attn_varlen_qkvpacked_func(qkv, cu, 7)
    ref = []
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
        q, k, v = (qkv[a:b, i].transpose(0, 1) for i in range(3))
        ref.append((torch.softmax(q @ k.transpose(-1, -2) / 8 ** 0.5, -1) @ v).transpose(0, 1))
    check(torch.allclose(out, torch.cat(ref), atol=1e-5), "varlen self-attention equals per-sequence attention")
    cq = torch.tensor([0, 2, 5], dtype=torch.int32, device=dev)
    ck = torch.tensor([0, 4, 10], dtype=torch.int32, device=dev)
    q, k, v = torch.randn(5, 2, 8, device=dev), torch.randn(10, 2, 8, device=dev), torch.randn(10, 2, 8, device=dev)
    o2 = tk.flash_attn_varlen_func(q, k, v, cq, ck, 3, 6)
    r2 = torch.cat([(torch.softmax(q[a:b].transpose(0, 1) @ k[c:d].transpose(0, 1).transpose(-1, -2) / 8 ** 0.5, -1)
                     @ v[c:d].transpose(0, 1)).transpose(0, 1) for (a, b), (c, d) in
                    zip([(0, 2), (2, 5)], [(0, 4), (4, 10)])])
    check(torch.allclose(o2, r2, atol=1e-5), "varlen cross-attention equals per-sequence attention")
    if dev == "cuda":
        o3 = tk.flash_attn_varlen_qkvpacked_func(qkv.half(), cu, 7)
        check(torch.allclose(o3.float(), out, atol=2e-2), "half-precision input as flash-attn is called")

    # scatter
    src = torch.randn(9, 3, device=dev)
    index = torch.tensor([0, 2, 2, 1, 0, 2, 1, 1, 0], device=dev)
    for red, fn in (("sum", "sum"), ("mean", "mean"), ("max", "amax")):
        ref = torch.stack([getattr(src[index == i], fn)(0) for i in range(3)])
        check(torch.allclose(tk.scatter(src, index, 0, reduce=red), ref, atol=1e-6), f"scatter {red}")
    order = index.argsort(stable=True)
    check(torch.allclose(tk.segment_csr(src[order], torch.tensor([0, 3, 6, 9], device=dev), "mean"),
                         torch.stack([src[index == i].mean(0) for i in range(3)]), atol=1e-6), "segment_csr mean")

    # submanifold conv vs dense 3-D convolution evaluated at active sites
    grid = torch.rand(2, 6, 7, 5) < 0.3
    idx = grid.nonzero().int().to(dev)                              # [batch, x, y, z]
    feats = torch.randn(idx.shape[0], 4, device=dev)
    conv = tk.SubMConv3d(4, 6, kernel_size=3, indice_key="s").to(dev)
    x = tk.SparseConvTensor(feats, idx, [6, 7, 5], 2)
    y = conv(x).features
    dense = torch.zeros(2, 4, 6, 7, 5, device=dev)
    b_, x_, y_, z_ = (idx[:, i].long() for i in range(4))
    dense[b_, :, x_, y_, z_] = feats
    ref = torch.nn.functional.conv3d(dense, conv.weight.permute(0, 4, 1, 2, 3), conv.bias, padding=1)
    ref = ref.permute(0, 2, 3, 4, 1)[b_, x_, y_, z_]
    check(torch.allclose(y, ref, atol=1e-5), f"submanifold conv equals dense conv at active sites (max err {(y - ref).abs().max():.2e})")
    check(("s", 3) in x.indice_dict and conv(x.replace_feature(feats)).features.shape == y.shape, "neighbour map cached by indice_key")
    y.sum().backward()
    check(conv.weight.grad is not None and conv.weight.grad.abs().sum() > 0, "gradients reach the kernel")
    conv5 = tk.SubMConv3d(4, 3, kernel_size=5).to(dev)
    check(conv5(tk.SparseConvTensor(feats, idx, [6, 7, 5], 2)).features.shape == (idx.shape[0], 3), "kernel 5 (stem)")

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print(f"ok - torch kernels on {dev}: varlen attention, scatter, submanifold conv == dense conv at active sites")
    return 0


if __name__ == "__main__":
    sys.exit(main())
