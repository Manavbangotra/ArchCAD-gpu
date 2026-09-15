"""
Pure-PyTorch GPU replacements for flash-attn, spconv and torch_scatter.

The papers' setting uses the CUDA kernels (cloud/Dockerfile, Linux). They have no
wheels for Windows / Python 3.12, so a Windows machine with an NVIDIA GPU (the
development RTX 3060) runs VecFormer with these instead. Unlike the CPU stand-ins
in checks/cpu_kernels.py, these compute the same function:

- attention: exact softmax attention through torch's scaled_dot_product_attention
  (memory-efficient kernel); variable-length sequences are grouped by length and
  batched, which fits PTv3's fixed-size serialized patches.
- scatter: torch scatter_reduce (exact; torch_scatter's "max" returns the same values).
- submanifold sparse 3-D convolution: every active voxel gathers its active
  neighbours through a hashed voxel lookup and applies one weight per kernel
  offset; inactive sites contribute nothing and outputs exist only at input sites,
  as in spconv's SubMConv3d. Neighbour maps are cached per indice_key on the
  tensor, as spconv does. Weights are stored as (out, k, k, k, in) like spconv 2.x;
  whether a kernel offset maps to the same weight index as spconv (correlation vs
  convolution order) is not verified, so checkpoints are not interchangeable with
  spconv-trained ones without a check. Training from scratch is unaffected.

    from utils import torch_kernels; torch_kernels.install()    # before importing the model
"""
import importlib.machinery
import itertools
import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- torch_scatter ----------------------------- #
def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
    assert dim == 0, "replacement supports dim=0"
    index = index.long()
    n = int(dim_size if dim_size is not None else (int(index.max()) + 1 if index.numel() else 0))
    shape = (n,) + tuple(src.shape[1:])
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    if reduce in ("sum", "add"):
        return src.new_zeros(shape).scatter_add(0, idx, src)
    mode = {"mean": "mean", "max": "amax", "min": "amin"}[reduce]
    return src.new_zeros(shape).scatter_reduce(0, idx, src, reduce=mode, include_self=False)


def segment_csr(src, indptr, reduce="sum"):
    counts = (indptr[1:] - indptr[:-1]).long()
    index = torch.repeat_interleave(torch.arange(len(counts), device=src.device), counts)
    return scatter(src, index, 0, dim_size=len(counts), reduce=reduce)


# ------------------------------- flash_attn ------------------------------ #
def _grouped_attention(q, k, v, cu_q, cu_k, scale, dropout_p):
    """q (Nq, H, D), k/v (Nk, H, D), sequence boundaries -> (Nq, H, D)."""
    out = q.new_empty(q.shape[0], q.shape[1], v.shape[-1])
    lq = (cu_q[1:] - cu_q[:-1]).tolist()
    lk = (cu_k[1:] - cu_k[:-1]).tolist()
    starts_q, starts_k = cu_q[:-1].tolist(), cu_k[:-1].tolist()
    groups = {}
    for i, key in enumerate(zip(lq, lk)):
        groups.setdefault(key, []).append(i)
    dtype = q.dtype if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) else torch.float32
    for (a, b), seqs in groups.items():
        if a == 0:
            continue
        iq = torch.stack([torch.arange(starts_q[s], starts_q[s] + a, device=q.device) for s in seqs])  # (G, a)
        ik = torch.stack([torch.arange(starts_k[s], starts_k[s] + b, device=q.device) for s in seqs])
        qq = q[iq].transpose(1, 2).to(dtype)          # (G, H, a, D)
        kk = k[ik].transpose(1, 2).to(dtype)
        vv = v[ik].transpose(1, 2).to(dtype)
        o = F.scaled_dot_product_attention(qq, kk, vv, dropout_p=dropout_p, scale=scale)
        out[iq.reshape(-1)] = o.transpose(1, 2).reshape(-1, o.shape[1], o.shape[-1]).to(out.dtype)
    return out


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, **_):
    return _grouped_attention(qkv[:, 0], qkv[:, 1], qkv[:, 2], cu_seqlens, cu_seqlens, softmax_scale, dropout_p)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           dropout_p=0.0, softmax_scale=None, **_):
    return _grouped_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, dropout_p)


# --------------------------------- spconv -------------------------------- #
class SparseModule(nn.Module):
    pass


class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size, indice_dict=None):
        self.features, self.indices = features, indices
        self.spatial_shape, self.batch_size = spatial_shape, batch_size
        self.indice_dict = {} if indice_dict is None else indice_dict

    def replace_feature(self, features):
        return SparseConvTensor(features, self.indices, self.spatial_shape, self.batch_size, self.indice_dict)


def neighbour_map(indices, kernel_size):
    """(N, 4) [batch, x, y, z] int -> (N, K^3) long index of each active neighbour, -1 if none.
    Offset order: itertools.product over x, y, z in range(-k//2 .. k//2)."""
    idx = indices.long()
    r = kernel_size // 2
    coords = idx[:, 1:] + r                                        # keep neighbours non-negative
    span = coords.max(0).values + r + 1
    b = idx[:, 0]

    def key(c):
        return ((b * span[0] + c[:, 0]) * span[1] + c[:, 1]) * span[2] + c[:, 2]

    keys = key(coords)
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    offsets = list(itertools.product(range(-r, r + 1), repeat=3))
    nbr = torch.full((idx.shape[0], len(offsets)), -1, dtype=torch.long, device=idx.device)
    for j, off in enumerate(offsets):
        c = coords + torch.tensor(off, device=idx.device)
        k = key(c)
        pos = torch.searchsorted(sorted_keys, k).clamp(max=sorted_keys.numel() - 1)
        hit = sorted_keys[pos] == k
        nbr[hit, j] = order[pos[hit]]
    return nbr


class SubMConv3d(SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=True,
                 indice_key=None, **_):
        super().__init__()
        assert stride == 1 and dilation == 1, "submanifold conv with stride 1 only"
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, kernel_size
        self.indice_key = indice_key
        self.weight = nn.Parameter(torch.empty(out_channels, kernel_size, kernel_size, kernel_size, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        fan_in = in_channels * kernel_size ** 3
        nn.init.uniform_(self.weight, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))

    def forward(self, x: SparseConvTensor):
        cache_key = (self.indice_key, self.kernel_size)
        nbr = x.indice_dict.get(cache_key) if self.indice_key is not None else None
        if nbr is None:
            with torch.no_grad():
                nbr = neighbour_map(x.indices, self.kernel_size)
            if self.indice_key is not None:
                x.indice_dict[cache_key] = nbr
        feats = x.features
        w = self.weight.reshape(self.out_channels, -1, self.in_channels).to(feats.dtype)   # (out, K, in)
        # gather each voxel's neighbourhood (zeros for inactive sites) and contract with the kernel
        padded = torch.cat([feats, feats.new_zeros(1, feats.shape[1])], 0)
        gathered = padded[torch.where(nbr >= 0, nbr, feats.shape[0])]                     # (N, K, in)
        out = torch.einsum("nki,oki->no", gathered, w)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return x.replace_feature(out)


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__dict__.update(attrs)
    return mod


def install(force=False):
    """Register the replacements under the kernel packages' names, unless the real ones import."""
    def missing(name):
        if force:
            return True
        try:
            __import__(name)
            return False
        except Exception:
            return True

    installed = []
    if missing("torch_scatter"):
        sys.modules["torch_scatter"] = _module("torch_scatter", scatter=scatter, segment_csr=segment_csr)
        installed.append("torch_scatter")
    if missing("flash_attn"):
        sys.modules["flash_attn"] = _module("flash_attn", flash_attn_varlen_qkvpacked_func=flash_attn_varlen_qkvpacked_func,
                                            flash_attn_varlen_func=flash_attn_varlen_func, __version__="torch-sdpa")
        installed.append("flash_attn")
    if missing("spconv"):
        modules = _module("spconv.pytorch.modules", SparseModule=SparseModule,
                          is_spconv_module=lambda m: isinstance(m, SparseModule))
        pytorch = _module("spconv.pytorch", SparseConvTensor=SparseConvTensor, SubMConv3d=SubMConv3d,
                          SparseModule=SparseModule, modules=modules)
        sys.modules["spconv"] = _module("spconv", pytorch=pytorch)
        sys.modules["spconv.pytorch"], sys.modules["spconv.pytorch.modules"] = pytorch, modules
        installed.append("spconv")
    return installed
