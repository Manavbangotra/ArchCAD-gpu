"""
Pure-PyTorch stand-ins for flash-attn, spconv and torch_scatter, for CPU checks only.

The model is trained with the real CUDA kernels (cloud/Dockerfile). These stand-ins
let the checks build the full VecFormer and run forward and backward on a laptop:
attention is exact (per-sequence softmax attention), scatter is exact, and the
sparse convolutions are point-wise linear layers (no neighbourhood), which keeps
every shape and code path but not the convolution's receptive field.

    import cpu_kernels; cpu_kernels.install()   # before importing the model
"""
import importlib.machinery
import sys
import types

import torch
import torch.nn as nn


# ----------------------------- torch_scatter ----------------------------- #
def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
    assert dim == 0, "stand-in supports dim=0"
    index = index.long()
    n = int(dim_size if dim_size is not None else (int(index.max()) + 1 if index.numel() else 0))
    shape = (n,) + tuple(src.shape[1:])
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    if reduce in ("sum", "add"):
        return src.new_zeros(shape).scatter_add(0, idx, src)
    if reduce == "mean":
        return src.new_zeros(shape).scatter_reduce(0, idx, src, reduce="mean", include_self=False)
    if reduce == "max":
        return src.new_zeros(shape).scatter_reduce(0, idx, src, reduce="amax", include_self=False)
    if reduce == "min":
        return src.new_zeros(shape).scatter_reduce(0, idx, src, reduce="amin", include_self=False)
    raise ValueError(reduce)


def segment_csr(src, indptr, reduce="sum"):
    counts = (indptr[1:] - indptr[:-1]).long()
    index = torch.repeat_interleave(torch.arange(len(counts), device=src.device), counts)
    return scatter(src, index, 0, dim_size=len(counts), reduce=reduce)


# ------------------------------- flash_attn ------------------------------ #
def _attend(q, k, v, scale):
    """q (Lq, H, D), k/v (Lk, H, D) -> (Lq, H, D)"""
    q, k, v = (t.float().transpose(0, 1) for t in (q, k, v))
    att = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
    return (att @ v).transpose(0, 1)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, **_):
    d = qkv.shape[-1]
    scale = softmax_scale if softmax_scale is not None else d ** -0.5
    out = []
    for a, b in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        out.append(_attend(qkv[a:b, 0], qkv[a:b, 1], qkv[a:b, 2], scale))
    return torch.cat(out, 0).to(qkv.dtype)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           dropout_p=0.0, softmax_scale=None, **_):
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    out = []
    for (qa, qb), (ka, kb) in zip(zip(cu_seqlens_q[:-1].tolist(), cu_seqlens_q[1:].tolist()),
                                  zip(cu_seqlens_k[:-1].tolist(), cu_seqlens_k[1:].tolist())):
        out.append(_attend(q[qa:qb], k[ka:kb], v[ka:kb], scale))
    return torch.cat(out, 0).to(q.dtype)


# --------------------------------- spconv -------------------------------- #
class SparseModule(nn.Module):
    pass


class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size):
        self.features, self.indices = features, indices
        self.spatial_shape, self.batch_size = spatial_shape, batch_size

    def replace_feature(self, features):
        return SparseConvTensor(features, self.indices, self.spatial_shape, self.batch_size)


class SubMConv3d(SparseModule):
    """Point-wise stand-in: a linear map of each voxel's features."""

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True, indice_key=None, **_):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: SparseConvTensor):
        return x.replace_feature(self.linear(x.features))


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)   # transformers probes find_spec
    mod.__dict__.update(attrs)
    return mod


def install():
    if "torch_scatter" not in sys.modules:
        sys.modules["torch_scatter"] = _module("torch_scatter", scatter=scatter, segment_csr=segment_csr)
    if "flash_attn" not in sys.modules:
        sys.modules["flash_attn"] = _module(
            "flash_attn", flash_attn_varlen_qkvpacked_func=flash_attn_varlen_qkvpacked_func,
            flash_attn_varlen_func=flash_attn_varlen_func)
    if "spconv" not in sys.modules:
        modules = _module("spconv.pytorch.modules", SparseModule=SparseModule,
                          is_spconv_module=lambda m: isinstance(m, SparseModule))
        pytorch = _module("spconv.pytorch", SparseConvTensor=SparseConvTensor, SubMConv3d=SubMConv3d,
                          SparseModule=SparseModule, modules=modules)
        sys.modules["spconv"] = _module("spconv", pytorch=pytorch)
        sys.modules["spconv.pytorch"], sys.modules["spconv.pytorch.modules"] = pytorch, modules
