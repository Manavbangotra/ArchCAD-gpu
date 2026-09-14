"""
Text modality for the line-based spotter, after TextCAD (arXiv 2607.12678, Sec. 3).

TACE (Type-Attribute Correlation Encoder)
    A text annotation is a type (embedding) plus up to `a` attributes (one MLP per
    numeric attribute, an embedding for the grade). The type token attends over its
    present attributes with masked multi-head cross-attention,
    S_t = MLP(concat(heads) + T_s), and the annotation's own geometry is added:
    X_t^0 = S_t + MLP(F_t). The paper uses D=32, H=4, a=4.

MSF (Multi-level Semantic Filtering), applied at every backbone stage l
    X_t^l = MLP_l(X_t^0)
    C^l   = (X_t W_q)(X_g W_k)^T / sqrt(D_l)      relevance of text m to line n
    r_m   = max_n C_{m,n}                          (within the same drawing)
    G^l   = hard-concrete gate on r (Louizos et al., L0 regularisation)
    X_t'  = G * (X_t W_v)
    X_g  <- X_g + CrossAttn(X_g -> X_t') with a distance bias, so nearby text dominates
    It also returns the expected number of open gates; the loss scales it by
    lambda_c (1e-4 in the paper).

Not specified by the paper and chosen here: the hard-concrete constants (Louizos
defaults tau=2/3, gamma=-0.1, zeta=1.1), the spatial term (a learned per-head
penalty on squared distance), and zero-initialised output projections, so that a
model with text starts exactly as the model without it.

Pure PyTorch (no flash-attn or spconv), so it runs and is tested anywhere.
"""
from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TextContext:
    """All text annotations of a batch, concatenated.

    feats: (T, D) TACE output; pos: (T, 2) normalised x, y in the lines' frame;
    batch: (T,) drawing index of each annotation.
    """
    feats: torch.Tensor
    pos: torch.Tensor
    batch: torch.Tensor


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(inplace=True), nn.Linear(h, o))


class TACE(nn.Module):
    def __init__(self, num_types: int, num_grades: int, dim: int = 32, heads: int = 4,
                 num_numeric: int = 3, geo_dim: int = 3):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.num_numeric = dim, heads, num_numeric
        self.type_emb = nn.Embedding(num_types, dim)
        self.attr_mlps = nn.ModuleList([_mlp(1, dim, dim) for _ in range(num_numeric)])
        self.grade_emb = nn.Embedding(num_grades, dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = _mlp(dim, dim, dim)
        self.geo = _mlp(geo_dim, dim, dim)

    def forward(self, types, attrs, grades, masks, geo):
        """types (T,), attrs (T, num_numeric), grades (T,), masks (T, num_numeric + 1) bool,
        geo (T, geo_dim) -> (T, dim)"""
        t = self.type_emb(types)                                            # (T, D)
        a = [mlp(attrs[:, j:j + 1]) for j, mlp in enumerate(self.attr_mlps)]
        a.append(self.grade_emb(grades))
        a = torch.stack(a, dim=1)                                           # (T, a, D)
        n, A, D = a.shape
        h, d = self.heads, D // self.heads
        q = self.q(t).view(n, 1, h, d).transpose(1, 2)                      # (T, h, 1, d)
        k = self.k(a).view(n, A, h, d).transpose(1, 2)                      # (T, h, A, d)
        v = self.v(a).view(n, A, h, d).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(d)                     # (T, h, 1, A)
        m = masks.bool().view(n, 1, 1, A)
        logits = logits.masked_fill(~m, float("-inf"))
        w = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)        # no attribute present
        heads_out = (w @ v).transpose(1, 2).reshape(n, D)
        return self.out(heads_out + t) + self.geo(geo)


def hard_concrete_gate(log_alpha, training, tau=2.0 / 3.0, gamma=-0.1, zeta=1.1):
    """Gate in [0, 1] per element, and its L0 complexity (expected open gates)."""
    if training:
        u = torch.rand_like(log_alpha).clamp_(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / tau)
    else:
        s = torch.sigmoid(log_alpha)
    gate = (s * (zeta - gamma) + gamma).clamp(0.0, 1.0)
    l0 = torch.sigmoid(log_alpha - tau * math.log(-gamma / zeta)).sum()
    return gate, l0


class MSFTextFusion(nn.Module):
    """One MSF level: filter text by relevance to this stage's lines, then fuse it in."""

    def __init__(self, line_dim: int, text_dim: int = 32, heads: int = 4, attn_dim: Optional[int] = None,
                 knn: int = 16):
        super().__init__()
        # Each line attends to its `knn` nearest annotations (0 = all). The distance
        # bias already makes far text negligible; this bounds memory at N * knn
        # instead of N * T (a 60k-segment US window with 500 notes).
        self.knn = knn
        attn_dim = attn_dim or max(heads * 8, min(line_dim, 128))
        attn_dim -= attn_dim % heads
        self.heads, self.attn_dim = heads, attn_dim
        self.text_proj = _mlp(text_dim, attn_dim, attn_dim)                # MLP_l
        self.rel_q = nn.Linear(attn_dim, attn_dim, bias=False)             # W_q (text)
        self.rel_k = nn.Linear(line_dim, attn_dim, bias=False)             # W_k (lines)
        self.val = nn.Linear(attn_dim, attn_dim)                           # W_v
        self.norm = nn.LayerNorm(line_dim)
        self.q = nn.Linear(line_dim, attn_dim)
        self.k = nn.Linear(attn_dim, attn_dim)
        self.v = nn.Linear(attn_dim, attn_dim)
        self.out = nn.Linear(attn_dim, line_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # learned per-head distance penalty; softplus keeps it positive (starts at 1)
        self.dist_scale = nn.Parameter(torch.full((heads,), math.log(math.e - 1)))

    def forward(self, feat, pos, batch, text: Optional[TextContext]):
        """feat (N, C), pos (N, 2), batch (N,) -> (fused feat (N, C), L0 complexity scalar)."""
        l0_total = feat.new_zeros(())
        if text is None or text.feats.numel() == 0:
            return feat, l0_total
        x_t = self.text_proj(text.feats)                                   # (T, A)
        out = torch.zeros_like(feat)
        h, d = self.heads, self.attn_dim // self.heads
        for b in torch.unique(batch).tolist():
            ln = (batch == b).nonzero(as_tuple=True)[0]
            tx = (text.batch == b).nonzero(as_tuple=True)[0]
            if tx.numel() == 0 or ln.numel() == 0:
                continue
            g_feat, g_pos = feat[ln], pos[ln]
            t_feat, t_pos = x_t[tx], text.pos[tx]
            rel = (self.rel_q(t_feat) @ self.rel_k(g_feat).T) / math.sqrt(self.attn_dim)   # (Tb, Nb)
            gate, l0 = hard_concrete_gate(rel.max(dim=1).values, self.training)
            l0_total = l0_total + l0
            kv = gate.unsqueeze(-1) * self.val(t_feat)                     # (Tb, A)
            q = self.q(self.norm(g_feat)).view(-1, h, d).transpose(0, 1)   # (h, Nb, d)
            k = self.k(kv).view(-1, h, d).transpose(0, 1)                  # (h, Tb, d)
            v = self.v(kv).view(-1, h, d).transpose(0, 1)
            penalty = F.softplus(self.dist_scale).view(h, 1, 1) * 100.0    # coords in [-0.5, 0.5]
            if self.knn and self.knn < tx.numel():
                with torch.no_grad():
                    near = torch.cdist(g_pos, t_pos).topk(self.knn, dim=1, largest=False).indices  # (Nb, k)
                dist2 = (g_pos.unsqueeze(1) - t_pos[near]).pow(2).sum(-1)  # (Nb, k)
                logits = (q.unsqueeze(-2) * k[:, near]).sum(-1) / math.sqrt(d)           # (h, Nb, k)
                att = torch.softmax(logits - penalty * dist2.unsqueeze(0), dim=-1)
                fused = (att.unsqueeze(-1) * v[:, near]).sum(-2)           # (h, Nb, d)
            else:
                dist2 = torch.cdist(g_pos, t_pos).pow(2)
                logits = q @ k.transpose(-1, -2) / math.sqrt(d)            # (h, Nb, Tb)
                att = torch.softmax(logits - penalty * dist2.unsqueeze(0), dim=-1)
                fused = att @ v                                            # (h, Nb, d)
            fused = fused.transpose(0, 1).reshape(-1, self.attn_dim)       # (Nb, A)
            out[ln] = self.out(fused)
        return feat + out, l0_total
