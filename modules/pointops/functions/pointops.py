"""
Pure-PyTorch CPU fallback for the `pointops` CUDA extension.

The upstream project depends on a compiled CUDA extension
(https://github.com/YodaEmbedding/pointops) that cannot be built or run without
an NVIDIA GPU. This module reimplements the subset of that API which DPSS
actually calls, using only stock PyTorch ops, so the network runs on CPU.

API compatibility notes (matched against the call sites in this repo):
  * `knnquery` returns `(idx, dist)` where `dist` is the *euclidean* distance
    (the CUDA kernel computes squared distance and the wrapper sqrt's it).
  * `queryandgroup` returns `(grouped, idx)` when `use_xyz=True` and just
    `grouped` when `use_xyz=False` — see modules/pointtransformer_utils.py:39-40.
  * All index tensors are int32 and hold *global* (not per-batch) indices, so
    callers can index the flat point tensors directly.

These are semantically equivalent to the CUDA kernels but considerably slower;
they are intended for CPU inference, debugging and small-scale runs.
"""

import torch


def _as_offsets(offset):
    """Convert an exclusive-end offset tensor to a list of (start, end) pairs."""
    ends = offset.detach().cpu().long().tolist()
    starts = [0] + ends[:-1]
    return list(zip(starts, ends))


def knnquery(nsample, xyz, new_xyz=None, offset=None, new_offset=None):
    """k-nearest-neighbour search, batched via offsets.

    Args:
        nsample: number of neighbours (k).
        xyz: (N, 3) support points.
        new_xyz: (M, 3) query points. Defaults to `xyz`.
        offset / new_offset: exclusive-end batch boundaries for xyz / new_xyz.

    Returns:
        idx: (M, nsample) int32 global indices into `xyz`.
        dist: (M, nsample) float euclidean distances.
    """
    if new_xyz is None:
        new_xyz = xyz
    if offset is None:
        offset = torch.tensor([xyz.shape[0]], dtype=torch.int32, device=xyz.device)
    if new_offset is None:
        new_offset = offset

    nsample = int(nsample)
    m = new_xyz.shape[0]
    idx = torch.zeros(m, nsample, dtype=torch.int32, device=xyz.device)
    dist = torch.zeros(m, nsample, dtype=torch.float32, device=xyz.device)

    for (s_src, e_src), (s_dst, e_dst) in zip(_as_offsets(offset), _as_offsets(new_offset)):
        src = xyz[s_src:e_src].float()          # (n_b, 3)
        dst = new_xyz[s_dst:e_dst].float()      # (m_b, 3)
        if src.shape[0] == 0 or dst.shape[0] == 0:
            continue

        # The matmul-based cdist path loses precision through catastrophic
        # cancellation: a point's distance to itself comes out ~1e-3 instead of
        # 0. `interpolation` weights by 1/(dist + 1e-8), so that error would
        # badly skew the weight of an exactly-coincident point. Force the
        # direct computation, which is exact and cheap at these sizes.
        d = torch.cdist(dst, src, compute_mode="donot_use_mm_for_euclid_dist")
        # The CUDA kernel silently clamps k to the number of available points;
        # replicate that instead of raising, so tiny drawings still work.
        k = min(nsample, src.shape[0])
        d_k, i_k = torch.topk(d, k, dim=1, largest=False)

        if k < nsample:
            # Pad by repeating the nearest neighbour, matching the kernel's
            # behaviour of leaving trailing slots pointing at valid points.
            d_k = torch.cat([d_k, d_k[:, :1].expand(-1, nsample - k)], dim=1)
            i_k = torch.cat([i_k, i_k[:, :1].expand(-1, nsample - k)], dim=1)

        idx[s_dst:e_dst] = (i_k + s_src).int()  # local -> global index
        dist[s_dst:e_dst] = d_k

    return idx, dist


def furthestsampling(xyz, offset, new_offset):
    """Farthest point sampling, batched via offsets.

    Returns (M,) int32 global indices into `xyz`, where M = new_offset[-1].
    """
    out = []
    for (s_src, e_src), (s_dst, e_dst) in zip(_as_offsets(offset), _as_offsets(new_offset)):
        n = e_src - s_src
        m = e_dst - s_dst
        if m <= 0:
            continue
        pts = xyz[s_src:e_src].float()

        if m >= n:  # nothing to drop
            sel = torch.arange(n, device=xyz.device)
            if m > n:  # pad by repeating the last point
                sel = torch.cat([sel, sel[-1:].expand(m - n)])
            out.append(sel + s_src)
            continue

        # Standard greedy FPS, seeded at index 0 like the CUDA kernel.
        sel = torch.zeros(m, dtype=torch.long, device=xyz.device)
        best = torch.full((n,), float("inf"), device=xyz.device)
        last = 0
        for i in range(1, m):
            d = torch.sum((pts - pts[last]) ** 2, dim=1)
            best = torch.minimum(best, d)
            last = int(torch.argmax(best).item())
            sel[i] = last
        out.append(sel + s_src)

    if not out:
        return torch.zeros(0, dtype=torch.int32, device=xyz.device)
    return torch.cat(out).int()


def sectorized_fps(xyz, offset, new_offset, num_sector=1):
    """FPS restricted to angular sectors around each batch's centroid.

    A training-time speed optimisation upstream; here it mainly preserves
    behavioural parity. Falls back to plain FPS when num_sector <= 1.
    """
    num_sector = int(num_sector)
    if num_sector <= 1:
        return furthestsampling(xyz, offset, new_offset)

    out = []
    for (s_src, e_src), (s_dst, e_dst) in zip(_as_offsets(offset), _as_offsets(new_offset)):
        n, m = e_src - s_src, e_dst - s_dst
        if m <= 0:
            continue
        pts = xyz[s_src:e_src].float()
        if m >= n:
            sel = torch.arange(n, device=xyz.device)
            if m > n:
                sel = torch.cat([sel, sel[-1:].expand(m - n)])
            out.append(sel + s_src)
            continue

        centred = pts[:, :2] - pts[:, :2].mean(0, keepdim=True)
        angle = torch.atan2(centred[:, 1], centred[:, 0])          # [-pi, pi]
        bucket = ((angle + torch.pi) / (2 * torch.pi) * num_sector).long().clamp(0, num_sector - 1)

        picked, taken = [], 0
        for s in range(num_sector):
            local = torch.nonzero(bucket == s, as_tuple=False).squeeze(1)
            if local.numel() == 0:
                continue
            # Allocate this sector's share, giving the last sector the remainder.
            quota = m - taken if s == num_sector - 1 else max(1, round(m * local.numel() / n))
            quota = min(quota, local.numel(), m - taken)
            if quota <= 0:
                continue
            sub_off = torch.tensor([local.numel()], dtype=torch.int32, device=xyz.device)
            sub_new = torch.tensor([quota], dtype=torch.int32, device=xyz.device)
            sub_sel = furthestsampling(pts[local], sub_off, sub_new).long()
            picked.append(local[sub_sel])
            taken += quota
            if taken >= m:
                break

        sel = torch.cat(picked) if picked else torch.arange(m, device=xyz.device)
        if sel.numel() < m:  # top up if rounding left us short
            pad = sel[-1:].expand(m - sel.numel())
            sel = torch.cat([sel, pad])
        out.append(sel[:m] + s_src)

    if not out:
        return torch.zeros(0, dtype=torch.int32, device=xyz.device)
    return torch.cat(out).int()


def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True):
    """Gather each query point's k neighbours and their features.

    Returns `(grouped, idx)` when use_xyz=True, else just `grouped` — this
    asymmetry matches how the model unpacks the result.
    """
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset)

    nsample = int(nsample)
    m = new_xyz.shape[0]
    flat = idx.reshape(-1).long()

    grouped_xyz = xyz[flat, :].view(m, nsample, 3) - new_xyz.unsqueeze(1)
    grouped_feat = feat[flat, :].view(m, nsample, feat.shape[1])

    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), dim=-1), idx
    return grouped_feat


def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    """Inverse-distance-weighted interpolation of `feat` onto `new_xyz`."""
    k = int(k)
    idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset)   # (m, k)
    dist_recip = 1.0 / (dist + 1e-8)
    weight = dist_recip / torch.sum(dist_recip, dim=1, keepdim=True)

    new_feat = torch.zeros(new_xyz.shape[0], feat.shape[1],
                           dtype=feat.dtype, device=feat.device)
    for i in range(k):
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1).to(feat.dtype)
    return new_feat


# Aliases kept for API parity with the CUDA extension.
knn_query = knnquery
furthest_sampling = furthestsampling
