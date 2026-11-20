#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5.7: batch the 6 cameras of one FPN scale into a single ``grid_sample``.

v5.5a left ``_sample`` looping over cameras, i.e. 3 scales x 6 cameras = 18
``grid_sample`` launches per frame.  Each launch only gathers
``N`` candidates x ``Cf`` x ``cand_size^2`` values from a tiny source map
(``Cf x Hf x Wf`` fits in L2), so the block is dominated by gather/launch
overhead rather than DRAM traffic.

This module fuses the camera axis into the batch axis:

    input  (B*Nc*N, Cf, Hf, Wf)   <- feat broadcast over the N candidates
    grid   (B*Nc*N, h, w, 2)      <- the same per-camera grids b55 built
    output (B, Nc, N, Cf, h, w)   -> accumulated in camera order

``grid_sample`` computes every output element independently, so the fused
call is **bit-identical** to the per-camera loop; the camera accumulation
still uses the same sequential ``addcmul_`` order as b55.  Nothing about the
grids, validity masks, support counts, variance or cost is changed.

Enable with ``V5_HEAD_CAMBATCH=1`` (requires ``V5_HEAD_GEO=1``); default off.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import b55_head_geo


def _sample_cambatch(bev, feat, geo):
    """Drop-in replacement for ``b55_head_geo._sample``."""
    N, h, w, B = geo["N"], geo["h"], geo["w"], geo["B"]
    dtype = feat.dtype
    feat = bev.proj(feat).float()
    Cf = feat.shape[1]
    Hf, Wf = feat.shape[-2], feat.shape[-1]
    feat = feat.reshape(B, feat.shape[0] // B, Cf, Hf, Wf)
    Nc = feat.shape[1]

    # one launch for all cameras of this scale
    inp = feat.unsqueeze(2).expand(B, Nc, N, Cf, Hf, Wf).reshape(
        B * Nc * N, Cf, Hf, Wf)
    grid = torch.stack(
        [g.reshape(B, N, h, w, 2) for g in geo["grids"]], dim=1).reshape(
            B * Nc * N, h, w, 2)
    out = F.grid_sample(
        inp, grid, mode="bilinear", padding_mode="zeros",
        align_corners=False,
    ).reshape(B, Nc, N, Cf, h, w)

    sum_feat = torch.zeros(B, N, Cf, h, w, device=feat.device,
                           dtype=torch.float32)
    sum_sq = torch.zeros_like(sum_feat)
    for c in range(Nc):
        sample = out[:, c]
        valid_f = geo["valids"][c]
        sum_feat.addcmul_(sample, valid_f)
        sum_sq.addcmul_(sample * sample, valid_f)

    hit = geo["hit"]
    mean = sum_feat / hit.clamp_min(1.0)
    meansq = sum_sq / hit.clamp_min(1.0)
    var = (meansq - mean * mean).clamp_min(0.0)
    cost = var.mean(dim=2, keepdim=True)
    support_penalty = 0.05 * (2.0 - hit).clamp_min(0.0)
    cost = torch.where(hit > 0, cost + support_penalty, torch.ones_like(cost))
    valid_out = (hit > 0).float()
    return mean.to(dtype=dtype), cost.to(dtype=dtype), valid_out.to(dtype=dtype)


def install():
    """Rebind the sampler used by b55's hoisted ``_candidate_condition``."""
    prev = b55_head_geo._sample
    b55_head_geo._sample = _sample_cambatch
    return prev
