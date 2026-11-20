#!/usr/bin/env python3
"""v8-A fine-grid Gaussian assembly: head maps -> RoGS GaussianModel2D.

All head maps are (B, C, F, F).  B is 1 per rank (renderer contract).  The
Gaussian XY is the fixed fine-grid centre plus the learned BEV offset scaled by
the fine cell; z, scale, rotation, opacity, semantic and the 12 SH coefficients
come straight off the fine grid.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from rogs_adapter import build_gaussians


def build_fine_gaussians(initial, base_xy, cell: float):
    """Args:
        initial: dict of (1,C,F,F) maps with keys z, xy, op, scale, rot, sem, sh.
        base_xy: (1, F*F, 2) fine-grid centres in the BDA frame.
        cell: fine-grid metres per cell.
    """
    if initial["z"].shape[0] != 1:
        raise NotImplementedError("The RoGS renderer contract is batch-size one per rank")
    device = initial["z"].device
    base_xy = base_xy.to(device=device, dtype=initial["z"].dtype)
    f = initial["z"].shape[-1]
    count = f * f
    if base_xy.reshape(-1, 2).shape[0] != count:
        raise ValueError("base_xy and fine Gaussian grid disagree")

    xy = (base_xy.reshape(-1, 2)
          + initial["xy"][0].permute(1, 2, 0).reshape(count, 2) * float(cell))
    z = initial["z"][0].permute(1, 2, 0).reshape(count, 1)
    sh = initial["sh"][0].permute(1, 2, 0).reshape(count, 12)
    sem = initial["sem"][0].permute(1, 2, 0).reshape(count, -1)
    scale = initial["scale"][0].permute(1, 2, 0).reshape(count, 2).clamp(-6.0, 2.0)
    rotation = F.normalize(
        initial["rot"][0].permute(1, 2, 0).reshape(count, 4), dim=-1, eps=1e-6
    )
    opacity = initial["op"][0].permute(1, 2, 0).reshape(count, 1)
    pc = build_gaussians(xy, z, sh, sem, scale, rotation, opacity)
    return pc, xy, z
