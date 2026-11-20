#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v7.2 (B): fused splat source layout (opt-in ``V7_SPLAT_FUSED=1``).

``b54_bevpool.splat_shared`` builds ``feat_all = context_p.unsqueeze(2) *
weights.unsqueeze(1)`` in ``(BN, C, 4, P)`` and then scatters
``feat_all[sl].permute(1, 0, 2, 3).reshape(C, -1)``.  The reshaped permuted
view is not contiguous, so that reshape *materialises a second full-size
source* (one extra read + write of ``BN*C*4*P`` floats) purely to reorder the
camera axis in front of the corner axis.

The scatter only needs columns ordered as (camera, corner, pixel).  Building
the source directly as ``(C, Nc, 4, P)`` from the permuted context gets that
order from a single broadcast multiply, so the reorder copy disappears and the
peak intermediate halves.

Two further redundancies in the same tail:
  * ``hit`` and ``z_den`` are the *same* sum of the same weights, accumulated
    with the same indices -- one of the two index_adds computes a duplicate;
  * the remaining scalar statistics (weight, z numerator) fit one two-channel
    accumulator, halving their launches.

Values and accumulation order are unchanged, so the output is bit-identical to
``splat_shared``; ``V7_B_VERIFY=1`` checks that on the first frame.
"""
from __future__ import annotations

import os

import torch

from continuous_lift_fpn import _bin_centers

_STATE = {"installed": False, "verified": False, "calls": 0}


def splat_fused(self, context, depth_prob, mats, sweep_index,
                return_depth_z=False):
    """Drop-in replacement for ``b54_bevpool.splat_shared``."""
    BN, C, fh, fw = context.shape
    Nc = self._nc
    B = BN // Nc

    device, dtype = context.device, context.dtype
    geo = getattr(self, "_v72_geo", None)
    if geo is None:
        Gx, Gy = int(self.voxel_num[0]), int(self.voxel_num[1])
        cell_x, cell_y = float(self.voxel_size[0]), float(self.voxel_size[1])
        geo = (Gx, Gy, cell_x, cell_y,
               float(self.voxel_coord[0] - cell_x / 2.0),
               float(self.voxel_coord[1] - cell_y / 2.0))
        self._v72_geo = geo
    Gx, Gy, cell_x, cell_y, xmin, ymin = geo

    centers = _bin_centers(self.d_bound, self.depth_channels, device,
                           dtype).view(1, -1, 1, 1)
    depth = (depth_prob * centers).sum(dim=1, keepdim=True)
    depth = depth.clamp(self.d_bound[0], self.d_bound[1])

    u, v = self._pixel_grid(fh, fw, device, dtype)
    P = u.numel()
    ones = torch.ones_like(u)
    pts = torch.stack([u.expand(BN, P), v.expand(BN, P),
                       depth.reshape(BN, P), ones.expand(BN, P)], dim=1)

    s2e = mats['sensor2ego_mats'][:, sweep_index].reshape(BN, 4, 4)
    intrin = mats['intrin_mats'][:, sweep_index].reshape(BN, 4, 4)
    ida = mats['ida_mats'][:, sweep_index].reshape(BN, 4, 4)
    bda = mats['bda_mat'].to(device=device, dtype=dtype)

    _ida_inv = mats.get('ida_inv')
    pts = (_ida_inv[:, sweep_index].reshape(BN, 4, 4) if _ida_inv is not None
           else torch.inverse(ida)).matmul(pts)

    zcam = pts[:, 2:3, :]
    pts = torch.cat([pts[:, 0:2, :] * zcam, pts[:, 2:, :]], dim=1)
    _intrin_inv = mats.get('intrin_inv')
    combine = s2e.matmul(
        _intrin_inv[:, sweep_index].reshape(BN, 4, 4)
        if _intrin_inv is not None else torch.inverse(intrin))
    pts = combine.matmul(pts)

    bda_rep = bda.unsqueeze(1).expand(B, Nc, 4, 4).reshape(B * Nc, 4, 4)
    pts = bda_rep.matmul(pts)
    xy = pts[:, :2, :]
    ego_z = pts[:, 2:3, :].detach().float()

    valid = (zcam.squeeze(1) > 1e-2)
    gx = (xy[:, 0] - xmin) / cell_x
    gy = (xy[:, 1] - ymin) / cell_y
    valid &= (gx >= 0.0) & (gx <= Gx - 1.0)
    valid &= (gy >= 0.0) & (gy <= Gy - 1.0)
    valid &= torch.isfinite(gx) & torch.isfinite(gy)

    x0 = torch.floor(gx)
    y0 = torch.floor(gy)
    wx = gx - x0
    wy = gy - y0
    x0c = x0.clamp(0, Gx - 1).long()
    x1c = (x0 + 1.0).clamp(0, Gx - 1).long()
    y0c = y0.clamp(0, Gy - 1).long()
    y1c = (y0 + 1.0).clamp(0, Gy - 1).long()

    vf = valid.to(dtype)
    weights = torch.stack([
        ((1 - wx) * (1 - wy)) * vf,
        (wx * (1 - wy)) * vf,
        ((1 - wx) * wy) * vf,
        (wx * wy) * vf,
    ], dim=1)                                                  # (BN, 4, P)
    lin = torch.stack([
        y0c * Gx + x0c, y0c * Gx + x1c, y1c * Gx + x0c, y1c * Gx + x1c
    ], dim=1)                                                  # (BN, 4, P)

    L = Gy * Gx
    ctx_bn = context.reshape(BN, C, P)
    bev_all = context.new_zeros(B * C, L)
    # channel 0: sum of weights (== hit == z_den), channel 1: z numerator
    stat_all = torch.zeros(2, B, L, dtype=torch.float32, device=device)

    for b in range(B):
        sl = slice(b * Nc, (b + 1) * Nc)
        idx = lin[sl].reshape(-1)
        # (C, Nc, P) -> (C, Nc, 4, P): reorder via broadcast multiply instead
        # of a separate permute+reshape copy of the whole source.
        feat = (ctx_bn[sl].permute(1, 0, 2).unsqueeze(2)
                * weights[sl].unsqueeze(0))
        bev_all[b * C:(b + 1) * C].index_add_(1, idx, feat.reshape(C, -1))
        w_b = weights[sl].detach().float()
        stat_all[:, b:b + 1].index_add_(
            2, idx, torch.stack([w_b, ego_z[sl] * w_b]).reshape(2, 1, -1))

    bev = bev_all.reshape(B, C, Gy, Gx)
    hit_raw = stat_all[0].reshape(B, 1, Gy, Gx)
    z_num = stat_all[1].reshape(B, 1, Gy, Gx)
    bev.div_(hit_raw.clamp_min(1.0))
    if return_depth_z:
        return bev, z_num, hit_raw
    return bev


def install(lift, verify=None):
    """Replace ``lift._splat_continuous`` (must run after b54_bevpool)."""
    if _STATE["installed"]:
        return lift._splat_continuous
    prev = lift._splat_continuous
    fused = splat_fused.__get__(lift, type(lift))
    if verify is None:
        verify = os.environ.get("V7_B_VERIFY", "0") == "1"

    def _logged(*a, **k):
        out = fused(*a, **k)
        if not _STATE["verified"]:
            ref = prev(*a, **k)
            _STATE["verified"] = True
            names = ("bev", "z_num", "z_den")
            pairs = (zip(ref, out) if isinstance(out, tuple) else [(ref, out)])
            msg = ", ".join(
                "%s=%.3e" % (names[i] if isinstance(out, tuple) else "bev",
                             float((r - o).abs().max()))
                for i, (r, o) in enumerate(pairs))
            print("V7_B verify vs b54.splat_shared: %s" % msg, flush=True)
        return out

    lift._splat_continuous = _logged if verify else fused
    _STATE["installed"] = True
    return prev
