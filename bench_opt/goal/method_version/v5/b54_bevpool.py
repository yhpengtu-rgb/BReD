#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5.4 BEV pooling: fold the camera axis into a shared (C, L) accumulator.

The v8-a17 continuous splat accumulates into a dense
``(BN, C, Gy, Gx)`` buffer and only afterwards sums the camera axis.  With
BN=6, C=512 and 256x256 cells that is an 805 MB zero-initialised tensor per
sweep, plus an equally large output for the camera sum and another one for
the ``/hit`` normalisation.

Because the camera axis is a *sum*, the accumulator can be shared across all
cameras: ``(C, L)`` = 134 MB.  All Nc cameras x 4 bilinear corners are then
fused into a single ``index_add_`` per output tensor, which removes the
camera sum and cuts the dominant memory traffic ~6x.

Only the final accumulation order changes (still index-ordered scatter, no
algebraic rewriting of the projection itself), so results differ from the
reference in the last float32 bits only.
"""
from __future__ import annotations

import torch

from continuous_lift_fpn import _bin_centers


def splat_shared(self, context, depth_prob, mats, sweep_index,
                 return_depth_z=False):
    """Drop-in replacement for ``ContinuousLiftFPN._splat_continuous``."""
    BN, C, fh, fw = context.shape
    Nc = self._nc
    B = BN // Nc

    device, dtype = context.device, context.dtype
    # The geometry buffers are registered tensors; float()/int() on them is a
    # host sync and therefore not CUDA-graph capturable.  Cache the Python
    # scalars once.
    geo = getattr(self, "_v54_geo", None)
    if geo is None:
        Gx, Gy = int(self.voxel_num[0]), int(self.voxel_num[1])
        cell_x, cell_y = float(self.voxel_size[0]), float(self.voxel_size[1])
        geo = (Gx, Gy, cell_x, cell_y,
               float(self.voxel_coord[0] - cell_x / 2.0),
               float(self.voxel_coord[1] - cell_y / 2.0))
        self._v54_geo = geo
    Gx, Gy, cell_x, cell_y, xmin, ymin = geo

    # Continuous expected depth per pixel: (BN, 1, fh, fw)
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

    # undo IDA (image post-IDA -> original image plane)
    # v7.3: cuSOLVER inverses are not capturable -> prefer precomputed buffers.
    _ida_inv = mats.get('ida_inv')
    pts = (_ida_inv[:, sweep_index].reshape(BN, 4, 4) if _ida_inv is not None
           else torch.inverse(ida)).matmul(pts)

    # camera unprojection: [x*z, y*z, z, 1] then s2e @ inv(K)
    zcam = pts[:, 2:3, :]
    pts = torch.cat([pts[:, 0:2, :] * zcam, pts[:, 2:, :]], dim=1)
    _intrin_inv = mats.get('intrin_inv')
    combine = s2e.matmul(
        _intrin_inv[:, sweep_index].reshape(BN, 4, 4)
        if _intrin_inv is not None else torch.inverse(intrin))
    pts = combine.matmul(pts)

    # expand+reshape: repeat_interleave stages its repeat index through
    # pageable CPU memory, which invalidates CUDA graph capture.
    bda_rep = bda.unsqueeze(1).expand(B, Nc, 4, 4).reshape(B * Nc, 4, 4)
    pts = bda_rep.matmul(pts)
    xy = pts[:, :2, :]
    # v8-a17: the depth-derived elevation is a detached pseudo-target.
    ego_z = pts[:, 2:3, :].detach().float()

    # validity: in front of camera, inside BEV extent, finite
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

    vmask = valid.to(dtype).unsqueeze(1)                      # (BN, 1, P)
    weights = torch.stack([
        ((1 - wx) * (1 - wy)) * valid.to(dtype),
        (wx * (1 - wy)) * valid.to(dtype),
        ((1 - wx) * wy) * valid.to(dtype),
        (wx * wy) * valid.to(dtype),
    ], dim=1)                                                  # (BN, 4, P)
    lin = torch.stack([
        y0c * Gx + x0c, y0c * Gx + x1c, y1c * Gx + x0c, y1c * Gx + x1c
    ], dim=1)                                                  # (BN, 4, P)

    L = Gy * Gx
    context_p = context.reshape(BN, C, P)
    # one fused scatter source: (BN, C, 4, P)
    feat_all = context_p.unsqueeze(2) * weights.unsqueeze(1)
    w_all = weights.detach().float()
    z_src = ego_z.unsqueeze(2) * w_all.unsqueeze(1)             # (BN, 1, 4, P)

    bev_all = context.new_zeros(B * C, L)
    hit_all = context.new_zeros(B, L)
    z_num_all = torch.zeros(B, L, dtype=torch.float32, device=device)
    z_den_all = torch.zeros(B, L, dtype=torch.float32, device=device)

    for b in range(B):
        sl = slice(b * Nc, (b + 1) * Nc)
        idx = lin[sl].reshape(-1)
        bev_all[b * C:(b + 1) * C].index_add_(
            1, idx, feat_all[sl].permute(1, 0, 2, 3).reshape(C, -1))
        hit_all[b:b + 1].index_add_(1, idx, weights[sl].reshape(1, -1))
        z_num_all[b:b + 1].index_add_(1, idx, z_src[sl].reshape(1, -1))
        z_den_all[b:b + 1].index_add_(1, idx, w_all[sl].reshape(1, -1))

    bev = bev_all.reshape(B, C, Gy, Gx)
    hit = hit_all.reshape(B, 1, Gy, Gx).clamp_min(1.0)
    bev.div_(hit)
    if return_depth_z:
        return (bev,
                z_num_all.reshape(B, 1, Gy, Gx),
                z_den_all.reshape(B, 1, Gy, Gx))
    return bev


# backward-compatible alias used by the profile probe
splat_reference_guard = splat_shared


def install(lift, mode="shared"):
    """Install the v5.4 splat on ``lift`` (returns the previous callable)."""
    if mode in ("base", "off", ""):
        return None
    if mode not in ("shared", "b54", "v54"):
        raise ValueError("unknown v5.4 pool mode: %r" % (mode,))
    prev = lift._splat_continuous
    lift._splat_continuous = splat_shared.__get__(lift, type(lift))
    return prev
