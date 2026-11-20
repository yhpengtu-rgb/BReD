#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v8-a4 continuous-depth forward-projection lifting backbone.

This subclasses the official ``BEVStereoLSSFPN`` and keeps **all** of its
pretrained submodules (image ResNet50, SECONDFPN neck, stereo depth net,
context down-sampler, similarity / mask nets) untouched.  Only the final voxel
pooling is replaced:

  * The official D-bin depth distribution is still produced by the official depth
    net and used for the perspective LiDAR depth BCE (form / z_cam / native x3.0
    unchanged).
  * The projected depth is the *expectation* over that distribution -- a
    continuous scalar per pixel.  No discrete-bin softmax weighting is used for
    the lift.
  * For every image pixel (forward, image -> BEV) we unproject with the predicted
    continuous depth, camera->ego, apply BDA, and splat the neck feature onto the
    256x256 BEV grid (0.4 m/cell) with a bilinear soft splat.  We never ray-cast
    from a BEV cell (that would require the cell elevation z, which is exactly
    what the head predicts -- the chicken-and-egg problem this design avoids).
  * The SECONDFPN neck already fuses C2/C3/C4/C5 multi-level features; the lifted
    context is that neck feature map.

The inherited official ``forward`` extracts context + depth score for every
sweep and dispatches to the overridden :meth:`_forward_single_sweep`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from bevdepth.layers.backbones.bevstereo_lss_fpn import BEVStereoLSSFPN


def _bin_centers(d_bound, depth_channels, device, dtype):
    d0, _, step = d_bound
    idx = torch.arange(depth_channels, device=device, dtype=dtype)
    return d0 + idx * step


class ContinuousLiftFPN(BEVStereoLSSFPN):
    """Official stereo backbone with a continuous forward-projection lift."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_official_backbone(self, official_backbone: nn.Module) -> None:
        """Copy official weights, but keep our own geometry buffers.

        The official backbone was built on the 128x128 / 256x704 grid; its
        registered ``voxel_num`` / ``voxel_coord`` / ``voxel_size`` / ``frustum``
        buffers must NOT overwrite the v8-a4 256x256 @ 0.4 m registration.
        """
        sd = official_backbone.state_dict()
        skip = {"voxel_num", "voxel_coord", "voxel_size", "frustum"}
        sd = {k: v for k, v in sd.items() if k.split(".")[-1] not in skip}
        self.load_state_dict(sd, strict=False)

    # ------------------------------------------------------------------ #
    # continuous forward splat (replaces voxel_pooling_train / inference)
    # ------------------------------------------------------------------ #
    def _pixel_grid(self, fh: int, fw: int, device, dtype):
        ogfH, ogfW = self.final_dim
        u = torch.linspace(0, ogfW - 1, fw, device=device, dtype=dtype)
        v = torch.linspace(0, ogfH - 1, fh, device=device, dtype=dtype)
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        return uu.reshape(-1), vv.reshape(-1)

    def _splat_continuous(self, context: torch.Tensor, depth_prob: torch.Tensor,
                          mats: dict, sweep_index: int,
                          return_depth_z: bool = False) -> torch.Tensor:
        """Forward-project one sweep's neck features onto the 2D BEV grid.

        Args:
            context: (B*Nc, C, fh, fw) neck feature.
            depth_prob: (B*Nc, D, fh, fw) depth bin probabilities.
            mats: official mats_dict.
            sweep_index: sweep index into mats tensors.

        Returns:
            bev: (B, C, Gy, Gx) accumulated top-down feature.
        """
        BN, C, fh, fw = context.shape
        Nc = self._nc
        B = BN // Nc

        device, dtype = context.device, context.dtype
        # The geometry buffers are registered tensors; float()/int() on them is
        # a host sync, which forbids CUDA graph capture.  Cache Python scalars.
        geo = getattr(self, "_v73_geo", None)
        if geo is None:
            Gx, Gy = int(self.voxel_num[0]), int(self.voxel_num[1])
            cell_x, cell_y = float(self.voxel_size[0]), float(self.voxel_size[1])
            geo = (Gx, Gy, cell_x, cell_y,
                   float(self.voxel_coord[0] - cell_x / 2.0),
                   float(self.voxel_coord[1] - cell_y / 2.0))
            self._v73_geo = geo
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

        # expand+reshape instead of repeat_interleave: the latter stages its
        # repeat index through pageable CPU memory, which invalidates CUDA
        # graph capture (same values, same order).
        bda_rep = bda.unsqueeze(1).expand(B, Nc, 4, 4).reshape(B * Nc, 4, 4)
        pts = bda_rep.matmul(pts)
        xy = pts[:, :2, :]
        # v8-a17: the depth-derived elevation is a detached pseudo-target.
        # Keep both values and aggregation weights out of the autograd graph.
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

        L = Gy * Gx
        bev = context.new_zeros(BN, C, L)
        hit = context.new_zeros(BN, 1, L)
        z_num = context.new_zeros(BN, 1, L, dtype=torch.float32)
        z_den = context.new_zeros(BN, 1, L, dtype=torch.float32)
        context_p = context.reshape(BN, C, P)

        corners = [
            ((1 - wx) * (1 - wy), x0c, y0c),
            (wx * (1 - wy), x1c, y0c),
            ((1 - wx) * wy, x0c, y1c),
            (wx * wy, x1c, y1c),
        ]
        for w, xi, yi in corners:
            w = w * valid.to(dtype)
            lin = yi * Gx + xi
            feat_w = context_p * w.unsqueeze(1)
            w_z = w.detach().float()
            weighted_z = ego_z * w_z.unsqueeze(1)
            for b in range(BN):
                bev[b].index_add_(1, lin[b], feat_w[b])
                hit[b].index_add_(1, lin[b], w[b].reshape(1, -1))
                z_num[b].index_add_(1, lin[b], weighted_z[b])
                z_den[b].index_add_(1, lin[b], w_z[b].reshape(1, -1))

        bev = bev.reshape(BN, C, Gy, Gx).reshape(B, Nc, C, Gy, Gx).sum(dim=1)
        hit = hit.reshape(B, Nc, 1, Gy, Gx).sum(dim=1).clamp_min(1.0)
        if return_depth_z:
            z_num = z_num.reshape(B, Nc, 1, Gy, Gx).sum(dim=1)
            z_den = z_den.reshape(B, Nc, 1, Gy, Gx).sum(dim=1)
            return bev / hit, z_num, z_den
        return bev / hit

    def _forward_single_sweep(self, sweep_index, context, mats_dict, depth_score,
                              is_return_depth=False):
        """Replacement for the official voxel-pooling single-sweep lift."""
        B, Nc, C, fh, fw = context.shape
        ctx = context.reshape(B * Nc, C, fh, fw)
        prob = depth_score.reshape(B * Nc, self.depth_channels, fh, fw)
        self._nc = Nc
        if getattr(self, "fp32_splat", False):
            with torch.autocast(device_type=ctx.device.type, enabled=False):
                bev, z_num, z_den = self._splat_continuous(
                    ctx.float(), prob.float(), mats_dict, sweep_index,
                    return_depth_z=True)
        else:
            bev, z_num, z_den = self._splat_continuous(
                ctx, prob, mats_dict, sweep_index, return_depth_z=True)
        if sweep_index == 0:
            self._depth_bev_z_num = z_num
            self._depth_bev_z_den = z_den
        else:
            self._depth_bev_z_num = self._depth_bev_z_num + z_num
            self._depth_bev_z_den = self._depth_bev_z_den + z_den
        if is_return_depth:
            return bev.contiguous(), depth_score
        return bev.contiguous()

    def get_depth_bev_z(self):
        """Detached expected-depth elevation on the fine BEV lattice."""
        if not hasattr(self, "_depth_bev_z_den"):
            raise RuntimeError("depth-BEV z requested before the lift forward")
        valid = self._depth_bev_z_den > 0
        z = self._depth_bev_z_num / self._depth_bev_z_den.clamp_min(1e-6)
        return z.detach(), valid.detach()
