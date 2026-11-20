#!/usr/bin/env python3
"""v8-A elevation-guided perspective query.

This is the *only* alignment between the BEV elevation grid and the
perspective texture feature map.  For every fine BEV cell center (x, y) we read
its elevation z, build a 3D point in the BDA/official frame, undo BDA, map
ego->camera, project through the (already IDA-combined) intrinsics and IDA to
the 256x704 image, bilinearly sample the 64x176 perspective texture feature and
aggregate over valid cameras.

Rules (locked):
  * out-of-bounds pixels or points behind a camera contribute ZERO feature and
    are excluded from the valid mask.  We never clamp coordinates to the image
    border.
  * elevation (and, by default, the XY centre) is detached before being used to
    query texture, so texture gradients do not flow back into geometry.
  * elevation (128 -> fine) and the perspective feature map live in different
    coordinate systems; there is no resize alignment of the feature map.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ElevationGuidedQuery(nn.Module):
    def __init__(
        self,
        pc_range: Tuple[float, float, float, float, float, float],
        fine_size: int = 256,
        img_h: int = 576,
        img_w: int = 1024,
        eps_depth: float = 1.0,
    ):
        super().__init__()
        self.pc_range = tuple(float(v) for v in pc_range)
        self.fine_size = int(fine_size)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.eps_depth = float(eps_depth)
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.cell = (x_max - x_min) / self.fine_size

    def fine_grid_centers(self, device, dtype):
        f = self.fine_size
        coords = (torch.arange(f, device=device, dtype=dtype) + 0.5) * self.cell
        coords = coords - (self.x_max - self.x_min) / 2.0  # pc_range is symmetric
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack((xx, yy), dim=-1)  # (F, F, 2)

    def forward(
        self,
        tex_feat: torch.Tensor,
        elevation: torch.Tensor,
        mats: Dict[str, torch.Tensor],
        xy_offset: torch.Tensor | None = None,
        detach: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tex_feat: (B, Nc, Ct, 64, 176) perspective texture feature.
            elevation: (B, 1, F, F) fine-grid elevation in the BDA frame.
            mats: official mats_dict (key-frame sweep 0).
            xy_offset: optional (B, 2, F, F) learnable BEV XY offset.
            detach: detach the query geometry (z and xy).  Default True.

        Returns:
            dict with:
              tex_bev: (B, Ct, F, F) aggregated texture feature.
              valid_cams: (B, 1, F, F) number of cameras that saw the cell.
              coords: (B, Nc, F, F, 2) image coords (for audit).
        """
        B, Nc, Ct, fH, fW = tex_feat.shape
        device = tex_feat.device
        dtype = tex_feat.dtype
        FS = self.fine_size
        if elevation.shape[-2:] != (FS, FS):
            raise ValueError(f"elevation must be {(FS,FS)}, got {tuple(elevation.shape)}")

        # Same dtype policy as v8-a9's MultiViewCandidateBEV: the whole
        # geometric chain (grid centres, homogeneous points, camera matrices,
        # pixel coordinates, grid_sample) runs in fp32.  Under bf16 autocast a
        # normalised coordinate in [-1, 1] has ~0.008 resolution, i.e. several
        # pixels of jitter at 1024 px width, which would silently move the
        # sampling location.  Only the returned feature is cast back.
        centers = self.fine_grid_centers(device, torch.float32)  # (F,F,2)
        xy = centers.permute(2, 0, 1).unsqueeze(0)  # (1,2,F,F)
        if xy_offset is not None:
            xy = xy + xy_offset.float()
        z = elevation.float()
        if detach:
            xy = xy.detach()
            z = z.detach()

        # (B, F, F, 3) point in BDA frame; pad homogeneous.
        xy_b = xy.permute(0, 2, 3, 1)  # (B,F,F,2)
        z_b = z.permute(0, 2, 3, 1)    # (B,F,F,1)
        pts = torch.cat((xy_b, z_b), dim=-1)  # (B,F,F,3)
        ones = torch.ones_like(pts[..., :1])
        pts_h = torch.cat((pts, ones), dim=-1)  # (B,F,F,4)

        bda = mats["bda_mat"].to(device=device, dtype=torch.float32)  # (B,4,4)
        inv_bda = torch.inverse(bda)
        # ego = inv(bda) @ pts_h
        pts_ego = torch.einsum("bij,bfwj->bfwi", inv_bda, pts_h)  # (B,F,F,4)

        intrin = mats["intrin_mats"][:, 0].to(device=device, dtype=torch.float32)
        ida = mats["ida_mats"][:, 0].to(device=device, dtype=torch.float32)
        s2e = mats["sensor2ego_mats"][:, 0].to(device=device, dtype=torch.float32)

        agg = torch.zeros(B, Ct, FS, FS, device=device, dtype=torch.float32)
        hit = torch.zeros(B, 1, FS, FS, device=device, dtype=torch.float32)
        coords_out = torch.zeros(B, Nc, FS, FS, 2, device=device,
                                 dtype=torch.float32)

        for c in range(Nc):
            inv_s2e = torch.inverse(s2e[:, c])  # (B,4,4)
            pts_cam = torch.einsum("bij,bfwj->bfwi", inv_s2e, pts_ego)  # (B,F,F,4)
            Xc = pts_cam[..., 0]
            Yc = pts_cam[..., 1]
            Zc = pts_cam[..., 2]

            K = intrin[:, c, :3, :3]  # (B,3,3)
            # [u0*Zc, v0*Zc, Zc]^T = K @ [Xc, Yc, Zc]
            u0 = (K[:, 0:1, 0:1].view(B, 1, 1) * Xc
                  + K[:, 0:1, 2:3].view(B, 1, 1) * Zc) / Zc.clamp_min(1e-8)
            v0 = (K[:, 1:2, 1:2].view(B, 1, 1) * Yc
                  + K[:, 1:2, 2:3].view(B, 1, 1) * Zc) / Zc.clamp_min(1e-8)

            ida_c = ida[:, c]  # (B,4,4)
            a = ida_c[:, 0:1, 0:1].view(B, 1, 1)
            b = ida_c[:, 0:1, 1:2].view(B, 1, 1)
            t = ida_c[:, 0:1, 3:4].view(B, 1, 1)
            d = ida_c[:, 1:2, 1:2].view(B, 1, 1)
            e = ida_c[:, 1:2, 0:1].view(B, 1, 1)
            f = ida_c[:, 1:2, 3:4].view(B, 1, 1)
            u_i = a * u0 + b * v0 + t
            v_i = e * u0 + d * v0 + f

            valid = ((Zc > self.eps_depth)
                     & (u_i >= 0) & (u_i < self.img_w)
                     & (v_i >= 0) & (v_i < self.img_h)
                     & torch.isfinite(u_i) & torch.isfinite(v_i))
            coords_out[:, c, :, :, 0] = u_i
            coords_out[:, c, :, :, 1] = v_i

            # grid_sample normalised coords in [-1, 1].  padding_mode='zeros'
            # already returns zero outside; we additionally mask behind-camera.
            gx = u_i / self.img_w * 2.0 - 1.0
            gy = v_i / self.img_h * 2.0 - 1.0
            grid = torch.stack((gx, gy), dim=-1)  # (B,F,F,2)
            sampled = F.grid_sample(
                tex_feat[:, c].float(), grid, mode="bilinear",
                padding_mode="zeros", align_corners=False,
            )  # (B,Ct,F,F) fp32
            valid_f = valid.unsqueeze(1).float()
            sampled = sampled * valid_f
            agg = agg + sampled
            hit = hit + valid_f

        tex_bev = agg / hit.clamp_min(1.0)
        return {
            "tex_bev": tex_bev.to(dtype=dtype),
            "valid_cams": hit.to(dtype=dtype),
            "coords": coords_out,
        }
