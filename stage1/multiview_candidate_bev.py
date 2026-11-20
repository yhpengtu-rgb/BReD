#!/usr/bin/env python3
"""Multi-view candidate BEV builder for the strict DiffMVS-style refiner.

Given a batch of multi-view image-backbone features and a set of per-cell
elevation candidates, this module projects every BEV cell centre at every
candidate elevation into every camera, bilinearly samples the multi-view
feature, and returns:

  * cand_feat: mean multi-view feature per candidate  [B,N,C,h,w]
  * cost:      variance across views per candidate      [B,N,1,h,w]
  * valid:     number of cameras that saw the point     [B,N,1,h,w]

This is the BEV-elevation analogue of DiffMVS's local depth cost volume:
the candidate axis is the local inverse-depth search, and the cost is a
multi-view feature-variance cost (DiffMVS uses group-wise correlation because
it has a fixed reference camera; a BEV cell can be seen by any camera, so
variance over all valid views is the direct analogue).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewCandidateBEV(nn.Module):
    def __init__(
        self,
        in_ch: int = 256,
        out_ch: int = 32,
        fine_size: int = 256,
        cand_size: int = 128,
        img_h: int = 576,
        img_w: int = 1024,
        pc_range: Tuple[float, float, float, float, float, float] = (
            -51.2, -51.2, -5.0, 51.2, 51.2, 3.0
        ),
        eps_depth: float = 1.0,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.fine_size = int(fine_size)
        self.cand_size = int(cand_size)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.eps_depth = float(eps_depth)
        self.pc_range = tuple(float(v) for v in pc_range)
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.cell = (x_max - x_min) / self.cand_size

        # Project high-channel image-backbone features to a compact cost/feature
        # dimensionality before the 6-camera / 6-candidate sampling loop.
        self.proj = nn.Conv2d(self.in_ch, self.out_ch, 1)

    def grid_centers(self, device, dtype):
        cs = self.cand_size
        coords = (torch.arange(cs, device=device, dtype=dtype) + 0.5) * self.cell
        coords = coords - (self.x_max - self.x_min) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack((xx, yy), dim=-1)  # (h,w,2)

    def forward(
        self,
        feat: torch.Tensor,          # (B*Nc, C_in, Hf, Wf)
        z_candidates: torch.Tensor,  # (B, N, h, w)
        mats: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if feat is None:
            raise ValueError("strict DiffMVS refiner requires multi-view image features")
        if feat.dim() != 4:
            raise ValueError(f"image feature must be 4D, got {tuple(feat.shape)}")

        intrin_all = mats["intrin_mats"]
        Nc = int(intrin_all.shape[2])
        if feat.shape[0] % Nc != 0:
            raise ValueError(f"image feature batch {feat.shape[0]} not divisible by Nc={Nc}")
        B = feat.shape[0] // Nc
        N = int(z_candidates.shape[1])
        h, w = int(z_candidates.shape[-2]), int(z_candidates.shape[-1])
        if (h, w) != (self.cand_size, self.cand_size):
            raise ValueError(f"z_candidates must be {(self.cand_size,self.cand_size)}, got {(h,w)}")

        device = feat.device
        dtype = feat.dtype
        # The projection is under autocast in training; sampling/accumulation is
        # kept fp32 for stable variance and grid_sample coordinates.
        feat = self.proj(feat).float()
        Cf = feat.shape[1]
        Hf, Wf = feat.shape[-2], feat.shape[-1]
        feat = feat.reshape(B, Nc, Cf, Hf, Wf)

        centers = self.grid_centers(device, torch.float32)  # (h,w,2)
        xy = centers.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)  # (B,2,h,w)
        xy = xy.detach()
        z = torch.nan_to_num(z_candidates.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        z = z.clamp(self.pc_range[2], self.pc_range[5])

        xy_b = xy.permute(0, 2, 3, 1)  # (B,h,w,2)
        z_b = z.unsqueeze(-1)          # (B,N,h,w,1)
        pts = torch.cat(
            (xy_b.unsqueeze(1).expand(B, N, h, w, 2), z_b), dim=-1
        )  # (B,N,h,w,3)
        ones = torch.ones_like(pts[..., :1])
        pts_h = torch.cat((pts, ones), dim=-1)  # (B,N,h,w,4)

        bda = mats["bda_mat"].to(device=device, dtype=torch.float32)  # (B,4,4)
        inv_bda = torch.inverse(bda)
        pts_ego = torch.einsum("bij,bnhwj->bnhwi", inv_bda, pts_h)  # (B,N,h,w,4)

        intrin = mats["intrin_mats"][:, 0].to(device=device, dtype=torch.float32)
        ida = mats["ida_mats"][:, 0].to(device=device, dtype=torch.float32)
        s2e = mats["sensor2ego_mats"][:, 0].to(device=device, dtype=torch.float32)

        sum_feat = torch.zeros(B, N, Cf, h, w, device=device, dtype=torch.float32)
        sum_sq = torch.zeros_like(sum_feat)
        hit = torch.zeros(B, N, 1, h, w, device=device, dtype=torch.float32)

        for c in range(Nc):
            inv_s2e = torch.inverse(s2e[:, c])
            pts_cam = torch.einsum("bij,bnhwj->bnhwi", inv_s2e, pts_ego)
            Xc, Yc, Zc = pts_cam[..., 0], pts_cam[..., 1], pts_cam[..., 2]

            K = intrin[:, c, :3, :3]
            u0 = (K[:, 0:1, 0:1].view(B, 1, 1) * Xc
                  + K[:, 0:1, 2:3].view(B, 1, 1) * Zc) / Zc.clamp_min(1e-8)
            v0 = (K[:, 1:2, 1:2].view(B, 1, 1) * Yc
                  + K[:, 1:2, 2:3].view(B, 1, 1) * Zc) / Zc.clamp_min(1e-8)

            # Intrinsics are already IDA-combined in the official mat dict; the
            # remaining IDA maps the ideal pixel to the resized image pixel.
            ida_c = ida[:, c]
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
            valid_f = valid.float().unsqueeze(2)  # (B,N,1,h,w)

            gx = u_i / self.img_w * 2.0 - 1.0
            gy = v_i / self.img_h * 2.0 - 1.0
            grid = torch.stack((gx, gy), dim=-1)  # (B,N,h,w,2)
            grid = grid.reshape(B * N, h, w, 2)

            # (B,C,Hf,Wf) -> (B*N,C,Hf,Wf)
            feat_c = feat[:, c].unsqueeze(1).expand(B, N, Cf, Hf, Wf)
            feat_c = feat_c.reshape(B * N, Cf, Hf, Wf)
            sample = F.grid_sample(
                feat_c, grid, mode="bilinear",
                padding_mode="zeros", align_corners=False,
            ).reshape(B, N, Cf, h, w)

            sum_feat = sum_feat + sample * valid_f
            sum_sq = sum_sq + sample * sample * valid_f
            hit = hit + valid_f

        mean = sum_feat / hit.clamp_min(1.0)
        meansq = sum_sq / hit.clamp_min(1.0)
        var = (meansq - mean * mean).clamp_min(0.0)
        cost = var.mean(dim=2, keepdim=True)  # (B,N,1,h,w)

        # A single view has zero variance and would look falsely perfect.
        # Penalise low-support candidates; keep the value finite for hit=0.
        support_penalty = 0.05 * (2.0 - hit).clamp_min(0.0)
        cost = torch.where(hit > 0, cost + support_penalty,
                           torch.ones_like(cost))
        valid_out = (hit > 0).float()
        return mean.to(dtype=dtype), cost.to(dtype=dtype), valid_out.to(dtype=dtype)
