#!/usr/bin/env python3
"""Strict DiffMVS-style residual-diffusion elevation refiner.

This version keeps the v8-a8 diffusion trunk but makes the inner K-iteration
loop follow the original DiffMVS semantics:

  * cumulative residual update:  r_k = r_{k-1} + delta_k
  * cumulative elevation:        z_k = z_coarse + res_max * tanh(r_k)
  * next iteration sees z_k (as a normalised cumulative residual), not delta_k
  * every iteration predicts a confidence map C_k from the post-GRU decoder
  * C_{k-1} sets the next elevation-candidate radius:
        R_k = R_min + (1 - C_{k-1}) * (R_max - R_min)
  * candidate elevations are projected through all cameras into image-backbone
    features, producing a local multi-view cost/feature volume
  * the candidate BEV/cost volume is the conditioning input of that iteration
  * C_k is supervised by the DiffMVS Laplace NLL on the metric elevation error
    (no confidence GT exists; the error itself is the supervision)

The candidate BEV is built at ``cand_size`` (default 128 = 0.8 m) and
bilinearly upsampled to the fine 256 lattice.  This is a compute/memory
adaptation only; the candidate search and confidence feedback are explicit.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from multiview_candidate_bev import MultiViewCandidateBEV


def sinusoidal_t_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype)
        / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _ConvGRU(nn.Module):
    """Bottleneck conv-GRU carrying inner-iteration hidden state."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Conv2d(channels * 2, channels, 3, 1, 1)
        self.update = nn.Conv2d(channels * 2, channels, 3, 1, 1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([h, x], dim=1)))
        u = torch.tanh(self.update(torch.cat([h, x], dim=1)))
        return (1.0 - g) * h + g * u


class CandidateConditionEncoder(nn.Module):
    """Cost-weighted aggregation of candidate BEV features.

    Inputs per candidate: feature, feature-variance cost, normalised offset,
    valid mask.  A shared 2D conv encodes each candidate; a cost softmin
    aggregates them into one residual condition.  The output projection is
    zero-initialised so a warm-started v8-a8 checkpoint starts unchanged and
    the candidate branch grows only if it reduces the loss.
    """

    def __init__(self, cand_ch: int = 32, hidden: int = 64,
                 num_candidates: int = 6):
        super().__init__()
        self.num_candidates = int(num_candidates)
        self.feat = nn.Sequential(
            nn.Conv2d(cand_ch + 3, hidden, 3, 1, 1),
            nn.SyncBatchNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden + 1, hidden, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, cand_feat: torch.Tensor, cost: torch.Tensor,
                valid: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
        B, N, C, h, w = cand_feat.shape
        inp = torch.cat([cand_feat, cost, dz.unsqueeze(2), valid], dim=2)
        inp = inp.reshape(B * N, C + 3, h, w)
        feat = self.feat(inp).reshape(B, N, -1, h, w)

        # Cost softmin over candidates.  Normalise per cell so the temperature
        # is not dominated by absolute feature scale.
        cost_norm = cost / cost.mean(dim=1, keepdim=True).clamp_min(1e-6)
        score = -cost_norm / 0.1
        score = score + torch.log(valid.clamp_min(1e-6))
        weights = torch.softmax(score, dim=1)
        agg = (feat * weights).sum(dim=1)
        cost_agg = (cost * weights).sum(dim=1)
        return self.out(torch.cat([agg, cost_agg], dim=1))


class MultiScaleBackboneFPN(nn.Module):
    """Minimal P2-P4 FPN built directly from ResNet c2/c3/c4 features."""

    def __init__(self, in_channels=(256, 512, 1024), out_channels=256):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(int(c), int(out_channels), 1) for c in in_channels
        ])

    def forward(self, feats):
        if len(feats) != 3:
            raise ValueError(f"expected c2/c3/c4, got {len(feats)} scales")
        p4 = self.lateral[2](feats[2])
        p3 = self.lateral[1](feats[1]) + F.interpolate(
            p4, size=feats[1].shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral[0](feats[0]) + F.interpolate(
            p3, size=feats[0].shape[-2:], mode="bilinear", align_corners=False)
        return (p2, p3, p4)


class ElevationRefiner(nn.Module):
    def __init__(self,
                 fine_ch: int = 128,
                 ctx_ch: int = 256,
                 hidden: int = 64,
                 k_iters: int = 1,
                 res_max_m: float = 0.3,
                 s_z: float = 0.167,
                 sigma_z: float = 0.5,
                 T: int = 1000,
                 num_candidates: int = 6,
                 cand_size: int = 128,
                 cand_ch: int = 32,
                 radius_base: float = 0.1,
                 radius_min: float = 0.04,
                 radius_max: float = 0.4,
                 z_min: float = -5.0,
                 z_max: float = 3.0,
                 img_h: int = int(os.environ.get("BRED_FINAL_H", 576)),
                 img_w: int = int(os.environ.get("BRED_FINAL_W", 1024))):
        super().__init__()
        self.k_iters = int(k_iters)
        self.res_max_m = float(res_max_m)
        self.s_z = float(s_z)
        self.sigma_z = float(sigma_z)
        self.clean_prior_prob = 0.5  # v8-a17: explicit t=0 depth-prior samples
        self.T = int(T)
        self.num_candidates = int(num_candidates)
        self.cand_size = int(cand_size)
        self.radius_base = float(radius_base)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

        # Morphology-B: elevation FiLM on the lifted context (z detached).
        self.elev_film = nn.Sequential(
            nn.Conv2d(1, ctx_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ctx_ch, ctx_ch * 2, 1),
        )

        # condition: fine + detached coarse z + elevation-aligned context
        self.cond = nn.Sequential(
            nn.Conv2d(fine_ch + 1 + ctx_ch, hidden, 3, 1, 1),
            nn.SyncBatchNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.t_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden * 2))

        self.enc1 = nn.Sequential(
            nn.Conv2d(hidden + 1, hidden, 3, 1, 1),
            nn.SyncBatchNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.down = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 2, 1),
            nn.SyncBatchNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.gru = _ConvGRU(hidden)
        self.up = nn.ConvTranspose2d(hidden, hidden, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 3, 1, 1),
            nn.SyncBatchNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.delta = nn.Conv2d(hidden, 1, 1)
        # DiffMVS predicts confidence from the U-Net decoder every iteration.
        self.conf = nn.Conv2d(hidden, 1, 1)

        # Multi-view candidate BEV / local cost volume.  The original sampler is
        # shared across P2/P3/P4 so its v8-a9 projection weights warm-start exactly.
        self.cand_bev = MultiViewCandidateBEV(
            in_ch=256, out_ch=int(cand_ch),
            fine_size=256, cand_size=self.cand_size,
            img_h=img_h, img_w=img_w,
            pc_range=(-51.2, -51.2, self.z_min, 51.2, 51.2, self.z_max),
        )
        self.ms_fpn = MultiScaleBackboneFPN(
            in_channels=(256, 512, 1024), out_channels=256)
        # Equal initial weights over P2/P3/P4; learned during v8-a15 fine-tuning.
        self.ms_scale_logits = nn.Parameter(torch.zeros(3))
        self.cand_cond = CandidateConditionEncoder(
            cand_ch=int(cand_ch), hidden=hidden,
            num_candidates=self.num_candidates)

        # Zero-init residual head: start from z_refined == z_coarse.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def _cosine_alpha_bar(self, t_norm: torch.Tensor) -> torch.Tensor:
        s = 0.008
        f = torch.cos(((t_norm + s) / (1.0 + s)) * math.pi * 0.5) ** 2
        return f.clamp(1e-4, 1.0)

    def _film_context(self, ctx: torch.Tensor, z_coarse: torch.Tensor) -> torch.Tensor:
        film = self.elev_film(z_coarse.detach())
        scale, shift = torch.chunk(film, 2, dim=1)
        return ctx * (1.0 + scale) + shift

    def _candidate_condition(self, img_feats,
                             mats: Dict[str, torch.Tensor],
                             z_cur: torch.Tensor,
                             conf_prev: Optional[torch.Tensor]) -> torch.Tensor:
        """Build and fuse P2/P3/P4 candidate BEVs at the current elevation."""
        cs = self.cand_size
        z_small = F.interpolate(z_cur, size=(cs, cs),
                                mode="bilinear", align_corners=False).squeeze(1)
        if conf_prev is None:
            radius = torch.full_like(z_small, self.radius_base)
        else:
            c_small = F.interpolate(conf_prev, size=(cs, cs),
                                    mode="bilinear", align_corners=False).squeeze(1)
            radius = self.radius_min \
                + (1.0 - c_small) * (self.radius_max - self.radius_min)

        offsets = torch.linspace(
            -1.0, 1.0, self.num_candidates,
            device=z_cur.device, dtype=z_cur.dtype
        ).view(1, self.num_candidates, 1, 1)
        z_cand = z_small.unsqueeze(1) + offsets * radius.unsqueeze(1)
        z_cand = z_cand.clamp(self.z_min, self.z_max)

        cand_feat = None
        cost = None
        valid = None
        scale_weights = torch.softmax(self.ms_scale_logits, dim=0)
        for scale_idx, feat in enumerate(self.ms_fpn(img_feats)):
            f_s, c_s, v_s = self.cand_bev(feat, z_cand, mats)
            w_s = scale_weights[scale_idx].to(dtype=f_s.dtype)
            cand_feat = f_s * w_s if cand_feat is None else cand_feat + f_s * w_s
            cost = c_s * w_s if cost is None else cost + c_s * w_s
            valid = v_s if valid is None else torch.maximum(valid, v_s)
        dz = (z_cand - z_small.unsqueeze(1)) / self.res_max_m
        cand_cond = self.cand_cond(cand_feat, cost, valid, dz)
        return F.interpolate(cand_cond, size=z_cur.shape[-2:],
                             mode="bilinear", align_corners=False)

    def forward(self, fine: torch.Tensor, ctx: torch.Tensor,
                z_coarse: torch.Tensor,
                img_feats,
                mats: Dict[str, torch.Tensor],
                refiner_target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_noise: bool = True,
                refiner_residual_override: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        B, _, H, W = fine.shape
        dtype, device = fine.dtype, fine.device

        ctx_aligned = self._film_context(ctx, z_coarse)

        sigma_field = self.sigma_z

        t_in: Optional[torch.Tensor] = None
        if refiner_residual_override is not None:
            # Evaluation-only outer-loop feedback.  The caller computes the
            # current F-C / C-F / abs(F-C) residual against the previous coarse,
            # then passes the previous fine as this call's new z_coarse.
            if self.training:
                raise ValueError("outer-loop residual override is evaluation-only")
            r0 = refiner_residual_override.detach().to(
                device=device, dtype=dtype).reshape(B, 1, H, W)
            if not bool(torch.isfinite(r0).all()):
                raise FloatingPointError("non-finite outer-loop residual override")
            x_t = r0
            t_in = torch.zeros(B, device=device, dtype=dtype)
            self.last_t_norm = t_in.detach().clone()
            self.last_r0_abs_mean = r0.detach().abs().mean()
            self.last_target_valid_fraction = torch.ones(
                (), device=device, dtype=dtype)
        elif refiner_target is not None:
            # v8-a17: refiner_target is normally the detached depth-BEV z.
            # Detach here as well so no caller can accidentally backpropagate
            # through the pseudo-target into the depth network.
            tz, tvalid = refiner_target
            tz = tz.detach().to(device=device, dtype=dtype).reshape(1, 1, H, W) \
                .expand(B, -1, -1, -1)
            tv = tvalid.detach().to(device=device).reshape(1, 1, H, W) \
                .expand(B, -1, -1, -1)
            r0 = torch.clamp((tz - z_coarse.detach()) / self.res_max_m, -1.0, 1.0)
            r0 = r0 * tv.to(r0.dtype)

            if self.training:
                # Half clean t=0 conditioning, half conventional q-sampling.
                # The clean samples exactly match deterministic inference.
                t = torch.randint(1, self.T + 1, (B,), device=device)
                clean = torch.rand(B, device=device) < self.clean_prior_prob
                t = t.masked_fill(clean, 0)
                t_norm = t.float() / self.T
                abar = self._cosine_alpha_bar(t_norm).view(B, 1, 1, 1)
                eps = torch.randn_like(r0) * sigma_field
                x_t = torch.sqrt(abar) * r0 + torch.sqrt(1.0 - abar) * eps
                x_t = torch.where(clean.view(B, 1, 1, 1), r0, x_t)
                t_in = t_norm
            else:
                # Deterministic depth-prior inference: x_0 is the normalised
                # (z_depth_bev - z_coarse) residual, with no random noise.
                x_t = r0
                t_in = torch.zeros(B, device=device, dtype=dtype)
            self.last_t_norm = t_in.detach().clone()
            self.last_r0_abs_mean = r0.detach().abs().mean()
            self.last_target_valid_fraction = tv.detach().float().mean()
        elif use_noise:
            # Legacy fallback only; v8-a17 supplies depth-BEV refiner_target.
            x_t = torch.randn(B, 1, H, W, device=device, dtype=dtype) \
                * sigma_field
            t_in = torch.ones(B, device=device, dtype=dtype)
        else:
            x_t = torch.zeros(B, 1, H, W, device=device, dtype=dtype)

        h = torch.zeros(B, self.cond[0].out_channels, H // 2, W // 2,
                        device=device, dtype=dtype)

        t_emb = None
        if t_in is not None:
            t_emb = self.t_mlp(sinusoidal_t_embedding(t_in, self.cond[0].out_channels))

        r = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        z_mid: List[torch.Tensor] = []
        conf_list: List[torch.Tensor] = []
        dec = None
        for k in range(self.k_iters):
            # DiffMVS detaches the cumulative state at the start of each inner
            # iteration; the intermediate loss still supervises each update.
            if k > 0:
                r = r.detach()
                x_t = x_t.detach()

            z_cur = z_coarse + self.res_max_m * torch.tanh(r)
            conf_prev = conf_list[-1].detach() if conf_list else None
            # DiffMVS feeds the cumulative estimate (inv_depth_new) to the
            # next encoder, not only the latest residual.  Re-encode z_cur each
            # iteration so the main condition tracks the accumulated elevation.
            cond_k = self.cond(torch.cat(
                [fine, z_cur.detach(), ctx_aligned], dim=1))
            cond_k = cond_k + self._candidate_condition(
                img_feats, mats, z_cur, conf_prev)

            e1 = self.enc1(torch.cat([cond_k, x_t], dim=1))
            d = self.down(e1)
            h = self.gru(h, d)
            u = self.up(h)
            dec = self.dec1(torch.cat([u, e1], dim=1))
            if t_emb is not None:
                scale, shift = torch.chunk(t_emb, 2, dim=1)
                dec = dec * (1.0 + scale[:, :, None, None]) \
                    + shift[:, :, None, None]

            delta = self.delta(dec)
            conf = torch.sigmoid(self.conf(dec))

            r = r + delta
            z_new = z_coarse + self.res_max_m * torch.tanh(r)
            z_mid.append(z_new)
            conf_list.append(conf)
            # Next iteration sees the cumulative estimate (DiffMVS
            # inv_depth_new), not the latest increment.
            x_t = (z_new - z_coarse) / self.res_max_m

        residual_m = self.res_max_m * torch.tanh(r)
        z_refined = z_coarse + residual_m
        return z_refined, z_mid, conf_list
