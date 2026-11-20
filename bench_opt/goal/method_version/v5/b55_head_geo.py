#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5.5: hoist the feature-independent geometry out of the 3-scale candidate loop.

``ElevationRefiner._candidate_condition`` calls ``MultiViewCandidateBEV`` once
per P2/P3/P4 level with the *same* ``z_cand`` and ``mats``.  Everything up to
the normalised sampling grid (ego/camera transforms, intrinsics, validity mask,
support count) depends only on the geometry, so it is computed three times per
frame for identical values.  Preparing it once is bit-identical and removes
roughly 2/3 of that work.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _prepare(bev, z_candidates, mats):
    """Precompute per-camera sampling grids, validity and support counts."""
    N = int(z_candidates.shape[1])
    h, w = int(z_candidates.shape[-2]), int(z_candidates.shape[-1])
    if (h, w) != (bev.cand_size, bev.cand_size):
        raise ValueError("z_candidates must be %dx%d, got %dx%d"
                         % (bev.cand_size, bev.cand_size, h, w))
    device = z_candidates.device
    intrin_all = mats["intrin_mats"]
    Nc = int(intrin_all.shape[2])
    B = int(z_candidates.shape[0])

    xy = bev.grid_centers(device, torch.float32).permute(2, 0, 1).unsqueeze(0)
    xy = xy.expand(B, -1, -1, -1).detach()
    z = torch.nan_to_num(z_candidates.detach().float(), nan=0.0,
                         posinf=0.0, neginf=0.0)
    z = z.clamp(bev.pc_range[2], bev.pc_range[5])

    xy_b = xy.permute(0, 2, 3, 1)
    pts = torch.cat((xy_b.unsqueeze(1).expand(B, N, h, w, 2),
                     z.unsqueeze(-1)), dim=-1)
    ones = torch.ones_like(pts[..., :1])
    pts_h = torch.cat((pts, ones), dim=-1)

    bda = mats["bda_mat"].to(device=device, dtype=torch.float32)
    inv_bda = mats.get("bda_inv")
    if inv_bda is None:
        inv_bda = torch.inverse(bda)
    pts_ego = torch.einsum("bij,bnhwj->bnhwi", inv_bda, pts_h)

    intrin = mats["intrin_mats"][:, 0].to(device=device, dtype=torch.float32)
    ida = mats["ida_mats"][:, 0].to(device=device, dtype=torch.float32)
    s2e = mats["sensor2ego_mats"][:, 0].to(device=device, dtype=torch.float32)

    grids, valids = [], []
    hit = torch.zeros(B, N, 1, h, w, device=device, dtype=torch.float32)
    _s2e_inv = mats.get("sensor2ego_inv")
    for c in range(Nc):
        inv_s2e = (_s2e_inv[:, 0, c] if _s2e_inv is not None
                   else torch.inverse(s2e[:, c]))
        pts_cam = torch.einsum("bij,bnhwj->bnhwi", inv_s2e, pts_ego)
        Xc, Yc, Zc = pts_cam[..., 0], pts_cam[..., 1], pts_cam[..., 2]
        K = intrin[:, c, :3, :3]
        u0 = (K[:, 0:1, 0:1].view(B, 1, 1) * Xc
              + K[:, 0:1, 2:3].view(B, 1, 1) * Zc) / Zc.clamp_min(1e-8)
        v0 = (K[:, 1:2, 1:2].view(B, 1, 1) * Yc
              + K[:, 1:2, 2:3].view(B, 1, 1) * Zc) / Zc.clamp_min(1e-8)
        ida_c = ida[:, c]
        a = ida_c[:, 0:1, 0:1].view(B, 1, 1)
        b = ida_c[:, 0:1, 1:2].view(B, 1, 1)
        t = ida_c[:, 0:1, 3:4].view(B, 1, 1)
        d = ida_c[:, 1:2, 1:2].view(B, 1, 1)
        e = ida_c[:, 1:2, 0:1].view(B, 1, 1)
        f = ida_c[:, 1:2, 3:4].view(B, 1, 1)
        u_i = a * u0 + b * v0 + t
        v_i = e * u0 + d * v0 + f
        valid = ((Zc > bev.eps_depth)
                 & (u_i >= 0) & (u_i < bev.img_w)
                 & (v_i >= 0) & (v_i < bev.img_h)
                 & torch.isfinite(u_i) & torch.isfinite(v_i))
        valid_f = valid.float().unsqueeze(2)
        gx = u_i / bev.img_w * 2.0 - 1.0
        gy = v_i / bev.img_h * 2.0 - 1.0
        grid = torch.stack((gx, gy), dim=-1).reshape(B * N, h, w, 2)
        grids.append(grid)
        valids.append(valid_f)
        hit = hit + valid_f
    return {"grids": grids, "valids": valids, "hit": hit,
            "N": N, "h": h, "w": w, "B": B}


def _sample(bev, feat, geo):
    """Run only the feature-dependent part: projection + grid_sample + moments."""
    N, h, w, B = geo["N"], geo["h"], geo["w"], geo["B"]
    dtype = feat.dtype
    feat = bev.proj(feat).float()
    Cf = feat.shape[1]
    Hf, Wf = feat.shape[-2], feat.shape[-1]
    feat = feat.reshape(B, feat.shape[0] // B, Cf, Hf, Wf)
    Nc = feat.shape[1]

    sum_feat = torch.zeros(B, N, Cf, h, w, device=feat.device, dtype=torch.float32)
    sum_sq = torch.zeros_like(sum_feat)
    for c in range(Nc):
        feat_c = feat[:, c].unsqueeze(1).expand(B, N, Cf, Hf, Wf)
        feat_c = feat_c.reshape(B * N, Cf, Hf, Wf)
        sample = F.grid_sample(
            feat_c, geo["grids"][c], mode="bilinear",
            padding_mode="zeros", align_corners=False,
        ).reshape(B, N, Cf, h, w)
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


def install(refiner):
    """Wrap ElevationRefiner._candidate_condition with a single geometry prep."""
    if getattr(refiner, "_v55_geo", False):
        return refiner
    bev = refiner.cand_bev
    ms_fpn = refiner.ms_fpn
    cand_cond = refiner.cand_cond
    cond_enc = cand_cond
    logits = refiner.ms_scale_logits

    def _candidate_condition(img_feats, mats, z_cur, conf_prev):
        cs = refiner.cand_size
        z_small = F.interpolate(z_cur, size=(cs, cs), mode="bilinear",
                                align_corners=False).squeeze(1)
        if conf_prev is None:
            radius = torch.full_like(z_small, refiner.radius_base)
        else:
            c_small = F.interpolate(conf_prev, size=(cs, cs), mode="bilinear",
                                    align_corners=False).squeeze(1)
            radius = refiner.radius_min + (1.0 - c_small) * (
                refiner.radius_max - refiner.radius_min)
        offsets = torch.linspace(
            -1.0, 1.0, refiner.num_candidates,
            device=z_cur.device, dtype=z_cur.dtype).view(1, refiner.num_candidates, 1, 1)
        z_cand = (z_small.unsqueeze(1) + offsets * radius.unsqueeze(1)).clamp(
            refiner.z_min, refiner.z_max)

        geo = _prepare(bev, z_cand, mats)
        cand_feat = None
        cost = None
        valid = None
        scale_weights = torch.softmax(logits, dim=0)
        for scale_idx, feat in enumerate(ms_fpn(img_feats)):
            f_s, c_s, v_s = _sample(bev, feat, geo)
            w_s = scale_weights[scale_idx].to(dtype=f_s.dtype)
            cand_feat = f_s * w_s if cand_feat is None else cand_feat + f_s * w_s
            cost = c_s * w_s if cost is None else cost + c_s * w_s
            valid = v_s if valid is None else torch.maximum(valid, v_s)
        dz = (z_cand - z_small.unsqueeze(1)) / refiner.res_max_m
        cand_cond = cond_enc(cand_feat, cost, valid, dz)
        return F.interpolate(cand_cond, size=z_cur.shape[-2:], mode="bilinear",
                             align_corners=False)

    refiner._candidate_condition = _candidate_condition
    refiner._v55_geo = True
    return refiner


def install_class():
    """Patch ElevationRefiner so the first call installs the hoisted path."""
    import rogs_refiner

    cls = rogs_refiner.ElevationRefiner
    orig = cls.forward

    def forward(self, *args, **kwargs):
        if not getattr(self, "_v55_geo", False):
            install(self)
        return orig(self, *args, **kwargs)

    cls.forward = forward
    return cls
