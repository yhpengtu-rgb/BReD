#!/usr/bin/env python3
"""Strict-equivalent stereo optimizations, isolated subclass mixin.

Changes vs vendor ``bevstereo_lss_fpn.py`` (formal vendor source untouched):

  * DepthNet: dead ``d_coords`` allocation removed (built, never used).
  * ``create_depth_sample_frustum``: static image-plane x/y grid cached per
    downsample factor; only the depth channel is rebuilt per call.
  * ``homo_warping``: per (ref, src) precompute
    ``A = K_src @ sensor2sensor @ inv(K_ref)`` and reuse across the 4 ranges
    x 3 EM iterations and the mask warp. Perspective division and
    ``ida_src`` remain outside A (cannot be folded before division).
  * ``_generate_cost_volume``: grouped channel reduce via einsum instead of
    materializing the full ``[B, G, C/G, D, H, W]`` product tensor.
  * ``_forward_stereo``: transform only the ref sweep's mu/sigma per range;
    ``mu.repeat`` replaced by ``expand``; depth coordinate base cached.

EM=3, num_ranges=4, 2 sweep x 6 cams, G=8, depth-bin order, masks and FP32
output semantics are unchanged. No inference dtype is changed here.
"""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F

from bevdepth.layers.backbones.bevstereo_lss_fpn import (  # noqa: E402
    BEVStereoLSSFPN, DepthNet)

_CORR_MOD = None
_CORR_LOAD_MS = None


def _load_corr_module():
    """Lazily build/load the fused grouped-correlation CUDA extension."""
    global _CORR_MOD, _CORR_LOAD_MS
    if _CORR_MOD is not None:
        return _CORR_MOD
    from torch.utils import cpp_extension
    here = os.path.dirname(os.path.abspath(__file__))
    build = os.path.join(here, "corr_ext", "build_corr")
    os.makedirs(build, exist_ok=True)
    _t0 = time.perf_counter()
    _CORR_MOD = cpp_extension.load(
        name="corr_ext",
        sources=[os.path.join(here, "corr_ext", "src", "grouped_corr.cu")],
        build_directory=build,
        verbose=False,
        extra_cuda_cflags=["--expt-relaxed-constexpr"],
    )
    _CORR_LOAD_MS = (time.perf_counter() - _t0) * 1e3
    return _CORR_MOD


class DepthNetOpt(DepthNet):
    """DepthNet with the dead d_coords allocation removed."""

    def forward(self, x, mats_dict, scale_depth_factor=1000.0):
        B, _, H, W = x.shape
        intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict['ida_mats'][:, 0:1, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
        bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4,
                                        4).repeat(1, 1, num_cams, 1, 1)
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth_feat = self.depth_se(x, depth_se)
        depth_feat = self.depth_feat_conv(depth_feat)
        mono_depth = self.mono_depth_net(depth_feat)
        mu_sigma_score = self.mu_sigma_range_net(depth_feat)
        # Vendor allocated `d_coords` here but never used it; removed.
        mu = mu_sigma_score[:, 0:self.num_ranges, ...]
        sigma = mu_sigma_score[:, self.num_ranges:2 * self.num_ranges, ...]
        range_score = mu_sigma_score[:,
                                     2 * self.num_ranges:3 * self.num_ranges,
                                     ...]
        sigma = F.elu(sigma) + 1.0 + 1e-10
        return x, context, mu, sigma, range_score, mono_depth


class BEVStereoOpt(BEVStereoLSSFPN):
    """Mixin with cached frustum grid, cached camera matrix A, fused reduce."""

    def __init__(self, *args, use_einsum_corr: bool = True,
                 use_A: bool = True, ref_only: bool = True,
                 use_cuda_corr: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._opt_cache: dict = {}
        self._frustum_grid: dict = {}
        self._d_coords_base: object = None
        self._use_einsum_corr = use_einsum_corr
        self._use_A = use_A
        self._ref_only = ref_only
        self._use_cuda_corr = use_cuda_corr

    def _configure_depth_net(self, depth_net_conf):
        return DepthNetOpt(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
            self.d_bound,
            self.num_ranges,
        )

    def clear_opt_cache(self):
        self._opt_cache.clear()

    # ------------------------------------------------------------------ #
    # static frustum image-plane grid
    # ------------------------------------------------------------------ #
    def create_depth_sample_frustum(self, depth_sample, downsample_factor=16):
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // downsample_factor, ogfW // downsample_factor
        batch_size, num_depth, _, _ = depth_sample.shape
        key = (downsample_factor, fH, fW, depth_sample.device,
               depth_sample.dtype)
        if key not in self._frustum_grid:
            x_coords = torch.linspace(
                0, ogfW - 1, fW, dtype=torch.float,
                device=depth_sample.device).view(1, 1, 1, fW)
            y_coords = torch.linspace(
                0, ogfH - 1, fH, dtype=torch.float,
                device=depth_sample.device).view(1, 1, fH, 1)
            self._frustum_grid[key] = (x_coords, y_coords)
        x_coords, y_coords = self._frustum_grid[key]
        x_coords = x_coords.expand(batch_size, num_depth, fH, fW)
        y_coords = y_coords.expand(batch_size, num_depth, fH, fW)
        paddings = torch.ones_like(depth_sample)
        frustum = torch.stack((x_coords, y_coords, depth_sample, paddings), -1)
        return frustum

    # ------------------------------------------------------------------ #
    # homo warping with cached A = K_src @ s2s @ inv(K_ref)
    # ------------------------------------------------------------------ #
    def _camera_A(self, key_intrin_mats, sweep_intrin_mats,
                  sensor2sensor_mats, B):
        k = ('A', id(key_intrin_mats), id(sweep_intrin_mats),
             id(sensor2sensor_mats))
        if k not in self._opt_cache:
            k_ref = key_intrin_mats.reshape(B, 4, 4)
            k_src = sweep_intrin_mats.reshape(B, 4, 4)
            s2s = sensor2sensor_mats.reshape(B, 4, 4)
            self._opt_cache[k] = k_src @ s2s @ k_ref.inverse()
        return self._opt_cache[k]

    def homo_warping(self, stereo_feat, key_intrin_mats, sweep_intrin_mats,
                     sensor2sensor_mats, key_ida_mats, sweep_ida_mats,
                     depth_sample, frustum):
        B, channels = stereo_feat.shape[0], stereo_feat.shape[1]
        height, width = stereo_feat.shape[2], stereo_feat.shape[3]
        with torch.no_grad():
            points = frustum
            points = points.reshape(points.shape[0], -1, points.shape[-1])
            points[..., 2] = 1

            k = ('ida', id(key_ida_mats))
            if k not in self._opt_cache:
                self._opt_cache[k] = key_ida_mats.reshape(B, 4, 4).inverse()
            ida_inv = self._opt_cache[k]
            points = ida_inv.unsqueeze(1) @ points.unsqueeze(-1)

            points[..., :3, :] *= depth_sample.reshape(B, -1, 1, 1)
            num_depth = frustum.shape[1]
            if self._use_A:
                A = self._camera_A(key_intrin_mats, sweep_intrin_mats,
                                   sensor2sensor_mats, B)
                points = A.unsqueeze(1) @ points
            else:
                k = ('kinv', id(key_intrin_mats))
                if k not in self._opt_cache:
                    self._opt_cache[k] = key_intrin_mats.reshape(B, 4, 4).inverse()
                kinv = self._opt_cache[k]
                points = kinv.unsqueeze(1) @ points
                points = sensor2sensor_mats.reshape(B, 4, 4).unsqueeze(1) @ points
                points = sweep_intrin_mats.reshape(B, 4, 4).unsqueeze(1) @ points
            points[..., :2, :] = points[..., :2, :] / points[..., 2:3, :]

            points = sweep_ida_mats.reshape(B, 4, 4).unsqueeze(1) @ points
            points = points.squeeze(-1)
            neg_mask = points[..., 2] < 1e-3
            points[..., 0][neg_mask] = width * self.stereo_downsample_factor
            points[..., 1][neg_mask] = height * self.stereo_downsample_factor
            points[..., 2][neg_mask] = 1
            proj_x_normalized = points[..., 0] / (
                (width * self.stereo_downsample_factor - 1) / 2) - 1
            proj_y_normalized = points[..., 1] / (
                (height * self.stereo_downsample_factor - 1) / 2) - 1
            grid = torch.stack([proj_x_normalized, proj_y_normalized], dim=2)

        warped_stereo_fea = F.grid_sample(
            stereo_feat,
            grid.view(B, num_depth * height, width, 2),
            mode='bilinear',
            padding_mode='zeros',
        )
        warped_stereo_fea = warped_stereo_fea.view(B, channels, num_depth,
                                                   height, width)
        return warped_stereo_fea

    # ------------------------------------------------------------------ #
    # grouped correlation fused reduce
    # ------------------------------------------------------------------ #
    def _generate_cost_volume(self, sweep_index, stereo_feats_all_sweeps,
                               mats_dict, depth_sample, depth_sample_frustum,
                               sensor2sensor_mats):
        batch_size, num_channels, height, width = \
            stereo_feats_all_sweeps[0].shape
        num_sweeps = len(stereo_feats_all_sweeps)
        depth_score_all_sweeps = list()
        Cg = num_channels // self.num_groups
        for idx in range(num_sweeps):
            if idx == sweep_index:
                continue
            warped_stereo_fea = self.homo_warping(
                stereo_feats_all_sweeps[idx],
                mats_dict['intrin_mats'][:, sweep_index, ...],
                mats_dict['intrin_mats'][:, idx, ...],
                sensor2sensor_mats[idx],
                mats_dict['ida_mats'][:, sweep_index, ...],
                mats_dict['ida_mats'][:, idx, ...],
                depth_sample,
                depth_sample_frustum.type_as(stereo_feats_all_sweeps[idx]),
            )
            warped_stereo_fea = warped_stereo_fea.reshape(
                batch_size, self.num_groups, Cg, self.num_samples, height,
                width)
            ref_stereo_feat = stereo_feats_all_sweeps[sweep_index].reshape(
                batch_size, self.num_groups, Cg, height, width)

            if self._use_cuda_corr:
                corr = _load_corr_module()
                ref_flat = stereo_feats_all_sweeps[sweep_index].contiguous()
                warp_flat = warped_stereo_fea.reshape(
                    batch_size, num_channels, self.num_samples,
                    height, width).contiguous()
                feat_cost = corr.grouped_corr(
                    ref_flat, warp_flat, self.num_groups)
            elif self._use_einsum_corr:
                BG = batch_size * self.num_groups
                ref_g = ref_stereo_feat.reshape(BG, Cg, height * width)
                warp_g = warped_stereo_fea.reshape(
                    BG, Cg, self.num_samples, height * width)
                # sum over grouped channels, then mean over Cg (same as vendor
                # torch.mean(..., axis=2)); einsum avoids the full product tensor.
                feat_cost = torch.einsum(
                    'gch,gcsh->gsh', ref_g, warp_g).div_(Cg)
                feat_cost = feat_cost.reshape(batch_size, self.num_groups,
                                              self.num_samples, height, width)
            else:
                feat_cost = torch.mean(
                    ref_stereo_feat.unsqueeze(3) * warped_stereo_fea, axis=2)
            depth_score = self.similarity_net(feat_cost).squeeze(1)
            depth_score_all_sweeps.append(depth_score)
        return torch.stack(depth_score_all_sweeps).mean(0)

    # ------------------------------------------------------------------ #
    # stereo depth: ref-only mu/sigma, expand, cached depth base
    # ------------------------------------------------------------------ #
    def _forward_stereo(self, sweep_index, stereo_feats_all_sweeps,
                        mono_depth_all_sweeps, mats_dict,
                        sensor2sensor_mats, mu_all_sweeps, sigma_all_sweeps,
                        range_score_all_sweeps, depth_feat_all_sweeps):
        batch_size_with_cams, _, feat_height, feat_width = \
            stereo_feats_all_sweeps[0].shape
        device = stereo_feats_all_sweeps[0].device
        if self._d_coords_base is None:
            self._d_coords_base = torch.arange(
                *self.d_bound, dtype=torch.float, device=device)
        d_coords = self._d_coords_base.reshape(
            1, -1, 1, 1).expand(
            batch_size_with_cams, self.depth_channels, feat_height,
            feat_width)
        stereo_depth = stereo_feats_all_sweeps[0].new_zeros(
            batch_size_with_cams, self.depth_channels, feat_height, feat_width)
        mask_score = stereo_feats_all_sweeps[0].new_zeros(
            batch_size_with_cams,
            self.depth_channels,
            feat_height * self.stereo_downsample_factor //
            self.downsample_factor,
            feat_width * self.stereo_downsample_factor //
            self.downsample_factor,
        )
        score_all_ranges = list()
        range_score = range_score_all_sweeps[sweep_index].softmax(1)
        for range_idx in range(self.num_ranges):
            range_start = self.range_list[range_idx][0]
            range_end = self.range_list[range_idx][1]
            if self._ref_only:
                # Only the ref sweep's mu/sigma feed the EM updates; the other
                # sweeps' per-range transforms are unused in this method.
                mu = mu_all_sweeps[sweep_index][
                    :, range_idx:range_idx + 1, ...].sigmoid() * (
                        range_end - range_start) + range_start
                sigma = sigma_all_sweeps[sweep_index][
                    :, range_idx:range_idx + 1, ...]
            else:
                mu_all_single = [
                    m[:, range_idx:range_idx + 1, ...].sigmoid() *
                    (range_end - range_start) + range_start
                    for m in mu_all_sweeps]
                sigma_all_single = [
                    s[:, range_idx:range_idx + 1, ...]
                    for s in sigma_all_sweeps]
                mu = mu_all_single[sweep_index]
                sigma = sigma_all_single[sweep_index]
            for _ in range(self.em_iteration):
                depth_sample = torch.cat(
                    [mu + sigma * k for k in self.k_list], 1)
                depth_sample_frustum = self.create_depth_sample_frustum(
                    depth_sample, self.stereo_downsample_factor)
                mu_score = self._generate_cost_volume(
                    sweep_index,
                    stereo_feats_all_sweeps,
                    mats_dict,
                    depth_sample,
                    depth_sample_frustum,
                    sensor2sensor_mats,
                )
                mu_score = mu_score.softmax(1)
                scale_factor = torch.clamp(
                    0.5 / (1e-4 + mu_score[:, self.num_samples //
                                           2:self.num_samples // 2 + 1, ...]),
                    min=0.1,
                    max=10)
                sigma = torch.clamp(sigma * scale_factor, min=0.1, max=10)
                mu = (depth_sample * mu_score).sum(1, keepdim=True)
                del depth_sample
                del depth_sample_frustum
            range_length = int(
                (range_end - range_start) // self.d_bound[2])
            if self.use_mask:
                depth_sample = F.avg_pool2d(
                    mu,
                    self.downsample_factor // self.stereo_downsample_factor,
                    self.downsample_factor // self.stereo_downsample_factor,
                )
                depth_sample_frustum = self.create_depth_sample_frustum(
                    depth_sample, self.downsample_factor)
                mask = self._forward_mask(
                    sweep_index,
                    mono_depth_all_sweeps,
                    mats_dict,
                    depth_sample,
                    depth_sample_frustum,
                    sensor2sensor_mats,
                )
                mask_score[:,
                           int((range_start - self.d_bound[0]) //
                               self.d_bound[2]):range_length +
                           int((range_start - self.d_bound[0]) //
                               self.d_bound[2]), ..., ] += mask
                del depth_sample
                del depth_sample_frustum
            sigma = torch.clamp(sigma, self.min_sigma)
            mu_repeated = mu.expand(
                -1, range_length, -1, -1)
            eps = 1e-6
            depth_score_single_range = (-1 / 2 * (
                (d_coords[:,
                          int((range_start - self.d_bound[0]) //
                              self.d_bound[2]):range_length +
                          int((range_start - self.d_bound[0]) //
                              self.d_bound[2]), ..., ] - mu_repeated) /
                torch.sqrt(sigma))**2)
            depth_score_single_range = depth_score_single_range.exp()
            score_all_ranges.append(mu_score.sum(1).unsqueeze(1))
            depth_score_single_range = depth_score_single_range / (
                sigma * math.sqrt(2 * math.pi) + eps)
            stereo_depth[:,
                         int((range_start - self.d_bound[0]) //
                             self.d_bound[2]):range_length +
                         int((range_start - self.d_bound[0]) //
                             self.d_bound[2]), ..., ] = (
                                 depth_score_single_range *
                                 range_score[:, range_idx:range_idx + 1, ...])
            del depth_score_single_range
            del mu_repeated
        if self.use_mask:
            return stereo_depth, mask_score
        else:
            return stereo_depth


def install_depth_merge(lift):
    """Wrap DepthNet so the two per-sweep calls run as one B=2,Nc=6 batch.

    Non-invasive: vendor ``forward`` loop is left untouched. The first sweep's
    depth_net call records its feat/mats and returns uninitialized placeholders;
    the second sweep triggers one batched depth_net call, splits it back, and
    fills the placeholders in place. The context down-sampler must be Identity
    (as in this baseline); any other module aborts instead of guessing.

    Merge mats: every row copies the exact sweep0 camera slice the original two
    calls read (vendor DepthNet indexes ``[:, 0:1]``); sweep1 matrices are not
    used. DepthNet weights/state and all stereo/matcher kernels are untouched.
    """
    dn = lift.depth_net
    if type(lift.context_downsample_net).__name__ != "Identity":
        raise RuntimeError("depth-merge requires Identity context_downsample_net")
    orig_dn = dn.forward
    state = {"pending": None}

    def _reset(_mod, _inp):
        state["pending"] = None

    def _merge_mats(mats, n_batch):
        # Keep the (batch, sweep, cams) layout DepthNet expects: batch rows =
        # the two sweep groups, cams = 6; flatten batch*cams matches x.
        merged = {}
        for key in ("intrin_mats", "ida_mats", "sensor2ego_mats"):
            merged[key] = mats[key][:, 0:1, ...].repeat(
                n_batch, 1, 1, 1, 1).contiguous()
        merged["bda_mat"] = mats["bda_mat"].repeat(n_batch, 1, 1).contiguous()
        # sensor2sensor is not read by DepthNet; pass through unchanged.
        merged["sensor2sensor_mats"] = mats["sensor2sensor_mats"]
        return merged

    def _dn_wrap(feat, mats_dict, scale_depth_factor=1000.0):
        n, c, h, w = feat.shape
        if state["pending"] is None:
            ctx_ch = dn.context_conv.out_channels
            mono_ch = dn.mono_depth_net[-1].out_channels
            nr = dn.num_ranges
            placeholders = [
                feat.new_empty(n, c, h, w),                 # depth_feat
                feat.new_empty(n, ctx_ch, h, w),            # context
                feat.new_empty(n, nr, 4 * h, 4 * w),       # mu
                feat.new_empty(n, nr, 4 * h, 4 * w),        # sigma
                feat.new_empty(n, nr, 4 * h, 4 * w),       # range_score
                feat.new_empty(n, mono_ch, h, w),          # mono_depth
            ]
            state["pending"] = {"feat": feat.detach(), "mats": mats_dict,
                                "ph": placeholders, "n": n}
            return tuple(placeholders)
        pending = state["pending"]
        x12 = torch.cat([pending["feat"], feat], dim=0).contiguous()
        n_batch = x12.shape[0] // pending["n"]
        mats12 = _merge_mats(pending["mats"], n_batch)
        out12 = tuple(orig_dn(x12, mats12, scale_depth_factor))
        n0 = pending["n"]
        for ph, row in zip(pending["ph"], out12):
            ph.copy_(row[:n0])
        state["pending"] = None
        return tuple(row[n0:] for row in out12)

    lift.register_forward_pre_hook(_reset)
    dn.forward = _dn_wrap


def install_img_cache(lift, capacity=4):
    """Bounded GPU LRU of image-only backbone+neck packs across adjacent frames.

    Caches only the ``get_cam_feats`` outputs (neck ``img_feats`` and detached
    top-scale ``stereo_feats``) keyed by exact tensor equality on the input image
    pack. DepthNet, stereo cost-volume, matcher and head are never cached;
    sensor2ego / intrinsic / IDA matrices follow their original per-sweep path.

    Non-invasive: vendor ``forward`` loop is untouched. A miss runs the original
    ``get_cam_feats`` and stores detached clones; a hit returns clones of the
    cached pack so the cache storage cannot be mutated downstream. Content
    equality is checked on GPU (no D2H digest/hash in the hot path).
    """
    orig_gcf = lift.get_cam_feats
    state = {"packs": [], "hits": 0, "misses": 0, "capacity": capacity,
             "calls": 0}

    def _same(a, b):
        return a.shape == b.shape and torch.equal(a, b)

    def wrap(imgs):
        # Vendor loop calls get_cam_feats twice per frame in fixed sweep order:
        # even call index is the key-frame sweep, odd is the secondary sweep.
        # The head's c2/c3/c4 hook state is captured on the key-frame backbone
        # call, so a key-frame call must never be cache-served (that would skip
        # img_backbone and leave the hooks unfired). Only the secondary sweep
        # may reuse a stored pack; key-frame misses still store their pack.
        is_key = (state["calls"] % 2 == 0)
        state["calls"] += 1
        packs = state["packs"]
        if not is_key:
            for idx, pack in enumerate(packs):
                if _same(imgs, pack["imgs"]):
                    packs.append(packs.pop(idx))  # LRU: move to most-recent
                    state["hits"] += 1
                    return pack["img_feats"].clone(), pack["stereo_feats"].clone()
        img_feats, stereo_feats = orig_gcf(imgs)
        packs.append({
            "imgs": imgs.detach().clone(),
            "img_feats": img_feats.detach().clone(),
            "stereo_feats": stereo_feats.detach().clone(),
        })
        while len(packs) > capacity:
            packs.pop(0)
        state["misses"] += 1
        return img_feats, stereo_feats

    lift.get_cam_feats = wrap
    lift._img_cache = state
