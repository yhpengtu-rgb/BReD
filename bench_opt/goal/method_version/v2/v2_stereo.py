#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2: 在 v1.1 (EM=1) 基础上压缩立体代价体的通道冗余。

本文件是 method_version/v2 的**新增代码**，不修改原始 BReD 代码树，也不修改 v1。
继承 ``V1StereoMixin``：EM 调度、槽位、mask、Gaussian 打分全部沿用 v1.1，
仅覆盖 ``_v1_generate_cost_volume``（相关性/SimNet 入口）。

v2.0 相关性通道切片（零训练、权重不变）
----------------------------------------
原始分组相关：把 512 通道切成 8 组、每组 64 通道，逐像素乘积后在组内取均值
    feat_cost[b,g,d,h,w] = mean_{c<Cg} ref[b,g,c,h,w] * warp[b,g,c,d,h,w]
再送入 8 通道输入的 SimNet（1x1x1 pointwise 3D conv）。

v2.0 只保留每组前 K 个通道的均值：
    feat_cost' = mean_{c<K} ref * warp,  K = 64 / 48 / 32 / 16
SimNet 的输入通道数（8）与所有权重完全不变，因此**无需训练、无新 ckpt**。
切片在 ``pre`` 阶段执行（在 homo_warping 之前按组抽通道）：
  * grid_sample 逐通道独立，先切片与"先 warp 后切片"在数学上逐元素相同；
  * 因此 ``pre`` 与 ``post`` 精度完全一致，但 ``pre`` 额外省下 homo grid_sample
    在 (64-K)/64 通道上的开销。

v2.2 mask 通道/扫描压缩（默认关闭）
-----------------------------------
``mask_sweeps=N`` 时，mask 只对前 N 个非关键 sweep 求均值（默认保持全部），
用于评估 11.2 s mask 预算里 sweep 维度的冗余，权重与输入通道数不变。

统计：v2_stats.json 记录每帧实际 warp 通道数、SimNet 输入通道数与调用次数。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from v1_stereo import V1StereoMixin


class V2StereoMixin(V1StereoMixin):
    """在 v1.1 EM=1 之上做相关性通道切片 / mask sweep 压缩。"""

    def v2_init(
        self,
        slot_idx,
        em_iters: int,
        *,
        corr_channels=None,
        slice_stage: str = "pre",
        mask_sweeps=None,
        corr_sweeps=None,
        mask_stride=None,
        mask_batch=False,
        **v1_kwargs,
    ) -> None:
        self.v1_init(slot_idx, em_iters, **v1_kwargs)
        if slice_stage not in ("pre", "post"):
            raise ValueError("slice_stage must be 'pre' or 'post'")
        self.v2_corr_channels = corr_channels
        self.v2_slice_stage = slice_stage
        self.v2_mask_sweeps = mask_sweeps
        self.v2_corr_sweeps = corr_sweeps
        self.v2_mask_stride = mask_stride
        self.v2_mask_batch = bool(mask_batch)
        self.v2_stats = {
            "corr_channels_requested": corr_channels,
            "corr_channels_used": None,
            "slice_stage": slice_stage,
            "mask_sweeps": mask_sweeps,
            "corr_sweeps": corr_sweeps,
            "mask_stride": mask_stride,
            "mask_batch": bool(mask_batch),
            "mask_batch_calls": 0,
            "cost_volume_calls": 0,
            "warp_channels_per_call": [],
            "ref_channels_per_call": [],
            "sweeps_per_call": [],
        }

    def _v2_group_slice(self, feat, batch_size, groups, ch_per_group, k,
                        height, width):
        """按组取前 k 通道并重排成连续通道，返回 (B, groups*k, H, W)。"""
        return feat.reshape(batch_size, groups, ch_per_group, height,
                            width)[:, :, :k].reshape(batch_size, groups * k,
                                                     height, width)

    # -- 覆盖 v1 的代价体入口；循环/矩阵/softmax 位置与 vendor 一致。
    def _v1_generate_cost_volume(
        self,
        sweep_index,
        stereo_feats_all_sweeps,
        mats_dict,
        depth_sample,
        depth_sample_frustum,
        sensor2sensor_mats,
    ):
        batch_size, num_channels, height, width = stereo_feats_all_sweeps[
            0].shape
        num_sweeps = len(stereo_feats_all_sweeps)
        d_slots = depth_sample.shape[1]
        groups = self.num_groups
        ch_per_group = num_channels // groups
        k = ch_per_group if self.v2_corr_channels is None else int(
            self.v2_corr_channels)
        k = max(1, min(k, ch_per_group))
        use_slice = k < ch_per_group

        depth_score_all_sweeps = list()
        key_feat = stereo_feats_all_sweeps[sweep_index]
        if use_slice:
            key_ref = self._v2_group_slice(key_feat, batch_size, groups,
                                           ch_per_group, k, height, width)
            ref_stereo_feat = key_ref.reshape(batch_size, groups, k, height,
                                              width)
        else:
            ref_stereo_feat = key_feat.reshape(batch_size, groups, ch_per_group,
                                               height, width)
        limit = None if self.v2_corr_sweeps is None else max(
            1, int(self.v2_corr_sweeps))
        n_used = 0
        for idx in range(num_sweeps):
            if idx == sweep_index:
                continue
            if limit is not None and n_used >= limit:
                break
            n_used += 1
            feat = stereo_feats_all_sweeps[idx]
            if use_slice and self.v2_slice_stage == "pre":
                feat_sel = self._v2_group_slice(feat, batch_size, groups,
                                                ch_per_group, k, height, width)
                warped_stereo_fea = self.homo_warping(
                    feat_sel,
                    mats_dict['intrin_mats'][:, sweep_index, ...],
                    mats_dict['intrin_mats'][:, idx, ...],
                    sensor2sensor_mats[idx],
                    mats_dict['ida_mats'][:, sweep_index, ...],
                    mats_dict['ida_mats'][:, idx, ...],
                    depth_sample,
                    depth_sample_frustum.type_as(feat_sel),
                )
                warped_stereo_fea = warped_stereo_fea.reshape(
                    batch_size, groups, k, d_slots, height, width)
            else:
                warped_stereo_fea = self.homo_warping(
                    feat,
                    mats_dict['intrin_mats'][:, sweep_index, ...],
                    mats_dict['intrin_mats'][:, idx, ...],
                    sensor2sensor_mats[idx],
                    mats_dict['ida_mats'][:, sweep_index, ...],
                    mats_dict['ida_mats'][:, idx, ...],
                    depth_sample,
                    depth_sample_frustum.type_as(feat),
                )
                warped_stereo_fea = warped_stereo_fea.reshape(
                    batch_size, groups, ch_per_group, d_slots, height, width)
                if use_slice:
                    warped_stereo_fea = warped_stereo_fea[:, :, :k]

            feat_cost = torch.mean(
                (ref_stereo_feat.unsqueeze(3) * warped_stereo_fea), axis=2)
            depth_score = self.similarity_net(feat_cost).squeeze(1)
            depth_score_all_sweeps.append(depth_score)

        self.v1_stats["cost_volume_calls"] += 1
        self.v1_stats["slots_per_call"].append(int(d_slots))
        self.v2_stats["cost_volume_calls"] += 1
        self.v2_stats["corr_channels_used"] = int(k)
        self.v2_stats["warp_channels_per_call"].append(
            int(groups * k) if (use_slice and self.v2_slice_stage == "pre")
            else int(num_channels))
        self.v2_stats["ref_channels_per_call"].append(int(groups * k))
        self.v2_stats["sweeps_per_call"].append(int(n_used))
        return torch.stack(depth_score_all_sweeps).mean(0)

    # -- v2.2：mask 路径压缩。mask 只在 depth_sample 上做一次 homo warp，
    #    再与关键帧 mono depth 拼接送 mask_net；两项都可降采样后上采样回原尺寸。
    def _forward_mask(
        self,
        sweep_index,
        mono_depth_all_sweeps,
        mats_dict,
        depth_sample,
        depth_sample_frustum,
        sensor2sensor_mats,
    ):
        if self.v2_mask_sweeps is None and self.v2_mask_stride is None:
            return super()._forward_mask(
                sweep_index, mono_depth_all_sweeps, mats_dict, depth_sample,
                depth_sample_frustum, sensor2sensor_mats)

        limit = None if self.v2_mask_sweeps is None else max(
            1, int(self.v2_mask_sweeps))
        stride = 1 if self.v2_mask_stride is None else max(
            1, int(self.v2_mask_stride))
        key_depth = mono_depth_all_sweeps[sweep_index]
        target_hw = key_depth.shape[-2:]
        ds, fr, key = depth_sample, depth_sample_frustum, key_depth
        if stride > 1:
            # 均匀池化后重算 frustum；create_depth_sample_frustum 由
            # final_dim // factor 决定网格尺寸，所以 factor 同步放大 stride 倍。
            ds = F.avg_pool2d(depth_sample, stride, stride)
            fr = self.create_depth_sample_frustum(
                ds, self.downsample_factor * stride)
            key = F.avg_pool2d(key_depth, stride, stride)

        num_sweeps = len(mono_depth_all_sweeps)
        mask_all_sweeps = list()
        used = 0
        for idx in range(num_sweeps):
            if idx == sweep_index:
                continue
            if limit is not None and used >= limit:
                break
            src = mono_depth_all_sweeps[idx]
            if stride > 1:
                src = F.avg_pool2d(src, stride, stride)
            warped_mono_depth = self.homo_warping(
                src,
                mats_dict['intrin_mats'][:, sweep_index, ...],
                mats_dict['intrin_mats'][:, idx, ...],
                sensor2sensor_mats[idx],
                mats_dict['ida_mats'][:, sweep_index, ...],
                mats_dict['ida_mats'][:, idx, ...],
                ds,
                fr.type_as(src),
            )
            mask = self.mask_net(
                torch.cat([key.detach(), warped_mono_depth.mean(2).detach()],
                          1))
            if stride > 1:
                mask = F.interpolate(mask, size=target_hw, mode='bilinear',
                                     align_corners=False)
            mask_all_sweeps.append(mask)
            used += 1
        return torch.stack(mask_all_sweeps).mean(0)

    # -- v2.2b：把 4 个 range 的 mask 拼成一次调用（每帧 8 次 -> 2 次）。
    #    先跑完所有 range 的 EM 得到 mu/sigma，再统一做 mask，最后统一写回
    #    Gaussian 深度分与 mask_score。逐 range 的数学与 v1 完全一致。
    def _forward_stereo(
        self,
        sweep_index,
        stereo_feats_all_sweeps,
        mono_depth_all_sweeps,
        mats_dict,
        sensor2sensor_mats,
        mu_all_sweeps,
        sigma_all_sweeps,
        range_score_all_sweeps,
        depth_feat_all_sweeps,
    ):
        if not self.v2_mask_batch:
            return super()._forward_stereo(
                sweep_index, stereo_feats_all_sweeps, mono_depth_all_sweeps,
                mats_dict, sensor2sensor_mats, mu_all_sweeps,
                sigma_all_sweeps, range_score_all_sweeps,
                depth_feat_all_sweeps)

        batch_size_with_cams, _, feat_height, feat_width = \
            stereo_feats_all_sweeps[0].shape
        device = stereo_feats_all_sweeps[0].device
        d_coords = torch.arange(*self.d_bound, dtype=torch.float,
                                device=device).reshape(1, -1, 1, 1)
        d_coords = d_coords.repeat(batch_size_with_cams, 1, feat_height,
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
        range_score = range_score_all_sweeps[sweep_index].softmax(1)
        bin_size = float(self.d_bound[2])

        mus, sigmas, range_scores, mask_depths, mask_frustums = [], [], [], [], []
        for range_idx in range(self.num_ranges):
            range_start = self.range_list[range_idx][0]
            range_end = self.range_list[range_idx][1]
            mu = (mu_all_sweeps[sweep_index][:, range_idx:range_idx + 1, ...]
                  .sigmoid() * (range_end - range_start) + range_start)
            sigma = sigma_all_sweeps[sweep_index][:, range_idx:range_idx + 1,
                                                  ...]

            max_iters = self.v1_em_iters
            if self.v1_range_score_tol is not None:
                if float(range_score[:, range_idx].max()) < \
                        self.v1_range_score_tol:
                    max_iters = min(max_iters, 1)
                    self.v1_stats["low_score_capped"] += 1

            slots_this_range = self.v1_slot_idx
            if (self.v1_adaptive_slots
                    and self.v1_slot_sigma_tol is not None
                    and len(self.v1_slot_idx) > 1):
                if float(sigma.max()) < self.v1_slot_sigma_tol:
                    slots_this_range = [self.v1_center_slot]
                    self.v1_stats["adaptive_slot_drops"] += 1

            k_list_sub = self.k_list[slots_this_range]
            center_local = slots_this_range.index(self.v1_center_slot)

            iters_done = 0
            mu_score = None
            for _ in range(max_iters):
                prev_mu = mu
                prev_sigma = sigma
                iters_done += 1
                depth_sample = torch.cat(
                    [mu + sigma * k for k in k_list_sub], 1)
                depth_sample_frustum = self.create_depth_sample_frustum(
                    depth_sample, self.stereo_downsample_factor)
                mu_score = self._v1_generate_cost_volume(
                    sweep_index,
                    stereo_feats_all_sweeps,
                    mats_dict,
                    depth_sample,
                    depth_sample_frustum,
                    sensor2sensor_mats,
                )
                mu_score = mu_score.softmax(1)
                scale_factor = torch.clamp(
                    0.5 / (1e-4 + mu_score[:, center_local:
                                           center_local + 1, ...]),
                    min=0.1,
                    max=10)
                sigma = torch.clamp(sigma * scale_factor, min=0.1, max=10)
                mu = (depth_sample * mu_score).sum(1, keepdim=True)
                del depth_sample
                del depth_sample_frustum

                if self.v1_adaptive_em and iters_done < max_iters:
                    dmu = float((mu - prev_mu).abs().max()) / bin_size
                    dsig = float(((sigma - prev_sigma).abs() /
                                  prev_sigma.clamp_min(1e-6)).max())
                    self.v1_stats["delta_mu"].append(dmu)
                    self.v1_stats["delta_sigma_rel"].append(dsig)
                    if dmu < self.v1_mu_tol and dsig < self.v1_sigma_tol:
                        self.v1_stats["early_exits"] += 1
                        break

            self.v1_stats["range_blocks"] += 1
            self.v1_stats["iters_per_range"].append(iters_done)
            mus.append(mu)
            sigmas.append(sigma)
            range_scores.append(mu_score.sum(1).unsqueeze(1))
            if self.use_mask:
                ratio = self.downsample_factor // self.stereo_downsample_factor
                ds = F.avg_pool2d(mu, ratio, ratio)
                mask_depths.append(ds)
                mask_frustums.append(
                    self.create_depth_sample_frustum(ds,
                                                     self.downsample_factor))

        masks = None
        if self.use_mask:
            masks = self._v2_forward_mask_batched(
                sweep_index, mono_depth_all_sweeps, mats_dict, mask_depths,
                mask_frustums, sensor2sensor_mats)

        for range_idx in range(self.num_ranges):
            range_start = self.range_list[range_idx][0]
            range_length = int(
                (self.range_list[range_idx][1] - range_start)
                // self.d_bound[2])
            start_idx = int((range_start - self.d_bound[0]) // self.d_bound[2])
            if self.use_mask:
                mask_score[:, start_idx:range_length +
                           start_idx, ..., ] += masks[range_idx]
            mu = mus[range_idx]
            sigma = torch.clamp(sigmas[range_idx], self.min_sigma)
            mu_repeated = mu.repeat(1, range_length, 1, 1)
            eps = 1e-6
            depth_score_single_range = (-1 / 2 * (
                (d_coords[:, start_idx:range_length + start_idx, ..., ] -
                 mu_repeated) / torch.sqrt(sigma))**2)
            depth_score_single_range = depth_score_single_range.exp()
            depth_score_single_range = depth_score_single_range / (
                sigma * math.sqrt(2 * math.pi) + eps)
            stereo_depth[:, start_idx:range_length +
                         start_idx, ..., ] = (
                             depth_score_single_range *
                             range_score[:, range_idx:range_idx + 1, ...])
            del depth_score_single_range
            del mu_repeated

        self.v1_stats["score_ranges"] = len(range_scores)
        if self.use_mask:
            return stereo_depth, mask_score
        return stereo_depth

    def _v2_forward_mask_batched(self, sweep_index, mono_depth_all_sweeps,
                                 mats_dict, depths, frustums,
                                 sensor2sensor_mats):
        """4 个 range 沿 batch 维拼接：一次 warp + 一次 mask_net。"""
        n_ranges = len(depths)
        key = mono_depth_all_sweeps[sweep_index]
        cams = key.shape[0]
        key_b = key.unsqueeze(0).expand(n_ranges, -1, -1, -1, -1).reshape(
            n_ranges * cams, *key.shape[1:])
        ds_b = torch.cat(depths, 0)
        fr_b = torch.cat(frustums, 0)
        outs = list()
        for idx in range(len(mono_depth_all_sweeps)):
            if idx == sweep_index:
                continue
            src = mono_depth_all_sweeps[idx]
            src_b = src.unsqueeze(0).expand(n_ranges, -1, -1, -1,
                                            -1).reshape(
                n_ranges * cams, *src.shape[1:])
            warped = self.homo_warping(
                src_b,
                mats_dict['intrin_mats'][:, sweep_index, ...].expand(
                    n_ranges, -1, -1, -1),
                mats_dict['intrin_mats'][:, idx, ...].expand(
                    n_ranges, -1, -1, -1),
                sensor2sensor_mats[idx].expand(n_ranges, -1, -1, -1),
                mats_dict['ida_mats'][:, sweep_index, ...].expand(
                    n_ranges, -1, -1, -1),
                mats_dict['ida_mats'][:, idx, ...].expand(
                    n_ranges, -1, -1, -1),
                ds_b,
                fr_b.type_as(src_b),
            )
            mask = self.mask_net(
                torch.cat([key_b.detach(), warped.mean(2).detach()], 1))
            outs.append(mask)
            self.v2_stats["mask_batch_calls"] += 1
        stacked = torch.stack(outs).mean(0)
        return [stacked[i * cams:(i + 1) * cams] for i in range(n_ranges)]

    def v2_stats_summary(self) -> dict:
        s = self.v2_stats
        warp = s["warp_channels_per_call"]
        ref = s["ref_channels_per_call"]
        return {
            "v2": {
                "corr_channels_requested": s["corr_channels_requested"],
                "corr_channels_used": s["corr_channels_used"],
                "slice_stage": s["slice_stage"],
                "mask_sweeps": s["mask_sweeps"],
                "corr_sweeps": s["corr_sweeps"],
                "mask_stride": s["mask_stride"],
                "mask_batch": s["mask_batch"],
                "mask_batch_calls": s["mask_batch_calls"],
                "cost_volume_calls": s["cost_volume_calls"],
                "sweeps_per_call": {
                    "min": min(s["sweeps_per_call"]) if s["sweeps_per_call"] else None,
                    "max": max(s["sweeps_per_call"]) if s["sweeps_per_call"] else None,
                    "distinct": sorted(set(s["sweeps_per_call"])),
                },
                "warp_channels": {
                    "min": min(warp) if warp else None,
                    "max": max(warp) if warp else None,
                    "distinct": sorted(set(warp)),
                },
                "ref_channels": {
                    "min": min(ref) if ref else None,
                    "max": max(ref) if ref else None,
                    "distinct": sorted(set(ref)),
                },
            }
        }
