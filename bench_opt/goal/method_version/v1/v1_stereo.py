#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v1: 推理侧调度改动 —— 深度候选槽位裁剪 + EM 早退。

本文件是 method_version/v1 的**新增代码**，不修改原始 BReD 代码树。
通过 mixin 继承挂接到 ``ContinuousLiftFPN``（其原始实现保持只读）。

语义边界：
  * 只改 EM 循环的调度：保留哪几个高斯候选槽位、跑几轮 EM；
  * 保留槽位上的数学与 vendor 完全一致：双线性 grid_sample、zeros padding、
    分组 channel 乘积/均值、1x1x1 pointwise similarity_net、槽位 softmax、
    Gaussian depth score、mask 路径均沿用原始算子与权重；
  * 不改深度 bin 总数 D 的定义，不改采样/bilinear/相似度数学，不改 D/EM 结构、
    mu/sigma 网络、SimNet、相机几何、mono/stereo 合成与任何权重值。

档位：
  * v1.0 parity : slots=[0,1,2], em_iters=3, 所有自适应开关关闭；
  * v1.1        : 范围级 EM 早退（mu/sigma 相对变化低于阈值即 break；
                  低 range_score 的 range 最多 1 轮）；
  * v1.2        : 在 v1.1 基础上，sigma 足够窄的 range 关闭外层槽位（3→2/1）；
  * v1.3        : tile 级门控，单独实现，不在本文件内。
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F


class V1StereoMixin:
    """挂接在 ContinuousLiftFPN 上的 v1 stereo 调度 mixin。"""

    def v1_init(
        self,
        slot_idx: List[int],
        em_iters: int,
        *,
        adaptive_em: bool = False,
        mu_tol: float = 0.02,
        sigma_tol: float = 0.05,
        range_score_tol: Optional[float] = None,
        adaptive_slots: bool = False,
        slot_sigma_tol: Optional[float] = None,
    ) -> None:
        self.v1_slot_idx = list(slot_idx)
        self.v1_em_iters = int(em_iters)
        self.v1_center_slot = self.num_samples // 2
        self.v1_center_local = self.v1_slot_idx.index(self.v1_center_slot)
        self.v1_adaptive_em = bool(adaptive_em)
        self.v1_mu_tol = float(mu_tol)
        self.v1_sigma_tol = float(sigma_tol)
        self.v1_range_score_tol = range_score_tol
        self.v1_adaptive_slots = bool(adaptive_slots)
        self.v1_slot_sigma_tol = slot_sigma_tol
        # 运行统计（Python 计数器；v1.0 不引入额外 GPU 同步）
        self.v1_stats = {
            "cost_volume_calls": 0,
            "range_blocks": 0,
            "slots_per_call": [],
            "iters_per_range": [],
            "early_exits": 0,
            "low_score_capped": 0,
            "adaptive_slot_drops": 0,
            "delta_mu": [],
            "delta_sigma_rel": [],
        }

    # -- 复制自 vendor 的成本体积，唯一差异：槽位数从 depth_sample.shape[1] 读取，
    #    不再硬编码 self.num_samples；其余 warp/分组/乘积/均值/simnet 调用链不变。
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
        depth_score_all_sweeps = list()
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
                batch_size, self.num_groups, num_channels // self.num_groups,
                d_slots, height, width)
            ref_stereo_feat = stereo_feats_all_sweeps[sweep_index].reshape(
                batch_size, self.num_groups, num_channels // self.num_groups,
                height, width)
            feat_cost = torch.mean(
                (ref_stereo_feat.unsqueeze(3) * warped_stereo_fea), axis=2)
            depth_score = self.similarity_net(feat_cost).squeeze(1)
            depth_score_all_sweeps.append(depth_score)
        self.v1_stats["cost_volume_calls"] += 1
        self.v1_stats["slots_per_call"].append(int(d_slots))
        return torch.stack(depth_score_all_sweeps).mean(0)

    # -- 复制自 vendor ``_forward_stereo``，差异仅在调度：
    #    1) 候选槽位按 v1_slot_idx 子集构造（v1.2 可按 range 自适应收缩）；
    #    2) EM 迭代次数由 v1_em_iters 上限控制（v1.1 可提前 break）；
    #    3) 中心槽位索引映射到子集内局部索引；
    #    4) 统计实际保留槽位数、实际 EM 轮数与跳过量。
    #    其余 range 映射、mask、Gaussian 打分、累加写回逻辑逐行保留。
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
        batch_size_with_cams, _, feat_height, feat_width = \
            stereo_feats_all_sweeps[0].shape
        device = stereo_feats_all_sweeps[0].device
        d_coords = torch.arange(*self.d_bound,
                                dtype=torch.float,
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
        score_all_ranges = list()
        range_score = range_score_all_sweeps[sweep_index].softmax(1)
        bin_size = float(self.d_bound[2])

        for range_idx in range(self.num_ranges):
            range_start = self.range_list[range_idx][0]
            mu_all_sweeps_single_range = [
                mu[:, range_idx:range_idx + 1, ...].sigmoid() *
                (self.range_list[range_idx][1] - self.range_list[range_idx][0])
                + range_start for mu in mu_all_sweeps
            ]
            sigma_all_sweeps_single_range = [
                sigma[:, range_idx:range_idx + 1, ...]
                for sigma in sigma_all_sweeps
            ]
            batch_size_with_cams, _, feat_height, feat_width = \
                stereo_feats_all_sweeps[0].shape
            mu = mu_all_sweeps_single_range[sweep_index]
            sigma = sigma_all_sweeps_single_range[sweep_index]

            # ---- v1.1/v1.2：范围级调度决策（均为整 range 决策，避免稠密逐像素空转）----
            max_iters = self.v1_em_iters
            if self.v1_range_score_tol is not None:
                score_max = float(range_score[:, range_idx].max())
                if score_max < self.v1_range_score_tol:
                    max_iters = min(max_iters, 1)
                    self.v1_stats["low_score_capped"] += 1

            slots_this_range = self.v1_slot_idx
            if (self.v1_adaptive_slots
                    and self.v1_slot_sigma_tol is not None
                    and len(self.v1_slot_idx) > 1):
                sigma_max = float(sigma.max())
                if sigma_max < self.v1_slot_sigma_tol:
                    slots_this_range = [self.v1_center_slot]
                    self.v1_stats["adaptive_slot_drops"] += 1

            k_list_sub = self.k_list[slots_this_range]
            center_local = slots_this_range.index(self.v1_center_slot)

            iters_done = 0
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

            range_length = int(
                (self.range_list[range_idx][1] - self.range_list[range_idx][0])
                // self.d_bound[2])
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
            mu_repeated = mu.repeat(1, range_length, 1, 1)
            eps = 1e-6
            depth_score_single_range = (-1 / 2 * (
                (d_coords[:,
                          int((range_start - self.d_bound[0]) //
                              self.d_bound[2]):range_length + int(
                                  (range_start - self.d_bound[0]) //
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

        self.v1_stats["score_ranges"] = len(score_all_ranges)
        if self.use_mask:
            return stereo_depth, mask_score
        else:
            return stereo_depth

    def v1_stats_summary(self) -> dict:
        s = self.v1_stats
        slots = s["slots_per_call"]
        iters = s["iters_per_range"]

        def _pct(xs, q):
            if not xs:
                return None
            ys = sorted(xs)
            return ys[min(len(ys) - 1, int(q * (len(ys) - 1)))]

        def _agg(xs):
            return {
                "n": len(xs),
                "p50": _pct(xs, 0.50),
                "p90": _pct(xs, 0.90),
                "max": max(xs) if xs else None,
            }

        return {
            "slot_idx": self.v1_slot_idx,
            "em_iters_cap": self.v1_em_iters,
            "adaptive_em": self.v1_adaptive_em,
            "mu_tol": self.v1_mu_tol,
            "sigma_tol": self.v1_sigma_tol,
            "range_score_tol": self.v1_range_score_tol,
            "adaptive_slots": self.v1_adaptive_slots,
            "slot_sigma_tol": self.v1_slot_sigma_tol,
            "cost_volume_calls": s["cost_volume_calls"],
            "range_blocks": s["range_blocks"],
            "early_exits": s["early_exits"],
            "low_score_capped": s["low_score_capped"],
            "adaptive_slot_drops": s["adaptive_slot_drops"],
            "slots_per_call": {
                "min": min(slots) if slots else None,
                "max": max(slots) if slots else None,
                "distinct": sorted(set(slots)),
            },
            "iters_per_range": {
                "min": min(iters) if iters else None,
                "max": max(iters) if iters else None,
                "distinct": sorted(set(iters)),
                "histogram": {str(k): iters.count(k)
                              for k in sorted(set(iters))},
            },
            "delta_mu_bins": _agg(s["delta_mu"]),
            "delta_sigma_rel": _agg(s["delta_sigma_rel"]),
        }
