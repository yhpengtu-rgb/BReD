#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5.3: skip the stereo-only mu/sigma/range branch of DepthNet.

In v5 the stereo cost volume is bypassed (``_forward_stereo`` returns a zero
residual), so ``mu`` / ``sigma`` / ``range_score`` are produced and then never
read.  ``mono_depth`` and ``context`` are the only tensors that reach the
output.  Dropping ``mu_sigma_range_net`` therefore leaves the v5 result
bit-identical while removing 4 x (512->512 3x3) convolutions per sweep.
"""
from __future__ import annotations

import torch


def _forward_no_mu_sigma(self, x, mats_dict, scale_depth_factor=1000.0):
    B, _, H, W = x.shape
    intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
    batch_size = intrins.shape[0]
    num_cams = intrins.shape[2]
    ida = mats_dict['ida_mats'][:, 0:1, ...]
    sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
    bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4, 4).repeat(
        1, 1, num_cams, 1, 1)
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
    empty = x.new_empty(0)
    return x, context, empty, empty, empty, mono_depth


def install(depth_net, mode="skip"):
    if mode in ("", "off", "base"):
        return None
    if mode != "skip":
        raise ValueError("unknown v5.3 depth mode: %r" % (mode,))
    prev = depth_net.forward
    depth_net.forward = _forward_no_mu_sigma.__get__(depth_net, type(depth_net))
    return prev
