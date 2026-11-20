#!/usr/bin/env python3
"""Official BEV image-to-BEV backbone + v8-a4 RoGS task head wrapper.

The continuous-lift backbone (ContinuousLiftFPN) is substituted for the official
BEVStereoLSSFPN before this wrapper is built.  The official two-key trunk/neck
weights are reused via the RoGS head.  Only the elevation head plus the lifted
backbone are trained in geometry-only mode.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Dict

import torch
import torch.nn as nn

from rogs_head import BEVStereoRoGSHead


class BEVStereoRoGSModel(nn.Module):
    def __init__(self, official_model, lift_backbone=None, fine_size: int = 256,
                 texture_channels: int = 64, query_detach: bool = True,
                 geometry_only: bool = False, freeze_official_bn: bool = False,
                 **head_kwargs):
        super().__init__()
        self.geometry_only = bool(geometry_only)
        self.freeze_official_bn = bool(freeze_official_bn)
        self.backbone = lift_backbone if lift_backbone is not None \
            else official_model.backbone
        self._img_feats: Dict[int, torch.Tensor] = {}
        self._hook_trace_enabled = False
        self._hook_trace = []
        self._hooks = []
        img_backbone = self.backbone.img_backbone
        if not self.geometry_only:
            for idx in (0, 1):
                self._hooks.append(img_backbone.register_forward_hook(self._make_hook(idx)))
        else:
            # v8-a15: retain key-frame c2/c3/c4 for the multi-scale FPN
            # candidate condition.  These are already computed by the official
            # image backbone; the hooks only retain the three feature maps.
            for idx in (0, 1, 2):
                self._hooks.append(img_backbone.register_forward_hook(self._make_hook(idx)))
        self.rogs_head = BEVStereoRoGSHead.from_official_head(
            official_model.head,
            fine_size=fine_size,
            texture_channels=texture_channels,
            query_detach=query_detach,
            geometry_only=self.geometry_only,
            **head_kwargs,
        )
        if self.geometry_only:
            # geometry-only: only the elevation head (geom_z) plus the lifted
            # backbone are trained.  texture / query / SH / semantic and the
            # non-elevation Gaussian geometry outputs (xy / opacity / scale /
            # rotation) are frozen and unused by the loss.
            for m in (self.rogs_head.texture_fpn, self.rogs_head.query,
                      self.rogs_head.sh_decoder, self.rogs_head.geom_xy,
                      self.rogs_head.opacity, self.rogs_head.scale,
                      self.rogs_head.rotation, self.rogs_head.semantic):
                for p in m.parameters():
                    p.requires_grad_(False)
        # Full-parameter fine-tune: the lifted backbone (image ResNet50, neck,
        # depth net) is trainable, including the stem conv1/BN that the official
        # model ships with requires_grad=False.
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            feats = output if isinstance(output, (tuple, list)) else (output,)
            if self._hook_trace_enabled:
                self._hook_trace.append(feats[idx])
            # BEVStereo calls img_backbone once per sweep.  The first call is
            # the key frame; later sweep calls must not overwrite the feature
            # used with key-frame mats by the candidate-BEV projection.
            if idx not in self._img_feats:
                self._img_feats[idx] = feats[idx]
        return hook

    def rogs_bn_state(self):
        buffers = dict(self.named_buffers())
        names = []
        for module_name, module in self.rogs_head.named_modules():
            official = (module_name == "trunk" or module_name.startswith("trunk.") or
                        module_name == "neck" or module_name.startswith("neck."))
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and not official:
                for buffer_name, _ in module.named_buffers():
                    names.append(f"rogs_head.{module_name}.{buffer_name}")
        return {name: buffers[name].detach().cpu().clone() for name in names}

    def copy_rogs_bn_state(self, state):
        expected = self.rogs_bn_state()
        if set(state) != set(expected):
            raise RuntimeError("Checkpoint RoGS BN buffers differ from the current head")
        buffers = dict(self.named_buffers())
        with torch.no_grad():
            for name, value in state.items():
                buffers[name].copy_(value.to(buffers[name].device, buffers[name].dtype))

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_official_bn:
            for root in (self.backbone, self.rogs_head.trunk, self.rogs_head.neck):
                for module in root.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
        return self

    def forward(self, imgs, mats_dict, timestamps=None, resolution=None,
                return_depth=False, refiner_target=None, refiner_use_noise=True,
                refiner_outer_loops=0, refiner_outer_seed=0, head_fp32=False,
                refiner_loop_residual="f_minus_c"):
        self._img_feats = {}
        self._hook_trace = []
        depth_bev_target = None
        if return_depth:
            bev, depth_pred = self.backbone(imgs, mats_dict, timestamps, is_return_depth=True)
            depth_bev_target = self.backbone.get_depth_bev_z()
            if refiner_target is None:
                # v8-a17: replace the GT diffusion target with the detached
                # predicted-depth BEV residual in both training and inference.
                refiner_target = depth_bev_target
        else:
            bev = self.backbone(imgs, mats_dict, timestamps)
            depth_pred = None
        if self.geometry_only:
            # v8-a15 multi-scale candidate condition: c2/c3/c4 at strides 4/8/16.
            c2 = [self._img_feats.get(idx) for idx in range(3)]
            if any(feat is None for feat in c2):
                raise RuntimeError("key-frame c2/c3/c4 hooks did not all fire")
            c3 = None
        else:
            c2 = self._img_feats[0]
            c3 = self._img_feats[1]
        if refiner_outer_loops > 0 and refiner_target is None:
            raise ValueError(
                "v8-a17 outer loops require the detached depth-BEV refiner_target; "
                "call with return_depth=True instead of using the noise fallback")
        head_context = (torch.autocast(device_type=bev.device.type, enabled=False)
                        if head_fp32 else nullcontext())
        with head_context:
            if head_fp32:
                bev = bev.float()
                c2 = [feat.float() for feat in c2] if self.geometry_only else c2.float()
                if c3 is not None:
                    c3 = c3.float()
                if refiner_target is not None:
                    refiner_target = tuple(x.float() for x in refiner_target)
            initial, diagnostics = self.rogs_head(
                bev, c2, c3, mats_dict,
                refiner_target=refiner_target,
                refiner_use_noise=refiner_use_noise,
                refiner_outer_loops=refiner_outer_loops,
                refiner_outer_seed=refiner_outer_seed,
                refiner_loop_residual=refiner_loop_residual)
        diagnostics["official_bev_shape"] = tuple(bev.shape)
        if not self.geometry_only:
            diagnostics["c2_shape"] = tuple(c2.shape)
            diagnostics["c3_shape"] = tuple(c3.shape)
        if depth_pred is not None:
            diagnostics["depth_pred"] = depth_pred
        if depth_bev_target is not None:
            diagnostics["depth_bev_z"] = depth_bev_target[0]
            diagnostics["depth_bev_valid"] = depth_bev_target[1]
            diagnostics["refiner_target_source"] = "detached_expected_depth_bev_t0"
        return initial, diagnostics
