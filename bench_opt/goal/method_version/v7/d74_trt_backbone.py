#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v7.4 (D): TensorRT engine for ResNet50 + SECONDFPN neck only.

Scope, per the v7 plan: the image backbone stack (``img_backbone`` +
``img_neck``) runs in one TensorRT engine, everything else stays in PyTorch.
The earlier attempts failed for two reasons that are addressed here:

  * **precision** -- the previous "FP32" engine left ``BuilderFlag.TF32`` at its
    TRT default, so Ampere convolutions ran in TF32 and the neck differed from
    eager by 3.8e-1.  This module loads an engine built with TF32 *cleared*
    (``engine/resnet50_neck_fp32_notf32.engine``), which is real FP32.
  * **copies / syncs** -- the engine is created once with fixed shapes, all
    bindings are preallocated device buffers, execution is enqueued on the
    caller's current stream (no ``synchronize``, no ``.cpu()``), and the output
    buffers are double-buffered so nothing has to be cloned: the chain keeps
    both sweeps' features alive at the same time.

The wrapper replaces ``ContinuousLiftFPN.get_cam_feats`` (the only place the
vendor code touches ``img_backbone``/``img_neck``).  The multi-scale features
``c2/c3/c4`` normally reach the RoGS head through forward hooks on
``img_backbone``; since that module is bypassed they are filled into
``model._img_feats`` from the engine outputs instead.

Enable with ``V7_TRT_BACKBONE=1``; ``V7_TRT_VERIFY=1`` prints the first-call
deviation against the eager ``get_cam_feats``.
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRED = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_DEFAULT_ENGINE = os.path.join(_HERE, "engine", "resnet50_neck_fp32_notf32.engine")
_OUT_NAMES = ("neck_feat", "c2", "c3", "c4")

_STATE = {"installed": False, "verified": False, "calls": 0}


class TRTBackbone:
    """Fixed-shape, double-buffered, zero-copy TensorRT execution wrapper."""

    def __init__(self, engine_path, log=False):
        trt_root = os.path.join(_BRED, "bench_opt", "trt_target")
        if os.path.isdir(trt_root) and trt_root not in sys.path:
            sys.path.insert(0, trt_root)
        import tensorrt as trt

        with open(engine_path, "rb") as f:
            ser = f.read()
        logger = trt.Logger(trt.Logger.INFO if log else trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(ser)
        if self.engine is None:
            raise RuntimeError("failed to deserialize %s" % engine_path)
        self.ctx = self.engine.create_execution_context()

        self.in_name = None
        shapes, dtypes = {}, {}
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shapes[name] = tuple(int(d) for d in self.engine.get_binding_shape(i))
            dtypes[name] = self.engine.get_binding_dtype(i)
            if self.engine.binding_is_input(i):
                if self.in_name is not None:
                    raise RuntimeError("engine has more than one input")
                self.in_name = name
        if dtypes[self.in_name] != trt.DataType.FLOAT:
            raise RuntimeError("expected an FP32 engine, got %s"
                               % dtypes[self.in_name])
        self.in_shape = shapes[self.in_name]
        self.inp = torch.empty(self.in_shape, dtype=torch.float32, device="cuda")

        # Two output sets: the vendor chain keeps sweep-0 features alive while
        # sweep 1 runs, so a single set would alias.
        self.sets = []
        for _ in range(2):
            s = {n: torch.empty(shapes[n], dtype=torch.float32, device="cuda")
                 for n in _OUT_NAMES}
            ptrs = [0] * self.engine.num_bindings
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                ptrs[i] = (self.inp if name == self.in_name else s[name]).data_ptr()
            self.sets.append((s, ptrs))
        self.slot = 0
        print("V7_TRT engine: in=%s out=%s"
              % (self.in_shape, {n: shapes[n] for n in _OUT_NAMES}), flush=True)

    def stream(self):
        return torch.cuda.current_stream().cuda_stream

    def run(self, x):
        """Enqueue one batch; returns the previous buffer set (no clones)."""
        self.inp.copy_(x)
        s, ptrs = self.sets[self.slot]
        self.slot ^= 1
        self.ctx.execute_async_v2(ptrs, self.stream())
        return s


def install(lift, model=None, engine_path=None, verify=None):
    """Replace ``lift.get_cam_feats`` with the TensorRT engine."""
    if _STATE["installed"]:
        return lift.get_cam_feats
    engine_path = engine_path or os.environ.get("V7_TRT_ENGINE", _DEFAULT_ENGINE)
    if verify is None:
        verify = os.environ.get("V7_TRT_VERIFY", "0") == "1"
    runner = TRTBackbone(engine_path)
    orig = lift.get_cam_feats

    def get_cam_feats(imgs):
        B, S, Nc, C, H, W = imgs.shape
        if B * S * Nc != runner.in_shape[0]:
            raise RuntimeError("V7_TRT: engine batch %d != %d*%d*%d"
                               % (runner.in_shape[0], B, S, Nc))
        outs = runner.run(imgs.reshape(B * S * Nc, C, H, W))
        if verify and not _STATE["verified"]:
            _STATE["verified"] = True
            ref_feat, ref_ctx = orig(imgs.contiguous())
            for tag, a, b in (("neck", ref_feat, outs["neck_feat"].reshape(
                    B, S, Nc, *outs["neck_feat"].shape[1:])),
                    ("c2", ref_ctx, outs["c2"])):
                print("V7_TRT verify %s max=%.3e mean=%.3e"
                      % (tag, float((a - b).abs().max()),
                         float((a - b).abs().mean())), flush=True)
        neck = outs["neck_feat"].reshape(B, S, Nc, *outs["neck_feat"].shape[1:])
        if model is not None and 0 not in model._img_feats:
            for idx, name in enumerate(("c2", "c3", "c4")):
                model._img_feats[idx] = outs[name].detach()
        _STATE["calls"] += 1
        return neck, outs["c2"]

    lift.get_cam_feats = get_cam_feats
    _STATE["installed"] = True
    _STATE["orig"] = orig
    return orig
