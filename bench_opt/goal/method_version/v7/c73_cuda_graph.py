#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v7.3: one CUDA Graph for the whole learned forward.

The old v5.8 experiment graphed only ResNet on a fixed batch-6 buffer, which
could not remove the launch gaps of neck/DepthNet/head and added copies.  This
version captures the entire per-frame computation that is static:

    imgs -> backbone (ResNet + neck + DepthNet + lift/splat/bev pool)
         -> RoGS head (encode_bev, geometry heads, refiner)
         -> (initial, diag) with z / z_coarse / z_mid

Everything downstream (XY lattice, world transform, RoadGT matching) stays
eager: the matcher already synchronises every frame, uses CPU numpy tables and
must never be captured.

Constraints handled here (PyTorch 1.13 CUDA graph semantics):
  * static device buffers for imgs / every mats tensor / timestamps, filled
    with ``copy_`` before each replay (no data-dependent shapes);
  * warm-up on a side stream (>=3 iterations) before capture;
  * capture-time code is allocation-free on the host and contains no
    ``.item()`` / ``.cpu()`` / sync: verified by reading the vendor backbone,
    b53/b54/b55 patches and rogs_refiner eval branch (deterministic, no noise);
  * ``refiner_outer_seeds`` is constant: outer loops (>0) are rejected because
    they fork RNG per frame;
  * replay runs on the caller's current stream, so the eager consumer code
    keeps its ordering.

Enable with ``V7_CUDA_GRAPH=1``; the module is imported by ``eval_v5.py``.
"""
from __future__ import annotations

import gc
import os

import torch

_STATE = {"installed": False, "calls": {}}   # calls: module -> _GraphCall

# 4x4 matrix inverses go through cuSOLVER, which is not capturable (host-side
# info read).  They are cheap: invert on the caller's stream each frame into a
# static buffer and let the captured graph read the precomputed result.
_INV_KEYS = (("sensor2sensor_mats", "sensor2sensor_inv"),
             ("intrin_mats", "intrin_inv"),
             ("ida_mats", "ida_inv"),
             ("sensor2ego_mats", "sensor2ego_inv"),
             ("bda_mat", "bda_inv"))


def _allowed_attrs():
    return ("imgs", "mats", "timestamps", "kwargs", "outputs")


class _GraphCall:
    """Capture-once, replay-per-frame wrapper around one model instance."""

    def __init__(self, model, warmup=3, verify=False, fwd=None):
        self.model = model
        self.fwd = fwd
        self.warmup = max(1, int(warmup))
        self.verify = verify
        self.graph = None
        self.outputs = None
        self.verified = False
        self.replays = 0

    # -- static buffers -----------------------------------------------------
    def _bind(self, imgs, mats, timestamps, kwargs):
        self.imgs = torch.empty_like(imgs)
        self.mats = {k: torch.empty_like(v) for k, v in mats.items()}
        self.inv_pairs = [(s, d) for s, d in _INV_KEYS if s in self.mats]
        for _s, d in self.inv_pairs:
            self.mats[d] = torch.empty_like(self.mats[_s])
        self.timestamps = torch.empty_like(timestamps)
        self.kwargs = dict(kwargs)
        self.dtype = imgs.dtype

    def _fill(self, imgs, mats, timestamps):
        self.imgs.copy_(imgs)
        for k, buf in self.mats.items():
            if k not in dict(self.inv_pairs).values():
                buf.copy_(mats[k])
        for src, dst in self.inv_pairs:
            torch.linalg.inv(self.mats[src], out=self.mats[dst])
        self.timestamps.copy_(timestamps)

    def _run(self):
        return self.fwd(self.model, self.imgs, self.mats, self.timestamps,
                        **self.kwargs)

    # -- capture ------------------------------------------------------------
    def _capture(self):
        side = torch.cuda.Stream(device=self.imgs.device)
        side.wait_stream(torch.cuda.current_stream(self.imgs.device))
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                self._run()
        torch.cuda.current_stream(self.imgs.device).wait_stream(side)
        torch.cuda.synchronize(self.imgs.device)

        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(self.imgs.device)
        gc.collect()
        torch.cuda.empty_cache()
        with torch.cuda.stream(side):
            self.graph.capture_begin()
            try:
                self.outputs = self._run()
            finally:
                self.graph.capture_end()
        self.graph.replay()
        print("V7_CG captured: imgs=%s mats=%s warmup=%d"
              % (tuple(self.imgs.shape),
                 {k: tuple(v.shape) for k, v in self.mats.items()},
                 self.warmup), flush=True)

    # -- verification -------------------------------------------------------
    def _verify_against_eager(self, imgs, mats, timestamps):
        mats_v = {k: v.clone() for k, v in mats.items()}
        for src, dst in self.inv_pairs:
            mats_v[dst] = torch.linalg.inv(mats_v[src])
        ref = self.fwd(self.model, imgs.clone(), mats_v,
                       timestamps.clone(), **self.kwargs)
        torch.cuda.synchronize(self.imgs.device)
        worst = []
        for tag, (a, b) in (("initial", (ref[0], self.outputs[0])),
                            ("diag", (ref[1], self.outputs[1]))):
            for k, va in a.items():
                vb = b.get(k)
                if torch.is_tensor(va) and torch.is_tensor(vb) \
                        and va.is_floating_point() and va.shape == vb.shape:
                    worst.append((float((va - vb).abs().max().item()), tag, k))
        worst.sort(reverse=True)
        print("V7_CG verify vs eager: %s"
              % ", ".join("%s.%s=%.3e" % (t, k, d) for d, t, k in worst[:8]),
              flush=True)
        self.verified = True

    # -- public -------------------------------------------------------------
    def __call__(self, imgs, mats, timestamps, **kwargs):
        if self.graph is None:
            loops = int(kwargs.get("refiner_outer_loops", 0))
            if loops > 0:
                raise RuntimeError(
                    "V7_CUDA_GRAPH does not support refiner_outer_loops>0 "
                    "(per-frame RNG fork cannot be captured)")
            self._bind(imgs, mats, timestamps, kwargs)
            self._fill(imgs, mats, timestamps)
            self._capture()
            if self.verify:
                self._verify_against_eager(imgs, mats, timestamps)
            self.replays = 1
            return self.outputs
        if imgs.shape != self.imgs.shape:
            raise RuntimeError("V7_CUDA_GRAPH static shape mismatch: %s vs %s"
                               % (tuple(imgs.shape), tuple(self.imgs.shape)))
        self._fill(imgs, mats, timestamps)
        self.graph.replay()
        self.replays += 1
        return self.outputs


def install(warmup=None, verify=None):
    """Patch ``BEVStereoRoGSModel.forward`` with the graphed version."""
    if _STATE["installed"]:
        return _STATE
    import bevstereo_rogs_model

    if warmup is None:
        warmup = int(os.environ.get("V7_CG_WARMUP", "3"))
    if verify is None:
        verify = os.environ.get("V7_CG_VERIFY", "0") == "1"

    orig = bevstereo_rogs_model.BEVStereoRoGSModel.forward

    def forward(self, imgs, mats_dict, timestamps=None, **kwargs):
        key = id(self)
        call = _STATE["calls"].get(key)
        if call is None or call.model is not self:
            call = _GraphCall(self, warmup=warmup, verify=verify, fwd=orig)
            _STATE["calls"][key] = call
        return call(imgs, mats_dict, timestamps, **kwargs)

    bevstereo_rogs_model.BEVStereoRoGSModel.forward = forward
    _STATE["installed"] = True
    _STATE["orig"] = orig
    return _STATE


def summary():
    return {k: {"replays": v.replays} for k, v in _STATE["calls"].items()}
