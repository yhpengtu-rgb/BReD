#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v8.1 (J): DepthNet pre-DCN trunk on a fixed-shape FP32 TensorRT engine.

Boundary chosen from the v8.1 per-block probe (2026-11-19, 41.3 ms/frame):

    engine : reduce_conv -> context{mlp, se, conv}
                        -> depth{mlp, se} -> BasicBlock -> BasicBlock -> ASPP
    eager  : DCN (depth_feat_conv[3]) -> mono_depth_net
             (+ mu_sigma_range_net when V5_SKIP_STEREO_HEAD is off)

DCN is only 2.6% of DepthNet's GPU time, so it stays eager: that keeps the
engine free of third-party plugins while covering ~97% of the arithmetic.
Everything the engine touches is static-shape FP32 with TF32 explicitly
cleared, matching the v7.4 ResNet engine precedent.

The engine is exported and built on the first forward from the *live* module,
so it always matches the loaded checkpoint; it is then cached on disk and later
processes only deserialize it.  Enable with ``V8_DN_TRT=1``;
``V8_DN_TRT_VERIFY=1`` prints the first-call deviation against the eager
subgraph, ``V8_DN_TRT_REBUILD=1`` forces a rebuild,
``V8_DN_TRT_ENGINE=<path>`` relocates the cache.
"""
from __future__ import annotations

import atexit
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRED = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_DEFAULT_ENGINE = os.path.join(_HERE, "engine",
                               "depthnet_predcn_fp32_notf32.engine")
_FP16_ENGINE = os.path.join(_HERE, "engine", "depthnet_predcn_fp16.engine")
_OUT_NAMES = ("reduce_feat", "context", "aspp_out")

_STATE = {"installed": False, "runner": None, "verified": False,
          "calls": 0}

# Opt-in in-chain timing (V8_DN_TRT_PROBE=1, graph off): CUDA events around the
# engine call and the eager DCN/mono tail, drained lazily, printed at exit.
_PROBE = {"q": [], "acc": {}, "n": 0}


def _drain_probe(keep=6):
    q = _PROBE["q"]
    n = max(0, len(q) - keep)
    for tag, e0, e1 in q[:n]:
        ms = _PROBE["acc"].get(tag, 0.0) + e0.elapsed_time(e1)
        _PROBE["acc"][tag] = ms
    del q[:n]


def _report_probe():
    if not _PROBE["n"]:
        return
    torch.cuda.synchronize()
    _drain_probe(0)
    n = float(_PROBE["n"])
    for tag, ms in sorted(_PROBE["acc"].items()):
        print("V8.1 DN TRT probe %-7s %6.2f ms/call %6.2f ms/frame"
              % (tag, ms / n, 2.0 * ms / n), flush=True)


class _PreDCN(torch.nn.Module):
    """The exported subgraph; holds references to the real DepthNet children."""

    def __init__(self, dn):
        super().__init__()
        self.bn = dn.bn
        self.reduce_conv = dn.reduce_conv
        self.context_mlp = dn.context_mlp
        self.context_se = dn.context_se
        self.context_conv = dn.context_conv
        self.depth_mlp = dn.depth_mlp
        self.depth_se = dn.depth_se
        self.bb0 = dn.depth_feat_conv[0]
        self.bb1 = dn.depth_feat_conv[1]
        self.aspp = dn.depth_feat_conv[2]

    def forward(self, x, mlp_raw):
        mlp = self.bn(mlp_raw)
        r = self.reduce_conv(x)
        cse = self.context_mlp(mlp)[..., None, None]
        context = self.context_conv(self.context_se(r, cse))
        dse = self.depth_mlp(mlp)[..., None, None]
        d = self.depth_se(r, dse)
        d = self.aspp(self.bb1(self.bb0(d)))
        return r, context, d


def _assemble_mlp(mats_dict):
    """Vendor DepthNet.forward camera-vector assembly, pre-``bn`` (6, 27)."""
    intrins = mats_dict["intrin_mats"][:, 0:1, ..., :3, :3]
    batch_size = intrins.shape[0]
    num_cams = intrins.shape[2]
    ida = mats_dict["ida_mats"][:, 0:1, ...]
    sensor2ego = mats_dict["sensor2ego_mats"][:, 0:1, ..., :3, :]
    bda = mats_dict["bda_mat"].view(batch_size, 1, 1, 4, 4).repeat(
        1, 1, num_cams, 1, 1)
    mlp = torch.cat(
        [torch.stack(
            [intrins[:, 0:1, ..., 0, 0], intrins[:, 0:1, ..., 1, 1],
             intrins[:, 0:1, ..., 0, 2], intrins[:, 0:1, ..., 1, 2],
             ida[:, 0:1, ..., 0, 0], ida[:, 0:1, ..., 0, 1],
             ida[:, 0:1, ..., 0, 3], ida[:, 0:1, ..., 1, 0],
             ida[:, 0:1, ..., 1, 1], ida[:, 0:1, ..., 1, 3],
             bda[:, 0:1, ..., 0, 0], bda[:, 0:1, ..., 0, 1],
             bda[:, 0:1, ..., 1, 0], bda[:, 0:1, ..., 1, 1],
             bda[:, 0:1, ..., 2, 2]], dim=-1),
         sensor2ego.view(batch_size, 1, num_cams, -1)], -1)
    return mlp.reshape(-1, mlp.shape[-1]).contiguous()


def _trt():
    trt_root = os.path.join(_BRED, "bench_opt", "trt_target")
    if os.path.isdir(trt_root) and trt_root not in sys.path:
        sys.path.insert(0, trt_root)
    import tensorrt as trt
    return trt


class DepthNetTRT:
    """Fixed-shape, double-buffered, zero-copy TensorRT execution wrapper."""

    def __init__(self, engine_path, log=False):
        trt = _trt()
        with open(engine_path, "rb") as f:
            ser = f.read()
        logger = trt.Logger(trt.Logger.INFO if log else trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(ser)
        if self.engine is None:
            raise RuntimeError("failed to deserialize %s" % engine_path)
        self.ctx = self.engine.create_execution_context()

        shapes = {}
        self.in_names = []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shapes[name] = tuple(int(d) for d in self.engine.get_binding_shape(i))
            if self.engine.binding_is_input(i):
                self.in_names.append(name)
        self.inp = {n: torch.empty(shapes[n], dtype=torch.float32,
                                   device="cuda") for n in self.in_names}

        # Two output sets: the caller keeps sweep-0 tensors alive while sweep 1
        # runs, so a single set would alias.
        self.sets = []
        for _ in range(2):
            s = {n: torch.empty(shapes[n], dtype=torch.float32, device="cuda")
                 for n in _OUT_NAMES}
            ptrs = [0] * self.engine.num_bindings
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                ptrs[i] = (self.inp[name] if name in self.inp
                           else s[name]).data_ptr()
            self.sets.append((s, ptrs))
        self.slot = 0
        print("V8.1 DN TRT engine: in=%s out=%s"
              % ({n: shapes[n] for n in self.in_names},
                 {n: shapes[n] for n in _OUT_NAMES}), flush=True)

    def run(self, x, mlp_raw):
        self.inp["x"].copy_(x)
        self.inp["mlp_raw"].copy_(mlp_raw)
        s, ptrs = self.sets[self.slot]
        self.slot ^= 1
        self.ctx.execute_async_v2(
            ptrs, torch.cuda.current_stream().cuda_stream)
        return s[_OUT_NAMES[0]], s[_OUT_NAMES[1]], s[_OUT_NAMES[2]]


def _build(module, x, mlp_raw, engine_path, fp16=False):
    """ONNX-export the live subgraph and build TF32-free FP32/FP16 engine."""
    import time

    import onnx
    import tensorrt as trt

    onnx_path = engine_path[:-len(".engine")] + ".onnx"
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            module, (x, mlp_raw), onnx_path,
            input_names=["x", "mlp_raw"], output_names=list(_OUT_NAMES),
            opset_version=11, do_constant_folding=True, verbose=False)
    onnx.checker.check_model(onnx.load(onnx_path))
    export_s = time.time() - t0

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError("V8.1 DN TRT: ONNX parse failed: %s" % (errs,))
    config = builder.create_builder_config()
    config.max_workspace_size = 4 << 30
    for fname in ("INT8", "TF32", "BF16"):
        flag = getattr(trt.BuilderFlag, fname, None)
        if flag is not None:
            config.clear_flag(flag)
    # FP16 I/O stays FLOAT; only the internal kernels are half.
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.clear_flag(trt.BuilderFlag.FP16)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("V8.1 DN TRT: build_serialized_network returned None")
    build_s = time.time() - t0
    with open(engine_path, "wb") as f:
        f.write(bytes(serialized))
    os.remove(onnx_path)
    print("V8.1 DN TRT built %s (export %.1f s, build %.1f s, %.1f MB)"
          % (engine_path, export_s, build_s,
             os.path.getsize(engine_path) / 1e6), flush=True)


def _verify(module, runner, x, mlp_raw):
    with torch.no_grad():
        ref = module(x, mlp_raw)
        outs = runner.run(x, mlp_raw)
        torch.cuda.synchronize()
    for tag, a, b in zip(_OUT_NAMES, ref, outs):
        d = (a - b).abs()
        print("V8.1 DN TRT verify %-11s max=%.3e mean=%.3e"
              % (tag, float(d.max()), float(d.mean())), flush=True)


def install(lift, verbose=True):
    """Replace ``lift.depth_net.forward`` with engine + eager DCN/mono tail."""
    if _STATE["installed"]:
        return _STATE
    dn = lift.depth_net
    fp16 = os.environ.get("V8_DN_TRT_FP16", "0") == "1"
    engine_path = os.environ.get(
        "V8_DN_TRT_ENGINE", _FP16_ENGINE if fp16 else _DEFAULT_ENGINE)
    skip_mu = os.environ.get("V5_SKIP_STEREO_HEAD", "0") == "1"
    verify = os.environ.get("V8_DN_TRT_VERIFY", "0") == "1"
    rebuild = os.environ.get("V8_DN_TRT_REBUILD", "0") == "1"
    num_ranges = dn.num_ranges
    module = _PreDCN(dn).eval()
    orig = dn.forward
    probe = os.environ.get("V8_DN_TRT_PROBE", "0") == "1"
    if probe:
        atexit.register(_report_probe)

    def forward(x, mats_dict, scale_depth_factor=1000.0):
        mlp_raw = _assemble_mlp(mats_dict)
        x = x.contiguous()
        if _STATE["runner"] is None:
            if rebuild or not os.path.isfile(engine_path):
                _build(module, x, mlp_raw, engine_path, fp16=fp16)
            _STATE["runner"] = DepthNetTRT(engine_path)
            if verify and not _STATE["verified"]:
                _STATE["verified"] = True
                _verify(module, _STATE["runner"], x, mlp_raw)
        if probe:
            _drain_probe()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            e0.record()
        r, context, d3 = _STATE["runner"].run(x, mlp_raw)
        if probe:
            e1.record()
        d4 = dn.depth_feat_conv[3](d3)          # DCN stays on the vendor path
        mono_depth = dn.mono_depth_net(d4)
        if probe:
            e2.record()
            _PROBE["q"] += [("engine", e0, e1), ("tail", e1, e2)]
            _PROBE["n"] += 1
        _STATE["calls"] += 1
        if skip_mu:
            empty = r.new_empty(0)
            return r, context, empty, empty, empty, mono_depth
        score = dn.mu_sigma_range_net(d4)
        mu = score[:, 0:num_ranges, ...]
        sigma = score[:, num_ranges:2 * num_ranges, ...]
        range_score = score[:, 2 * num_ranges:3 * num_ranges, ...]
        sigma = F.elu(sigma) + 1.0 + 1e-10
        return r, context, mu, sigma, range_score, mono_depth

    dn.forward = forward
    _STATE["installed"] = True
    _STATE["orig"] = orig
    if verbose:
        print("V8.1 DN TRT installed (pre-DCN trunk in engine, DCN eager)",
              flush=True)
    return _STATE
