#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5 evaluation: mono-only lift + optional fine-tuned depth branch.

Runs the unmodified v2 entry (protocol identical to v2.2b), and after the
frozen v8a17 checkpoint is loaded:

  * always installs the mono-only stereo bypass used by the v5 fine-tune;
  * --v5-depth-ckpt loads {"depth_net": state_dict} from the v5 fine-tune.

The bypass is the same one measured in phase 2: depth_score = softmax(mono).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MV = os.path.dirname(HERE)
for _p in (MV + "/v2", MV + "/v1", HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_v2  # noqa: E402


def mono_stereo(ref_idx, feats, monos, mats, s2s, mu, sigma, rs, dfs,
                *a, **k):
    m = monos[ref_idx]
    return (torch.zeros_like(m, dtype=torch.float32),
            torch.ones((), device=m.device, dtype=torch.float32))


def make_mono_stereo(temp=1.0):
    """Zero stereo residual; temp != 1 rescales the mono logits.

    depth_score == (mono + stereo) .softmax(1), so returning
    mono*(1/temp - 1) yields softmax(mono / temp) exactly.
    """
    if temp == 1.0:
        return mono_stereo

    def _f(ref_idx, feats, monos, mats, s2s, mu, sigma, rs, dfs, *a, **k):
        m = monos[ref_idx]
        resid = m.to(torch.float32) * (1.0 / temp - 1.0)
        return resid, torch.ones((), device=m.device, dtype=torch.float32)
    return _f


def make_bin_bias(bias):
    """depth_score = softmax(mono + bias): return the bias as the residual."""
    b = bias.view(1, -1, 1, 1)
    cached = {"b": None}

    def _f(ref_idx, feats, monos, mats, s2s, mu, sigma, rs, dfs, *a, **k):
        m = monos[ref_idx]
        # cache the device copy: a per-call host->device copy is a sync point
        # and would invalidate CUDA graph capture (v7.3).
        if cached["b"] is None or cached["b"].device != m.device:
            cached["b"] = b.to(m.device, torch.float32)
        return (cached["b"].expand_as(m).contiguous(),
                torch.ones((), device=m.device, dtype=torch.float32))
    return _f


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--v5-depth-ckpt", default="")
    ap.add_argument("--v5-bin-bias", default="",
                    help="v5 bias-only ckpt: 112 logit offsets")
    ap.add_argument("--v5-temp", type=float, default=1.0,
                    help="mono logit temperature (1.0 = plain mono-only)")
    mine, rest = ap.parse_known_args()

    ck = None
    if mine.v5_depth_ckpt:
        ck = torch.load(mine.v5_depth_ckpt, map_location="cpu")
        assert ck.get("mono_only", True), "checkpoint is not a mono-only ckpt"
        print("V5_EVAL depth ckpt epoch=%s scene=%s loss=%s"
              % (ck.get("epoch"), ck.get("scene"), ck.get("loss")), flush=True)

    bias = None
    if mine.v5_bin_bias:
        bck = torch.load(mine.v5_bin_bias, map_location="cpu")
        bias = bck["bin_bias"]
        print("V5_EVAL bin bias epoch=%s loss=%s norm=%.4f"
              % (bck.get("epoch"), bck.get("loss"), float(bias.norm())),
              flush=True)

    state = {"done": False}
    orig_lsd = torch.nn.Module.load_state_dict

    def patched_lsd(self, state_dict, *a, **k):
        res = orig_lsd(self, state_dict, *a, **k)
        if not state["done"] and isinstance(self, eval_v2.BEVStereoRoGSModel):
            state["done"] = True
            lift = self.backbone
            lift.depth_downsample_net = torch.nn.Identity()
            lift._forward_stereo = (make_bin_bias(bias) if bias is not None
                                    else make_mono_stereo(mine.v5_temp))
            if mine.v5_temp != 1.0:
                print("V5_EVAL mono temperature %.3f" % mine.v5_temp,
                      flush=True)
            pool_mode = os.environ.get("V5_B54_POOL", "")
            if pool_mode:
                import b54_bevpool
                b54_bevpool.install(lift, pool_mode)
                print("V5_EVAL bev pool mode=%s" % pool_mode, flush=True)
            if os.environ.get("V7_IMG_PRECACHE"):
                import i76_img_precache
                i76_img_precache.install(
                    cache_dir=os.environ.get("V8_IP_CACHE_DIR") or None,
                    verify=bool(os.environ.get("V7_IP_VERIFY")),
                    mode=os.environ["V7_IMG_PRECACHE"])
            if os.environ.get("V7_SPLAT_FUSED"):
                import c72_splat_fused
                c72_splat_fused.install(lift)
                print("V5_EVAL splat fused source layout (v7.2)", flush=True)
            if os.environ.get("V7_TRT_BACKBONE"):
                import d74_trt_backbone
                if (os.environ.get("V8_TRT_FP16", "0") == "1"
                        and not os.environ.get("V7_TRT_ENGINE")):
                    os.environ["V7_TRT_ENGINE"] = os.path.join(
                        HERE, "..", "v8", "engine",
                        "resnet50_neck_fp16.engine")
                    print("V5_EVAL ResNet50+neck engine FP16 (v8.5)", flush=True)
                d74_trt_backbone.install(lift, self)
                print("V5_EVAL ResNet50+neck on TensorRT (v7.4)", flush=True)
            if os.environ.get("V5_SKIP_STEREO_HEAD"):
                import b53_depth_skip
                b53_depth_skip.install(lift.depth_net, "skip")
                print("V5_EVAL depth head: skip mu/sigma/range", flush=True)
            if os.environ.get("V8_DN_TRT"):
                import j81_depthnet_trt
                j81_depthnet_trt.install(lift)
                print("V5_EVAL depth_net pre-DCN trunk on TensorRT (v8.1)",
                      flush=True)
            if os.environ.get("V8_DN_STUB"):
                import j81_depthnet_probe
                j81_depthnet_probe.install_stub(lift.depth_net)
                print("V5_EVAL depth_net stubbed (v8.1 share A/B)", flush=True)
            elif os.environ.get("V8_DN_PROBE"):
                import j81_depthnet_probe
                j81_depthnet_probe.install(lift.depth_net)
                print("V5_EVAL depth_net per-block probe (v8.1)", flush=True)
            if ck is not None:
                miss, unexp = lift.depth_net.load_state_dict(
                    ck["depth_net"], strict=False)
                if miss or unexp:
                    raise RuntimeError("depth ckpt mismatch miss=%s unexp=%s"
                                       % (miss[:5], unexp[:5]))
                print("V5_EVAL depth branch installed", flush=True)
            cand3_ckpt = os.environ.get("V5_CAND3", "")
            if cand3_ckpt:
                # v5.5b: 3-candidate refiner distilled from the 6-candidate head.
                cand3 = torch.load(cand3_ckpt, map_location="cpu")
                refiner = self.rogs_head.refiner
                old = refiner.cand_cond
                dev = next(refiner.parameters()).device
                cand_ch = old.feat[0].in_channels - 3
                hidden = old.out.out_channels
                new_cond = type(old)(cand_ch=cand_ch, hidden=hidden,
                                     num_candidates=int(cand3["num_candidates"]))
                new_cond.load_state_dict(cand3["cand_cond"])
                new_cond = new_cond.to(dev).eval()
                for p in new_cond.parameters():
                    p.requires_grad_(False)
                refiner.num_candidates = int(cand3["num_candidates"])
                refiner.cand_cond = new_cond
                print("V5_EVAL cand3 student epoch=%s loss=%s cands=%d"
                      % (cand3.get("epoch"), cand3.get("loss"),
                         refiner.num_candidates), flush=True)
        return res

    torch.nn.Module.load_state_dict = patched_lsd
    if os.environ.get("V5_HEAD_GEO"):
        import b55_head_geo
        b55_head_geo.install_class()
        print("V5_EVAL head geometry hoisted (v5.5)", flush=True)
    if os.environ.get("V5_HEAD_CAMBATCH"):
        if not os.environ.get("V5_HEAD_GEO"):
            raise SystemExit("V5_HEAD_CAMBATCH requires V5_HEAD_GEO=1")
        import b57_head_cambatch
        b57_head_cambatch.install()
        print("V5_EVAL head sample: camera-batched grid_sample (v5.7)",
              flush=True)
    if os.environ.get("V7_CUDA_GRAPH"):
        import c73_cuda_graph
        c73_cuda_graph.install()
        print("V5_EVAL cuda graph: whole learned forward captured (v7.3)",
              flush=True)
    if str(os.environ.get("V7_IMG_PRECACHE", "")).lower() in ("u8", "uint8"):
        import i76_img_precache
        i76_img_precache.install_gpu_norm()
        print("V5_EVAL uint8 image cache + gpu normalise (v8.3)", flush=True)
    if os.environ.get("V5_HEAD_PROFILE"):
        import b55_head_profile
        b55_head_profile.install_class()
        print("V5_EVAL head profiling on", flush=True)
    sys.argv = [os.path.join(MV, "v2", "eval_v2.py")] + rest
    eval_v2.main()


if __name__ == "__main__":
    main()
