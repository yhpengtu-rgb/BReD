#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v8-a4 standalone DDP training (geometry-only elevation + road-only depth).

Locked data / precision policy (unchanged from v8-a3 unless noted):
  * No augmentation: IDA fully OFF (deterministic uniform resize 0.64, no
    crop/flip/rotate/color); BDA forced identity; CBGS OFF (raw scene-0001).
  * Input resolution: 576x1024 (H,W), uniform 0.64 shrink of raw 900x1600;
    576/32=18, 1024/32=32.  Train == eval.
  * BEV lattice: 256x256 @ 0.4 m/cell over +/-51.2 m.  Elevation head is 1:1
    with that lattice (no upsampling).
  * Continuous-depth forward-projection lift (ContinuousLiftFPN) replaces the
    official discrete-bin voxel pooling.  Depth supervision keeps the official
    perspective BCE form, z_cam labels, native x3.0.
  * bf16 autocast around the forward; BN stats / EMA / optimizer master stay
    fp32.  The depth BCE helper itself runs in fp32.
  * Full-parameter fine-tune, no LoRA, freeze_official_bn=false; DDP 2 cards,
    batch 1 per card; 6 epochs; scene-0001 single scene.
"""
from __future__ import annotations

import argparse, json, os, sys, time, threading, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from bevstereo_official import BRED_ROOT, collate_fn as official_collate, load_experiment_class

# Patch the official experiment's input size / BEV grid BEFORE the model is
# built, so the image backbone, depth lifting and IDA all agree on one size.
import bevdepth.exps.nuscenes.base_exp as base_exp  # noqa: E402

FINAL_H, FINAL_W = 576, 1024
RESIZE = 0.64
BEV_CELL = 0.4
FINE_SIZE = 256
X_BOUND = [-51.2, 51.2, BEV_CELL]
Y_BOUND = [-51.2, 51.2, BEV_CELL]

base_exp.final_dim = (FINAL_H, FINAL_W)
base_exp.backbone_conf["final_dim"] = (FINAL_H, FINAL_W)
base_exp.backbone_conf["x_bound"] = X_BOUND
base_exp.backbone_conf["y_bound"] = Y_BOUND
base_exp.ida_aug_conf["final_dim"] = (FINAL_H, FINAL_W)

from continuous_lift_fpn import ContinuousLiftFPN
from bevstereo_rogs_model import BEVStereoRoGSModel
from rogs_fine_target import make_fine_bev_target
from roadgt_provider import RoadGTProvider
from rogs_lidar_loss_v8a9 import compute_v8a9_loss
from rogs_depth_aux import compute_official_depth_aux
from rogs_gradnorm import GradNorm
from nusc_det_dataset_road import NuscDetDatasetRoadV8a4

# bf16 safety patches (same rationale as v8-a3): bilinear interp / grid_sample /
# DCN lack bf16 kernels in this torch/mmcv build; run those in fp32, cast back.
import torch.nn.functional as _F
_orig_interp = _F.interpolate
_orig_grid = _F.grid_sample


def _bf16_safe_interp(input, *args, **kwargs):
    if (input.dtype == torch.bfloat16
            and kwargs.get("mode", "nearest") in ("bilinear", "bicubic", "trilinear")):
        return _orig_interp(input.float(), *args, **kwargs).to(torch.bfloat16)
    return _orig_interp(input, *args, **kwargs)


def _bf16_safe_grid(input, *args, **kwargs):
    if input.dtype == torch.bfloat16 and kwargs.get("mode", "bilinear") == "bilinear":
        return _orig_grid(input.float(), *args, **kwargs).to(torch.bfloat16)
    return _orig_grid(input, *args, **kwargs)


_F.interpolate = _bf16_safe_interp
_F.grid_sample = _bf16_safe_grid

import mmcv.ops.deform_conv as _dc
_orig_dcn = _dc.deform_conv2d


def _bf16_safe_dcn(input, offset, *args, **kwargs):
    if input.dtype == torch.bfloat16:
        return _orig_dcn(input.float(), offset.float(), *args, **kwargs).to(torch.bfloat16)
    return _orig_dcn(input, offset, *args, **kwargs)


_dc.deform_conv2d = _bf16_safe_dcn

# DepthAggregation boundary dtype patch (kept for parity; the continuous lift
# bypasses the 3D aggregation net, but the helper may still be referenced).
import bevdepth.layers.backbones.base_lss_fpn as _blfp
_DA = _blfp.DepthAggregation
_orig_da_fwd = _DA.forward


def _da_fwd(self, x):
    bf = x.dtype == torch.bfloat16
    out = _orig_da_fwd(self, x.float() if bf else x)
    return out.to(x.dtype) if bf else out


_DA.forward = _da_fwd


def build_zero_aug_conf(ida_aug_conf, bda_aug_conf):
    import copy
    ida = copy.deepcopy(ida_aug_conf)
    bda = copy.deepcopy(bda_aug_conf)
    ida["resize_lim"] = (RESIZE, RESIZE)
    ida["final_dim"] = (FINAL_H, FINAL_W)
    ida["rot_lim"] = (0.0, 0.0)
    ida["rand_flip"] = False
    ida["bot_pct_lim"] = (0.0, 0.0)
    bda["rot_lim"] = (0.0, 0.0)
    bda["scale_lim"] = (1.0, 1.0)
    bda["flip_dx_ratio"] = 0.0
    bda["flip_dy_ratio"] = 0.0
    return ida, bda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--info-path", required=True)
    ap.add_argument("--official-checkpoint", required=True)
    ap.add_argument("--init-checkpoint", default="",
                    help="optional v8-a4 last.pt to warm-start coarse head "
                         "(strict=False; refiner keys are freshly initialised)")
    ap.add_argument("--roadgt-cache", required=True)
    ap.add_argument("--scene-name", default="scene-0001")
    ap.add_argument("--allow-frozen-val-scene", action="store_true",
                    help="explicit user opt-in to train on a scene listed as frozen val")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--fine-size", type=int, default=FINE_SIZE)
    ap.add_argument("--z-surround", type=float, default=60.0)
    ap.add_argument("--official-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=2e-4)
    ap.add_argument("--gradnorm-lr", type=float, default=5e-3)
    ap.add_argument("--loss-rate", type=float, default=0.9)
    ap.add_argument("--lambda-c", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop after N steps per rank (0 = full epochs); health gate")
    ap.add_argument("--debug-candidates", action="store_true",
                    help="print first-step per-iteration candidate BEV feature differences")
    ap.add_argument("--debug-hook", action="store_true",
                    help="verify key-frame hook retention and per-camera feature contribution")
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    out_dir = os.path.join(str(BRED_ROOT / "results" / "stage1"), args.run_name)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    Exp = load_experiment_class()
    lightning = Exp(gpus=1, batch_size_per_device=1,
                    data_root=str(BRED_ROOT / "assets" / "datasets" / "nuScenes"),
                    default_root_dir=out_dir)
    _sd = torch.load(args.official_checkpoint, map_location="cpu")["state_dict"]
    _sd.pop("model.backbone.frustum", None)
    lightning.load_state_dict(_sd, strict=False)

    # Swap the discrete-bin stereo backbone for the continuous-lift backbone.
    # The subclass keeps the official module names, so the official checkpoint
    # weights transfer with a strict=False load (frustum buffer is dropped).
    bc = lightning.backbone_conf
    num_sweeps = 1 + len(lightning.key_idxes)
    lift = ContinuousLiftFPN(**bc)
    lift.load_official_backbone(lightning.model.backbone)
    lightning.model.backbone = lift

    # geometry-only model; official BN NOT frozen; only elevation + depth.
    model = BEVStereoRoGSModel(lightning.model, lift_backbone=lift,
                               fine_size=args.fine_size,
                               geometry_only=True,
                               freeze_official_bn=False).to(device)

    # All BatchNorm (our new modules AND the official ResNet/neck/depth-net/head
    # trunk-neck BNs built by mmdet) are converted to SyncBatchNorm before DDP.
    # batch=1 per card makes ordinary BN running stats collapse; cross-card
    # synchronised BN keeps batch statistics meaningful.  BN stays in train
    # (freeze_official_bn=False); BN buffers/EMA/optimizer master stay fp32.
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model._hook_trace_enabled = bool(args.debug_hook and rank == 0)
    # Full-parameter fine-tune: keep every backbone parameter trainable (mmdet
    # ResNet can re-freeze its stem on mode switches).
    for p in model.backbone.parameters():
        p.requires_grad_(True)

    # Warm-start from v8-a4 coarse head (strict=False): refiner keys absent
    # in the checkpoint stay at their zero-init residual head.
    if args.init_checkpoint:
        _init = torch.load(args.init_checkpoint, map_location="cpu")
        _init_sd = _init.get("model", _init)
        _miss, _unexp = model.load_state_dict(_init_sd, strict=False)
        _conf_keys = {
            "rogs_head.refiner.conf.weight",
            "rogs_head.refiner.conf.bias",
        }
        _bad_unexp = [k for k in _unexp if k not in _conf_keys]
        if _bad_unexp:
            raise RuntimeError(f"unexpected init keys: {_bad_unexp[:8]}")
        if rank == 0:
            print(f"[init] loaded warm-start: {len(_miss)} missing, "
                  f"{len([k for k in _unexp if k in _conf_keys])} ignored conf keys",
                  flush=True)

    # Gradient checkpointing is DISABLED by default: GradNorm runs two retained
    # backward passes (autograd.grad retain_graph=True) over the graph plus the
    # final model backward, and a non-reentrant checkpoint segment expects a
    # single backward -- a second retained pass can deadlock inside the checkpoint
    # (observed: hang between the two autograd.grad calls at it3).  Peak
    # activation memory is small (~5.2 GB / 24 GB), so checkpointing is not
    # needed.  Re-enable only with USE_CHECKPOINT=1.
    import torch.utils.checkpoint as cp
    _ib = model.backbone.img_backbone
    _ib_orig_fwd = _ib.forward
    if os.environ.get("USE_CHECKPOINT", "0") == "1":
        _ib.forward = lambda x: cp.checkpoint(_ib_orig_fwd, x, use_reentrant=False)
    else:
        _ib.forward = _ib_orig_fwd

    ida_aug_conf, bda_aug_conf = build_zero_aug_conf(
        lightning.ida_aug_conf, lightning.bda_aug_conf)

    common = dict(ida_aug_conf=ida_aug_conf, bda_aug_conf=bda_aug_conf,
                  data_root=str(BRED_ROOT / "assets" / "datasets" / "nuScenes"),
                  info_paths=args.info_path, is_train=True,
                  return_depth=True,
                  sweep_idxes=lightning.sweep_idxes, key_idxes=lightning.key_idxes,
                  use_fusion=lightning.use_fusion)
    ds = NuscDetDatasetRoadV8a4(**common, classes=lightning.class_names,
                                use_cbgs=False)
    if args.allow_frozen_val_scene:
        import roadgt_provider as _rgp
        _rgp.FROZEN_VAL_SCENES = frozenset(
            s for s in _rgp.FROZEN_VAL_SCENES if s != args.scene_name)
        print(f"[scene] frozen-val guard explicitly lifted for {args.scene_name}",
              flush=True)
    provider = RoadGTProvider(args.roadgt_cache, args.scene_name)

    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=1, sampler=sampler, num_workers=args.workers,
                        collate_fn=lambda s: official_collate(s, is_return_depth=True))

    g, hd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (hd if n.startswith("rogs_head.") else g).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": g, "lr": args.official_lr},
         {"params": hd, "lr": args.head_lr}],
        weight_decay=args.weight_decay)

    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if rank == 0 and args.debug_candidates:
        _probe = {"k": 0, "prev_z": None, "prev_feat": None}
        _cand_forward = ddp.module.rogs_head.refiner.cand_bev.forward
        def _debug_cand_forward(feat, z_candidates, mats_arg):
            out = _cand_forward(feat, z_candidates, mats_arg)
            k = _probe["k"]
            if k < ddp.module.rogs_head.refiner.k_iters:
                zc = z_candidates.detach().float()
                cf = out[0].detach().float()
                radius = 0.5 * (zc.amax(dim=1) - zc.amin(dim=1))
                valid_ratio = float(out[2].detach().float().mean())
                feat_std = float(cf.std())
                print(f"REAL_CAND k={k} z_center_mean={float(zc[:, zc.shape[1]//2].mean()):.6f} "
                      f"radius_mean={float(radius.mean()):.6f} feat_std={feat_std:.8f} "
                      f"valid_ratio={valid_ratio:.6f}", flush=True)
                if valid_ratio <= 0 or feat_std <= 0:
                    raise RuntimeError(f"invalid/zero real candidate BEV feature at k={k}")
                if _probe["prev_z"] is not None:
                    zdiff = (zc - _probe["prev_z"]).abs().mean()
                    fdiff = (cf - _probe["prev_feat"]).abs().mean()
                    print(f"REAL_CAND_DIFF k={k-1}->{k} z_mean_abs={float(zdiff):.8f} "
                          f"feat_mean_abs={float(fdiff):.8f}", flush=True)
                    if float(zdiff) <= 1e-10 or float(fdiff) <= 1e-10:
                        raise RuntimeError(f"candidate/feature did not change at k={k}")
                _probe["prev_z"] = zc.clone()
                _probe["prev_feat"] = cf.clone()
                _probe["k"] = k + 1
                if _probe["k"] == ddp.module.rogs_head.refiner.k_iters:
                    print("REAL_CANDIDATE_MULTIVIEW_TRACE_PASSED", flush=True)
            return out
        ddp.module.rogs_head.refiner.cand_bev.forward = _debug_cand_forward

    # GradNorm task-weight balancer: independent w_elev / w_depth, optimised by
    # a dedicated small optimizer against the shared img_neck reference params.
    gradnorm = GradNorm(n_tasks=2, lr=args.gradnorm_lr).to(device)

    # Per-step watchdog: abort (os._exit) if one step exceeds STEP_TIMEOUT s,
    # so a hung forward/backward cannot occupy the GPUs indefinitely.
    _step = {"done": True, "t0": time.perf_counter()}
    _deadline = float(os.environ.get("STEP_TIMEOUT", "120"))

    def _watchdog():
        while True:
            time.sleep(5)
            if (not _step["done"]
                    and time.perf_counter() - _step["t0"] > _deadline):
                print(f"[watchdog] step > {_deadline:.0f}s, aborting rank {rank}",
                      flush=True)
                # Dump EVERY thread's current frame (main thread included),
                # not just this watchdog thread.
                for tid, frm in sys._current_frames().items():
                    print(f"--- thread {tid} ---", flush=True)
                    traceback.print_stack(frm)
                sys.stdout.flush()
                os._exit(2)

    threading.Thread(target=_watchdog, daemon=True).start()

    cfg = {"geometry_only": True, "use_cbgs": False,
           "input_final_dim": [FINAL_H, FINAL_W], "resize": RESIZE,
           "bev_cell_m": BEV_CELL, "fine_size": args.fine_size,
           "lift": "continuous_forward_projection",
           "refiner": "diffusion_cosine_x0_convgru_film_multiview_conf",
           "refiner_k": 4,
           "refiner_sigma_z": 0.5,
           "refiner_candidates": 6,
           "refiner_candidate_bev_size": 128,
           "refiner_candidate_radius_m": [0.04, 0.2, 0.4],
           "confidence_loss": {"lambda_c": args.lambda_c, "loss_rate": args.loss_rate},
           "ida_aug": {"resize_lim": [RESIZE, RESIZE], "final_dim": [FINAL_H, FINAL_W],
                       "rand_flip": False, "rot_lim": [0.0, 0.0]},
           "bda": "identity", "autocast": "bf16", "freeze_official_bn": False,
           "gradnorm": {"lr": args.gradnorm_lr, "alpha": 0.5, "target_sum": 2.0},
           "ema": False, "world_size": world,
           "global_samples_per_epoch": len(ds)}
    if rank == 0:
        json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    hook_debug_done = False
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        ddp.train()
        for it, batch in enumerate(loader):
            imgs = batch[0].to(device)
            mats = {k: v.to(device) for k, v in batch[1].items()}
            ts = batch[2].to(device)
            img_meta = batch[3][0]
            cell = BEV_CELL
            target = make_fine_bev_target(
                provider, img_meta, mats["bda_mat"][0], device,
                fine_size=args.fine_size, cell=cell, z_surround=args.z_surround)

            _step["t0"] = time.perf_counter()
            _step["done"] = False
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            def _now():
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                return time.perf_counter()

            t_start = _now()
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                initial, diag = ddp(
                    imgs, mats, ts, return_depth=True,
                    refiner_target=(target["target_z"], target["target_valid"]))
                if rank == 0:
                    print(f"[stage] backbone+head forward {_now()-t_start:.2f}s",
                          flush=True)
                if rank == 0 and args.debug_hook and not hook_debug_done:
                    calls = model._hook_trace
                    kept = model._img_feats[0]
                    print(f"REAL_HOOK calls={len(calls)} num_sweeps={imgs.shape[1]}", flush=True)
                    assert len(calls) == imgs.shape[1], "unexpected img_backbone call count"
                    assert kept.data_ptr() == calls[0].data_ptr(), "kept c2 is not first key-frame output"
                    first_last = (calls[0].float() - calls[-1].float()).abs().mean()
                    kept_first = (kept.float() - calls[0].float()).abs().max()
                    print(f"REAL_HOOK kept_first_max={float(kept_first):.8f} "
                          f"first_last_mean_abs={float(first_last):.8f}", flush=True)
                    assert float(kept_first) == 0.0
                    assert float(first_last) > 1e-8, "sweep outputs identical; hook overwrite not testable"
                    B0, Nc0 = imgs.shape[0], imgs.shape[2]
                    cam_feat = kept.reshape(B0, Nc0, *kept.shape[1:])
                    cam_diffs = [float((cam_feat[:, c].float() - cam_feat[:, 0].float()).abs().mean())
                                 for c in range(1, Nc0)]
                    print("REAL_HOOK camera_vs_cam0=" + ",".join(f"{x:.8f}" for x in cam_diffs), flush=True)
                    assert sum(x > 1e-8 for x in cam_diffs) >= 2, "camera features appear duplicated"

                    # Counterfactual per-camera ablation on the same retained c2.
                    # If several camera features really contribute to candidate BEV,
                    # zeroing each corresponding camera must change the sampled output.
                    ref = model.rogs_head.refiner
                    z0 = diag["z_coarse"].detach()
                    zs = torch.nn.functional.interpolate(
                        z0, size=(ref.cand_size, ref.cand_size),
                        mode="bilinear", align_corners=False).squeeze(1)
                    offsets = torch.linspace(-1.0, 1.0, ref.num_candidates,
                                             device=z0.device, dtype=z0.dtype).view(1, -1, 1, 1)
                    zcand = (zs.unsqueeze(1) + offsets * 0.2).clamp(ref.z_min, ref.z_max)
                    effects = []
                    with torch.no_grad():
                        base = ref.cand_bev(kept, zcand, mats)[0].float()
                        for c in range(Nc0):
                            ablated = kept.clone().reshape(B0, Nc0, *kept.shape[1:])
                            ablated[:, c].zero_()
                            ablated = ablated.reshape_as(kept)
                            out = ref.cand_bev(ablated, zcand, mats)[0].float()
                            effects.append(float((out - base).abs().mean()))
                    print("REAL_HOOK ablation_effects=" + ",".join(f"{x:.8f}" for x in effects), flush=True)
                    assert sum(x > 1e-10 for x in effects) >= 2, "candidate BEV does not use multiple cameras"
                    print("REAL_KEYFRAME_MULTIVIEW_HOOK_PASSED", flush=True)
                    hook_debug_done = True
                elev = compute_v8a9_loss(
                    initial, diag, target["base_xy"], target["target_z"],
                    target["target_valid"], cell,
                    beta=args.loss_rate, lambda_c=args.lambda_c)
                dlabels = batch[6].to(device).float()
                dpred = diag["depth_pred"].float()
                # Native official depth BCE, already x3; do not rescale.
                depth = compute_official_depth_aux(lightning, dlabels, dpred)
            if rank == 0:
                print(f"[stage] elev+depth loss {_now()-t_start:.2f}s", flush=True)

            # GradNorm norms are computed on two INDEPENDENT forward replicas
            # over the raw (un-DDP-reduced) module, each with a single
            # non-retained autograd.grad.  The main DDP graph is only ever
            # back-propagated once (total.backward); no retained second
            # backward touches the DDP + SyncBN collective (that deadlocked
            # world_size>1 at it2/it3).
            t_gn = _now()
            shared_neck = list(ddp.module.backbone.img_neck.parameters())
            mem_before_gn = (torch.cuda.memory_allocated(device)
                             if device.type == "cuda" else 0)
            g_list = []
            with ddp.no_sync():
                # Replica A: elevation task.
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    i_e, d_e = ddp.module(
                        imgs, mats, ts, return_depth=True,
                        refiner_target=(target["target_z"], target["target_valid"]))
                    el_r = compute_v8a9_loss(
                        i_e, d_e, target["base_xy"], target["target_z"],
                        target["target_valid"], cell,
                        beta=args.loss_rate, lambda_c=args.lambda_c)
                if os.environ.get("DEBUG_GN"):
                    print(f"[gn] replica elev grad start", flush=True)
                g0 = GradNorm.task_grad_norm(
                    el_r["elevation_loss"], shared_neck, retain_graph=False)
                del i_e, d_e, el_r
                # Replica B: depth task (fresh forward, fresh graph).
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    i_d, d_d = ddp.module(
                        imgs, mats, ts, return_depth=True,
                        refiner_use_noise=False)
                    dp_r = compute_official_depth_aux(
                        lightning, dlabels, d_d["depth_pred"].float())
                if os.environ.get("DEBUG_GN"):
                    print(f"[gn] replica depth grad start", flush=True)
                g1 = GradNorm.task_grad_norm(
                    dp_r, shared_neck, retain_graph=False)
                del i_d, d_d, dp_r
            gn = gradnorm.update_norms(
                [elev["elevation_loss"], depth], torch.stack([g0, g1]))
            mem_after_gn = (torch.cuda.memory_allocated(device)
                           if device.type == "cuda" else 0)
            gn["mem_delta_mib"] = float((mem_after_gn - mem_before_gn) / 1024**2)
            assert torch.isfinite(gn["weight"]).all(), "GradNorm weight non-finite"
            assert (gn["weight"] > 0).all(), "GradNorm weight non-positive"
            assert abs(gn["weight_sum"] - 2.0) < 1e-3, "GradNorm weights do not sum to 2"
            assert torch.isfinite(gn["grad_norm"]).all(), "GradNorm grad-norm non-finite"
            if rank == 0:
                print(f"[stage] gradnorm update {_now()-t_gn:.2f}s "
                      f"g={[round(float(x),3) for x in gn['grad_norm']]} "
                      f"w={[round(float(x),4) for x in gn['weight']]}", flush=True)

            total = gradnorm.weighted_total(
                [elev["elevation_loss"], depth])
            t_bwd = _now()
            total.backward()
            _conf_grad = ddp.module.rogs_head.refiner.conf.weight.grad
            conf_grad_abs = (float(_conf_grad.detach().abs().sum())
                             if _conf_grad is not None else float("nan"))
            torch.nn.utils.clip_grad_norm_(
                [p for p in ddp.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            _step["done"] = True
            if rank == 0:
                print(f"[stage] backward+step {_now()-t_bwd:.2f}s", flush=True)

            w = gn["weight"]
            g = gn["grad_norm"]
            if it % args.log_every == 0 and rank == 0:
                peak = (torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda" else 0.0)
                print(f"e{epoch} it{it} total={float(total):.4f} "
                      f"elev_rmse={float(elev['elevation_rmse']):.4f} "
                      f"elev_total={float(elev['elevation_loss']):.4f} "
                      f"conf_nll={float(elev['conf_term']):.4f} "
                      f"depth_x3={float(depth):.4f} "
                      f"w_elev={float(w[0]):.4f} w_depth={float(w[1]):.4f} "
                      f"3*w_depth={float(3.0 * w[1]):.4f} "
                      f"g_elev={float(g[0]):.4f} g_depth={float(g[1]):.4f} "
                      f"gn_loss={float(gn['gradnorm_loss']):.4f} "
                      f"gn_mem_delta_mib={float(gn['mem_delta_mib']):.2f} "
                      f"conf_grad_abs={conf_grad_abs:.6f} "
                      f"peak_mib={peak:.1f}", flush=True)
            if args.max_steps > 0 and it + 1 >= args.max_steps:
                break
        if rank == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()},
                       os.path.join(out_dir, f"epoch_{epoch:03d}.pt"))
        if args.max_steps > 0:
            break
    if rank == 0:
        torch.save({"epoch": args.epochs, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()},
                   os.path.join(out_dir, "last.pt"))


if __name__ == "__main__":
    main()
