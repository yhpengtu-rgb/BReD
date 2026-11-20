#!/usr/bin/env python3
"""Evaluate v8-a17 depth-prior coarse/fine elevation with the RoGS z metric.

Protocol (same as the formal v8-a9 scene-0523 evaluation):
  * deterministic 576x1024 input, resize 0.64, identity BDA, CBGS off;
  * 256x256 Gaussian lattice at 0.4 m/cell, running BN statistics;
  * transform predicted Gaussian centres to world coordinates;
  * query = Gaussian world XY, reference = raw RoadGT world XY;
  * pytorch3d ball_query K=1, radius=0.1 m;
  * compare world z for matched Gaussians only; unmatched Gaussians are ignored.

One backbone forward produces diag['z_coarse'] and the first initial['z'].
Optional outer loops feed each z_fine back as the next refiner z_coarse while
reusing the image/BEV conditions; every loop is transformed and matched
independently because its z also enters the rigid world-coordinate transform.
The primary result pools squared residuals over all matched Gaussians from all
391 frames; per-frame RMSE mean is secondary.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bevstereo_official import BRED_ROOT, REPO, collate_fn as official_collate, \
    load_experiment_class

FINAL_H, FINAL_W = 576, 1024
RESIZE = 0.64
BEV_CELL = 0.4
FINE_SIZE = 256
X_BOUND = [-51.2, 51.2, BEV_CELL]
Y_BOUND = [-51.2, 51.2, BEV_CELL]

import bevdepth.exps.nuscenes.base_exp as base_exp  # noqa: E402

base_exp.final_dim = (FINAL_H, FINAL_W)
base_exp.backbone_conf["final_dim"] = (FINAL_H, FINAL_W)
base_exp.backbone_conf["x_bound"] = X_BOUND
base_exp.backbone_conf["y_bound"] = Y_BOUND
base_exp.ida_aug_conf["final_dim"] = (FINAL_H, FINAL_W)

from continuous_lift_fpn import ContinuousLiftFPN  # noqa: E402
from bevstereo_rogs_model import BEVStereoRoGSModel  # noqa: E402
from rogs_gaussian_build import build_fine_gaussians  # noqa: E402
from rogs_fine_target import make_fine_bev_target  # noqa: E402
from roadgt_provider import RoadGTProvider  # noqa: E402
from nusc_det_dataset_road import NuscDetDatasetRoadV8a4  # noqa: E402
from pytorch3d.ops import ball_query  # noqa: E402
from pytorch3d.ops.utils import masked_gather  # noqa: E402
from plyfile import PlyData  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402


def build_zero_aug_conf(ida_aug_conf, bda_aug_conf):
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


class StageStats:
    def __init__(self, name: str):
        self.name = name
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.matched = 0
        self.candidates = 0
        self.frame_rmse: List[float] = []
        self.frame_mae: List[float] = []
        self.frame_matched: List[int] = []
        self.frame_candidates: List[int] = []
        self.abs_residuals: List[np.ndarray] = []

    def update(self, error: torch.Tensor, candidate_count: int) -> None:
        if error.ndim != 1:
            error = error.reshape(-1)
        n = int(error.numel())
        self.candidates += int(candidate_count)
        self.matched += n
        self.frame_matched.append(n)
        self.frame_candidates.append(int(candidate_count))
        if n:
            e64 = error.double()
            self.abs_sum += float(e64.abs().sum().item())
            self.sq_sum += float((e64 ** 2).sum().item())
            self.frame_rmse.append(float(torch.sqrt((e64 ** 2).mean()).item()))
            self.frame_mae.append(float(e64.abs().mean().item()))
            self.abs_residuals.append(error.detach().abs().float().cpu().numpy())
        else:
            self.frame_rmse.append(float("nan"))
            self.frame_mae.append(float("nan"))

    def summary(self) -> Dict:
        pooled_rmse = math.sqrt(self.sq_sum / max(self.matched, 1))
        pooled_mae = self.abs_sum / max(self.matched, 1)
        coverage = [n / max(total, 1) for n, total in
                    zip(self.frame_matched, self.frame_candidates)]
        if self.abs_residuals:
            abs_e = np.concatenate(self.abs_residuals)
            q50, q90, q95, q99 = np.percentile(abs_e, [50, 90, 95, 99])
        else:
            q50 = q90 = q95 = q99 = float("nan")
        return {
            "matched_points": self.matched,
            "candidate_points": self.candidates,
            "matched_fraction": self.matched / max(self.candidates, 1),
            "Z_RMSE_m": pooled_rmse,
            "Z_MAE_m": pooled_mae,
            "frame_rmse_mean_m": float(np.nanmean(self.frame_rmse)),
            "frame_mae_mean_m": float(np.nanmean(self.frame_mae)),
            "frame_coverage_mean": float(np.mean(coverage)),
            "abs_error_p50_m": float(q50),
            "abs_error_p90_m": float(q90),
            "abs_error_p95_m": float(q95),
            "abs_error_p99_m": float(q99),
            "per_frame": [
                {
                    "frame": i,
                    "matched": self.frame_matched[i],
                    "candidates": self.frame_candidates[i],
                    "rmse_m": self.frame_rmse[i],
                    "mae_m": self.frame_mae[i],
                }
                for i in range(len(self.frame_matched))
            ],
        }


def world_gaussians(gaussian_xyz: torch.Tensor, meta: Dict,
                    bda_mat: torch.Tensor) -> torch.Tensor:
    translation = np.asarray(meta["ego2global_translation"], dtype=np.float64)
    rotation = Quaternion(np.asarray(meta["ego2global_rotation"],
                                    dtype=np.float64)).rotation_matrix
    bda = bda_mat.detach().cpu().numpy().astype(np.float64)
    bda_r = torch.as_tensor(bda[:3, :3], device=gaussian_xyz.device,
                            dtype=torch.float64)
    bda_t = torch.as_tensor(bda[:3, 3], device=gaussian_xyz.device,
                            dtype=torch.float64)
    g_ego = (gaussian_xyz.double() - bda_t) @ bda_r
    g_world = (g_ego @ torch.as_tensor(rotation.T, device=gaussian_xyz.device,
                                      dtype=torch.float64)
               + torch.as_tensor(translation, device=gaussian_xyz.device,
                                 dtype=torch.float64))
    return g_world.float()


def matched_world_z_error(g_world: torch.Tensor, native_xyz_global: np.ndarray,
                          chunk: int = 32768) -> torch.Tensor:
    xy_min = g_world[:, :2].min(dim=0).values.cpu().numpy() - 0.11
    xy_max = g_world[:, :2].max(dim=0).values.cpu().numpy() + 0.11
    inside = ((native_xyz_global[:, 0] >= xy_min[0]) &
              (native_xyz_global[:, 0] <= xy_max[0]) &
              (native_xyz_global[:, 1] >= xy_min[1]) &
              (native_xyz_global[:, 1] <= xy_max[1]))
    lidar_xyz = torch.from_numpy(native_xyz_global[inside].astype(np.float32)).to(
        g_world.device)
    if not len(lidar_xyz):
        return g_world.new_empty((0,), dtype=torch.float32)
    idx_parts = []
    for start in range(0, g_world.shape[0], chunk):
        query_xy = g_world[start:start + chunk, :2]
        idx_parts.append(ball_query(
            query_xy[None, :, :], lidar_xyz[None, :, :2],
            K=1, return_nn=False, radius=0.1).idx)
    near_idx = torch.cat(idx_parts, dim=1)
    valid = (near_idx.squeeze(0) != -1).reshape(-1)
    gt_z = masked_gather(lidar_xyz[None, :, 2:3], near_idx).reshape(-1)
    gt_z = gt_z[valid]
    pred_z = g_world[:, 2][valid]
    return pred_z - gt_z


def fixed_base_xy(fine_size: int, cell: float, device: torch.device) -> torch.Tensor:
    """Bit-identical fixed lattice from make_fine_bev_target, without GT work."""
    coords = (np.arange(fine_size, dtype=np.float64) + 0.5) * cell \
        - cell * fine_size / 2.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    base_xy = np.stack((xx, yy), axis=-1).reshape(-1, 2)
    return torch.from_numpy(base_xy.astype(np.float32)).reshape(
        1, fine_size * fine_size, 2).to(device)


def eval_xy_z(initial: Dict, base_xy: torch.Tensor, cell: float):
    """Only the XY/Z operations from build_fine_gaussians, with identical math."""
    z_map = initial["z"]
    count = z_map.shape[-1] * z_map.shape[-1]
    base = base_xy.to(device=z_map.device, dtype=z_map.dtype).reshape(-1, 2)
    xy = base + initial["xy"][0].permute(1, 2, 0).reshape(count, 2) * float(cell)
    z = z_map[0].permute(1, 2, 0).reshape(count, 1)
    return xy, z


def matched_world_z_error_pair(g_worlds: torch.Tensor,
                               native_xyz_global: np.ndarray,
                               matcher: str = "ball") -> List[torch.Tensor]:
    """Batched coarse/fine matching over their exact union RoadGT candidate set."""
    xy_min = g_worlds[:, :, :2].amin(dim=(0, 1)).cpu().numpy() - 0.11
    xy_max = g_worlds[:, :, :2].amax(dim=(0, 1)).cpu().numpy() + 0.11
    inside = ((native_xyz_global[:, 0] >= xy_min[0]) &
              (native_xyz_global[:, 0] <= xy_max[0]) &
              (native_xyz_global[:, 1] >= xy_min[1]) &
              (native_xyz_global[:, 1] <= xy_max[1]))
    lidar_xyz = torch.from_numpy(native_xyz_global[inside].astype(np.float32)).to(
        g_worlds.device)
    stages = int(g_worlds.shape[0])
    if not len(lidar_xyz):
        return [g_worlds.new_empty((0,), dtype=torch.float32) for _ in range(stages)]
    query_xy = g_worlds[:, :, :2]
    reference_xy = lidar_xyz[None, :, :2].expand(stages, -1, -1)
    if matcher != "ball":
        raise ValueError(f"unsupported precision-safe matcher: {matcher}")
    near_idx = ball_query(query_xy, reference_xy, K=1, return_nn=False,
                          radius=0.1).idx
    valid = (near_idx.squeeze(-1) != -1)
    reference_z = lidar_xyz[None, :, 2:3].expand(stages, -1, -1)
    gt_z = masked_gather(reference_z, near_idx).reshape(stages, -1)
    pred_z = g_worlds[:, :, 2]
    return [(pred_z[k] - gt_z[k])[valid[k]] for k in range(stages)]


def matched_world_z_error_tiled(g_worlds: torch.Tensor,
                                native_xyz_global: np.ndarray,
                                tile_size: int) -> List[torch.Tensor]:
    """Exact RoGS ball_query with spatially reduced, order-preserving references.

    Every tile keeps the native RoadGT order.  Its reference bbox contains all
    points that can be within 0.1 m of any query in that tile, so the first
    radius-inlier is exactly the same native point as in the full-frame query.
    """
    stages, count, _ = g_worlds.shape
    fine_size = int(math.isqrt(count))
    if fine_size * fine_size != count or fine_size % tile_size:
        raise ValueError("tile_size must divide the square Gaussian lattice")
    xy_min = g_worlds[:, :, :2].amin(dim=(0, 1)).cpu().numpy() - 0.11
    xy_max = g_worlds[:, :, :2].amax(dim=(0, 1)).cpu().numpy() + 0.11
    inside = ((native_xyz_global[:, 0] >= xy_min[0]) &
              (native_xyz_global[:, 0] <= xy_max[0]) &
              (native_xyz_global[:, 1] >= xy_min[1]) &
              (native_xyz_global[:, 1] <= xy_max[1]))
    candidates = native_xyz_global[inside].astype(np.float32)
    if not len(candidates):
        return [g_worlds.new_empty((0,), dtype=torch.float32) for _ in range(2)]
    candidate_xy = candidates[:, :2]
    world_cpu = g_worlds.cpu().numpy()
    tiles = []
    for stage in range(stages):
        for r0 in range(0, fine_size, tile_size):
            for c0 in range(0, fine_size, tile_size):
                rr, cc = np.meshgrid(np.arange(r0, r0 + tile_size),
                                     np.arange(c0, c0 + tile_size),
                                     indexing="ij")
                indices = (rr * fine_size + cc).reshape(-1)
                query = world_cpu[stage, indices, :2]
                mn = query.min(axis=0) - 0.11
                mx = query.max(axis=0) + 0.11
                mask = ((candidate_xy[:, 0] >= mn[0]) &
                        (candidate_xy[:, 0] <= mx[0]) &
                        (candidate_xy[:, 1] >= mn[1]) &
                        (candidate_xy[:, 1] <= mx[1]))
                reference = candidates[mask]
                tiles.append((stage, indices, query, reference))
    queries_per_tile = tile_size * tile_size
    batch_size = len(tiles)
    max_reference = max(len(tile[3]) for tile in tiles)
    p1 = np.zeros((batch_size, queries_per_tile, 2), dtype=np.float32)
    p2 = np.zeros((batch_size, max_reference, 2), dtype=np.float32)
    p2z = np.zeros((batch_size, max_reference, 1), dtype=np.float32)
    pred = np.zeros((batch_size, queries_per_tile), dtype=np.float32)
    lengths2 = np.zeros((batch_size,), dtype=np.int64)
    for tile_idx, (stage, indices, query, reference) in enumerate(tiles):
        p1[tile_idx] = query
        pred[tile_idx] = world_cpu[stage, indices, 2]
        lengths2[tile_idx] = len(reference)
        if len(reference):
            p2[tile_idx, :len(reference)] = reference[:, :2]
            p2z[tile_idx, :len(reference), 0] = reference[:, 2]
    device = g_worlds.device
    near_idx = ball_query(
        torch.from_numpy(p1).to(device), torch.from_numpy(p2).to(device),
        lengths2=torch.from_numpy(lengths2).to(device),
        K=1, return_nn=False, radius=0.1).idx
    valid = near_idx.squeeze(-1) != -1
    gt_z = masked_gather(torch.from_numpy(p2z).to(device), near_idx).reshape(
        batch_size, queries_per_tile)
    error_all = torch.from_numpy(pred).to(device) - gt_z
    tiles_per_stage = batch_size // stages
    output = []
    for stage in range(stages):
        stage_slice = slice(stage * tiles_per_stage, (stage + 1) * tiles_per_stage)
        stage_valid = valid[stage_slice]
        output.append(error_all[stage_slice][stage_valid])
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--info-path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--official-checkpoint", required=True)
    ap.add_argument("--roadgt-cache", required=True)
    ap.add_argument("--native-road-ply", required=True)
    ap.add_argument("--scene-name", default="scene-0523")
    ap.add_argument("--allow-frozen-val-scene", action="store_true")
    ap.add_argument("--data-root", default=str(BRED_ROOT / "assets" / "datasets" / "nuScenes"))
    ap.add_argument("--fine-size", type=int, default=FINE_SIZE)
    ap.add_argument("--cell-m", type=float, default=BEV_CELL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--expected-frames", type=int, default=391)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--z-surround", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forward-dtype", choices=("fp32", "fp16", "fp16_mixed"), default="fp32",
                    help="fp16=full autocast; fp16_mixed=fp16 backbone with fp32 elevation/refiner head; metrics remain fp32")
    ap.add_argument("--fast", action="store_true",
                    help="precision-preserving eval: skip unused GT/Gaussian work, "
                         "batched coarse/fine matching, no per-frame empty_cache")
    ap.add_argument("--matcher", choices=("ball", "ball_tiled"), default="ball_tiled")
    ap.add_argument("--tile-size", type=int, default=128)
    ap.add_argument("--outer-loops", type=int, choices=(0, 1, 2, 3, 4, 5), default=0,
                    help="number of z_fine -> next z_coarse feedback loops; 0 keeps original fine")
    ap.add_argument("--loop-residual", choices=("f_minus_c", "c_minus_f", "abs"),
                    default="f_minus_c", help="feedback residual policy for loop 1 and above")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    if not math.isclose(args.fine_size * args.cell_m, 102.4,
                        rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("fine_size * cell_m must preserve the 102.4 m BEV extent")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if args.forward_dtype in ("fp16", "fp16_mixed") and device.type != "cuda":
        raise ValueError("fp16 forward evaluation requires a CUDA device")

    Exp = load_experiment_class()
    lightning = Exp(gpus=0, batch_size_per_device=1, data_root=args.data_root,
                    default_root_dir=str(BRED_ROOT / "results" / "stage1" / "lightning_eval"))
    official_sd = torch.load(args.official_checkpoint, map_location="cpu")["state_dict"]
    official_sd.pop("model.backbone.frustum", None)
    lightning.load_state_dict(official_sd, strict=False)
    lift = ContinuousLiftFPN(**lightning.backbone_conf)
    lift.fp32_splat = args.forward_dtype == "fp16_mixed"
    lift.load_official_backbone(lightning.model.backbone)
    lightning.model.backbone = lift
    model = BEVStereoRoGSModel(lightning.model, lift_backbone=lift,
                               fine_size=args.fine_size, geometry_only=True,
                               freeze_official_bn=False).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:12]}, "
                           f"unexpected={unexpected[:12]}")
    refiner = model.rogs_head.refiner
    if refiner.k_iters != 1 or not math.isclose(refiner.radius_base, 0.1) \
            or not math.isclose(refiner.res_max_m, 0.3):
        raise RuntimeError("evaluation model is not K=1/radius=0.10/res_max=0.30")
    print(f"LOADED epoch={ckpt.get('epoch')} missing=0 unexpected=0 "
          f"K={refiner.k_iters} radius_base={refiner.radius_base} "
          f"res_max_m={refiner.res_max_m} fine_size={args.fine_size} "
          f"cell_m={args.cell_m}", flush=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    ida_aug_conf, bda_aug_conf = build_zero_aug_conf(
        lightning.ida_aug_conf, lightning.bda_aug_conf)
    common = dict(ida_aug_conf=ida_aug_conf, bda_aug_conf=bda_aug_conf,
                  data_root=args.data_root, info_paths=args.info_path,
                  is_train=False, return_depth=False,
                  sweep_idxes=lightning.sweep_idxes,
                  key_idxes=lightning.key_idxes,
                  use_fusion=lightning.use_fusion)
    dataset = NuscDetDatasetRoadV8a4(**common, classes=lightning.class_names,
                                    use_cbgs=False)
    if len(dataset) != args.expected_frames:
        raise RuntimeError(f"expected {args.expected_frames} frames, got {len(dataset)}")
    loader_kwargs = dict(batch_size=1, shuffle=False, num_workers=args.workers,
                         collate_fn=lambda s: official_collate(s, is_return_depth=False),
                         pin_memory=args.fast)
    if args.fast and args.workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(dataset, **loader_kwargs)
    provider = None
    if not args.fast:
        if args.allow_frozen_val_scene:
            import roadgt_provider as rgp
            rgp.FROZEN_VAL_SCENES = frozenset(
                s for s in rgp.FROZEN_VAL_SCENES if s != args.scene_name)
        provider = RoadGTProvider(args.roadgt_cache, args.scene_name)
    base_xy_fast = fixed_base_xy(args.fine_size, args.cell_m, device) \
        if args.fast else None
    road = PlyData.read(args.native_road_ply).elements[0]
    native_xyz_global = np.stack((np.asarray(road["x"]), np.asarray(road["y"]),
                                  np.asarray(road["z"])), axis=1).astype(np.float64)
    if not np.isfinite(native_xyz_global).all():
        raise ValueError("native RoadGT contains non-finite XYZ")
    print(f"DATASET frames={len(dataset)} NATIVE_ROAD_POINTS={len(native_xyz_global)}",
          flush=True)

    fine_names = (["fine"] if args.outer_loops == 0 else
                  ["fine_no_loop"] +
                  [f"fine_loop{k}" for k in range(1, args.outer_loops + 1)])
    stage_names = ["coarse"] + fine_names
    stats = {name: StageStats(name) for name in stage_names}
    frames = 0
    eval_start = time.perf_counter()
    grad_context = torch.inference_mode() if args.fast else torch.no_grad()
    with grad_context:
        for i, batch in enumerate(loader):
            if args.max_frames and i >= args.max_frames:
                break
            imgs = batch[0].to(device, non_blocking=args.fast)
            mats = {k: v.to(device, non_blocking=args.fast)
                    for k, v in batch[1].items()}
            timestamps = batch[2].to(device, non_blocking=args.fast)
            meta = batch[3][0]
            if args.fast:
                base_xy = base_xy_fast
            else:
                target = make_fine_bev_target(
                    provider, meta, mats["bda_mat"][0], device,
                    fine_size=args.fine_size, cell=args.cell_m,
                    z_surround=args.z_surround)
                base_xy = target["base_xy"]
            if args.profile and device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_start = time.perf_counter()
            forward_context = (torch.autocast(device_type="cuda", dtype=torch.float16)
                               if args.forward_dtype in ("fp16", "fp16_mixed") else nullcontext())
            with forward_context:
                initial, diag = model(imgs, mats, timestamps, return_depth=True,
                                      refiner_use_noise=True,
                                      refiner_outer_loops=args.outer_loops,
                                      refiner_outer_seed=args.seed + i * 1009,
                                      head_fp32=args.forward_dtype == "fp16_mixed",
                                      refiner_loop_residual=args.loop_residual)
            # Half precision is confined to the learned model.  Reconstruction,
            # world transforms, ball_query and error accumulation stay fp32.
            initial = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point()
                           else v) for k, v in initial.items()}
            diag["z_coarse"] = diag["z_coarse"].float()
            if args.outer_loops > 0:
                diag["z_outer"] = [z.float() for z in diag["z_outer"]]
            if args.profile and device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_end = time.perf_counter()
            coarse_initial = dict(initial)
            coarse_initial["z"] = diag["z_coarse"]
            stage_initials = [("coarse", coarse_initial)]
            for loop_idx, name in enumerate(fine_names):
                loop_initial = dict(initial)
                if args.outer_loops > 0:
                    loop_initial["z"] = diag["z_outer"][loop_idx]
                stage_initials.append((name, loop_initial))
            stage_xyz = []
            for name, stage_initial in stage_initials:
                if args.fast:
                    xy, z = eval_xy_z(stage_initial, base_xy, args.cell_m)
                else:
                    _, xy, z = build_fine_gaussians(
                        stage_initial, base_xy, args.cell_m)
                if not torch.isfinite(z).all():
                    raise FloatingPointError(f"non-finite {name} elevation at frame {i}")
                stage_xyz.append((xy, z))
            xy_reference = stage_xyz[0][0]
            if any(not torch.equal(xy, xy_reference) for xy, _ in stage_xyz[1:]):
                raise RuntimeError("coarse/fine Gaussian XY differ")

            if args.fast:
                # Keep the validated two-stage matcher memory bound.  The
                # original no-loop fine is paired with coarse; later loops paired.
                for start in range(0, len(stage_xyz), 2):
                    g_worlds = []
                    for xy, z in stage_xyz[start:start + 2]:
                        gaussian_xyz = torch.cat((xy, z), dim=1).float()
                        g_worlds.append(world_gaussians(
                            gaussian_xyz, meta, mats["bda_mat"][0]))
                    g_worlds = torch.stack(g_worlds, dim=0)
                    if args.matcher == "ball_tiled":
                        errors = matched_world_z_error_tiled(
                            g_worlds, native_xyz_global, tile_size=args.tile_size)
                    else:
                        errors = matched_world_z_error_pair(
                            g_worlds, native_xyz_global, matcher=args.matcher)
                    for name, error in zip(stage_names[start:start + 2], errors):
                        stats[name].update(error, int(g_worlds.shape[1]))
                    del g_worlds, errors
                if args.profile and device.type == "cuda":
                    torch.cuda.synchronize(device)
                match_end = time.perf_counter()
            else:
                for (name, _), (xy, z) in zip(stage_initials, stage_xyz):
                    gaussian_xyz = torch.cat((xy, z), dim=1).float()
                    g_world = world_gaussians(gaussian_xyz, meta, mats["bda_mat"][0])
                    error = matched_world_z_error(g_world, native_xyz_global)
                    stats[name].update(error, int(g_world.shape[0]))
                    del gaussian_xyz, g_world, error
                match_end = time.perf_counter()
            frames += 1
            if i % 10 == 0 or i + 1 == len(loader):
                rmse_text = " ".join(
                    f"{name}_rmse={math.sqrt(stats[name].sq_sum / max(stats[name].matched, 1)):.8f}"
                    for name in stage_names)
                count_text = " ".join(
                    f"{name}_n={stats[name].matched}" for name in stage_names)
                print(f"EVAL frame={i + 1}/{len(loader)} {rmse_text} {count_text} "
                      f"forward_ms={(forward_end - forward_start) * 1e3:.1f} "
                      f"match_ms={(match_end - forward_end) * 1e3:.1f} "
                      f"elapsed_s={time.perf_counter() - eval_start:.1f}", flush=True)
            del initial, diag, coarse_initial, stage_initials, stage_xyz
            if not args.fast:
                torch.cuda.empty_cache()
    eval_elapsed = time.perf_counter() - eval_start

    expected_run_frames = args.max_frames or args.expected_frames
    if frames != expected_run_frames:
        raise RuntimeError(f"evaluation incomplete: {frames}/{expected_run_frames}")
    result = {
        "scene_name": args.scene_name,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "frames": frames,
        "fine_size": args.fine_size,
        "cell_m": args.cell_m,
        "bn_stats": "running",
        "forward_dtype": args.forward_dtype,
        "metric_dtype": "fp32",
        "refiner_input": "detached_expected_depth_bev_t0",
        "refiner_k": refiner.k_iters,
        "refiner_outer_loops": args.outer_loops,
        "loop_residual": args.loop_residual if args.outer_loops > 0 else None,
        "outer_loop_rule": "fine_no_loop uses the original detached depth-BEV residual at t=0; each loop computes the selected F-C / C-F / abs(F-C) feedback against the previous coarse, then sets coarse=latest fine and rebuilds FiLM and multi-scale candidate cond from it at t=0; camdepth is not reused",
        "radius_base_m": refiner.radius_base,
        "res_max_m": refiner.res_max_m,
        "fast": args.fast,
        "matcher": args.matcher if args.fast else "ball_chunked_independent",
        "tile_size": args.tile_size if args.matcher == "ball_tiled" else None,
        "eval_elapsed_s": eval_elapsed,
        "metric_protocol": "RoGS eval_z_metric: world Gaussian XY -> raw RoadGT world XY, ball_query K=1 radius=0.1m, world-z residual, unmatched Gaussians ignored",
        "aggregation": "residual-pooled over all matched Gaussians; per-frame mean is secondary",
        "native_road_points": len(native_xyz_global),
        "coarse": stats["coarse"].summary(),
    }
    for name in fine_names:
        result[name] = stats[name].summary()
    compact = {
        "frames": frames,
        "coarse_rmse": result["coarse"]["Z_RMSE_m"],
        "coarse_mae": result["coarse"]["Z_MAE_m"],
        "coarse_matched": result["coarse"]["matched_points"],
    }
    if args.outer_loops == 0:
        result["fine_minus_coarse_rmse_m"] = (
            result["fine"]["Z_RMSE_m"] - result["coarse"]["Z_RMSE_m"])
        compact.update({
            "fine_rmse": result["fine"]["Z_RMSE_m"],
            "fine_mae": result["fine"]["Z_MAE_m"],
            "fine_matched": result["fine"]["matched_points"],
        })
    else:
        result["rmse_delta_vs_no_loop_m"] = {
            name: result[name]["Z_RMSE_m"] - result["fine_no_loop"]["Z_RMSE_m"]
            for name in fine_names[1:]
        }
        for name in fine_names:
            compact.update({
                f"{name}_rmse": result[name]["Z_RMSE_m"],
                f"{name}_mae": result[name]["Z_MAE_m"],
                f"{name}_matched": result[name]["matched_points"],
            })
    compact["out_json"] = os.path.abspath(args.out_json)
    out_json = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2, allow_nan=True)
    print("V8A17_MS_ROGS_Z_EVAL " + json.dumps(compact), flush=True)


if __name__ == "__main__":
    main()
