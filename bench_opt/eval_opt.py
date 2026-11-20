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
import statistics
import sys
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

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


class GTBank:
    """GPU-resident native RoadGT with a spatial-tiled matcher.

    Semantics kept identical to `matched_world_z_error_tiled`: reference
    points stay in native RoadGT order, bbox margin 0.11 m, ball_query K=1,
    radius 0.1 m, lengths2 padding, -1 validity filter. Coarse/fine share the
    same XY (formal equality assertion above), so the spatial match is
    computed once per tile and reused for both stages; only pred z differs.
    """

    GRID_H = 0.1
    GRID_R = 0.1

    def __init__(self, native_xyz_global: np.ndarray, device: torch.device,
                 dump_dir: Optional[str] = None, use_tree: bool = False,
                 use_grid: bool = False, use_cuda: bool = False,
                 dump_frames: Optional[set] = None):
        self.device = device
        self.xy = torch.from_numpy(
            native_xyz_global[:, :2].astype(np.float32)).to(device)
        self.z = torch.from_numpy(
            native_xyz_global[:, 2].astype(np.float32)).to(device)
        self.native_np = native_xyz_global.astype(np.float64)
        self.native_z = native_xyz_global[:, 2].astype(np.float32)
        self.tree = None
        if use_tree:
            from scipy.spatial import cKDTree
            self.tree = cKDTree(native_xyz_global[:, :2].astype(np.float32))
        self.grid = None
        if use_grid:
            self.grid = self._build_grid()
        self.cuda_index = None
        self.extension_load_ms: Optional[float] = None
        if use_cuda:
            from matcher_ext.matcher_lib import CudaGridMatcher
            _t0 = time.perf_counter()
            self.cuda_index = CudaGridMatcher(self.xy, self.z)
            torch.cuda.synchronize(self.device)
            self.extension_load_ms = (time.perf_counter() - _t0) * 1e3
        self.match_ms: List[float] = []
        self.dump_dir = dump_dir
        self.dump_frames = dump_frames
        self.dump_records: List[Dict] = []

    def _want_dump(self, frame: Optional[int]) -> bool:
        if not self.dump_dir:
            return False
        if self.dump_frames is None:
            return True
        return frame is not None and frame in self.dump_frames

    def _build_grid(self) -> Dict:
        """Static CSR uniform grid over native RoadGT XY.

        Sorted by (bin, native) with a stable sort so each bucket keeps its
        ascending native order.  Duplicate XY are retained (never deduped).
        Origin = reference min XY; FP32 bucketing.
        """
        h = self.GRID_H
        origin = self.xy.min(dim=0).values
        cell = torch.floor((self.xy - origin) / h).long()
        nc = cell.max(dim=0).values + 1
        cid = cell[:, 0] * nc[1] + cell[:, 1]
        order = torch.argsort(cid, stable=True)
        scid = cid[order]
        uniq, cnt = torch.unique_consecutive(scid, return_counts=True)
        start = torch.cumsum(cnt, dim=0) - cnt
        return {
            "origin": origin, "nc": nc, "uniq": uniq, "start": start,
            "cnt": cnt, "order": order,
            "sorted_xy": self.xy[order], "sorted_z": self.z[order],
            "kcap": int(cnt.max().item()),
        }

    @staticmethod
    def precompute_tiles(fine_size: int, tile_size: int, device) -> List[torch.Tensor]:
        tiles = []
        for r0 in range(0, fine_size, tile_size):
            for c0 in range(0, fine_size, tile_size):
                rr, cc = np.meshgrid(np.arange(r0, r0 + tile_size),
                                     np.arange(c0, c0 + tile_size),
                                     indexing="ij")
                idx = (rr * fine_size + cc).reshape(-1)
                tiles.append(torch.from_numpy(idx).to(device))
        return tiles

    def match(self, g_worlds: torch.Tensor,
              tile_indexes: List[torch.Tensor],
              frame: Optional[int] = None) -> List[torch.Tensor]:
        stages, count, _ = g_worlds.shape
        # Global candidate filter uses the union bbox over all stages, exactly
        # as the original tiled matcher.  Per-stage tile queries use that
        # stage's own world XY: the BDA/ego rotation mixes z into XY, so
        # coarse and fine world XY differ slightly and must NOT be shared.
        union_xy = g_worlds[:, :, :2].reshape(-1, 2)
        mn_u = union_xy.min(dim=0).values - 0.11
        mx_u = union_xy.max(dim=0).values + 0.11
        inside = ((self.xy[:, 0] >= mn_u[0]) & (self.xy[:, 0] <= mx_u[0]) &
                  (self.xy[:, 1] >= mn_u[1]) & (self.xy[:, 1] <= mx_u[1]))
        ref_idx = torch.nonzero(inside, as_tuple=False).squeeze(-1)
        ref_xy = self.xy[ref_idx]

        spatial = len(tile_indexes)
        qpt = tile_indexes[0].shape[0]
        out: List[torch.Tensor] = []
        for stage in range(stages):
            q_xy = g_worlds[stage, :, :2]
            p1 = torch.zeros(spatial, qpt, 2, dtype=torch.float32,
                             device=self.device)
            tile_ref_idx: List[torch.Tensor] = []
            tile_q_idx: List[torch.Tensor] = []
            for t, indices in enumerate(tile_indexes):
                qt = q_xy[indices]
                p1[t] = qt
                tmn = qt.min(dim=0).values - 0.11
                tmx = qt.max(dim=0).values + 0.11
                m = ((ref_xy[:, 0] >= tmn[0]) & (ref_xy[:, 0] <= tmx[0]) &
                     (ref_xy[:, 1] >= tmn[1]) & (ref_xy[:, 1] <= tmx[1]))
                ri = ref_idx[m]
                tile_ref_idx.append(ri)
                tile_q_idx.append(indices)
            max_ref = max(1, max(ri.shape[0] for ri in tile_ref_idx))
            p2 = torch.zeros(spatial, max_ref, 2, dtype=torch.float32,
                             device=self.device)
            p2z = torch.zeros(spatial, max_ref, 1, dtype=torch.float32,
                              device=self.device)
            lengths2 = torch.zeros(spatial, dtype=torch.int64, device=self.device)
            for t, ri in enumerate(tile_ref_idx):
                p2[t, :ri.shape[0]] = self.xy[ri]
                p2z[t, :ri.shape[0], 0] = self.z[ri]
                lengths2[t] = ri.shape[0]

            near_idx = ball_query(p1, p2, lengths2=lengths2, K=1,
                                  return_nn=False, radius=0.1).idx
            valid = near_idx.squeeze(-1) != -1
            gt_z = masked_gather(p2z, near_idx).reshape(spatial, qpt)
            native_chosen = torch.full((count,), -1, dtype=torch.int64,
                                       device=self.device)
            for t, (qi, ri) in enumerate(zip(tile_q_idx, tile_ref_idx)):
                sel = near_idx[t].squeeze(-1)
                v = valid[t]
                native_chosen[qi[v]] = ri[sel[v]]

            pred_z = g_worlds[stage, :, 2]
            pred_tile = torch.zeros(spatial, qpt, dtype=torch.float32,
                                    device=self.device)
            for t, qi in enumerate(tile_q_idx):
                pred_tile[t] = pred_z[qi]
            out.append((pred_tile - gt_z)[valid])
            if self.dump_dir:
                self.dump_records.append({
                    "frame": -1 if frame is None else frame,
                    "stage": stage,
                    "xy": g_worlds[stage, :, :2].detach().cpu().numpy(),
                    "native": native_chosen.detach().cpu().numpy(),
                    "pred_z": pred_z.detach().cpu().numpy(),
                })
        return out

    def match_tree(self, g_worlds: torch.Tensor,
                   frame: Optional[int] = None) -> List[torch.Tensor]:
        """Spatial-index matcher with proven ball_query selection semantics.

        Real ball_query (pytorch3d) returns the *first reference in native
        order within radius*, not the nearest.  Validated per-point against
        the real GPU matcher on dumped frames: chosen native index matches
        exactly.  KDTree returns the in-radius superset; we select the
        smallest native RoadGT index.  Radius 0.1 m, K=1, unmatched ignored,
        residual accumulated FP32.  Coarse/fine query independently.
        """
        if self.tree is None:
            raise RuntimeError("tree index not built")
        stages, count, _ = g_worlds.shape
        pred = g_worlds[:, :, 2].detach().cpu().numpy()
        out: List[torch.Tensor] = []
        for stage in range(stages):
            q = g_worlds[stage, :, :2].detach().cpu().numpy().astype(np.float32)
            # Cheap nearest prefilter: only queries with any in-radius GT
            # enter the list ball query.  Nearest index is NOT used for the
            # final choice; final choice remains min native index within radius.
            _, near_idx = self.tree.query(q, k=1, distance_upper_bound=0.1)
            near = near_idx != self.tree.n
            native = np.full(count, -1, dtype=np.int64)
            if near.any():
                sub = self.tree.query_ball_point(q[near], r=0.1)
                near_pos = np.nonzero(near)[0]
                for k, cand in enumerate(sub):
                    if cand:
                        native[near_pos[k]] = min(cand)
            valid = native >= 0
            gt_z = self.native_z[native[valid]]
            res = (pred[stage][valid] - gt_z).astype(np.float32)
            out.append(torch.from_numpy(res).to(self.device))
            if self.dump_dir:
                self.dump_records.append({
                    "frame": -1 if frame is None else frame,
                    "stage": stage,
                    "xy": g_worlds[stage, :, :2].detach().cpu().numpy(),
                    "native": native,
                })
        return out

    def match_grid(self, g_worlds: torch.Tensor,
                   frame: Optional[int] = None) -> List[torch.Tensor]:
        """GPU uniform-grid matcher, proven ball_query selection semantics.

        Static CSR grid (bin, native) stable-sorted.  Per query the full 3x3
        neighbor cells form the candidate superset; the grid does NOT prune by
        distance or tighten the 0.11 margin.  Radius judgment and selection
        happen on GPU in FP32: among in-radius references choose the smallest
        native RoadGT index (argmin native, never argmin distance).  Duplicate
        XY retained; empty/out-of-bounds cells yield native=-1; coarse/fine
        query independently.  Residuals stay FP32.
        """
        if self.grid is None:
            raise RuntimeError("grid index not built")
        torch.cuda.synchronize(self.device)
        _t0 = time.perf_counter()
        G = self.grid
        h = self.GRID_H
        radius2 = self.GRID_R * self.GRID_R
        stages, count, _ = g_worlds.shape
        kcap = G["kcap"]
        nc = G["nc"]
        offs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        out: List[torch.Tensor] = []
        for stage in range(stages):
            q = g_worlds[stage, :, :2]
            qc = torch.floor((q - G["origin"]) / h).long()
            Q = q.shape[0]
            cn = torch.full((Q, 9, kcap), -1, dtype=torch.int64,
                            device=self.device)
            cxy = torch.zeros((Q, 9, kcap, 2), dtype=torch.float32,
                              device=self.device)
            for k, (dx, dy) in enumerate(offs):
                cc = qc[:, 0] + dx
                rr = qc[:, 1] + dy
                valid = (cc >= 0) & (cc < nc[0]) & (rr >= 0) & (rr < nc[1])
                c = cc * nc[1] + rr
                pos = torch.searchsorted(G["uniq"], c).clamp(
                    max=G["uniq"].shape[0] - 1)
                found = valid & (G["uniq"][pos] == c)
                st = G["start"][pos]
                cn2 = G["cnt"][pos]
                j = torch.arange(kcap, device=self.device)
                inb = j[None, :] < cn2[:, None]
                g = (st[:, None] + j[None, :]).clamp(
                    max=G["order"].shape[0] - 1)
                nat = G["order"][g]
                xy = G["sorted_xy"][g]
                cn[:, k] = torch.where(found[:, None] & inb, nat,
                                       torch.full_like(nat, -1))
                cxy[:, k] = torch.where(
                    found[:, None, None] & inb[:, :, None], xy,
                    torch.zeros_like(xy))
            cnr = cn.reshape(Q, -1)
            cyr = cxy.reshape(Q, -1, 2)
            d2 = (((cyr - G["origin"]) - (q[:, None, :] - G["origin"])) ** 2
                  ).sum(-1)
            inr = (cnr >= 0) & (d2 <= radius2)
            big = torch.iinfo(torch.int64).max
            nm = torch.where(inr, cnr, torch.full_like(cnr, big))
            sel = nm.argmin(dim=1)
            chosen = torch.gather(cnr, 1, sel[:, None]).squeeze(1)
            no_hit = torch.gather(nm, 1, sel[:, None]).squeeze(1) == big
            chosen = torch.where(no_hit, torch.full_like(chosen, -1), chosen)
            torch.cuda.synchronize(self.device)
            self.match_ms.append((time.perf_counter() - _t0) * 1e3)
            valid = chosen >= 0
            gt_z = self.z[chosen[valid]]
            pred_z = g_worlds[stage, :, 2]
            out.append((pred_z[valid] - gt_z))
            if self._want_dump(frame):
                self.dump_records.append({
                    "frame": -1 if frame is None else frame,
                    "stage": stage,
                    "xy": g_worlds[stage, :, :2].detach().cpu().numpy(),
                    "native": chosen.detach().cpu().numpy(),
                })
            del cn, cxy, cnr, cyr, d2, inr, nm
        return out

    def match_cuda(self, g_worlds: torch.Tensor,
                   frame: Optional[int] = None) -> List[torch.Tensor]:
        """Direct CUDA CSR matcher; same selection semantics as match_grid.

        Per stage the kernel scans the 3x3 neighbor cells on the static CSR
        index and writes the smallest in-radius native index.  No candidate
        tensor / mask / gather is materialized on the torch side.  Coarse and
        fine query their own world XY independently; residuals stay FP32.
        """
        if self.cuda_index is None:
            raise RuntimeError("cuda grid index not built")
        stages, count, _ = g_worlds.shape
        out: List[torch.Tensor] = []
        for stage in range(stages):
            q = g_worlds[stage, :, :2]
            torch.cuda.synchronize(self.device)
            _t0 = time.perf_counter()
            chosen = self.cuda_index.match(q)
            torch.cuda.synchronize(self.device)
            self.match_ms.append((time.perf_counter() - _t0) * 1e3)
            valid = chosen >= 0
            gt_z = self.z[chosen[valid]]
            pred_z = g_worlds[stage, :, 2]
            out.append((pred_z[valid] - gt_z))
            if self._want_dump(frame):
                self.dump_records.append({
                    "frame": -1 if frame is None else frame,
                    "stage": stage,
                    "xy": g_worlds[stage, :, :2].detach().cpu().numpy(),
                    "native": chosen.detach().cpu().numpy(),
                    "pred_z": pred_z.detach().cpu().numpy(),
                })
        return out


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
    ap.add_argument("--forward-dtype", choices=("fp32", "fp16", "fp16_mixed", "bf16"), default="fp32",
                    help="fp16=full autocast; fp16_mixed=fp16 backbone with fp32 elevation/refiner head; "
                         "bf16=full autocast bfloat16; metrics remain fp32")
    ap.add_argument("--fast", action="store_true",
                    help="precision-preserving eval: skip unused GT/Gaussian work, "
                         "batched coarse/fine matching, no per-frame empty_cache")
    ap.add_argument("--matcher", choices=("ball", "ball_tiled", "ball_tiled_opt", "ball_tree_opt", "ball_grid_gpu", "ball_grid_cuda"),
                    default="ball_tiled")
    ap.add_argument("--tile-size", type=int, default=128)
    ap.add_argument("--dump-native-dir", default=None,
                    help="debug: dump real ball_query chosen native index + XY per frame")
    ap.add_argument("--dump-native-frames", default=None,
                    help="comma-separated pre-registered frame indices to dump; empty = all frames")
    ap.add_argument("--outer-loops", type=int, choices=(0, 1, 2, 3, 4, 5), default=0,
                    help="number of z_fine -> next z_coarse feedback loops; 0 keeps original fine")
    ap.add_argument("--loop-residual", choices=("f_minus_c", "c_minus_f", "abs"),
                    default="f_minus_c", help="feedback residual policy for loop 1 and above")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--cuda-profiler", action="store_true",
                    help="diagnostic: torch.profiler CUDA/CPU kernel breakdown for first N frames")
    ap.add_argument("--prof-out", default=None)
    ap.add_argument("--split-timing", action="store_true",
                    help="diagnostic: time backbone vs rogs_head separately")
    ap.add_argument("--stage-trace-out", default=None,
                    help="diagnostic: host perf_counter milestone breakdown JSON")
    ap.add_argument("--channels-last", action="store_true",
                    help="approx: conv memory_format channels_last")
    ap.add_argument("--micro-batch", action="store_true",
                    help="standalone backbone image-batch microbenchmark; no E2E")
    ap.add_argument("--micro-out", default=None)
    ap.add_argument("--lift-opt", action="store_true",
                    help="strict-equivalent stereo micro-opt subclass")
    ap.add_argument("--corr-einsum", action="store_true",
                    help="with --lift-opt: grouped correlation via einsum")
    ap.add_argument("--no-A", action="store_true",
                    help="with --lift-opt: keep sequential camera matmuls")
    ap.add_argument("--no-ref-only", action="store_true",
                    help="with --lift-opt: transform all sweeps' mu/sigma")
    ap.add_argument("--cuda-corr", action="store_true",
                    help="with --lift-opt: fused grouped-correlation CUDA kernel")
    ap.add_argument("--stereo-probe", action="store_true",
                    help="deep stereo cost-volume timing/shape probe")
    ap.add_argument("--stereo-out", default=None)
    ap.add_argument("--schedule-probe", action="store_true",
                    help="diagnostic: per-frame fetch/H2D/forward/match/stats timing with CUDA events; no per-phase synchronize")
    ap.add_argument("--schedule-out", default=None)
    ap.add_argument("--depth-merge", action="store_true",
                    help="candidate: merge the two per-sweep DepthNet calls into one B=2,Nc=6 batch (A/B probe)")
    ap.add_argument("--img-cache", action="store_true",
                    help="candidate: bounded GPU LRU of image-only backbone+neck packs across adjacent frames (A/B probe)")
    ap.add_argument("--trt-engine", default=None,
                    help="candidate A/B: replace ResNet50+neck with serialized FP32 TRT "
                         "engine; DepthNet/stereo/head/matcher stay on vendor path")
    ap.add_argument("--dn-trt-engine", default=None,
                    help="candidate A/B: replace DepthNet pre-DCN trunk+context "
                         "branch with serialized FP32 TRT engine; DCN/mono_depth/"
                         "mu_sigma heads, stereo/head/matcher stay vendor path")
    ap.add_argument("--cuda-graph-backbone", action="store_true",
                    help="candidate A/B: capture vendor FP32 ResNet50+neck in a CUDA "
                         "Graph on a fixed batch-6 buffer; DepthNet/stereo/head/matcher "
                         "stay on vendor path")
    ap.add_argument("--reuse-audit", action="store_true",
                    help="audit-only: record per-sweep/per-ref backbone, depth_net, stereo inputs/outputs and compare sweep0 vs sweep1")
    ap.add_argument("--reuse-out", default=None)
    ap.add_argument("--probe-frames", type=int, default=5)
    ap.add_argument("--homo-dump-dir", default=None,
                    help="debug: dump real homo inputs/outputs per call (vendor reference)")
    ap.add_argument("--homo-fuse", action="store_true",
                    help="fast candidate: replace homo projection chain with homo_fuse kernel")
    ap.add_argument("--homo-real-out", default=None,
                    help="JSON path for homo real-dump metadata")
    ap.add_argument("--homo-fallback", action="store_true",
                    help="fast candidate: kernel + vendor-order fallback for near-zero points")
    ap.add_argument("--homo-clean", action="store_true",
                    help="single-path timing only: no dual vendor/fused live comparison")
    ap.add_argument("--tf32", action="store_true",
                    help="enable TF32 matmul/cudnn (low-risk FP32 switch)")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="enable cudnn.benchmark autotuner")
    args = ap.parse_args()

    _trace: Dict[str, float] = {}
    if args.stage_trace_out:
        _trace["main_entry"] = time.perf_counter()

    if not math.isclose(args.fine_size * args.cell_m, 102.4,
                        rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("fine_size * cell_m must preserve the 102.4 m BEV extent")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if args.forward_dtype in ("fp16", "fp16_mixed", "bf16") and device.type != "cuda":
        raise ValueError("half/bf16 forward evaluation requires a CUDA device")

    Exp = load_experiment_class()
    lightning = Exp(gpus=0, batch_size_per_device=1, data_root=args.data_root,
                    default_root_dir=str(BRED_ROOT / "results" / "stage1" / "lightning_eval"))
    official_sd = torch.load(args.official_checkpoint, map_location="cpu")["state_dict"]
    official_sd.pop("model.backbone.frustum", None)
    lightning.load_state_dict(official_sd, strict=False)
    if args.stage_trace_out:
        _trace["official_ckpt"] = time.perf_counter()
    if args.lift_opt:
        from lift_opt import BEVStereoOpt

        class LiftOpt(ContinuousLiftFPN, BEVStereoOpt):
            def __init__(self, *a, **k):
                k["use_einsum_corr"] = args.corr_einsum
                k["use_A"] = not args.no_A
                k["ref_only"] = not args.no_ref_only
                k["use_cuda_corr"] = args.cuda_corr
                super().__init__(*a, **k)

        lift = LiftOpt(**lightning.backbone_conf)
    else:
        lift = ContinuousLiftFPN(**lightning.backbone_conf)
    lift.fp32_splat = args.forward_dtype == "fp16_mixed"
    lift.load_official_backbone(lightning.model.backbone)
    lightning.model.backbone = lift
    model = BEVStereoRoGSModel(lightning.model, lift_backbone=lift,
                               fine_size=args.fine_size, geometry_only=True,
                               freeze_official_bn=False).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.stage_trace_out:
        _trace["model_built"] = time.perf_counter()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if args.stage_trace_out:
        _trace["v8a17_ckpt"] = time.perf_counter()
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
    if args.lift_opt:
        lift._opt_cache = {}
        lift._frustum_grid = {}

        def _clear_opt_cache(_mod, _inp):
            lift._opt_cache.clear()

        model.backbone.register_forward_pre_hook(_clear_opt_cache)
    if args.depth_merge:
        from lift_opt import install_depth_merge
        install_depth_merge(lift)
        print("DEPTH_MERGE installed: two sweep DepthNet calls batched B=2,Nc=6",
              flush=True)
    if args.img_cache:
        from lift_opt import install_img_cache
        install_img_cache(lift, capacity=4)
        print("IMG_CACHE installed: bounded GPU LRU of backbone+neck packs (cap=4)",
              flush=True)
    if args.trt_engine:
        sys.path.insert(0, str(BRED_ROOT / "bench_opt" / "probe" / "trt_graph"))
        from trt_backbone import install_trt_backbone
        _trt_runner = install_trt_backbone(lift, args.trt_engine, device,
                                            rogs_model=model)
        print("TRT_BACKBONE installed engine=%s load_s=%.3f"
              % (args.trt_engine, _trt_runner.load_s), flush=True)
    if args.dn_trt_engine:
        sys.path.insert(0, str(BRED_ROOT / "bench_opt" / "probe" / "dn_trt"))
        from dn_trt_install import install_dn_trt
        _dn_runner = install_dn_trt(lift, args.dn_trt_engine, device)
        print("DN_TRT installed engine=%s load_s=%.3f"
              % (args.dn_trt_engine, _dn_runner.load_s), flush=True)
    if args.cuda_graph_backbone:
        sys.path.insert(0, str(BRED_ROOT / "bench_opt" / "probe" / "graph_bb"))
        from graph_backbone import install_graph_backbone
        _graph_bb = install_graph_backbone(lift, device, rogs_model=model)
        print("CUDA_GRAPH_BACKBONE installed capture_s=%.3f"
              % _graph_bb.capture_s, flush=True)
    model.eval()
    if args.channels_last:
        for _m in model.modules():
            if isinstance(_m, nn.Conv2d):
                _m.weight.data = _m.weight.data.to(
                    memory_format=torch.channels_last)
    for param in model.parameters():
        param.requires_grad_(False)

    # --- reuse audit: per-sweep/per-ref backbone / depth_net / stereo inputs/outputs ---
    reuse_audit = args.reuse_audit and device.type == "cuda"
    reuse_frames: List[dict] = []
    if reuse_audit:
        def _rsummary(t):
            if not torch.is_tensor(t):
                return None
            tc = t.detach().float().cpu()
            return {"shape": list(t.shape), "dtype": str(t.dtype),
                    "device": str(t.device), "numel": t.numel(),
                    "mean": round(tc.mean().item(), 6),
                    "absmax": round(tc.abs().max().item(), 6)}

        def _rcmp(a, b):
            if not (torch.is_tensor(a) and torch.is_tensor(b)):
                return {"equal": None, "note": "non-tensor"}
            if a.shape != b.shape:
                return {"equal": False, "shape_a": list(a.shape),
                        "shape_b": list(b.shape)}
            ac, bc = a.detach().cpu(), b.detach().cpu()
            eq = torch.equal(ac, bc)
            d = (ac.float() - bc.float()).abs()
            return {"equal": bool(eq), "shape": list(a.shape),
                    "maxdiff": round(d.max().item(), 8),
                    "meandiff": round(d.mean().item(), 8)}

        def _clone_cpu(t):
            return t.detach().cpu().clone() if torch.is_tensor(t) else t

        reuse_state = {"cam": [], "depth": [], "stereo": []}
        _orig_cam = lift.get_cam_feats

        def _cam_wrap(imgs):
            img_feats, stereo_feats = _orig_cam(imgs)
            reuse_state["cam"].append({
                "input_imgs": _rsummary(imgs),
                "out_img_feats": _rsummary(img_feats),
                "out_stereo": _rsummary(stereo_feats),
                "_img_cpu": _clone_cpu(imgs),
                "_imgfeat_cpu": _clone_cpu(img_feats),
                "_stereo_cpu": _clone_cpu(stereo_feats)})
            return img_feats, stereo_feats

        _orig_dn = lift.depth_net.forward

        def _dn_wrap(feat, mats_dict):
            out = tuple(_orig_dn(feat, mats_dict))
            mats_snap = {}
            for mk, mv in mats_dict.items():
                if torch.is_tensor(mv):
                    mats_snap[mk] = {"summary": _rsummary(mv),
                                     "_cpu": _clone_cpu(mv)}
                else:
                    mats_snap[mk] = {"summary": str(type(mv))}
            reuse_state["depth"].append({
                "input_feat": _rsummary(feat),
                "mats": mats_snap,
                "out": [_rsummary(o) for o in out],
                "_feat_cpu": _clone_cpu(feat),
                "_mats_cpu": {k: v.get("_cpu") for k, v in mats_snap.items()},
                "_out_cpu": [_clone_cpu(o) for o in out]})
            return out

        _orig_st = lift._forward_stereo

        def _st_wrap(ref_idx, stereo_feats_all_sweeps, mono_depth_all_sweeps,
                     mats_dict, sensor2sensor_mats, mu_all_sweeps,
                     sigma_all_sweeps, range_score_all_sweeps,
                     depth_feat_all_sweeps):
            out = _orig_st(ref_idx, stereo_feats_all_sweeps,
                           mono_depth_all_sweeps, mats_dict,
                           sensor2sensor_mats, mu_all_sweeps,
                           sigma_all_sweeps, range_score_all_sweeps,
                           depth_feat_all_sweeps)
            stereo_depth = out[0] if isinstance(out, tuple) else out
            reuse_state["stereo"].append({
                "ref_idx": ref_idx,
                "stereo_feats": [_rsummary(x) for x in stereo_feats_all_sweeps],
                "mono_depth": [_rsummary(x) for x in mono_depth_all_sweeps],
                "mu": [_rsummary(x) for x in mu_all_sweeps],
                "sigma": [_rsummary(x) for x in sigma_all_sweeps],
                "range_score": [_rsummary(x) for x in range_score_all_sweeps],
                "depth_feat": [_rsummary(x) for x in depth_feat_all_sweeps],
                "sensor2sensor": [_rsummary(x) for x in sensor2sensor_mats],
                "out_stereo_depth": _rsummary(stereo_depth),
                "_stereo_cpu": [_clone_cpu(x) for x in stereo_feats_all_sweeps],
                "_mono_cpu": [_clone_cpu(x) for x in mono_depth_all_sweeps],
                "_mu_cpu": [_clone_cpu(x) for x in mu_all_sweeps],
                "_sigma_cpu": [_clone_cpu(x) for x in sigma_all_sweeps],
                "_range_cpu": [_clone_cpu(x) for x in range_score_all_sweeps],
                "_df_cpu": [_clone_cpu(x) for x in depth_feat_all_sweeps],
                "_s2s_cpu": [_clone_cpu(x) for x in sensor2sensor_mats],
                "_out_cpu": _clone_cpu(stereo_depth)})
            return out

        lift.get_cam_feats = _cam_wrap
        lift.depth_net.forward = _dn_wrap
        lift._forward_stereo = _st_wrap

    # --- real homo dump / fused fast-candidate hook (independent bench_opt) ---
    _homo_ctx = {"frame": -1, "token": None, "calls": 0, "records": []}
    _homo_stats = {"per_call": [], "vendor_ms": [], "fused_ms": [],
                    "fallback_ms": [], "near_count": [], "near_prop": [],
                    "mask_disagree_calls": 0}
    _homo_dump_dir = args.homo_dump_dir
    if _homo_dump_dir:
        os.makedirs(_homo_dump_dir, exist_ok=True)
    _homo_mode = "fallback" if args.homo_fallback else (
        "fuse" if args.homo_fuse else "vendor")
    if _homo_dump_dir or args.homo_fuse or args.homo_fallback or args.homo_clean:
        _orig_homo = model.backbone.homo_warping
        _dsf = getattr(model.backbone, "stereo_downsample_factor", 8)

        def _homo_real_wrap(feat, K_ref, K_src, s2s, ida_ref, ida_src,
                             depth, frustum):
            from homo_ext.homo_lib import (vendor_project, fused_project,
                                            fallback_project, compare_call)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            if _homo_mode == "fuse":
                out = fused_project(feat, K_ref, K_src, s2s, ida_ref,
                                    ida_src, depth, frustum, dsf=_dsf)
                warped_out = out["warped"]
            elif _homo_mode == "fallback":
                out = fallback_project(feat, K_ref, K_src, s2s, ida_ref,
                                        ida_src, depth, frustum, dsf=_dsf)
                warped_out = out["warped"]
            else:
                out = None
                warped_out = _orig_homo(feat, K_ref, K_src, s2s, ida_ref,
                                        ida_src, depth, frustum)
            torch.cuda.synchronize(device)
            backend_ms = (time.perf_counter() - t0) * 1e3

            if _homo_dump_dir and not args.homo_clean:
                vref = vendor_project(feat, K_ref, K_src, s2s, ida_ref,
                                       ida_src, depth, frustum, dsf=_dsf)
                candidate = out if out is not None else vref
                cmp = compare_call(vref, candidate)
                cmp["frame"] = _homo_ctx["frame"]
                cmp["call"] = _homo_ctx["calls"]
                cmp["token"] = _homo_ctx["token"]
                cmp["mode"] = _homo_mode
                cmp["backend_ms"] = backend_ms
                if _homo_mode == "fallback":
                    cmp["near_count"] = out["near_count"]
                    cmp["near_prop"] = out["near_prop"]
                cmp["orig_vs_vref_warped_max_abs"] = float(
                    (warped_out - vref["warped"]).abs().max())
                _homo_stats["per_call"].append(cmp)
                if not cmp["mask_agree"]:
                    _homo_stats["mask_disagree_calls"] += 1
                if _homo_ctx["calls"] < 2:
                    torch.save({
                        "K_ref": K_ref.detach().cpu().contiguous(),
                        "K_src": K_src.detach().cpu().contiguous(),
                        "s2s": s2s.detach().cpu().contiguous(),
                        "ida_ref": ida_ref.detach().cpu().contiguous(),
                        "ida_src": ida_src.detach().cpu().contiguous(),
                        "depth": depth.detach().cpu().contiguous(),
                        "frustum": frustum.detach().cpu().contiguous(),
                        "grid_ref": vref["grid"].detach().cpu().contiguous(),
                        "grid_fuse": candidate["grid"].detach().cpu().contiguous(),
                        "points_ref": vref["points"].detach().cpu().contiguous(),
                        "points_fuse": candidate["points"].detach().cpu().contiguous(),
                        "mask_ref": vref["neg_mask"].detach().cpu().contiguous(),
                        "mask_fuse": candidate["neg_mask"].detach().cpu().contiguous(),
                        "z_cam": vref["z_cam"].detach().cpu().contiguous(),
                    }, os.path.join(
                        _homo_dump_dir,
                        f"sample_f{_homo_ctx['frame']:03d}_c{_homo_ctx['calls']:03d}.pt"))
                _homo_ctx["records"].append({
                    "frame": _homo_ctx["frame"],
                    "token": _homo_ctx["token"],
                    "call": _homo_ctx["calls"],
                    "mode": _homo_mode,
                    "shape": {"feat": list(feat.shape),
                              "depth": list(depth.shape),
                              "frustum": list(frustum.shape)}})
                _homo_ctx["calls"] += 1
            elif args.homo_clean:
                if _homo_mode == "fuse":
                    _homo_stats["fused_ms"].append(backend_ms)
                elif _homo_mode == "fallback":
                    _homo_stats["fallback_ms"].append(backend_ms)
                    _homo_stats["near_count"].append(out["near_count"])
                    _homo_stats["near_prop"].append(out["near_prop"])
                else:
                    _homo_stats["vendor_ms"].append(backend_ms)
            return warped_out

        model.backbone.homo_warping = _homo_real_wrap

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

    if args.micro_batch:
        batch = next(iter(loader))
        imgs0 = batch[0]
        B, S, Nc, C, H, W = imgs0.shape
        base_M = B * S * Nc
        bb = model.backbone.img_backbone
        neck = model.backbone.img_neck
        torch.backends.cudnn.benchmark = False
        micro: Dict[str, object] = {
            "shape": {"B": B, "S": S, "Nc": Nc, "C": C, "H": H, "W": W,
                      "base_M": base_M},
            "dtype": "fp32", "checkpoint": args.checkpoint,
            "cudnn_benchmark": False, "runs": {}}
        for M in (12, 24, 48, 72, 96):
            x = torch.randn(M, C, H, W, device=device)
            try:
                for _ in range(2):
                    feats = bb(x)
                    _ = neck(feats)[0]
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                ts = []
                for _ in range(5):
                    torch.cuda.synchronize(device)
                    t = time.perf_counter()
                    feats = bb(x)
                    _ = neck(feats)[0]
                    torch.cuda.synchronize(device)
                    ts.append(time.perf_counter() - t)
                micro["runs"][str(M)] = {
                    "ok": True,
                    "mean_ms": round(statistics.mean(ts) * 1e3, 2),
                    "median_ms": round(statistics.median(ts) * 1e3, 2),
                    "std_ms": round(statistics.stdev(ts) * 1e3, 2) if len(ts) > 1 else 0.0,
                    "images_per_s": round(M / statistics.mean(ts), 1),
                    "peak_alloc_mib": round(torch.cuda.max_memory_allocated(device) / 1048576, 1),
                    "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1048576, 1),
                }
            except torch.cuda.OutOfMemoryError as e:
                micro["runs"][str(M)] = {"ok": False, "oom": str(e)[:200]}
                torch.cuda.empty_cache()
        out = args.micro_out or str(BRED_ROOT / "bench_opt" / "probe" / "micro_batch.json")
        json.dump(micro, open(out, "w"), indent=2)
        print("MICRO " + json.dumps(micro), flush=True)
        return

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
    if args.stage_trace_out:
        _trace["data_meta"] = time.perf_counter()

    split_t: Dict[str, List[float]] = {
        "backbone": [], "head": [], "resnet": [], "neck": [],
        "depth": [], "splat": [], "stereo": [], "homo": [], "simnet": []}
    if args.split_timing and device.type == "cuda":
        _t0: Dict[str, float] = {}

        def _make_hook(part: str):
            def _pre(mod, _inp):
                torch.cuda.synchronize(device)
                _t0[part] = time.perf_counter()

            def _post(mod, _inp, _out):
                torch.cuda.synchronize(device)
                split_t[part].append(time.perf_counter() - _t0[part])

            return _pre, _post

        for part, mod in (("backbone", model.backbone),
                          ("head", model.rogs_head)):
            pre, post = _make_hook(part)
            mod.register_forward_pre_hook(pre)
            mod.register_forward_hook(post)
        for part, attr in (("resnet", "img_backbone"), ("neck", "img_neck"),
                           ("depth", "depth_net"), ("simnet", "similarity_net")):
            if hasattr(model.backbone, attr):
                pre, post = _make_hook(part)
                getattr(model.backbone, attr).register_forward_pre_hook(pre)
                getattr(model.backbone, attr).register_forward_hook(post)
        if hasattr(model.backbone, "_splat_continuous"):
            _orig_splat = model.backbone._splat_continuous

            def _splat_wrap(*a, **k):
                torch.cuda.synchronize(device)
                t = time.perf_counter()
                r = _orig_splat(*a, **k)
                torch.cuda.synchronize(device)
                split_t["splat"].append(time.perf_counter() - t)
                return r

            model.backbone._splat_continuous = _splat_wrap
        if hasattr(model.backbone, "_forward_stereo"):
            _orig_stereo = model.backbone._forward_stereo

            def _stereo_wrap(*a, **k):
                torch.cuda.synchronize(device)
                t = time.perf_counter()
                r = _orig_stereo(*a, **k)
                torch.cuda.synchronize(device)
                split_t["stereo"].append(time.perf_counter() - t)
                return r

            model.backbone._forward_stereo = _stereo_wrap
        if hasattr(model.backbone, "homo_warping"):
            _orig_homo = model.backbone.homo_warping

            def _homo_wrap(*a, **k):
                torch.cuda.synchronize(device)
                t = time.perf_counter()
                r = _orig_homo(*a, **k)
                torch.cuda.synchronize(device)
                split_t["homo"].append(time.perf_counter() - t)
                return r

            model.backbone.homo_warping = _homo_wrap

    stereo_rec: Dict[str, List[float]] = {
        "stereo": [], "cost": [], "homo": [], "grid": [], "frustum": [],
        "mask": [], "simnet": []}
    stereo_shapes: Dict[str, dict] = {}
    stereo_counts: Dict[str, int] = {}
    if args.stereo_probe and device.type == "cuda":
        import torch.nn.functional as F

        def _sh(t):
            return list(t.shape) if torch.is_tensor(t) else None

        def _wrap_method(part: str, obj, name: str, capture=None):
            orig = getattr(obj, name)

            def wrapped(*a, **k):
                torch.cuda.synchronize(device)
                t = time.perf_counter()
                r = orig(*a, **k)
                torch.cuda.synchronize(device)
                stereo_rec[part].append(time.perf_counter() - t)
                stereo_counts[part] = stereo_counts.get(part, 0) + 1
                if capture and part not in stereo_shapes:
                    try:
                        stereo_shapes[part] = capture(a, r)
                    except Exception:
                        pass
                return r

            setattr(obj, name, wrapped)

        lift = model.backbone
        _wrap_method("stereo", lift, "_forward_stereo",
                     lambda a, r: {"out": _sh(r[0] if isinstance(r, tuple) else r)})
        _wrap_method("cost", lift, "_generate_cost_volume",
                     lambda a, r: {"depth_sample": _sh(a[3]), "out": _sh(r)})
        _wrap_method("frustum", lift, "create_depth_sample_frustum",
                     lambda a, r: {"depth_sample": _sh(a[0]), "out": _sh(r)})
        _wrap_method("mask", lift, "_forward_mask",
                     lambda a, r: {"out": _sh(r)})
        _wrap_method("homo", lift, "homo_warping",
                     lambda a, r: {"feat": _sh(a[0]), "depth": _sh(a[6]),
                                  "out": _sh(r)})
        _wrap_method("simnet", lift.similarity_net, "forward",
                      lambda a, r: {"in": _sh(a[0]), "out": _sh(r)})

        _orig_gs = F.grid_sample

        def _gs_wrap(*a, **k):
            torch.cuda.synchronize(device)
            t = time.perf_counter()
            r = _orig_gs(*a, **k)
            torch.cuda.synchronize(device)
            stereo_rec["grid"].append(time.perf_counter() - t)
            stereo_counts["grid"] = stereo_counts.get("grid", 0) + 1
            if "grid" not in stereo_shapes:
                stereo_shapes["grid"] = {"input": _sh(a[0]), "grid": _sh(a[1]),
                                         "out": _sh(r)}
            return r

        F.grid_sample = _gs_wrap

    gt_bank = None
    tile_indexes = None
    _dump_frames = None
    if args.dump_native_frames:
        _dump_frames = {int(x) for x in args.dump_native_frames.split(",") if x.strip()}
    if args.matcher in ("ball_tiled_opt", "ball_tree_opt", "ball_grid_gpu", "ball_grid_cuda"):
        gt_bank = GTBank(native_xyz_global, device, args.dump_native_dir,
                         use_tree=args.matcher == "ball_tree_opt",
                         use_grid=args.matcher == "ball_grid_gpu",
                         use_cuda=args.matcher == "ball_grid_cuda",
                         dump_frames=_dump_frames)
        tile_indexes = GTBank.precompute_tiles(
            args.fine_size, args.tile_size, device)
        print(f"GT_BANK points={gt_bank.xy.shape[0]} spatial_tiles={len(tile_indexes)}",
              flush=True)
        if args.stage_trace_out:
            _trace["ext_loaded"] = time.perf_counter()

    fine_names = (["fine"] if args.outer_loops == 0 else
                  ["fine_no_loop"] +
                  [f"fine_loop{k}" for k in range(1, args.outer_loops + 1)])
    stage_names = ["coarse"] + fine_names
    stats = {name: StageStats(name) for name in stage_names}
    frames = 0
    sched_probe = args.schedule_probe and device.type == "cuda"
    sched_frames: List[dict] = []
    _sched_ev: list = []
    _sched_prev_end: Optional[float] = None
    _sched_t0 = time.perf_counter()
    eval_start = time.perf_counter()
    if args.stage_trace_out:
        _trace["eval_start"] = eval_start
    _prof = None
    if args.cuda_profiler and device.type == "cuda":
        from torch.profiler import profile, ProfilerActivity
        _prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=False)
        _prof.__enter__()
    grad_context = torch.inference_mode() if args.fast else torch.no_grad()
    with grad_context:
        for i, batch in enumerate(loader):
            if args.max_frames and i >= args.max_frames:
                break
            _t_body = time.perf_counter()
            _t_fetch_gap = (None if _sched_prev_end is None
                            else _t_body - _sched_prev_end)
            imgs = batch[0].to(device, non_blocking=args.fast)
            mats = {k: v.to(device, non_blocking=args.fast)
                    for k, v in batch[1].items()}
            timestamps = batch[2].to(device, non_blocking=args.fast)
            _t_h2d = time.perf_counter()
            meta = batch[3][0]
            _homo_ctx["frame"] = i
            _homo_ctx["token"] = meta.get("token") if isinstance(meta, dict) else None
            if args.fast:
                base_xy = base_xy_fast
            else:
                target = make_fine_bev_target(
                    provider, meta, mats["bda_mat"][0], device,
                    fine_size=args.fine_size, cell=args.cell_m,
                    z_surround=args.z_surround)
                base_xy = target["base_xy"]
            if reuse_audit:
                reuse_state["cam"].clear()
                reuse_state["depth"].clear()
                reuse_state["stereo"].clear()
            if args.profile and device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_start = time.perf_counter()
            _ev0 = torch.cuda.Event(enable_timing=True) if sched_probe else None
            _ev1 = torch.cuda.Event(enable_timing=True) if sched_probe else None
            if _ev0 is not None:
                _ev0.record()
            _acast_dtype = {"fp16": torch.float16, "fp16_mixed": torch.float16,
                            "bf16": torch.bfloat16}.get(args.forward_dtype)
            forward_context = (torch.autocast(device_type="cuda", dtype=_acast_dtype)
                               if _acast_dtype is not None else nullcontext())
            with forward_context:
                initial, diag = model(imgs, mats, timestamps, return_depth=True,
                                      refiner_use_noise=True,
                                      refiner_outer_loops=args.outer_loops,
                                      refiner_outer_seed=args.seed + i * 1009,
                                      head_fp32=args.forward_dtype == "fp16_mixed",
                                      refiner_loop_residual=args.loop_residual)
            if _ev1 is not None:
                _ev1.record()
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
            if reuse_audit:
                _rec = reuse_state
                _cam_cmp = {"n_calls": len(_rec["cam"])}
                if len(_rec["cam"]) >= 2:
                    _ca, _cb = _rec["cam"][0], _rec["cam"][1]
                    _cam_cmp.update({
                        "input_imgs_sweep0_vs_sweep1": _rcmp(_ca["_img_cpu"], _cb["_img_cpu"]),
                        "out_img_feats": _rcmp(_ca["_imgfeat_cpu"], _cb["_imgfeat_cpu"]),
                        "out_stereo_feats": _rcmp(_ca["_stereo_cpu"], _cb["_stereo_cpu"])})
                _dep_cmp = {"n_calls": len(_rec["depth"])}
                if len(_rec["depth"]) >= 2:
                    _da, _db = _rec["depth"][0], _rec["depth"][1]
                    _dep_cmp["input_feat"] = _rcmp(_da["_feat_cpu"], _db["_feat_cpu"])
                    _dep_cmp["mats"] = {}
                    _SWEEP_MATS = {"sensor2ego_mats", "intrin_mats", "ida_mats",
                                   "sensor2sensor_mats"}
                    for _mk in _da["_mats_cpu"]:
                        _ma, _mb = _da["_mats_cpu"][_mk], _db["_mats_cpu"][_mk]
                        if not torch.is_tensor(_ma):
                            _dep_cmp["mats"][_mk] = _rcmp(_ma, _mb)
                        elif _mk in _SWEEP_MATS:
                            # vendor DepthNet reads sweep0 slice for both calls
                            # (intrin/ida/sensor2ego [:, 0:1]); compare same slice.
                            _dep_cmp["mats"][_mk] = _rcmp(_ma[:, 0], _mb[:, 0])
                        else:
                            # bda_mat has no sweep dim; compare whole tensor.
                            _dep_cmp["mats"][_mk] = _rcmp(_ma, _mb)
                    _dep_cmp["out"] = [_rcmp(_o0, _o1) for _o0, _o1 in
                                       zip(_da["_out_cpu"], _db["_out_cpu"])]
                _st_cmp = {"n_calls": len(_rec["stereo"])}
                if len(_rec["stereo"]) >= 2:
                    _sa, _sb = _rec["stereo"][0], _rec["stereo"][1]
                    _st_cmp["ref_idx"] = {"a": _sa["ref_idx"], "b": _sb["ref_idx"],
                                          "equal": _sa["ref_idx"] == _sb["ref_idx"]}
                    _st_cmp["shared_stereo_feats_per_sweep"] = [
                        _rcmp(x, y) for x, y in zip(_sa["_stereo_cpu"], _sb["_stereo_cpu"])]
                    _st_cmp["shared_mono_per_sweep"] = [
                        _rcmp(x, y) for x, y in zip(_sa["_mono_cpu"], _sb["_mono_cpu"])]
                    _st_cmp["mu_ref0_vs_ref1"] = _rcmp(
                        _sa["_mu_cpu"][_sa["ref_idx"]], _sb["_mu_cpu"][_sb["ref_idx"]])
                    _st_cmp["mono_ref0_vs_ref1"] = _rcmp(
                        _sa["_mono_cpu"][_sa["ref_idx"]], _sb["_mono_cpu"][_sb["ref_idx"]])
                    _st_cmp["range_ref0_vs_ref1"] = _rcmp(
                        _sa["_range_cpu"][_sa["ref_idx"]], _sb["_range_cpu"][_sb["ref_idx"]])
                    _st_cmp["s2s_ref0_vs_ref1"] = _rcmp(
                        _sa["_s2s_cpu"][_sa["ref_idx"]], _sb["_s2s_cpu"][_sb["ref_idx"]])
                    _st_cmp["out_stereo_depth"] = _rcmp(_sa["_out_cpu"], _sb["_out_cpu"])
                reuse_frames.append({
                    "frame": i,
                    "token": meta.get("token") if isinstance(meta, dict) else None,
                    "cam": _cam_cmp, "depth": _dep_cmp, "stereo": _st_cmp,
                    "cam_summaries": [{"input_imgs": c["input_imgs"],
                                       "out_img_feats": c["out_img_feats"],
                                       "out_stereo": c["out_stereo"]} for c in _rec["cam"]],
                    "depth_summaries": [{"input_feat": d["input_feat"],
                                         "out": d["out"]} for d in _rec["depth"]],
                    "stereo_summaries": [{"ref_idx": s["ref_idx"],
                                          "out_stereo_depth": s["out_stereo_depth"]}
                                         for s in _rec["stereo"]]})
            coarse_initial = dict(initial)
            coarse_initial["z"] = diag["z_coarse"]
            _t_match_end = forward_end
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
                    elif args.matcher == "ball_tiled_opt":
                        errors = gt_bank.match(g_worlds, tile_indexes, frame=i)
                    elif args.matcher == "ball_tree_opt":
                        errors = gt_bank.match_tree(g_worlds, frame=i)
                    elif args.matcher == "ball_grid_gpu":
                        errors = gt_bank.match_grid(g_worlds, frame=i)
                    elif args.matcher == "ball_grid_cuda":
                        errors = gt_bank.match_cuda(g_worlds, frame=i)
                    else:
                        errors = matched_world_z_error_pair(
                            g_worlds, native_xyz_global, matcher=args.matcher)
                    _t_match_end = time.perf_counter()
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
                    _t_match_end = time.perf_counter()
                    stats[name].update(error, int(g_world.shape[0]))
                    del gaussian_xyz, g_world, error
                match_end = time.perf_counter()
            _t_stats_end = time.perf_counter()
            if sched_probe:
                sched_frames.append({
                    "frame": i,
                    "body_t_s": round(_t_body - _sched_t0, 6),
                    "fetch_gap_s": (round(_t_fetch_gap, 6)
                                    if _t_fetch_gap is not None else None),
                    "h2d_ms": round((_t_h2d - _t_body) * 1e3, 3),
                    "pre_fwd_ms": round((forward_start - _t_h2d) * 1e3, 3),
                    "fwd_host_ms": round((forward_end - forward_start) * 1e3, 3),
                    "match_ms": round((_t_match_end - forward_end) * 1e3, 3),
                    "stats_ms": round((_t_stats_end - _t_match_end) * 1e3, 3),
                    "frame_total_ms": round((_t_stats_end - _t_body) * 1e3, 3),
                    "stats_end_t_s": round(_t_stats_end - _sched_t0, 6),
                })
                _sched_ev.append((_ev0, _ev1))
            _sched_prev_end = _t_stats_end
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
    if args.stage_trace_out:
        _trace["eval_end"] = time.perf_counter()
    if _prof is not None:
        _prof.__exit__(None, None, None)
        evts = _prof.key_averages()
        kernels = []
        tot_cuda_us = 0.0
        n_cuda_ev = 0
        for e in evts:
            cuda_us = getattr(e, "self_device_time_total",
                              getattr(e, "device_time_total", 0)) or 0
            if e.device_type.name == "CUDA" or cuda_us > 0:
                kernels.append({
                    "name": e.key,
                    "count": e.count,
                    "cuda_time_total_us": cuda_us,
                    "cuda_time_avg_us": round(cuda_us / max(e.count, 1), 3),
                    "cpu_time_total_us": e.self_cpu_time_total,
                })
                tot_cuda_us += cuda_us
                n_cuda_ev += e.count
        kernels.sort(key=lambda x: -x["cuda_time_total_us"])
        prof_out = args.prof_out or str(BRED_ROOT / "bench_opt" / "probe" / "cuda_profiler.json")
        json.dump({"frames_profiled": min(args.probe_frames, frames),
                   "eval_elapsed_s": eval_elapsed,
                   "total_cuda_time_ms": round(tot_cuda_us / 1000, 3),
                   "cuda_kernel_events": n_cuda_ev,
                   "top": kernels[:40]},
                  open(prof_out, "w"), indent=2)
        print(f"CUDA_PROFILER cuda_ms={tot_cuda_us/1000:.1f} events={n_cuda_ev} "
              f"out={prof_out}", flush=True)
    if args.split_timing:
        for part in ("backbone", "head", "resnet", "neck", "depth", "splat",
                     "stereo", "homo", "simnet"):
            vals = split_t[part]
            if vals:
                print(f"SPLIT {part} n={len(vals)} mean_ms={statistics.mean(vals)*1e3:.1f} "
                      f"median_ms={statistics.median(vals)*1e3:.1f} total_s={sum(vals):.1f}",
                      flush=True)

    if args.stereo_probe:
        def _summ(vals):
            if not vals:
                return None
            return {"n": len(vals),
                    "mean_ms": round(statistics.mean(vals) * 1e3, 3),
                    "median_ms": round(statistics.median(vals) * 1e3, 3),
                    "max_ms": round(max(vals) * 1e3, 3),
                    "total_s": round(sum(vals), 3)}
        probe = {"frames": frames, "device": str(device),
                 "counts": stereo_counts, "shapes": stereo_shapes,
                 "timing": {k: _summ(v) for k, v in stereo_rec.items()}}
        outp = args.stereo_out or str(BRED_ROOT / "bench_opt" / "probe" /
                                      "stereo_probe.json")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        json.dump(probe, open(outp, "w"), indent=2)
        print("STEREO_PROBE " + json.dumps(probe["timing"]), flush=True)

    if sched_probe:
        torch.cuda.synchronize(device)
        for rec, (e0, e1) in zip(sched_frames, _sched_ev):
            if e0 is not None and e1 is not None:
                rec["fwd_ev_ms"] = round(e0.elapsed_time(e1), 3)

        def _sched_col(key):
            vals = [r[key] for r in sched_frames if r.get(key) is not None]
            if not vals:
                return None
            return {"n": len(vals),
                    "mean_ms": round(statistics.mean(vals), 3),
                    "median_ms": round(statistics.median(vals), 3),
                    "max_ms": round(max(vals), 3),
                    "total_ms": round(sum(vals), 3)}

        sched_agg = {k: _sched_col(k) for k in (
            "fetch_gap_s", "h2d_ms", "pre_fwd_ms", "fwd_host_ms", "fwd_ev_ms",
            "match_ms", "stats_ms", "frame_total_ms")}
        sched_record = {
            "frames": frames,
            "device": str(device),
            "loader": {"num_workers": args.workers,
                       "persistent_workers": bool(args.fast and args.workers > 0),
                       "prefetch_factor": 4 if (args.fast and args.workers > 0) else None,
                       "pin_memory": bool(args.fast),
                       "h2d_non_blocking": bool(args.fast),
                       "copy_stream": "default stream; no dedicated copy stream",
                       "shuffle": False, "batch_size": 1},
            "model": {"matcher": args.matcher, "fast": args.fast,
                      "forward_dtype": args.forward_dtype, "tf32": args.tf32,
                      "cudnn_benchmark": args.cudnn_benchmark},
            "eval_elapsed_s": eval_elapsed,
            "aggregate": sched_agg,
            "per_frame": sched_frames,
        }
        sched_out = args.schedule_out or str(BRED_ROOT / "bench_opt" / "probe" /
                                             "schedule_probe.json")
        os.makedirs(os.path.dirname(sched_out), exist_ok=True)
        json.dump(sched_record, open(sched_out, "w"), indent=2)
        print("SCHED_PROBE " + json.dumps(sched_agg), flush=True)

    if reuse_audit:
        reuse_chain = {
            "entry": "BEVStereoRoGSModel.forward -> backbone(ContinuousLiftFPN.forward) once per frame",
            "backbone_forward": [
                "for sweep_index in range(num_sweeps=2): get_cam_feats(imgs[:, sweep_index]) -> img_backbone(ResNet50) + img_neck(SECONDFPN)",
                "same sweep: depth_net(img_feats, mats_dict) -> depth_feat/context/mu/sigma/range_score/mono_depth",
                "for ref_idx in range(num_sweeps=2): _forward_stereo(ref_idx, stereo_feats_all_sweeps, mono_depth_all_sweeps, mats_dict, sensor2sensor_mats, mu/sigma/range/depth_feat)",
                "_forward_single_sweep per sweep; head (rogs_head) runs once on lifted BEV; coarse/fine z come from that single head forward, not two backbone calls"],
            "mapper": {
                "img_backbone/neck calls": "one per sweep (S=2), NOT coarse/fine stages",
                "DepthNet calls": "one per sweep (S=2)",
                "_forward_stereo calls": "one per ref sweep (S=2); each internally loops src sweeps for grouped correlation",
                "coarse/fine": "head outputs in one backbone+head forward; matcher is separate per stage"},
            "config_snapshot": {"matcher": args.matcher, "fast": args.fast,
                                "forward_dtype": args.forward_dtype, "tf32": args.tf32,
                                "cudnn_benchmark": args.cudnn_benchmark}}
        reuse_record = {"chain": reuse_chain, "frames": reuse_frames}
        reuse_out = args.reuse_out or str(BRED_ROOT / "bench_opt" / "probe" /
                                         "reuse_audit5.json")
        os.makedirs(os.path.dirname(reuse_out), exist_ok=True)
        json.dump(reuse_record, open(reuse_out, "w"), indent=2)
        print("REUSE_AUDIT frames=" + str(len(reuse_frames)), flush=True)

    if gt_bank is not None and gt_bank.dump_records:
        os.makedirs(gt_bank.dump_dir, exist_ok=True)
        for rec in gt_bank.dump_records:
            _extra = {"pred_z": rec["pred_z"]} if "pred_z" in rec else {}
            np.savez(
                os.path.join(gt_bank.dump_dir,
                             f"frame_{rec['frame']:04d}_stage{rec['stage']}.npz"),
                xy=rec["xy"], native=rec["native"], **_extra)

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
        "tf32": args.tf32,
        "cudnn_benchmark": args.cudnn_benchmark,
        "matcher": args.matcher if args.fast else "ball_chunked_independent",
        "tile_size": args.tile_size if args.matcher in ("ball_tiled", "ball_tiled_opt") else None,
        "eval_elapsed_s": eval_elapsed,
        "metric_protocol": "RoGS eval_z_metric: world Gaussian XY -> raw RoadGT world XY, ball_query K=1 radius=0.1m, world-z residual, unmatched Gaussians ignored",
        "aggregation": "residual-pooled over all matched Gaussians; per-frame mean is secondary",
        "native_road_points": len(native_xyz_global),
        "coarse": stats["coarse"].summary(),
    }
    for name in fine_names:
        result[name] = stats[name].summary()
    if gt_bank is not None and gt_bank.match_ms:
        _mm = gt_bank.match_ms
        _arr = np.asarray(_mm, dtype=np.float64)
        result["matcher_ms"] = {
            "n_calls": len(_mm),
            "mean_ms": round(statistics.mean(_mm), 4),
            "median_ms": round(statistics.median(_mm), 4),
            "p25_ms": round(float(np.percentile(_arr, 25)), 4),
            "p75_ms": round(float(np.percentile(_arr, 75)), 4),
            "p90_ms": round(float(np.percentile(_arr, 90)), 4),
            "max_ms": round(max(_mm), 4),
            "total_ms": round(sum(_mm), 4),
        }
    if gt_bank is not None and gt_bank.extension_load_ms is not None:
        result["extension_load_ms"] = round(gt_bank.extension_load_ms, 4)
    result["dump_native_frames"] = sorted(_dump_frames) if _dump_frames else None
    if getattr(lift, "_img_cache", None) is not None:
        _ic = lift._img_cache
        result["img_cache"] = {"capacity": _ic["capacity"], "hits": _ic["hits"],
                               "misses": _ic["misses"],
                               "packs": len(_ic["packs"])}
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
    if args.homo_real_out or _homo_dump_dir:
        import hashlib
        def _sha(p):
            try:
                return hashlib.sha256(open(p, "rb").read()).hexdigest()
            except Exception:
                return None
        _here = os.path.dirname(os.path.abspath(__file__))
        homo_meta = {
            "fuse": bool(args.homo_fuse),
            "fallback": bool(args.homo_fallback),
            "clean": bool(args.homo_clean),
            "mode": _homo_mode,
            "dump_dir": _homo_dump_dir,
            "calls": _homo_ctx["calls"],
            "records": _homo_ctx["records"],
            "checkpoint": args.checkpoint,
            "code_sha256": {
                "eval_opt.py": _sha(os.path.join(_here, "eval_opt.py")),
                "lift_opt.py": _sha(os.path.join(_here, "lift_opt.py")),
                "homo_fuse.cu": _sha(os.path.join(
                    _here, "homo_ext", "src", "homo_fuse.cu")),
                "homo_lib.py": _sha(os.path.join(
                    _here, "homo_ext", "homo_lib.py")),
            },
        }
        def _ms_agg(xs):
            xs = [x for x in xs if x is not None]
            if not xs:
                return None
            return {"n": len(xs), "mean_ms": round(statistics.mean(xs), 4),
                    "median_ms": round(statistics.median(xs), 4),
                    "total_ms": round(sum(xs), 4)}
        homo_meta["timing"] = {
            "vendor_ms": _ms_agg(_homo_stats["vendor_ms"]),
            "fused_ms": _ms_agg(_homo_stats["fused_ms"]),
            "fallback_ms": _ms_agg(_homo_stats["fallback_ms"]),
            "mask_disagree_calls": _homo_stats["mask_disagree_calls"],
        }
        if _homo_stats["near_count"]:
            homo_meta["near_selection"] = {
                "n_calls": len(_homo_stats["near_count"]),
                "near_count_total": int(sum(_homo_stats["near_count"])),
                "near_prop_mean": round(statistics.mean(_homo_stats["near_prop"]), 6),
                "near_prop_max": round(max(_homo_stats["near_prop"]), 6),
            }
        homo_meta["per_call"] = _homo_stats["per_call"]
        _homo_out = args.homo_real_out or os.path.join(
            _homo_dump_dir, "meta.json")
        os.makedirs(os.path.dirname(_homo_out), exist_ok=True)
        json.dump(homo_meta, open(_homo_out, "w"), indent=2, allow_nan=True)
    compact["out_json"] = os.path.abspath(args.out_json)
    out_json = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2, allow_nan=True)
    print("V8A17_MS_ROGS_Z_EVAL " + json.dumps(compact), flush=True)

    if args.stage_trace_out and _trace:
        _trace["results_written"] = time.perf_counter()

        def _seg(a, b):
            return round(_trace[b] - _trace[a], 6) if a in _trace and b in _trace else None

        _trace_record = {
            "milestones_s_from_main_entry": {
                k: round(v - _trace["main_entry"], 6) for k, v in _trace.items()},
            "segments_s": {
                "arg_parse_to_official_ckpt": _seg("main_entry", "official_ckpt"),
                "official_ckpt_to_model_build": _seg("official_ckpt", "model_built"),
                "model_build_to_v8a17_ckpt": _seg("model_built", "v8a17_ckpt"),
                "v8a17_ckpt_to_data_meta": _seg("v8a17_ckpt", "data_meta"),
                "data_meta_to_ext_load": _seg("data_meta", "ext_loaded"),
                "ext_load_to_eval_start": _seg("ext_loaded", "eval_start"),
                "eval_elapsed": _seg("eval_start", "eval_end"),
                "eval_end_to_results_write": _seg("eval_end", "results_written"),
            },
            "in_process_total_s": round(
                _trace["results_written"] - _trace["main_entry"], 6)
            if "results_written" in _trace else None,
            "note": "python interpreter startup + imports are the wall residual "
                    "between launcher spawn and main_entry; not measured in-process.",
        }
        if args.lift_opt:
            try:
                import lift_opt as _lo
                _trace_record["corr_ext_load_ms"] = getattr(
                    _lo, "_CORR_LOAD_MS", None)
            except Exception:
                _trace_record["corr_ext_load_ms"] = None
        _tout = os.path.abspath(args.stage_trace_out)
        os.makedirs(os.path.dirname(_tout), exist_ok=True)
        json.dump(_trace_record, open(_tout, "w"), indent=1)
        print("STAGE_TRACE " + _tout, flush=True)


if __name__ == "__main__":
    main()
