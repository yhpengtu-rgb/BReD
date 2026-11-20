"""Scene-level v8-a17 elevation fusion for the official RoGS Road grid.

For each frame, exported global XYZ is rigidly transformed into RoGS' first-
chassis reference frame.  A nearest source point within one source-cell
circumradius supplies one Z per RoGS vertex (nearest-neighbour upsampling).
The final elevation is the median over frames; uncovered vertices retain the
official nearest-trajectory height.  This function runs before training_setup:
it supplies initialization only and never freezes or supervises RoGS Z.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.spatial import cKDTree


def fuse_v10a1_elevation(vertices_xy, trajectory_z, ref_pose_inv,
                         elevation_dir: str, cache_dir: str,
                         source_cell_m: float = 0.4,
                         expected_frames: int = 391,
                         query_workers: int = -1):
    target_xy = np.asarray(vertices_xy, dtype=np.float64)
    trajectory_z = np.asarray(trajectory_z, dtype=np.float32)
    if target_xy.ndim != 2 or target_xy.shape[1] != 2:
        raise ValueError(f"vertices_xy must be (N,2), got {target_xy.shape}")
    if trajectory_z.shape != (len(target_xy),):
        raise ValueError("trajectory_z and vertices_xy disagree")
    ref_pose_inv = np.asarray(ref_pose_inv, dtype=np.float64)
    if ref_pose_inv.shape != (4, 4) or not np.isfinite(ref_pose_inv).all():
        raise ValueError("invalid RoGS ref_pose_inv")

    source = Path(elevation_dir)
    with open(source / "manifest.json") as handle:
        source_manifest = json.load(handle)
    if source_manifest["experiment"] != "v10-a1":
        raise ValueError("elevation source is not v10-a1")
    if source_manifest["coordinate_frame"] != "nuScenes global":
        raise ValueError("elevation source must be in nuScenes global coordinates")
    if source_manifest["source_grid"]["cell_m"] != source_cell_m:
        raise ValueError("source cell size mismatch")
    frame_records = source_manifest["frames"]
    if len(frame_records) != expected_frames or source_manifest["frame_count"] != expected_frames:
        raise ValueError(f"expected {expected_frames} exported frames, got {len(frame_records)}")
    if source_manifest["points_per_frame"] != 256 * 256:
        raise ValueError("v10-a1 expects the v8-a17 256x256 source grid")

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    fused_path = cache / "fused_z.npy"
    coverage_path = cache / "coverage_count.npy"
    stats_path = cache / "fusion_stats.json"
    if fused_path.exists() and coverage_path.exists() and stats_path.exists():
        with open(stats_path) as handle:
            previous = json.load(handle)
        if (previous["vertex_count"] == len(target_xy)
                and previous["frame_count"] == expected_frames
                and previous["source_checkpoint_sha256"] == source_manifest["source_checkpoint_sha256"]):
            return np.load(fused_path), previous

    values_path = cache / "vertex_frame_z.npy"
    values = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.float32,
                                       shape=(expected_frames, len(target_xy)))
    max_distance = source_cell_m / math.sqrt(2.0) + 1e-6
    started = time.perf_counter()

    for frame_index, record in enumerate(frame_records):
        xyz_global = np.load(source / record["file"], allow_pickle=False)
        if xyz_global.shape != (source_manifest["points_per_frame"], 3):
            raise ValueError(f"bad source shape at frame {frame_index}: {xyz_global.shape}")
        if not np.isfinite(xyz_global).all():
            raise FloatingPointError(f"non-finite source at frame {frame_index}")
        xyz_ref = xyz_global @ ref_pose_inv[:3, :3].T + ref_pose_inv[:3, 3]
        tree = cKDTree(xyz_ref[:, :2])
        distance, nearest = tree.query(target_xy, k=1,
                                       distance_upper_bound=max_distance,
                                       workers=query_workers)
        valid = np.isfinite(distance) & (nearest < len(xyz_ref))
        row = np.full(len(target_xy), np.nan, dtype=np.float32)
        row[valid] = xyz_ref[nearest[valid], 2]
        values[frame_index] = row
        if frame_index % 10 == 0 or frame_index + 1 == expected_frames:
            print(f"V10A1_FUSION frame={frame_index + 1}/{expected_frames} "
                  f"covered={int(valid.sum())}/{len(target_xy)} "
                  f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)
    values.flush()

    fused = trajectory_z.copy()
    coverage = np.zeros(len(target_xy), dtype=np.int32)
    chunk_size = 65536
    for start in range(0, len(target_xy), chunk_size):
        end = min(start + chunk_size, len(target_xy))
        block = np.asarray(values[:, start:end])
        coverage[start:end] = np.isfinite(block).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(block, axis=0).astype(np.float32)
        covered = coverage[start:end] > 0
        fused[start:end][covered] = median[covered]
    if not np.isfinite(fused).all():
        raise FloatingPointError("non-finite fused elevation")

    covered = coverage > 0
    stats = {
        "experiment": "v10-a1",
        "coordinate_frame": "RoGS first-chassis reference",
        "source_checkpoint": source_manifest["source_checkpoint"],
        "source_checkpoint_sha256": source_manifest["source_checkpoint_sha256"],
        "frame_count": expected_frames,
        "vertex_count": int(len(target_xy)),
        "covered_vertices": int(covered.sum()),
        "uncovered_vertices": int((~covered).sum()),
        "covered_fraction": float(covered.mean()),
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "coverage_mean": float(coverage.mean()),
        "upsampling": "nearest source XY per frame in RoGS reference frame",
        "source_cell_m": source_cell_m,
        "max_source_distance_m": max_distance,
        "multi_frame_aggregation": "median over valid per-frame Z",
        "uncovered_policy": "retain official nearest-trajectory Z",
        "initialization_only": True,
        "rogs_z_trainable_after_initialization": True,
        "fused_z_min_m": float(fused.min()),
        "fused_z_max_m": float(fused.max()),
        "median_absolute_change_vs_trajectory_m": float(np.median(np.abs(fused - trajectory_z))),
        "elapsed_s": time.perf_counter() - started,
    }
    np.save(fused_path, fused, allow_pickle=False)
    np.save(coverage_path, coverage, allow_pickle=False)
    with open(stats_path, "w") as handle:
        json.dump(stats, handle, indent=2)
    print("V10A1_FUSION_DONE " + json.dumps(stats), flush=True)
    return fused, stats
