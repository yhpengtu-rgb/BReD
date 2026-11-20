"""Fuse frozen v8-a17 z-fine candidate BEV features onto the RoGS grid.

Only fixed geometric conditioning is produced here.  There is no optimizer,
gradient, GT supervision, or Gaussian parameter update in this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree


GEOMETRY_FEATURE_DIM = 96


def _grid_sha256(vertices) -> str:
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    array = np.ascontiguousarray(vertices, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_v11a4_geometry_prior(
    vertices, ref_pose_inv, export_dir, cache_dir,
    source_cell_m=0.8, expected_frames=391, query_workers=16,
):
    """Return (N,96) float16 features, (N,) validity and auditable statistics."""
    grid_hash = _grid_sha256(vertices)
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {vertices.shape}")
    target_xy = vertices[:, :2].astype(np.float64)
    ref_pose_inv = np.asarray(ref_pose_inv, dtype=np.float64)
    if ref_pose_inv.shape != (4, 4) or not np.isfinite(ref_pose_inv).all():
        raise ValueError("invalid RoGS ref_pose_inv")

    source = Path(export_dir)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["experiment"] != "v11-a4-geometry-export":
        raise ValueError("geometry export experiment mismatch")
    if manifest["coordinate_frame"] != "nuScenes global":
        raise ValueError("geometry features must use nuScenes global coordinates")
    if manifest["feature_dim"] != GEOMETRY_FEATURE_DIM:
        raise ValueError("geometry feature dimension mismatch")
    if manifest["source_grid"]["cell_m"] != source_cell_m:
        raise ValueError("geometry source cell mismatch")
    records = manifest["frames"]
    if len(records) != expected_frames or manifest["frame_count"] != expected_frames:
        raise ValueError(f"expected {expected_frames} geometry frames, got {len(records)}")

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    fused_path = cache / "geometric_features_f16.npy"
    valid_path = cache / "geometric_valid.npy"
    stats_path = cache / "geometric_fusion_stats.json"
    if fused_path.exists() and valid_path.exists() and stats_path.exists():
        previous = json.loads(stats_path.read_text())
        if (previous["grid_sha256"] == grid_hash
                and previous["frame_count"] == expected_frames
                and previous["source_checkpoint_sha256"] == manifest["source_checkpoint_sha256"]
                and previous["feature_dim"] == GEOMETRY_FEATURE_DIM):
            return np.load(fused_path), np.load(valid_path), previous

    count = len(vertices)
    feature_sum = np.zeros((count, GEOMETRY_FEATURE_DIM), dtype=np.float32)
    coverage = np.zeros(count, dtype=np.int32)
    max_distance = source_cell_m / math.sqrt(2.0) + 1e-6
    started = time.perf_counter()

    for frame_index, record in enumerate(records):
        with np.load(source / record["file"], allow_pickle=False) as data:
            xyz_global = data["xyz_global"]
            features = data["features"]
            source_valid = data["valid"]
        if xyz_global.shape != (manifest["points_per_frame"], 3):
            raise ValueError(f"bad geometry coordinates at frame {frame_index}")
        if features.shape != (manifest["points_per_frame"], GEOMETRY_FEATURE_DIM):
            raise ValueError(f"bad geometry features at frame {frame_index}")
        if source_valid.shape != (manifest["points_per_frame"],):
            raise ValueError(f"bad geometry validity at frame {frame_index}")
        if not np.isfinite(xyz_global).all() or not np.isfinite(features).all():
            raise FloatingPointError(f"non-finite geometry source at frame {frame_index}")
        xyz_ref = xyz_global @ ref_pose_inv[:3, :3].T + ref_pose_inv[:3, 3]
        tree = cKDTree(xyz_ref[:, :2])
        distance, nearest = tree.query(
            target_xy, k=1, distance_upper_bound=max_distance,
            workers=query_workers,
        )
        source_nearest_valid = source_valid[np.minimum(nearest, len(source_valid) - 1)]
        valid = np.isfinite(distance) & (nearest < len(xyz_ref)) & source_nearest_valid
        indices = np.flatnonzero(valid)
        if len(indices):
            feature_sum[indices] += features[nearest[indices]].astype(np.float32)
            coverage[indices] += 1
        if frame_index % 10 == 0 or frame_index + 1 == expected_frames:
            print(
                f"V11A4_GEOMETRY_FUSION frame={frame_index + 1}/{expected_frames} "
                f"covered={len(indices)}/{count} "
                f"elapsed_s={time.perf_counter() - started:.1f}", flush=True,
            )

    covered = coverage > 0
    fused = np.zeros((count, GEOMETRY_FEATURE_DIM), dtype=np.float32)
    fused[covered] = feature_sum[covered] / coverage[covered, None]
    if not np.isfinite(fused).all():
        raise FloatingPointError("non-finite fused geometry prior")
    fused_f16 = fused.astype(np.float16)
    np.save(fused_path, fused_f16, allow_pickle=False)
    np.save(valid_path, covered, allow_pickle=False)
    stats = {
        "experiment": "v11-a4",
        "parent": "v11-a3",
        "coordinate_frame": "RoGS first-chassis reference",
        "source_checkpoint": manifest["source_checkpoint"],
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "grid_sha256": grid_hash,
        "frame_count": expected_frames,
        "vertex_count": int(count),
        "feature_dim": GEOMETRY_FEATURE_DIM,
        "head_prior_dim": GEOMETRY_FEATURE_DIM + 1,
        "z_channel": "fixed v11-a3 initial_z appended by the residual refiner",
        "scale_channels": [32, 32, 32],
        "scale_names": ["P2", "P3", "P4"],
        "candidate_rule": manifest["candidate_rule"],
        "candidate_aggregation": manifest["candidate_aggregation"],
        "upsampling": "nearest valid source XY per frame in RoGS reference frame",
        "multi_frame_aggregation": "per-channel arithmetic mean over valid frames",
        "source_cell_m": source_cell_m,
        "max_source_distance_m": max_distance,
        "covered_vertices": int(covered.sum()),
        "uncovered_vertices": int((~covered).sum()),
        "covered_fraction": float(covered.mean()),
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "coverage_mean": float(coverage.mean()),
        "feature_abs_mean": float(np.abs(fused[covered]).mean()) if covered.any() else 0.0,
        "feature_abs_max": float(np.abs(fused).max()),
        "uncovered_policy": "zero features and prior_valid=False",
        "initialization_only": True,
        "diffusion_trainable": False,
        "base_gaussian_trainable": False,
        "elapsed_s": time.perf_counter() - started,
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print("V11A4_GEOMETRY_FUSION_DONE " + json.dumps(stats), flush=True)
    return fused_f16, covered, stats
