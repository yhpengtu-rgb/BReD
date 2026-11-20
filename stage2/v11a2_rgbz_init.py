"""v11-a2 fixed RGBZ initialization from official RoGS training views.

Z is inherited from the frozen v8-a17/v10-a1 fusion.  Each RGBZ vertex is
projected with the official world2cam transform into every official training
view.  Bilinear RGB samples with valid road masks are aggregated by median.
No tensor in this initializer requires gradients and no optimizer is created.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F


def _grid_sha256(vertices):
    array = np.ascontiguousarray(vertices.detach().cpu().numpy(), dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def rgb_probability_to_logits(rgb, epsilon=1e-4):
    """Convert observed [0,1] RGB to RoGS' raw sigmoid-logit parameterization."""
    probability = np.asarray(rgb, dtype=np.float32)
    if probability.ndim != 2 or probability.shape[1] != 3:
        raise ValueError(f"RGB must be (N,3), got {probability.shape}")
    if not np.isfinite(probability).all():
        raise FloatingPointError("non-finite RGB initialization")
    probability = np.clip(probability, epsilon, 1.0 - epsilon)
    return np.log(probability / (1.0 - probability)).astype(np.float32)


def _sample_chw(feature, grid):
    count = grid.shape[0]
    return F.grid_sample(
        feature.unsqueeze(0), grid.reshape(1, count, 1, 2),
        mode="bilinear", padding_mode="zeros", align_corners=True,
    ).reshape(feature.shape[0], count).transpose(0, 1)


def fuse_v11a2_rgb(vertices, dataset, cache_dir, median_chunk=4096):
    """Return fixed per-vertex RGB probabilities and auditable cache statistics."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    fused_path = cache / "initial_rgb_probability.npy"
    coverage_path = cache / "rgb_coverage_count.npy"
    stats_path = cache / "rgb_fusion_stats.json"
    grid_hash = _grid_sha256(vertices)
    view_count = len(dataset)
    if fused_path.exists() and coverage_path.exists() and stats_path.exists():
        previous = json.loads(stats_path.read_text())
        if (previous["grid_sha256"] == grid_hash
                and previous["view_count"] == view_count
                and previous["aggregation"] == "median_valid_road_rgb"):
            return np.load(fused_path), previous

    count = vertices.shape[0]
    values_path = cache / "rgb_observations_u8.npy"
    valid_path = cache / "rgb_observations_valid.npy"
    values = np.lib.format.open_memmap(
        values_path, mode="w+", dtype=np.uint8, shape=(count, view_count, 3)
    )
    valid_observations = np.lib.format.open_memmap(
        valid_path, mode="w+", dtype=np.bool_, shape=(count, view_count)
    )
    device = vertices.device
    xyz = vertices.detach()
    started = time.perf_counter()

    for view_index in range(view_count):
        sample = dataset[view_index]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = sample["R"]
        c2w[:3, 3] = sample["T"]
        world2cam = np.linalg.inv(c2w)
        rotation = torch.from_numpy(world2cam[:3, :3]).to(device=device, dtype=xyz.dtype)
        translation = torch.from_numpy(world2cam[:3, 3]).to(device=device, dtype=xyz.dtype)
        intrinsic = torch.from_numpy(sample["K"]).to(device=device, dtype=xyz.dtype)
        xyz_cam = xyz @ rotation.transpose(0, 1) + translation
        projected = xyz_cam @ intrinsic.transpose(0, 1)
        depth = projected[:, 2]
        safe_depth = depth.clamp_min(1e-6)
        u = projected[:, 0] / safe_depth
        v = projected[:, 1] / safe_depth
        width, height = int(sample["W"]), int(sample["H"])
        grid = torch.stack((2 * u / (width - 1) - 1, 2 * v / (height - 1) - 1), dim=-1)
        valid = (depth > 1.0) & (depth < 20.0)
        valid &= (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)

        image = torch.from_numpy(sample["image"]).to(device=device).permute(2, 0, 1)
        sampled_rgb = _sample_chw(image, grid)
        if "mask" not in sample:
            raise KeyError("v11-a2 RGB initialization requires the official road mask")
        road_mask = torch.from_numpy(sample["mask"]).to(device=device)[None]
        sampled_mask = _sample_chw(road_mask, grid)[:, 0]
        valid &= sampled_mask >= 0.5
        values[:, view_index, :] = (
            sampled_rgb.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
        )
        valid_observations[:, view_index] = valid.cpu().numpy()
        if view_index % 50 == 0 or view_index + 1 == view_count:
            print(
                f"V11A2_RGB frame={view_index + 1}/{view_count} "
                f"valid={int(valid.sum())}/{count} elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
    values.flush()
    valid_observations.flush()

    fused = np.full((count, 3), 0.5, dtype=np.float32)
    coverage = np.zeros(count, dtype=np.int32)
    for start in range(0, count, median_chunk):
        end = min(start + median_chunk, count)
        valid_block = np.asarray(valid_observations[start:end])
        value_block = np.asarray(values[start:end])
        coverage[start:end] = valid_block.sum(axis=1)
        masked = np.where(valid_block[..., None], value_block, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(masked, axis=1).astype(np.float32) / 255.0
        covered = coverage[start:end] > 0
        fused[start:end][covered] = median[covered]
    if not np.isfinite(fused).all():
        raise FloatingPointError("non-finite fused RGB")
    np.save(fused_path, fused, allow_pickle=False)
    np.save(coverage_path, coverage, allow_pickle=False)
    covered = coverage > 0
    stats = {
        "experiment": "v11-a2",
        "parent": "v11-a1 E2",
        "coordinate_frame": "RoGS first-chassis reference",
        "grid_sha256": grid_hash,
        "vertex_count": int(count),
        "view_count": int(view_count),
        "views": "all official RoGS keyframe+sweep training images",
        "projection": "explicit world2cam followed by official resized/cropped K",
        "rgb_sampling": "bilinear RGB and road-mask sampling at each RGBZ vertex",
        "aggregation": "median_valid_road_rgb",
        "valid_depth_m": [1.0, 20.0],
        "road_mask_threshold": 0.5,
        "uncovered_policy": "retain v11-a1 E2 zero RGB logits (probability 0.5)",
        "covered_vertices": int(covered.sum()),
        "uncovered_vertices": int((~covered).sum()),
        "covered_fraction": float(covered.mean()),
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "coverage_mean": float(coverage.mean()),
        "rgb_probability_min": float(fused[covered].min()) if covered.any() else 0.5,
        "rgb_probability_max": float(fused[covered].max()) if covered.any() else 0.5,
        "rgb_logit_clamp": 1e-4,
        "initialization_only": True,
        "base_gaussian_trainable": False,
        "elapsed_s": time.perf_counter() - started,
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print("V11A2_RGB_DONE " + json.dumps(stats), flush=True)
    return fused, stats
