#!/usr/bin/env python3
"""Official BEVDepth key-frame perspective LiDAR depth auxiliary.

Returns the official native depth BCE unchanged (label z_cam, native x3.0
factor already inside ``lightning.get_depth_loss``).  The value is NOT
multiplied or divided by 3 here; GradNorm consumes this raw x3 quantity
directly, so g_depth / r_depth / depth0 all use the same x3 scale.
"""
import torch


def compute_official_depth_aux(lightning, depth_labels, depth_predictions):
    """Return the official native depth loss (already x3)."""
    if depth_predictions is None:
        raise RuntimeError("Depth auxiliary requested without key-frame depth predictions")
    if depth_labels.ndim == 5:
        # Official base_exp.py: only the key frame contributes depth supervision.
        depth_labels = depth_labels[:, 0, ...]
    if depth_labels.ndim != 4 or depth_predictions.ndim != 4:
        raise ValueError(
            f"Unexpected depth shapes: labels={tuple(depth_labels.shape)}, "
            f"predictions={tuple(depth_predictions.shape)}")
    if depth_predictions.shape[1] != lightning.depth_channels:
        raise ValueError("Depth prediction channels differ from the official experiment")
    expected_hw = (depth_labels.shape[-2] // lightning.downsample_factor,
                   depth_labels.shape[-1] // lightning.downsample_factor)
    if tuple(depth_predictions.shape[-2:]) != expected_hw:
        raise ValueError(
            f"Depth prediction grid {tuple(depth_predictions.shape[-2:])} does not match "
            f"official downsampled LiDAR grid {expected_hw}")
    native_loss = lightning.get_depth_loss(depth_labels, depth_predictions)
    if not bool(torch.isfinite(native_loss)):
        raise FloatingPointError("Non-finite official perspective depth loss")
    return native_loss
