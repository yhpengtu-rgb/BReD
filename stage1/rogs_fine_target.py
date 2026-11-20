#!/usr/bin/env python3
"""Build v8-A fine-grid RoadGT targets (full-BEV lattice) from a RoadGTProvider.

Mirrors RoadGTProvider.make_target's coordinate chain (BDA frame lattice ->
inverse BDA -> inverse ego pose -> sparse cache; cache XYZ -> ego -> BDA for
target Z) but on a configurable fine lattice over the symmetric BEV range.
"""
from __future__ import annotations

import numpy as np
import torch

from rogs_adapter import make_cell_center_ortho_camera


def make_fine_bev_target(provider, img_metas, bda_mat, device,
                         fine_size: int = 256, cell: float = 0.4,
                         center_xy=(0.0, 0.0), z_surround: float = 60.0):
    from pyquaternion import Quaternion
    translation = provider._as_numpy(img_metas["ego2global_translation"]).astype(np.float64)
    quaternion = provider._as_numpy(img_metas["ego2global_rotation"]).astype(np.float64)
    rot = Quaternion(quaternion).rotation_matrix
    bda = provider._as_numpy(bda_mat).astype(np.float64)

    center = np.asarray(center_xy, dtype=np.float64)
    coords = (np.arange(fine_size, dtype=np.float64) + 0.5) * cell - cell * fine_size / 2.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    base_xy = np.stack((xx, yy), axis=-1).reshape(-1, 2) + center[None, :]

    base_xyz = np.concatenate((base_xy, np.zeros((base_xy.shape[0], 1))), axis=1)
    inv_bda = np.linalg.inv(bda)
    q_ego = base_xyz @ inv_bda[:3, :3].T + inv_bda[:3, 3][None, :]
    q_global = q_ego @ rot.T + translation[None, :]

    nx, ny = provider.nx, provider.ny
    px = np.rint((q_global[:, 0] - provider.min_xy[0]) / provider.resolution).astype(np.int64)
    py = np.rint((q_global[:, 1] - provider.min_xy[1]) / provider.resolution).astype(np.int64)
    inside = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
    rows = np.full(base_xy.shape[0], -1, dtype=np.int64)
    rows[inside] = provider.row_by_flat[py[inside] * nx + px[inside]]
    valid = rows >= 0
    valid[valid] &= provider.tgt_valid[rows[valid]]

    rgb = np.zeros((base_xy.shape[0], 3), dtype=np.float32)
    height = np.zeros(base_xy.shape[0], dtype=np.float32)
    label = np.full(base_xy.shape[0], 255, dtype=np.int64)
    if valid.any():
        src = rows[valid]
        rgb[valid] = provider.tgt_rgb[src]
        src_global = np.stack((
            provider.min_xy[0] + px[valid] * provider.resolution,
            provider.min_xy[1] + py[valid] * provider.resolution,
            provider.tgt_z[src]), axis=1).astype(np.float64)
        src_ego = (src_global - translation[None, :]) @ rot
        src_final = src_ego @ bda[:3, :3].T + bda[:3, 3][None, :]
        height[valid] = src_final[:, 2].astype(np.float32)
        label[valid] = provider.tgt_label[src]

    base_xy_t = torch.from_numpy(base_xy.astype(np.float32)).reshape(1, fine_size * fine_size, 2).to(device)
    tensors = {
        "base_xy": base_xy_t,
        "target_rgb": torch.from_numpy(rgb.T.reshape(3, fine_size, fine_size)).to(device),
        "target_z": torch.from_numpy(height.reshape(fine_size, fine_size)).to(device),
        "target_label": torch.from_numpy(label.reshape(fine_size, fine_size)).to(device),
        "target_valid": torch.from_numpy(valid.reshape(fine_size, fine_size)).to(device),
        "resolution": cell,
    }
    tensors["camera"] = make_cell_center_ortho_camera(base_xy_t, fine_size, cell, device)
    return tensors
