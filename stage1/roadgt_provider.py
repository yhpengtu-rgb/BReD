#!/usr/bin/env python3
"""RoadGT sparse-cache targets aligned to the official BEVStereo BDA frame."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from pyquaternion import Quaternion

from rogs_adapter import make_cell_center_ortho_camera


FROZEN_VAL_SCENES = frozenset({
    "scene-0064", "scene-0212", "scene-0523", "scene-0655", "scene-0856",
})


class RoadGTProvider:
    """Load one scene cache and emit fixed-size targets in final BEV coordinates.

    The target lattice is axis-aligned *after* BDA.  Query cell centers are
    inverse-transformed to the original ego frame for sparse nearest-cell
    lookup; their Z values are then forward-transformed through the same
    world->ego->BDA chain.  This avoids rotating the orthographic camera.
    """

    def __init__(self, cache_path, scene_name, tile_size=32, resolution=0.05,
                 pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)):
        if scene_name in FROZEN_VAL_SCENES:
            raise ValueError(f"Frozen validation scene is not available for training: {scene_name}")
        self.cache_path = Path(cache_path)
        self.scene_name = scene_name
        self.tile_size = int(tile_size)
        self.resolution = float(resolution)
        self.pc_range = tuple(float(value) for value in pc_range)
        if self.tile_size <= 0 or self.resolution <= 0:
            raise ValueError("tile_size and resolution must be positive")
        with np.load(self.cache_path) as data:
            self.xy = data["xy"].astype(np.float32)
            self.tgt_z = data["tgt_z"].astype(np.float32)
            self.tgt_rgb = data["tgt_rgb"].astype(np.float32)
            self.tgt_label = data["tgt_label"].astype(np.int64)
            self.tgt_valid = data["tgt_valid"].astype(bool)
            self.min_xy = data["min_xy"].astype(np.float32)
            self.max_xy = data["max_xy"].astype(np.float32)
            self.grid_shape = data["grid_shape"].astype(np.int64)
            cache_resolution = float(data["resolution"])
        if abs(cache_resolution - self.resolution) > 1e-6:
            raise ValueError((cache_resolution, self.resolution))
        count = len(self.xy)
        if not all(len(array) == count for array in
                   (self.tgt_z, self.tgt_rgb, self.tgt_label, self.tgt_valid)):
            raise ValueError("Sparse RoadGT arrays have inconsistent lengths")
        if self.tgt_rgb.shape != (count, 3):
            raise ValueError(self.tgt_rgb.shape)
        nx, ny = (int(self.grid_shape[0]), int(self.grid_shape[1]))
        px = np.rint((self.xy[:, 0] - self.min_xy[0]) / self.resolution).astype(np.int64)
        py = np.rint((self.xy[:, 1] - self.min_xy[1]) / self.resolution).astype(np.int64)
        inside = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
        if not bool(inside.all()):
            raise ValueError("Sparse cache contains cells outside its declared grid")
        self.row_by_flat = np.full(nx * ny, -1, dtype=np.int32)
        self.row_by_flat[py * nx + px] = np.arange(count, dtype=np.int32)
        self.nx, self.ny = nx, ny

    @staticmethod
    def _as_numpy(matrix):
        if isinstance(matrix, torch.Tensor):
            return matrix.detach().cpu().numpy()
        return np.asarray(matrix)

    def _poses(self, img_metas, bda_mat):
        translation = self._as_numpy(img_metas["ego2global_translation"]).astype(np.float64)
        quaternion = self._as_numpy(img_metas["ego2global_rotation"]).astype(np.float64)
        ego_to_global_rotation = Quaternion(quaternion).rotation_matrix
        bda = self._as_numpy(bda_mat).astype(np.float64)
        if bda.shape != (4, 4):
            raise ValueError(f"Expected one official 4x4 bda_mat, got {bda.shape}")
        if not np.isfinite(bda).all() or abs(np.linalg.det(bda[:2, :2])) < 1e-8:
            raise ValueError("Invalid BDA matrix")
        return translation, ego_to_global_rotation, bda

    def _choose_center(self, xy_final, token, tile_index, max_candidates=32,
                       surround=10.0):
        valid = self.tgt_valid.copy()
        valid &= np.isfinite(self.tgt_rgb).all(axis=1)
        valid &= np.isfinite(self.tgt_z)
        valid &= (self.tgt_label >= 0) & (self.tgt_label < 7)
        valid &= (xy_final[:, 0] >= self.pc_range[0]) & (xy_final[:, 0] <= self.pc_range[3])
        valid &= (xy_final[:, 1] >= self.pc_range[1]) & (xy_final[:, 1] <= self.pc_range[4])
        if surround is not None:
            # Match RoGS with-LiDAR's current-camera Z supervision region.
            # Official BDA has zero translation, hence ego XY remains zero.
            valid &= (np.abs(xy_final[:, 0]) <= float(surround))
            valid &= (np.abs(xy_final[:, 1]) <= float(surround))
        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            raise RuntimeError("No valid RoadGT cells inside the official BEV range")
        seed_material = f"{self.scene_name}:{token}:{tile_index}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        candidate_indices = rng.choice(indices, size=min(max_candidates, len(indices)), replace=False)
        points = xy_final[valid]
        labels = self.tgt_label[valid]
        half = self.tile_size * self.resolution / 2.0
        candidates = []
        for index in candidate_indices:
            center = xy_final[index].copy()
            inside = ((np.abs(points[:, 0] - center[0]) <= half) &
                      (np.abs(points[:, 1] - center[1]) <= half))
            count = int(inside.sum())
            diversity = int(np.unique(labels[inside]).size)
            distance = float(np.linalg.norm(center))
            candidates.append((count, diversity, -distance, center))
        candidates.sort(key=lambda item: item[:3], reverse=True)
        # GT availability controls training sampling only.  It is never an
        # evaluation score or a measure of reconstruction quality.
        top = candidates[:min(4, len(candidates))]
        chosen = top[int(tile_index) % len(top)]
        return chosen[3], {"candidate_valid_cells": chosen[0],
                           "candidate_label_diversity": chosen[1],
                           "candidate_count": len(candidates)}

    def make_target(self, img_metas, bda_mat, device, tile_index=0,
                    tile_center=None, min_valid_cells=64, z_surround=10.0,
                    tile_size=None):
        token = str(img_metas["token"])
        translation, rotation, bda = self._poses(img_metas, bda_mat)
        linear = bda[:2, :2]
        offset = bda[:2, 3]
        if tile_center is None:
            xyz_global = np.concatenate((self.xy.astype(np.float64),
                                         self.tgt_z[:, None].astype(np.float64)), axis=1)
            xyz_ego = (xyz_global - translation[None, :]) @ rotation
            xyz_final = xyz_ego @ bda[:3, :3].T + bda[:3, 3][None, :]
            xy_final = xyz_final[:, :2]
            center, selection = self._choose_center(
                xy_final, token, tile_index, surround=z_surround
            )
        else:
            center = self._as_numpy(tile_center).astype(np.float64).reshape(2)
            selection = {"candidate_valid_cells": None,
                         "candidate_label_diversity": None,
                         "candidate_count": 0}
        size = self.tile_size if tile_size is None else int(tile_size)
        if size <= 0:
            raise ValueError("tile_size must be positive")
        if size != self.tile_size and tile_center is None:
            raise ValueError("Evaluation-only tile_size override requires an explicit tile_center")
        resolution = self.resolution
        coordinates = (np.arange(size, dtype=np.float64) + 0.5) * resolution
        coordinates = coordinates - size * resolution / 2.0
        yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
        base_xy = np.stack((xx, yy), axis=-1).reshape(-1, 2) + center[None, :]

        # Inverse BDA for lookup, then inverse ego pose to global cache space.
        base_final_xyz = np.concatenate((base_xy, np.zeros((size * size, 1))), axis=1)
        inverse_bda = np.linalg.inv(bda)
        query_ego = base_final_xyz @ inverse_bda[:3, :3].T + inverse_bda[:3, 3][None, :]
        query_global = query_ego @ rotation.T + translation[None, :]
        px = np.rint((query_global[:, 0] - self.min_xy[0]) / resolution).astype(np.int64)
        py = np.rint((query_global[:, 1] - self.min_xy[1]) / resolution).astype(np.int64)
        inside = (px >= 0) & (px < self.nx) & (py >= 0) & (py < self.ny)
        rows = np.full(size * size, -1, dtype=np.int64)
        rows[inside] = self.row_by_flat[py[inside] * self.nx + px[inside]]
        valid = rows >= 0
        valid[valid] &= self.tgt_valid[rows[valid]]

        rgb = np.zeros((size * size, 3), dtype=np.float32)
        height = np.zeros(size * size, dtype=np.float32)
        label = np.full(size * size, 255, dtype=np.int64)
        if bool(valid.any()):
            source_rows = rows[valid]
            rgb[valid] = self.tgt_rgb[source_rows]
            source_z = self.tgt_z[source_rows]
            # Use the cache grid centers for a complete XYZ forward chain.
            source_global = np.stack((self.min_xy[0] + px[valid] * resolution,
                                      self.min_xy[1] + py[valid] * resolution,
                                      source_z), axis=1).astype(np.float64)
            source_ego = (source_global - translation[None, :]) @ rotation
            source_final = source_ego @ bda[:3, :3].T + bda[:3, 3][None, :]
            height[valid] = source_final[:, 2].astype(np.float32)
            label[valid] = self.tgt_label[source_rows]
        valid_count = int(valid.sum())
        if valid_count < int(min_valid_cells):
            raise RuntimeError(f"RoadGT tile has only {valid_count} valid cells; minimum is {min_valid_cells}")
        bda_scale = float(np.sqrt(abs(np.linalg.det(linear))))
        metadata = {
            "scene": self.scene_name,
            "sample_token": token,
            "tile_index": int(tile_index),
            "tile_center_bev": center.tolist(),
            "tile_size": size,
            "resolution_bev_m": resolution,
            "bda_xy_scale": bda_scale,
            "equivalent_source_query_spacing_m": resolution / bda_scale,
            "valid_cells": valid_count,
            "z_supervision_surround_m": None if tile_center is not None else z_surround,
            "label_values": np.unique(label[valid]).tolist(),
            "selection": selection,
            "coordinate_chain": "final BDA tile -> inverse BDA -> inverse ego -> sparse cache; cache XYZ -> ego -> BDA target Z",
        }
        tensors = {
            "base_xy": torch.from_numpy(base_xy.astype(np.float32)).reshape(1, size * size, 2).to(device),
            "target_rgb": torch.from_numpy(rgb.T.reshape(3, size, size)).to(device),
            "target_z": torch.from_numpy(height.reshape(size, size)).to(device),
            "target_label": torch.from_numpy(label.reshape(size, size)).to(device),
            "target_valid": torch.from_numpy(valid.reshape(size, size)).to(device),
            "resolution": resolution,
            "metadata": metadata,
        }
        tensors["camera"] = make_cell_center_ortho_camera(
            tensors["base_xy"], size, resolution, torch.device(device),
        )
        return tensors
