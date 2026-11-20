#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v8-a4 road-only depth dataset.

Input policy (locked):
  * Final resolution 576 x 1024 (H, W), uniform 0.64 shrink of the raw 900x1600;
    576/32=18, 1024/32=32.  Zero crop, zero flip, zero rotate, zero colour jitter.
    Train and eval use the identical deterministic resize only.
  * The road depth cache (`v8a4_road_depth_cache`) already stores cam_depth in the
    576x1024 resized coordinate space (seg resized, LiDAR coords x0.64).  The
    downstream rasterisation therefore does NOT resize/crop a second time -- it
    only drops points outside the final image and rasterises at final resolution.

Road filtering: official key-frame perspective BCE + z_cam labels unchanged;
non-road LiDAR returns are dropped before rasterisation via the per-camera
nuScenes_clip seg PNGs (KEEP_LABELS).
"""
import os
from pathlib import Path

import numpy as np
import torch
import mmcv
from PIL import Image
from pyquaternion import Quaternion

from bevdepth.datasets.nusc_det_dataset import NuscDetDataset

BRED_ROOT = Path(__file__).resolve().parents[1]
SEG_ROOT = str(BRED_ROOT / "assets" / "datasets" / "nuScenes_clip")
KEEP_LABELS = {2, 8, 9, 13, 14, 15, 23, 24, 29, 41}
CACHE_ROOT = os.environ.get("BRED_ROAD_CACHE",
                            str(BRED_ROOT / "assets" / "cache" / "road_depth"))
RESIZE = float(os.environ.get("BRED_RESIZE", 0.64))
FINAL_H, FINAL_W = 576, 1024


class NuscDetDatasetRoadV8a4(NuscDetDataset):

    # ------------------------------------------------------------------ #
    # IDA: deterministic uniform resize, no crop/flip/rotate/colour
    # ------------------------------------------------------------------ #
    def sample_ida_augmentation(self):
        fH, fW = self.ida_aug_conf['final_dim']
        resize = RESIZE
        resize_dims = (fW, fH)
        crop = (0, 0, fW, fH)
        flip = False
        rotate_ida = 0.0
        return resize, resize_dims, crop, flip, rotate_ida

    # ------------------------------------------------------------------ #
    # road seg filter + cache (cache coords are final 576x1024 space)
    # ------------------------------------------------------------------ #
    def _load_road_seg(self, cam_info):
        rel = cam_info["filename"]
        sub, cam, base = rel.split("/")
        if not base.endswith(".jpg"):
            return None
        seg_path = os.path.join(SEG_ROOT, sub, "seg_" + cam,
                                base[:-4] + ".png")
        if not os.path.exists(seg_path):
            return None
        return np.array(Image.open(seg_path))

    def _cache_path(self, cam_info):
        rel = cam_info["filename"]
        if "/" not in rel:
            return None
        sub, cam, base = rel.split("/")
        if not base.endswith(".jpg"):
            return None
        return os.path.join(CACHE_ROOT, sub, cam, base[:-4] + ".npz")

    def _write_cache(self, path, arr):
        d = os.path.dirname(path)
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            return
        tmp = path + ".tmp%d.npz" % os.getpid()
        try:
            np.savez_compressed(tmp, cam_depth=arr)
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def get_lidar_depth(self, lidar_points, img, lidar_info, cam_info):
        cache_path = self._cache_path(cam_info)
        if cache_path is not None and os.path.exists(cache_path):
            try:
                return np.load(cache_path)["cam_depth"].astype(np.float32)
            except Exception:
                pass
        cam_depth = super().get_lidar_depth(lidar_points, img, lidar_info,
                                            cam_info)
        if cam_depth.shape[0] == 0:
            return cam_depth
        seg = self._load_road_seg(cam_info)
        if seg is None:
            road_depth = np.zeros((0, 3), dtype=np.float32)
        else:
            x = cam_depth[:, 0].astype(np.int64)
            y = cam_depth[:, 1].astype(np.int64)
            inb = (x >= 0) & (x < seg.shape[1]) & (y >= 0) & (y < seg.shape[0])
            x = np.clip(x, 0, seg.shape[1] - 1)
            y = np.clip(y, 0, seg.shape[0] - 1)
            keep = inb & np.isin(seg[y, x], list(KEEP_LABELS))
            road_depth = cam_depth[keep]
            # project raw-image coords into the final 576x1024 cache space
            road_depth = road_depth.copy()
            road_depth[:, :2] *= RESIZE
        if cache_path is not None:
            self._write_cache(cache_path, road_depth)
        return road_depth

    def _rasterize_final(self, cam_depth, final_dim):
        """Rasterise final-coordinate road depth straight at final resolution.

        No resize / crop / flip / rotate is applied (cache already aligned to
        the 576x1024 input).  Only in-bounds points are written.
        """
        H, W = final_dim
        depth_map = np.zeros((H, W), dtype=np.float32)
        if cam_depth.shape[0] == 0:
            return torch.Tensor(depth_map)
        x = cam_depth[:, 0]
        y = cam_depth[:, 1]
        d = cam_depth[:, 2]
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        xi = np.clip(x[inb].astype(np.int64), 0, W - 1)
        yi = np.clip(y[inb].astype(np.int64), 0, H - 1)
        depth_map[yi, xi] = d[inb].astype(np.float32)
        return torch.Tensor(depth_map)

    # ------------------------------------------------------------------ #
    # copied from the official NuscDetDataset.get_image; only the depth
    # rasterisation call is replaced by `_rasterize_final`.
    # ------------------------------------------------------------------ #
    def get_image(self, cam_infos, cams, lidar_infos=None):
        assert len(cam_infos) > 0
        sweep_imgs = list()
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_ida_mats = list()
        sweep_sensor2sensor_mats = list()
        sweep_timestamps = list()
        sweep_lidar_depth = list()
        if self.return_depth or self.use_fusion:
            sweep_lidar_points = list()
            for lidar_info in lidar_infos:
                lidar_path = lidar_info['LIDAR_TOP']['filename']
                lidar_points = np.fromfile(os.path.join(
                    self.data_root, lidar_path),
                                           dtype=np.float32,
                                           count=-1).reshape(-1, 5)[..., :4]
                sweep_lidar_points.append(lidar_points)
        for cam in cams:
            imgs = list()
            sensor2ego_mats = list()
            intrin_mats = list()
            ida_mats = list()
            sensor2sensor_mats = list()
            timestamps = list()
            lidar_depth = list()
            key_info = cam_infos[0]
            resize, resize_dims, crop, flip, \
                rotate_ida = self.sample_ida_augmentation()
            for sweep_idx, cam_info in enumerate(cam_infos):
                img = Image.open(
                    os.path.join(self.data_root, cam_info[cam]['filename']))
                w, x, y, z = cam_info[cam]['calibrated_sensor']['rotation']
                sweepsensor2sweepego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepsensor2sweepego_tran = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['translation'])
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
                w, x, y, z = cam_info[cam]['ego_pose']['rotation']
                sweepego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepego2global_tran = torch.Tensor(
                    cam_info[cam]['ego_pose']['translation'])
                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3, -1] = sweepego2global_tran

                w, x, y, z = key_info[cam]['ego_pose']['rotation']
                keyego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keyego2global_tran = torch.Tensor(
                    key_info[cam]['ego_pose']['translation'])
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3, -1] = keyego2global_tran
                global2keyego = keyego2global.inverse()

                w, x, y, z = key_info[cam]['calibrated_sensor']['rotation']
                keysensor2keyego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keysensor2keyego_tran = torch.Tensor(
                    key_info[cam]['calibrated_sensor']['translation'])
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3, -1] = keysensor2keyego_tran
                keyego2keysensor = keysensor2keyego.inverse()
                keysensor2sweepsensor = (
                    keyego2keysensor @ global2keyego @ sweepego2global
                    @ sweepsensor2sweepego).inverse()
                sweepsensor2keyego = global2keyego @ sweepego2global @\
                    sweepsensor2sweepego
                sensor2ego_mats.append(sweepsensor2keyego)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[:3, :3] = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['camera_intrinsic'])
                if self.return_depth and (self.use_fusion or sweep_idx == 0):
                    point_depth = self.get_lidar_depth(
                        sweep_lidar_points[sweep_idx], img,
                        lidar_infos[sweep_idx], cam_info[cam])
                    # cache already lives in final 576x1024 coords: no second
                    # resize/crop/flip/rotate, rasterise at final resolution.
                    point_depth_augmented = self._rasterize_final(
                        point_depth, self.ida_aug_conf['final_dim'])
                    lidar_depth.append(point_depth_augmented)
                img, ida_mat = self.img_transform_local(
                    img, resize, resize_dims, crop, flip, rotate_ida)
                ida_mats.append(ida_mat)
                img = mmcv.imnormalize(np.array(img), self.img_mean,
                                       self.img_std, self.to_rgb)
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs.append(img)
                intrin_mats.append(intrin_mat)
                timestamps.append(cam_info[cam]['timestamp'])
            sweep_imgs.append(torch.stack(imgs))
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
            sweep_timestamps.append(torch.tensor(timestamps))
            if self.return_depth:
                sweep_lidar_depth.append(torch.stack(lidar_depth))
        ego2global_rotation = np.mean(
            [key_info[cam]['ego_pose']['rotation'] for cam in cams], 0)
        ego2global_translation = np.mean(
            [key_info[cam]['ego_pose']['translation'] for cam in cams], 0)
        img_metas = dict(
            box_type_3d=__import__(
                'mmdet3d.core.bbox.structures.lidar_box3d',
                fromlist=['LiDARInstance3DBoxes']).LiDARInstance3DBoxes,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
        )

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0),
            img_metas,
        ]
        if self.return_depth:
            # Match official: (Nc, NS, H, W) -> (NS, Nc, H, W) so the camera
            # and sweep axes line up with the 6-camera depth prediction.
            ret_list.append(torch.stack(sweep_lidar_depth).permute(1, 0, 2, 3))
        return ret_list

    @staticmethod
    def img_transform_local(img, resize, resize_dims, crop, flip, rotate):
        """Mirror the official img_transform for the zero-aug path."""
        from bevdepth.datasets.nusc_det_dataset import img_transform
        return img_transform(img, resize, resize_dims, crop, flip, rotate)
