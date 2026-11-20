#!/usr/bin/env python3
"""Prepare one scene without changing the locked BReD two-stage protocol.

Builds the official all-frame BEVDepth infos, the sparse RoadGT supervision
cache, and a scene-specific stage2 config. All inputs and outputs stay under
BReD unless explicit paths are supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch
import yaml
from nuscenes.nuscenes import NuScenes
from plyfile import PlyData
from pyquaternion import Quaternion
from pytorch3d.ops.knn import knn_points

BRED_ROOT = Path(__file__).resolve().parents[1]
CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK",
        "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
LABEL_REMAP = np.full(256, 6, dtype=np.uint8)
for raw in (7, 8, 14, 23, 24):
    LABEL_REMAP[raw] = 1
for raw in (2, 9):
    LABEL_REMAP[raw] = 2
for raw in (41, 13):
    LABEL_REMAP[raw] = 3
LABEL_REMAP[15] = 4
LABEL_REMAP[29] = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_infos(nusc, scene):
    samples = {}
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        samples[sample["token"]] = sample
        token = sample["next"] or None
    by_cam = {cam: [] for cam in CAMS}
    lidars = []
    for data in nusc.sample_data:
        if data["sample_token"] not in samples:
            continue
        if data["channel"] in by_cam:
            by_cam[data["channel"]].append(data)
        elif data["channel"] == "LIDAR_TOP":
            lidars.append(data)
    for values in by_cam.values():
        values.sort(key=lambda item: item["timestamp"])
    lidars.sort(key=lambda item: item["timestamp"])

    def entry(data, parent, camera):
        result = {
            "sample_token": parent,
            "ego_pose": nusc.get("ego_pose", data["ego_pose_token"]),
            "timestamp": data["timestamp"],
            "is_key_frame": data["is_key_frame"],
            "filename": data["filename"],
            "calibrated_sensor": nusc.get("calibrated_sensor", data["calibrated_sensor_token"]),
        }
        if camera:
            result.update(height=data["height"], width=data["width"])
        return result

    infos = []
    for lidar in lidars:
        parent = lidar["sample_token"]
        cam_tokens = {}
        cam_infos = {}
        for cam in CAMS:
            if lidar["is_key_frame"]:
                data = nusc.get("sample_data", samples[parent]["data"][cam])
            else:
                data = min(by_cam[cam], key=lambda item: abs(item["timestamp"] - lidar["timestamp"]))
            cam_tokens[cam] = data["token"]
            cam_infos[cam] = entry(data, parent, True)
        annotations = []
        for ann_token in samples[parent]["anns"]:
            annotation = dict(nusc.get("sample_annotation", ann_token))
            annotation["velocity"] = nusc.box_velocity(ann_token)
            annotations.append(annotation)
        infos.append({
            "sample_token": parent,
            "timestamp": lidar["timestamp"],
            "scene_token": scene["token"],
            "cam_infos": cam_infos,
            "lidar_infos": {"LIDAR_TOP": entry(lidar, parent, False)},
            "cam_sweeps": [], "lidar_sweeps": [], "ann_infos": annotations,
            "parent_sample_token": parent,
            "allframe_anchor": {
                "reason": "keyframe_lidar" if lidar["is_key_frame"] else "sweep_lidar",
                "reference_sample_data_token": lidar["token"],
                "camera_sample_data_tokens": cam_tokens,
                "lidar_sample_data_token": lidar["token"],
            },
        })
    return infos


def validate_files(infos, data_root: Path, image_root: Path):
    missing = []
    for info in infos:
        lidar = data_root / info["lidar_infos"]["LIDAR_TOP"]["filename"]
        if not lidar.is_file():
            missing.append(str(lidar))
        for cam in CAMS:
            relative = info["cam_infos"][cam]["filename"]
            if not (data_root / relative).is_file():
                missing.append(str(data_root / relative))
            seg = image_root / relative.replace("/CAM", "/seg_CAM").replace(".jpg", ".png")
            if not seg.is_file():
                missing.append(str(seg))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} scene files; first: {missing[:8]}")


def keyframe_poses(nusc, scene):
    poses = []
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego = nusc.get("ego_pose", data["ego_pose_token"])
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Quaternion(ego["rotation"]).rotation_matrix
        matrix[:3, 3] = ego["translation"]
        poses.append(matrix)
        token = sample["next"]
    if not poses:
        raise RuntimeError("Scene has no keyframe poses")
    return np.stack(poses)


def build_cache(nusc, scene, ply_path: Path, output: Path, device: str):
    poses = keyframe_poses(nusc, scene)
    pose_xy = poses[:, :2, 3]
    # Accepted cache: 24 m outer ROI and 24 m trajectory dilation. Keep the
    # bounds float64 during linspace; they are cast only when serialized.
    min_xy = pose_xy.min(axis=0) - 24.0
    max_xy = pose_xy.max(axis=0) + 24.0
    resolution = 0.05
    nx = int(round((max_xy[0] - min_xy[0]) / resolution)) + 1
    ny = int(round((max_xy[1] - min_xy[1]) / resolution)) + 1
    xs = np.linspace(min_xy[0], max_xy[0], nx, dtype=np.float32)
    ys = np.linspace(min_xy[1], max_xy[1], ny, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    vertices = np.stack((gx.reshape(-1), gy.reshape(-1)), axis=1)
    px = np.clip(np.round((pose_xy[:, 0] - min_xy[0]) / resolution).astype(np.int64), 0, nx - 1)
    py = np.clip(np.round((pose_xy[:, 1] - min_xy[1]) / resolution).astype(np.int64), 0, ny - 1)
    mask = np.zeros((nx, ny), dtype=np.uint8)
    mask[px, py] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (480, 480))
    keep = cv2.dilate(mask, kernel, iterations=1).reshape(-1).astype(bool)
    xy = vertices[keep].astype(np.float32)

    road = PlyData.read(str(ply_path)).elements[0]
    xyz = np.stack((np.asarray(road["x"]), np.asarray(road["y"]), np.asarray(road["z"])), axis=1).astype(np.float32)
    rgb = np.stack((np.asarray(road["r"]), np.asarray(road["g"]), np.asarray(road["b"])), axis=1).astype(np.float32)
    if rgb.size and rgb.max() > 1.5:
        rgb /= 255.0
    raw_label = np.asarray(road["label"]).astype(np.int64)
    label = LABEL_REMAP[np.clip(raw_label, 0, 255)].astype(np.int64)
    if not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
        raise FloatingPointError("RoadGT PLY contains non-finite values")

    query = torch.from_numpy(xy).to(device)
    target = torch.from_numpy(xyz[:, :2]).to(device)
    knn = knn_points(query[None], target[None], K=1, return_nn=False)
    distances = knn.dists[0, :, 0].cpu().numpy()
    indices = knn.idx[0, :, 0].cpu().numpy()
    valid = distances < 0.1 ** 2
    tgt_z = xyz[indices, 2]
    tgt_rgb = rgb[indices]
    tgt_label = label[indices]
    tgt_z[~valid] = 0.0
    tgt_rgb[~valid] = 0.0
    tgt_label[~valid] = 6
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(
        temporary, xy=xy, tgt_z=tgt_z.astype(np.float32), tgt_rgb=tgt_rgb.astype(np.float32),
        tgt_label=tgt_label, tgt_valid=valid, min_xy=min_xy.astype(np.float32),
        max_xy=max_xy.astype(np.float32),
        center_xy=(min_xy.astype(np.float32) + max_xy.astype(np.float32)) / np.float32(2.0),
        grid_shape=np.array((nx, ny)), resolution=np.float32(resolution),
        ref_pose=np.eye(4, dtype=np.float32),
    )
    os.replace(temporary, output)
    return {"grid_shape": [nx, ny], "sparse_vertices": int(len(xy)),
            "valid_cells": int(valid.sum()), "road_points": int(len(xyz))}


def write_stage2_config(scene: str, scene_id: str, frames: int, output: Path):
    base = BRED_ROOT / "stage2/configs/scene0523.yaml"
    config = yaml.safe_load(base.read_text())
    cache = f"assets/cache/{scene_id}"
    config["output"] = f"results/stage2/{scene_id}_rogs_wolidar_6ep"
    config["dataset"]["clip_list"] = [scene]
    config["model"]["v10a1_expected_frames"] = frames
    config["model"]["v11a4_geometry_expected_frames"] = frames
    for key in ("v10a1_elevation_dir", "v10a1_cache_dir", "v11a2_rgb_cache_dir",
                "v11a4_geometry_export_dir", "v11a4_geometry_cache_dir"):
        leaf = config["model"][key].split("/")[-1]
        config["model"][key] = f"{cache}/{leaf}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data-root", default=str(BRED_ROOT / "assets/datasets/nuScenes"))
    parser.add_argument("--image-root", default=str(BRED_ROOT / "assets/datasets/nuScenes_clip"))
    parser.add_argument("--road-ply", default="")
    parser.add_argument("--info-out", default="")
    parser.add_argument("--cache-out", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if len(args.scene) != 10 or not args.scene.startswith("scene-") or not args.scene[6:].isdigit():
        raise ValueError("Expected scene-XXXX")
    scene_id = args.scene.replace("-", "")
    data_root, image_root = Path(args.data_root), Path(args.image_root)
    ply = Path(args.road_ply) if args.road_ply else BRED_ROOT / f"assets/datasets/nuScenes_road_gt/{args.scene}.ply"
    info_out = Path(args.info_out) if args.info_out else BRED_ROOT / f"assets/metadata/{scene_id}_allframe_infos.pkl"
    cache_out = Path(args.cache_out) if args.cache_out else BRED_ROOT / f"assets/metadata/{scene_id}_roadgt_rogs_cache.npz"
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(data_root), verbose=False)
    scene = next((item for item in nusc.scene if item["name"] == args.scene), None)
    if scene is None:
        raise FileNotFoundError(args.scene)

    if info_out.exists() and not args.force:
        with info_out.open("rb") as handle:
            infos = pickle.load(handle)
    else:
        infos = build_infos(nusc, scene)
        info_out.parent.mkdir(parents=True, exist_ok=True)
        temporary = info_out.with_suffix(info_out.suffix + f".tmp{os.getpid()}")
        with temporary.open("wb") as handle:
            pickle.dump(infos, handle)
        os.replace(temporary, info_out)
    validate_files(infos, data_root, image_root)
    cache_stats = None
    if not cache_out.exists() or args.force:
        if not ply.is_file():
            raise FileNotFoundError(ply)
        cache_stats = build_cache(nusc, scene, ply, cache_out, args.device)
    if args.scene != "scene-0523":
        write_stage2_config(args.scene, scene_id, len(infos), BRED_ROOT / f"stage2/configs/{scene_id}.yaml")
    summary = {
        "scene": args.scene, "frames": len(infos),
        "keyframes": sum(bool(item["allframe_anchor"]["reason"] == "keyframe_lidar") for item in infos),
        "info": str(info_out), "info_sha256": sha256(info_out),
        "roadgt_cache": str(cache_out), "roadgt_cache_sha256": sha256(cache_out),
        "road_ply": str(ply), "cache_build": cache_stats,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
