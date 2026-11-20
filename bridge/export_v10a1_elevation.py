#!/usr/bin/env python3
"""Export frozen v8-a17 no-loop fine elevation as global-frame XYZ.

This is an inference-only helper.  It deliberately ignores learned XY offsets:
RoGS owns its fixed scene-grid XY, while v8-a17 supplies elevation only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

BRED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRED_ROOT / "stage1"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eval_v8a17_ms_rogs_z import (
    BEV_CELL, FINE_SIZE, REPO, BEVStereoRoGSModel, ContinuousLiftFPN,
    NuscDetDatasetRoadV8a4, build_zero_aug_conf, fixed_base_xy,
    load_experiment_class, official_collate, world_gaussians,
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--info-path", default=str(BRED_ROOT / "assets/metadata/scene0523_allframe_infos.pkl"))
    parser.add_argument("--scene-name", default="scene-0523")
    parser.add_argument("--checkpoint", default=str(BRED_ROOT / "assets/checkpoints/v8a17_ms_last.pt"))
    parser.add_argument("--official-checkpoint", default=str(BRED_ROOT / "assets/checkpoints/bevstereo_ema_cbgs.pth"))
    parser.add_argument("--out-dir", default=str(BRED_ROOT / "assets/cache/elevation_export"))
    parser.add_argument("--data-root", default=str(BRED_ROOT / "assets/datasets/nuScenes"))
    parser.add_argument("--expected-frames", type=int, default=391)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    expected = args.max_frames or args.expected_frames

    Exp = load_experiment_class()
    lightning = Exp(gpus=0, batch_size_per_device=1, data_root=args.data_root,
                    default_root_dir=str(BRED_ROOT / "results/bridge/lightning_elevation"))
    official_sd = torch.load(args.official_checkpoint, map_location="cpu")["state_dict"]
    official_sd.pop("model.backbone.frustum", None)
    lightning.load_state_dict(official_sd, strict=False)
    lift = ContinuousLiftFPN(**lightning.backbone_conf)
    lift.load_official_backbone(lightning.model.backbone)
    lightning.model.backbone = lift
    model = BEVStereoRoGSModel(lightning.model, lift_backbone=lift,
                               fine_size=FINE_SIZE, geometry_only=True,
                               freeze_official_bn=False).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if checkpoint.get("epoch") != 6:
        raise RuntimeError(f"v10-a1 requires v8-a17 last.pt epoch 6, got {checkpoint.get('epoch')}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    ida, bda = build_zero_aug_conf(lightning.ida_aug_conf, lightning.bda_aug_conf)
    dataset = NuscDetDatasetRoadV8a4(
        ida_aug_conf=ida, bda_aug_conf=bda, data_root=args.data_root,
        info_paths=args.info_path, is_train=False, return_depth=False,
        sweep_idxes=lightning.sweep_idxes, key_idxes=lightning.key_idxes,
        use_fusion=lightning.use_fusion, classes=lightning.class_names,
        use_cbgs=False)
    if len(dataset) != args.expected_frames:
        raise RuntimeError(f"expected {args.expected_frames} frames, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers else None,
                        collate_fn=lambda samples: official_collate(samples, is_return_depth=False))
    base_xy = fixed_base_xy(FINE_SIZE, BEV_CELL, device)
    started = time.perf_counter()
    records = []

    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= expected:
                break
            output_path = frames_dir / f"frame_{index:06d}.npy"
            imgs = batch[0].to(device, non_blocking=True)
            mats = {key: value.to(device, non_blocking=True)
                    for key, value in batch[1].items()}
            timestamps = batch[2].to(device, non_blocking=True)
            meta = batch[3][0]
            initial, _ = model(imgs, mats, timestamps, return_depth=True,
                               refiner_use_noise=True, refiner_outer_loops=0,
                               refiner_outer_seed=args.seed + index * 1009)
            # Fixed lattice + fine no-loop Z.  No learned XY, Gaussian parameters,
            # RGB projection, coarse output, or outer-loop output enters RoGS.
            xy = base_xy.reshape(-1, 2).to(dtype=initial["z"].dtype)
            z = initial["z"][0].permute(1, 2, 0).reshape(-1, 1).float()
            xyz_bda = torch.cat((xy.float(), z), dim=1)
            xyz_global = world_gaussians(xyz_bda, meta, mats["bda_mat"][0])
            xyz_global = xyz_global.float().cpu().numpy()
            if xyz_global.shape != (FINE_SIZE * FINE_SIZE, 3) or not np.isfinite(xyz_global).all():
                raise FloatingPointError(f"invalid elevation at frame {index}: {xyz_global.shape}")
            temporary = output_path.with_suffix(".tmp.npy")
            np.save(temporary, xyz_global, allow_pickle=False)
            os.replace(temporary, output_path)
            records.append({"frame": index, "file": f"frames/{output_path.name}",
                            "points": int(len(xyz_global))})
            if index % 10 == 0 or index + 1 == expected:
                print(f"V10A1_EXPORT frame={index + 1}/{expected} "
                      f"z_min={xyz_global[:, 2].min():.6f} z_max={xyz_global[:, 2].max():.6f} "
                      f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)

    if len(records) != expected:
        raise RuntimeError(f"incomplete export: {len(records)}/{expected}")
    manifest = {
        "experiment": "v10-a1",
        "scene_name": args.scene_name,
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "source_checkpoint_epoch": int(checkpoint["epoch"]),
        "frames": records,
        "frame_count": len(records),
        "points_per_frame": FINE_SIZE * FINE_SIZE,
        "coordinate_frame": "nuScenes global",
        "source_grid": {"size": FINE_SIZE, "cell_m": BEV_CELL,
                        "xy": "fixed cell centres; learned XY offsets ignored"},
        "elevation": "initial[z] after one frozen v8-a17 residual-refiner forward; no outer loop",
        "dtype": "float32",
        "bn_stats": "running",
        "forward_dtype": "fp32",
        "diffusion_trainable": False,
        "rogs_z_trainable_after_initialization": True,
        "elapsed_s": time.perf_counter() - started,
    }
    temporary = out_dir / "manifest.tmp.json"
    with open(temporary, "w") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(temporary, out_dir / "manifest.json")
    print(f"V10A1_EXPORT_DONE frames={len(records)} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
