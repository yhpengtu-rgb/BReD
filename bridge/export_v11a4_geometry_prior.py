#!/usr/bin/env python3
"""Export frozen v8-a17 z-fine candidate BEV geometry features for v11-a4.

The v8-a17 source tree and checkpoint are read-only.  For every frame, six
candidates are expanded around the no-loop z-fine elevation with the final
confidence radius.  P2/P3/P4 candidate BEVs are built by the original
MultiViewCandidateBEV path and aggregated independently by the original cost
softmin rule, preserving 32 channels per scale (96 channels total).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
BRED_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zfine_candidate_features(refiner, image_features, z_fine, confidence, mats):
    """Return independent P2/P3/P4 cost-softmin BEVs and cell validity."""
    if z_fine.shape[0] != 1 or confidence.shape[0] != 1:
        raise ValueError("the formal exporter expects batch size one")
    size = refiner.cand_size
    z_small = F.interpolate(
        z_fine, size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(1)
    c_small = F.interpolate(
        confidence, size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(1)
    radius = refiner.radius_min + (1.0 - c_small) * (
        refiner.radius_max - refiner.radius_min
    )
    offsets = torch.linspace(
        -1.0, 1.0, refiner.num_candidates,
        device=z_fine.device, dtype=z_fine.dtype,
    ).view(1, refiner.num_candidates, 1, 1)
    z_candidates = z_small.unsqueeze(1) + offsets * radius.unsqueeze(1)
    z_candidates = z_candidates.clamp(refiner.z_min, refiner.z_max)

    scale_features = []
    cell_valid = None
    for feature in refiner.ms_fpn(image_features):
        candidate, cost, valid = refiner.cand_bev(feature, z_candidates, mats)
        # Apply the candidate cost softmin independently at each scale, before
        # any separate learned condition fusion so scale identity remains.
        cost_float = cost.float()
        cost_norm = cost_float / cost_float.mean(dim=1, keepdim=True).clamp_min(1e-6)
        score = -cost_norm / 0.1 + torch.log(valid.float().clamp_min(1e-6))
        weights = torch.softmax(score, dim=1)
        aggregated = (candidate.float() * weights).sum(dim=1)
        if aggregated.shape != (1, 32, size, size):
            raise ValueError(f"unexpected candidate BEV shape: {tuple(aggregated.shape)}")
        if not torch.isfinite(aggregated).all():
            raise FloatingPointError("non-finite z-fine candidate BEV feature")
        scale_features.append(aggregated[0].permute(1, 2, 0).reshape(-1, 32))
        visible = valid.sum(dim=1) > 0
        cell_valid = visible if cell_valid is None else torch.maximum(cell_valid, visible)
    features = torch.cat(scale_features, dim=-1)
    return features, cell_valid.reshape(-1), z_small.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8a17-source", default=str(BRED_ROOT / "stage1"))
    parser.add_argument("--info-path", default=str(BRED_ROOT / "assets/metadata/scene0523_allframe_infos.pkl"))
    parser.add_argument("--scene-name", default="scene-0523")
    parser.add_argument("--checkpoint", default=str(BRED_ROOT / "assets/checkpoints/v8a17_ms_last.pt"))
    parser.add_argument("--official-checkpoint", default=str(BRED_ROOT / "assets/checkpoints/bevstereo_ema_cbgs.pth"))
    parser.add_argument("--out-dir", default=str(BRED_ROOT / "assets/cache/geometry_export"))
    parser.add_argument("--data-root", default=str(BRED_ROOT / "assets/datasets/nuScenes"))
    parser.add_argument("--expected-frames", type=int, default=391)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = os.path.abspath(args.v8a17_source)
    sys.path.insert(0, source)
    from eval_v8a17_ms_rogs_z import (  # noqa: E402
        BEV_CELL, FINE_SIZE, REPO, BEVStereoRoGSModel, ContinuousLiftFPN,
        NuscDetDatasetRoadV8a4, build_zero_aug_conf, fixed_base_xy,
        load_experiment_class, official_collate, world_gaussians,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    expected = args.max_frames or args.expected_frames
    data_root = args.data_root

    Exp = load_experiment_class()
    lightning = Exp(
        gpus=0, batch_size_per_device=1, data_root=data_root,
        default_root_dir=str(BRED_ROOT / "results/bridge/lightning_geometry"),
    )
    official_sd = torch.load(args.official_checkpoint, map_location="cpu")["state_dict"]
    official_sd.pop("model.backbone.frustum", None)
    lightning.load_state_dict(official_sd, strict=False)
    lift = ContinuousLiftFPN(**lightning.backbone_conf)
    lift.load_official_backbone(lightning.model.backbone)
    lightning.model.backbone = lift
    model = BEVStereoRoGSModel(
        lightning.model, lift_backbone=lift, fine_size=FINE_SIZE,
        geometry_only=True, freeze_official_bn=False,
    ).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if checkpoint.get("epoch") != 6:
        raise RuntimeError(f"v11-a4 requires v8-a17 epoch 6, got {checkpoint.get('epoch')}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    refiner = model.rogs_head.refiner
    if refiner.num_candidates != 6 or refiner.cand_size != 128:
        raise RuntimeError("unexpected v8-a17 candidate configuration")

    ida, bda = build_zero_aug_conf(lightning.ida_aug_conf, lightning.bda_aug_conf)
    dataset = NuscDetDatasetRoadV8a4(
        ida_aug_conf=ida, bda_aug_conf=bda, data_root=data_root,
        info_paths=args.info_path, is_train=False, return_depth=False,
        sweep_idxes=lightning.sweep_idxes, key_idxes=lightning.key_idxes,
        use_fusion=lightning.use_fusion, classes=lightning.class_names,
        use_cbgs=False,
    )
    if len(dataset) != args.expected_frames:
        raise RuntimeError(f"expected {args.expected_frames} frames, got {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers else None,
        collate_fn=lambda samples: official_collate(samples, is_return_depth=False),
    )
    candidate_size = refiner.cand_size
    candidate_cell = 102.4 / candidate_size
    base_xy = fixed_base_xy(candidate_size, candidate_cell, device)
    started = time.perf_counter()
    records = []

    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= expected:
                break
            output_path = frames_dir / f"frame_{index:06d}.npz"
            imgs = batch[0].to(device, non_blocking=True)
            mats = {
                key: value.to(device, non_blocking=True)
                for key, value in batch[1].items()
            }
            timestamps = batch[2].to(device, non_blocking=True)
            meta = batch[3][0]
            initial, diagnostics = model(
                imgs, mats, timestamps, return_depth=True,
                refiner_use_noise=True, refiner_outer_loops=0,
                refiner_outer_seed=args.seed + index * 1009,
            )
            image_features = [model._img_feats.get(i) for i in range(3)]
            if any(feature is None for feature in image_features):
                raise RuntimeError("v8-a17 P2/P3/P4 hooks did not all fire")
            features, valid, z_small = zfine_candidate_features(
                refiner, image_features, initial["z"], diagnostics["conf"][-1], mats,
            )
            xyz_bda = torch.cat(
                (base_xy.reshape(-1, 2).float(), z_small.reshape(-1, 1).float()), dim=1
            )
            xyz_global = world_gaussians(xyz_bda, meta, mats["bda_mat"][0])
            xyz_np = xyz_global.float().cpu().numpy()
            feature_np = features.float().cpu().numpy().astype(np.float16)
            valid_np = valid.cpu().numpy().astype(np.bool_)
            if xyz_np.shape != (candidate_size * candidate_size, 3):
                raise ValueError(f"bad coordinate shape: {xyz_np.shape}")
            if feature_np.shape != (candidate_size * candidate_size, 96):
                raise ValueError(f"bad feature shape: {feature_np.shape}")
            if not np.isfinite(xyz_np).all() or not np.isfinite(feature_np).all():
                raise FloatingPointError(f"non-finite export at frame {index}")
            temporary = output_path.with_suffix(".tmp.npz")
            np.savez(temporary, xyz_global=xyz_np, features=feature_np, valid=valid_np)
            os.replace(temporary, output_path)
            records.append({
                "frame": index,
                "file": f"frames/{output_path.name}",
                "points": int(len(xyz_np)),
                "feature_dim": 96,
                "valid_fraction": float(valid_np.mean()),
                "feature_abs_mean": float(np.abs(feature_np).mean()),
            })
            if index % 10 == 0 or index + 1 == expected:
                print(
                    f"V11A4_GEOMETRY_EXPORT frame={index + 1}/{expected} "
                    f"valid={valid_np.mean():.6f} "
                    f"feature_abs_mean={np.abs(feature_np).mean():.8f} "
                    f"elapsed_s={time.perf_counter() - started:.1f}", flush=True,
                )

    if len(records) != expected:
        raise RuntimeError(f"incomplete export: {len(records)}/{expected}")
    manifest = {
        "experiment": "v11-a4-geometry-export",
        "scene_name": args.scene_name,
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "source_checkpoint_epoch": int(checkpoint["epoch"]),
        "source_tree": source,
        "frames": records,
        "frame_count": len(records),
        "points_per_frame": candidate_size * candidate_size,
        "feature_dim": 96,
        "scale_channels": [32, 32, 32],
        "scale_names": ["P2", "P3", "P4"],
        "candidate_count": refiner.num_candidates,
        "candidate_rule": "six elevations around no-loop z-fine with final-confidence radius",
        "candidate_aggregation": "independent per-scale cost softmin",
        "coordinate_frame": "nuScenes global",
        "source_grid": {
            "size": candidate_size, "cell_m": candidate_cell,
            "xy": "fixed cell centres; learned XY offsets ignored",
        },
        "dtype": {"xyz_global": "float32", "features": "float16", "valid": "bool"},
        "bn_stats": "running",
        "forward_dtype": "fp32",
        "frozen_inference_only": True,
        "elapsed_s": time.perf_counter() - started,
    }
    temporary = out_dir / "manifest.tmp.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, out_dir / "manifest.json")
    print(f"V11A4_GEOMETRY_EXPORT_DONE frames={len(records)} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
