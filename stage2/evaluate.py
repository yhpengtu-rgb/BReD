"""独立评测 BReD 第二阶段保存的逐轮 checkpoint。

checkpoint 已包含当轮两步残差物化后的 Gaussian；此处不再次运行残差头，
只重放正式 BEV 渲染与 eval_bev_metric / eval_z_metric 口径。
"""
import argparse
import json
from pathlib import Path

import addict
import cv2
import numpy as np
import torch
import yaml

import bred_paths
from datasets.nusc import NuscDataset
from evalution import eval_bev_metric, eval_z_metric
from models.exposure_model import ExposureModel
from models.gaussian_model import GaussianModel2D
from models.road import Road
from train import gt_render, set_randomness
from utils.image import render_semantic
from utils.render import render_blocks


def load_config(path):
    with open(path) as handle:
        cfg = addict.Dict(yaml.safe_load(handle))
    return bred_paths.resolve_config_paths(cfg)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def evaluate(config_path, run_dir, epoch=None, output_dir=None):
    cfg = load_config(config_path)
    run_dir = Path(run_dir).resolve()
    output_root = Path(output_dir).resolve() if output_dir else run_dir / "standalone_eval"
    output_root.mkdir(parents=True, exist_ok=True)
    set_randomness(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dataset = NuscDataset(cfg.dataset, use_label=cfg.optimization.seg_loss_weight > 0,
                          use_depth=cfg.optimization.depth_loss_weight > 0)
    road = Road(cfg.model, dataset, device=device)
    gaussians = GaussianModel2D(cfg.model)
    cfg.optimization.position_lr_max_steps = len(dataset) * cfg.optimization.epochs
    exposure = ExposureModel(num_camera=len(dataset.camera_names)).to(device)
    background = torch.tensor([1, 1, 1] if cfg.model.white_background else [0, 0, 0],
                              dtype=torch.float32, device=device)
    bev_cam = road.bev_camera
    road_pointcloud = dataset.road_pointcloud.copy()
    for key, value in road_pointcloud.items():
        road_pointcloud[key] = torch.from_numpy(value.astype(np.float32)).to(device)
    gt_root = run_dir / "road_point" if (run_dir / "road_point" / "bev_mask.png").exists() else output_root / "road_point"
    if not (gt_root / "bev_mask.png").exists():
        gt_render(road_pointcloud, dataset.label_remaps, dataset.filted_color_map,
                  road.min_xy, road.max_xy, bev_cam.cam2world[2, 3],
                  wh=(bev_cam.image_width, bev_cam.image_height), save_root=str(gt_root),
                  device=device)

    if epoch is None:
        checkpoints = sorted(run_dir.glob("epoch_*.pth"))
    else:
        checkpoints = [run_dir / f"epoch_{epoch:03d}.pth"]
    if not checkpoints:
        raise FileNotFoundError(f"No epoch_XXX.pth checkpoints in {run_dir}")

    all_metrics = []
    for checkpoint_path in checkpoints:
        epoch_number = int(checkpoint_path.stem.split("_")[-1])
        current_root = output_root / f"epoch_{epoch_number:03d}"
        current_root.mkdir(parents=True, exist_ok=True)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        gaussians.restore(checkpoint, cfg.optimization)
        exposure_path = run_dir / f"exposure_epoch_{epoch_number:03d}.pth"
        if cfg.model.use_exposure:
            exposure.load_state_dict(torch.load(exposure_path, map_location=device))
        exposure.eval()
        with torch.no_grad():
            bev_pkg = render_blocks(bev_cam, gaussians, cfg.pipeline, background, render_type="rgb")
            src_bev_image, bev_depth, bev_mask = bev_pkg["render"], bev_pkg["depth"], bev_pkg["mask"]
            bev_mask = bev_mask.astype(np.uint8)
            bev_mask = cv2.erode(bev_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))) > 0
            cv2.imwrite(str(current_root / "bev_mask.png"), bev_mask.astype(np.uint8) * 255)
            bev_image = exposure.apply_numpy(0, src_bev_image) if cfg.model.use_exposure else src_bev_image
            bev_image = (bev_image.transpose(1, 2, 0) * 255).astype(np.uint8)
            bev_image = cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGRA)
            bev_image[~bev_mask] = 0
            cv2.imwrite(str(current_root / "bev_image.png"), bev_image)
            label_pkg = render_blocks(bev_cam, gaussians, cfg.pipeline, background, render_type="label")
            bev_label = np.argmax(label_pkg["render"].transpose(1, 2, 0), axis=-1)
            cv2.imwrite(str(current_root / "bev_label.png"), bev_label)
            bev_label_vis = cv2.cvtColor(render_semantic(bev_label, dataset.filted_color_map), cv2.COLOR_RGB2BGRA)
            bev_label_vis[~bev_mask] = 0
            cv2.imwrite(str(current_root / "bev_label_vis.png"), bev_label_vis)
            np.save(current_root / "bev_height.npy", bev_cam.cam2world[2, 3].item() - bev_depth[0])
            bev_metric = eval_bev_metric(str(gt_root), str(current_root), dataset.num_class)
            z_rmse = eval_z_metric(road_pointcloud["xyz"], gaussians.get_xyz)
        metrics = {"epoch": epoch_number, "checkpoint": str(checkpoint_path),
                   "bev": bev_metric, "z_rmse_m": float(z_rmse)}
        (current_root / "metrics.json").write_text(json.dumps(metrics, indent=2, default=json_default) + "\n")
        all_metrics.append(metrics)
        print(json.dumps(metrics, default=json_default), flush=True)
    (output_root / "metrics_all.json").write_text(json.dumps(all_metrics, indent=2, default=json_default) + "\n")
    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs/scene0523.yaml"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    evaluate(args.config, args.run_dir, args.epoch, args.output_dir)


if __name__ == "__main__":
    main()
