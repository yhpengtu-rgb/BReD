"""One-real-view RoGS integration smoke for v11-a9 bias-free G97 only."""
import argparse

import addict
import bred_paths
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from diff_gaussian_rasterization.scene.cameras import PerspectiveCamera

from datasets.nusc import NuscDataset
from models.exposure_model import ExposureModel
from models.gaussian_model import GaussianModel2D
from models.gifsplat_residual import GIFSplatResidualRefiner
from models.loss import CELossWithMask, L1MaskedLoss
from models.road import Road


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scene0523.yaml")
    parser.add_argument("--materialize-views", type=int, default=0)
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = bred_paths.resolve_config_paths(addict.Dict(yaml.safe_load(handle)))
    device = torch.device("cuda:0")
    dataset = NuscDataset(cfg.dataset, use_label=True, use_depth=False)
    road = Road(cfg.model, dataset, device=device)
    gaussians = GaussianModel2D(cfg.model)
    gaussians.init_2d_gaussian(
        road.vertices, road.rotation, road.rgb, road.label, road.resolution,
        road.ref_pose, dataset.cameras_extent,
    )
    exposure = ExposureModel(num_camera=len(dataset.camera_names)).to(device)
    refiner = GIFSplatResidualRefiner(
        gaussians, cfg.model.gifsplat_residual,
        geometric_prior=road.v11a4_geometry_prior,
        geometric_prior_valid=road.v11a4_geometry_valid,
    ).to(device)
    assert torch.equal(refiner.initial_z, road.vertices[:, 2:3])
    assert refiner.geometry_prior.shape == (road.vertices.shape[0], 97)
    assert refiner.geometry_prior_valid.any()
    assert not hasattr(road, "v11a5_texture_prior")
    assert not hasattr(road, "v11a6_semantic_prior")
    assert road.v11a4_geometry_stats is not None and road.v11a2_rgbz_stats is not None
    assert torch.isfinite(road.rgb).all() and road.rgb.abs().max() > 0
    for name in ("v11a4_geometry_prior", "v11a4_geometry_valid"):
        setattr(road, name, None)
    torch.cuda.empty_cache()
    sample = next(iter(DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)))
    for key, value in sample.items():
        sample[key] = value[0] if key == "image_name" else value[0].to(device)
    cam = PerspectiveCamera(
        sample["R"], sample["T"], sample["K"], sample["W"], sample["H"], 1, 20, device
    )
    bg = torch.zeros(3, device=device)
    stages = refiner.training_stages(
        cam, gaussians, sample, cfg.pipeline, bg, exposure, sample["cam_idx"].item()
    )
    numerator, mass = refiner.observation_numerator_mass(
        refiner.initial_view(), sample, stages[0].render_pkg, stages[0].label_pkg,
        exposure, sample["cam_idx"].item(),
    )
    assert torch.isfinite(numerator).all() and mass.sum() > 0
    l1 = L1MaskedLoss()
    ce = CELossWithMask()

    def compute_loss(stage_outputs):
        total = 0.0
        for stage, weight in zip(stage_outputs, (0.4, 0.6)):
            mask = ((stage.render_pkg["depth"][0] > 1) & (stage.render_pkg["depth"][0] < 20)).float()
            mask = mask * sample["mask"]
            rgb = exposure(sample["cam_idx"].item(), stage.render_pkg["render"]).permute(1, 2, 0)
            seg = stage.label_pkg["render"].permute(1, 2, 0)
            total = total + weight * l1(rgb, sample["image"], mask[:, :, None]).mean()
            total = total + weight * 0.06 * ce(
                seg.reshape(-1, seg.shape[-1]), sample["label"].long().reshape(-1), mask.reshape(-1)
            )
        return total

    total_loss = compute_loss(stages)
    total_loss.backward()
    assert torch.isfinite(total_loss)
    for output_head in refiner.head.output_heads():
        assert output_head.weight.grad is not None and output_head.weight.grad.abs().sum() > 0
    refiner.optimizer.step()
    refiner.optimizer.zero_grad(set_to_none=True)
    exposure.zero_grad(set_to_none=True)
    output_weight_sum = sum(
        output_head.weight.abs().sum().item() for output_head in refiner.head.output_heads()
    )
    post_step_stages = refiner.training_stages(
        cam, gaussians, sample, cfg.pipeline, bg, exposure, sample["cam_idx"].item()
    )
    post_step_loss = compute_loss(post_step_stages)
    post_step_loss.backward()
    assert refiner.head.step_embedding.weight.grad.abs().sum() > 0
    assert not hasattr(refiner.head, "fusion")
    for index, encoder in enumerate(refiner.head.linear_encoders()):
        assert encoder[-1].weight.grad is not None and encoder[-1].weight.grad.abs().sum() > 0, index
    assert refiner.geometry_prior.grad is None
    refiner.optimizer.zero_grad(set_to_none=True)
    exposure.zero_grad(set_to_none=True)
    result = {
        "real_view_forward_backward": "PASS",
        "v10a1_z_exact": "PASS",
        "v11a2_rgbz_initialization": road.v11a2_rgbz_stats,
        "v11a4_geometry_prior": road.v11a4_geometry_stats,
        "texture_and_semantic_removed": "PASS",
        "all_11_branch_gradients": "PASS",
        "no_shared_fusion": "PASS",
        "loss": total_loss.item(),
        "post_step_loss": post_step_loss.item(),
        "observed_gaussians": int((mass[:, 0] > 0).sum()),
        "num_gaussians": mass.shape[0],
        "head_weight_abs_sum_after_step": output_weight_sum,
        "head_structure": {
            "latent_dim": refiner.head.latent_dim,
            "evidence_dim": refiner.head.evidence_dim,
            "geometric_prior_dim": refiner.head.geometric_prior_dim,
            "prior_mode": "bias_free_geometry97_only",
            "prior_valid_shape": [mass.shape[0], 1],
            "branch_dim": refiner.head.branch_dim,
            "concat_dim": refiner.head.concat_dim,
            "independent_linear_groups": len(refiner.head.linear_encoders()),
            "output_head_type": "three_independent_linear_heads",
            "output_dim": refiner.head.output_dim,
            "steps": refiner.steps,
            "trainable_head_parameters": sum(p.numel() for p in refiner.head.parameters()),
        },
    }
    if args.materialize_views > 0:
        stats = refiner.materialize(
            Subset(dataset, range(args.materialize_views)), cfg.pipeline, bg, exposure, device
        )
        initial_fields = {
            "z": refiner.initial_z, "rgb": refiner.initial_rgb,
            "label": refiner.initial_label, "scaling": refiner.initial_scaling,
            "rotation": refiner.initial_rotation, "opacity": refiner.initial_opacity,
        }
        current_fields = {
            "z": gaussians._z, "rgb": gaussians._rgb, "label": gaussians._label,
            "scaling": gaussians._scaling_xy, "rotation": gaussians._rotation,
            "opacity": gaussians._opacity,
        }
        changes = {}
        for name, current in current_fields.items():
            assert torch.isfinite(current).all(), name
            changes[name] = (current - initial_fields[name]).abs().max().item()
            assert changes[name] > 0, name
        result["materialize_smoke"] = "PASS"
        result["materialize_stats"] = stats
        result["max_abs_changes"] = changes
    print(result)


if __name__ == "__main__":
    main()
