"""GPU contract checks for v11-a9: G-only 384D concat and linear heads."""
from types import SimpleNamespace
import torch
from torch import nn
from models.gaussian_model import GaussianModel2D
from models.gifsplat_residual import GIFSplatResidualRefiner


def main():
    torch.manual_seed(17)
    count, classes, geometry_dim, branch_dim = 16, 7, 97, 32
    cfg = SimpleNamespace(sh_degree=3, use_rgb=True, opt_xy=False)
    gaussians = GaussianModel2D(cfg)
    xyz = torch.randn(count, 3, device="cuda")
    xyz[:, 2] = torch.linspace(-0.3, 0.4, count, device="cuda")
    rotation = torch.zeros(count, 4, device="cuda")
    rotation[:, 0] = 1
    gaussians.init_2d_gaussian(
        xyz, rotation, torch.randn(count, 3, device="cuda"),
        torch.randn(count, classes, device="cuda"), 0.05,
        torch.eye(4, device="cuda"), 10.0,
    )
    refiner = GIFSplatResidualRefiner(
        gaussians,
        {"steps": 2, "branch_dim": branch_dim, "geometric_prior_dim": geometry_dim,
         "residual_lr": 0.001, "z_delta_scale": 0.05,
         "rgb_delta_scale": 1.0, "label_delta_scale": 1.0},
        geometric_prior=torch.randn(count, geometry_dim - 1, device="cuda"),
        geometric_prior_valid=torch.ones(count, dtype=torch.bool, device="cuda"),
    )
    head = refiner.head
    assert not hasattr(head, "fusion")
    assert not hasattr(head, "texture_encoder")
    assert not hasattr(head, "semantic_encoder")
    assert len(head.linear_encoders()) == 11
    assert head.concat_dim == 384
    assert all(isinstance(output, nn.Linear) for output in head.output_heads())
    assert all(output.in_features == 384 for output in head.output_heads())
    assert all(encoder[-1].out_features == 32 for encoder in head.linear_encoders())
    assert all(len(encoder) == 1 and isinstance(encoder[0], nn.Linear)
               for encoder in tuple(head.state_encoders) + tuple(head.evidence_encoders))
    assert head.geometric_encoder[0].bias is None
    assert head.geometric_encoder[1].bias is None
    assert sum(p.numel() for p in head.parameters()) == 11507
    assert torch.equal(refiner.initial_z, xyz[:, 2:3])
    fixed_priors, fixed_valid = refiner.prior_inputs()
    assert len(fixed_priors) == 1 and fixed_priors[0].shape == (count, geometry_dim)
    assert fixed_valid.shape == (count, 1) and fixed_valid.all()
    assert torch.equal(fixed_priors[0][:, -1:], refiner.initial_z)
    evidence = torch.randn(count, 3 + classes + 1, device="cuda")
    initial = refiner.initial_view()
    refined = refiner.apply_residual(initial, evidence, step=0)
    for name in ("z", "rgb", "label", "scaling", "rotation", "opacity"):
        assert torch.equal(
            getattr(refined, f"_refined_{name}"), getattr(initial, f"_refined_{name}")
        ), name

    with torch.no_grad():
        for output_head in head.output_heads():
            output_head.bias.fill_(0.01)
    refined = refiner.apply_residual(initial, evidence, step=0)
    assert torch.allclose(refined._refined_z, initial._refined_z + 0.0005, atol=1e-6)
    assert torch.allclose(refined._refined_rgb, initial._refined_rgb + 0.01, atol=1e-6)
    assert torch.allclose(refined._refined_scaling, initial._refined_scaling + 0.001, atol=1e-6)
    with torch.no_grad():
        for output_head in head.output_heads():
            output_head.bias.fill_(100)
    saturated = refiner.apply_residual(initial, evidence, step=0)
    assert torch.allclose(saturated._refined_z, initial._refined_z + 0.05)
    assert torch.allclose(saturated._refined_scaling, initial._refined_scaling + 0.1)
    assert torch.allclose(saturated._refined_rotation, initial._refined_rotation + 0.1)
    detached = saturated.detached()
    assert not detached._refined_z.requires_grad

    with torch.no_grad():
        for output_head in head.output_heads():
            output_head.weight.fill_(0.01)
            output_head.bias.zero_()
    latent = refiner.tokens(initial)
    out_step0 = head(latent, evidence, step=0)
    out_step1 = head(latent, evidence, step=1)
    assert not torch.allclose(out_step0, out_step1)
    zero_geometry = torch.zeros(count, geometry_dim, device="cuda")
    zero_valid = torch.zeros(count, 1, device="cuda")
    one_valid = torch.ones(count, 1, device="cuda")
    out_zero_invalid = head(
        latent, evidence, step=0, priors=(zero_geometry,), prior_valid=zero_valid
    )
    assert torch.equal(out_step0, out_zero_invalid)
    out_zero_valid = head(
        latent, evidence, step=0, priors=(zero_geometry,), prior_valid=one_valid
    )
    assert torch.equal(out_step0, out_zero_valid)

    random_geometry = torch.randn(count, geometry_dim, device="cuda")
    out_masked = head(
        latent, evidence, step=0, priors=(random_geometry,), prior_valid=zero_valid
    )
    assert torch.equal(out_step0, out_masked)
    out_with_geometry = head(
        latent, evidence, step=0, priors=(random_geometry,), prior_valid=one_valid
    )
    reference_groups = [encoder(group) for encoder, group in zip(
        tuple(head.state_encoders) + tuple(head.evidence_encoders),
        torch.split(latent, head.state_dims, dim=-1) + torch.split(evidence, head.evidence_dims, dim=-1),
    )]
    reference_groups.append(head.geometric_encoder(random_geometry))
    reference_groups.append(head.step_embedding.weight[0].expand_as(reference_groups[0]))
    reference_concat = torch.cat(reference_groups, dim=-1)
    assert reference_concat.shape == (count, 384)
    reference_output = torch.cat(
        tuple(output_head(reference_concat) for output_head in head.output_heads()), dim=-1
    )
    torch.testing.assert_close(out_with_geometry, reference_output, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(out_step0, out_with_geometry)
    refined_with_geometry = refiner.apply_residual(
        initial, evidence, step=0, priors=fixed_priors, prior_valid=fixed_valid
    )
    refined_without_geometry = refiner.apply_residual(initial, evidence, step=0)
    assert not torch.allclose(
        refined_with_geometry._refined_z, refined_without_geometry._refined_z
    )
    refiner.optimizer.zero_grad(set_to_none=True)
    loss = out_with_geometry.square().mean() + out_step1.square().mean()
    loss.backward()
    for output_head in head.output_heads():
        assert output_head.weight.grad.abs().sum() > 0
    for index, encoder in enumerate(head.linear_encoders()):
        assert encoder[-1].weight.grad is not None and encoder[-1].weight.grad.abs().sum() > 0, index
    assert head.step_embedding.weight.grad.abs().sum() > 0
    for parameter in (
        gaussians._z, gaussians._rgb, gaussians._label,
        gaussians._scaling_xy, gaussians._rotation, gaussians._opacity,
    ):
        assert parameter.grad is None and not parameter.requires_grad
    print({
        "architecture": "11_independent_groups_plus_step_concat_384_to_3_linear_heads",
        "sliced_linear_equals_explicit_concat": "PASS",
        "bias_free_geometry_zero_in_zero_out": "PASS",
        "geometry_valid_mask": "PASS",
        "texture_and_semantic_removed": "PASS",
        "zero_initialization_and_residual_bounds": "PASS",
        "all_11_encoder_gradients": "PASS",
        "base_parameters_frozen": "PASS",
        "trainable_head_parameters": sum(p.numel() for p in head.parameters()),
    })


if __name__ == "__main__":
    main()
