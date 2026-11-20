"""Minimal GIFSplat-inspired residual refinement for RoGS.

This is not the official GIFSplat implementation.  Observation evidence is
sampled at projected Gaussian centers rather than pooled with rasterization
weights.  The residual head is trained online with the official RoGS losses.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from diff_gaussian_rasterization.scene.cameras import PerspectiveCamera

from utils.render import render


class RefinedGaussianView:
    """Delegate fixed Gaussian fields and expose differentiable residuals."""

    def __init__(self, base, z, rgb, label, scaling, rotation, opacity):
        self._base = base
        self._refined_z = z
        self._refined_rgb = rgb
        self._refined_label = label
        self._refined_scaling = scaling
        self._refined_rotation = rotation
        self._refined_opacity = opacity

    def __getattr__(self, name):
        return getattr(self._base, name)

    @property
    def get_xyz(self):
        return torch.cat((self._base._xy, self._refined_z), dim=-1)

    @property
    def get_rgb(self):
        return self._base.rgb_activation(self._refined_rgb)

    @property
    def get_label(self):
        return self._base.label_activation(self._refined_label, dim=-1)

    def get_rgb_mask(self, mask):
        return self._base.rgb_activation(self._refined_rgb[mask])

    def get_label_mask(self, mask):
        return self._base.label_activation(self._refined_label[mask], dim=-1)

    @property
    def get_scaling(self):
        scale_xy = self._base.scaling_activation(self._refined_scaling)
        return torch.cat((scale_xy, torch.zeros_like(scale_xy[:, :1])), dim=-1)

    @property
    def get_rotation(self):
        return self._base.rotation_activation(self._refined_rotation)

    @property
    def get_opacity(self):
        return self._base.opacity_activation(self._refined_opacity)

    def get_scaling_mask(self, mask):
        scale_xy = self._base.scaling_activation(self._refined_scaling[mask])
        return torch.cat((scale_xy, torch.zeros_like(scale_xy[:, :1])), dim=-1)

    def get_rotation_mask(self, mask):
        return self._base.rotation_activation(self._refined_rotation[mask])

    def get_opacity_mask(self, mask):
        return self._base.opacity_activation(self._refined_opacity[mask])

    def detached(self):
        return RefinedGaussianView(
            self._base,
            self._refined_z.detach(), self._refined_rgb.detach(),
            self._refined_label.detach(), self._refined_scaling.detach(),
            self._refined_rotation.detach(), self._refined_opacity.detach(),
        )

    def get_covariance_mask(self, mask, scaling_modifier=1):
        return self._base.covariance_activation(
            self.get_scaling_mask(mask), scaling_modifier, self._refined_rotation[mask]
        )


class ConditionalResidualHead(nn.Module):
    """Independent linear group mappings, concatenation and linear heads.

    Eleven groups are independently mapped to branch_dim channels: seven
    Gaussian-state groups, three observation groups and geometry G. Texture
    and semantic priors are absent. A learned step embedding is concatenated
    as control input, with no shared fusion MLP or cross-modal nonlinearity.
    """

    def __init__(self, latent_dim, evidence_dim, num_classes, branch_dim, steps,
                 geometric_prior_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.evidence_dim = evidence_dim
        self.geometric_prior_dim = int(geometric_prior_dim)
        self.branch_dim = branch_dim
        self.output_dim = 1 + 2 + 4 + 1 + 3 + num_classes
        self.state_dims = (2, 1, 2, 4, 1, 3, num_classes)
        self.evidence_dims = (3, num_classes, 1)
        self.state_encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, branch_dim)) for dim in self.state_dims
        )
        self.evidence_encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, branch_dim)) for dim in self.evidence_dims
        )
        # No affine bias anywhere in G: zero input remains zero after training.
        # Older PyTorch lacks LayerNorm(bias=False), so remove only its bias.
        geometry_norm = nn.LayerNorm(self.geometric_prior_dim)
        geometry_norm.bias = None
        self.geometric_encoder = nn.Sequential(
            geometry_norm,
            nn.Linear(self.geometric_prior_dim, branch_dim, bias=False),
        )
        self.step_embedding = nn.Embedding(steps, branch_dim)
        # 7 state + 3 evidence + G + step = 12 concatenated branches.
        self.concat_dim = 12 * branch_dim
        self.geometry_head = nn.Linear(self.concat_dim, 1 + 2 + 4)
        self.appearance_head = nn.Linear(self.concat_dim, 1 + 3)
        self.semantic_head = nn.Linear(self.concat_dim, num_classes)
        for output_head in self.output_heads():
            nn.init.zeros_(output_head.weight)
            nn.init.zeros_(output_head.bias)

    def output_heads(self):
        return self.geometry_head, self.appearance_head, self.semantic_head

    def linear_encoders(self):
        return tuple(self.state_encoders) + tuple(self.evidence_encoders) + (
            self.geometric_encoder,
        )

    def _predict_concat_linear(self, output_head, encoded, step):
        # Exactly Linear(cat(encoded, step_embedding)), without materializing [N,384].
        weight_groups = output_head.weight.split(self.branch_dim, dim=-1)
        result = F.linear(encoded[0], weight_groups[0], output_head.bias)
        for feature, weight in zip(encoded[1:], weight_groups[1:-1]):
            result = result + F.linear(feature, weight)
        return result + F.linear(self.step_embedding.weight[step], weight_groups[-1])

    def forward(self, latent, evidence, step, priors=None, prior_valid=None):
        if latent.shape[:-1] != evidence.shape[:-1]:
            raise ValueError("latent/evidence point dimensions must match")
        if latent.shape[-1] != self.latent_dim or evidence.shape[-1] != self.evidence_dim:
            raise ValueError("unexpected latent or evidence dimensions")
        if not 0 <= step < self.step_embedding.num_embeddings:
            raise ValueError(f"invalid residual step: {step}")
        state_groups = torch.split(latent, self.state_dims, dim=-1)
        evidence_groups = torch.split(evidence, self.evidence_dims, dim=-1)
        encoded = [
            encoder(group) for encoder, group in zip(self.state_encoders, state_groups)
        ]
        encoded.extend(
            encoder(group) for encoder, group in zip(self.evidence_encoders, evidence_groups)
        )
        if priors is not None:
            if len(priors) != 1 or prior_valid is None:
                raise ValueError("geometry and an [N,1] validity tensor are required together")
            if prior_valid.shape != (latent.shape[0], 1):
                raise ValueError("prior_valid must have shape [N,1]")
        elif prior_valid is not None:
            raise ValueError("prior_valid without priors is ambiguous")
        prior = priors[0] if priors is not None else None
        if prior is None:
            encoded.append(latent.new_zeros((latent.shape[0], self.branch_dim)))
        else:
            if prior.shape != (latent.shape[0], self.geometric_prior_dim):
                raise ValueError(
                    f"geometry must have shape [N,{self.geometric_prior_dim}], got {tuple(prior.shape)}"
                )
            valid = prior_valid[:, 0].to(latent.dtype).unsqueeze(-1)
            encoded.append(self.geometric_encoder(prior) * valid)
        return torch.cat(
            tuple(self._predict_concat_linear(output_head, encoded, step)
                  for output_head in self.output_heads()),
            dim=-1,
        )


@dataclass
class StageOutput:
    gaussians: RefinedGaussianView
    render_pkg: dict
    label_pkg: dict


class GIFSplatResidualRefiner(nn.Module):
    """v11-a9: frozen bias-free G=97 plus state/evidence and linear heads."""

    def __init__(self, gaussians, cfg, geometric_prior=None, geometric_prior_valid=None):
        super().__init__()
        if not gaussians.use_rgb:
            raise ValueError("v11-a9 requires use_rgb=True")
        object.__setattr__(self, "gaussians", gaussians)
        self.cfg = cfg
        self.steps = int(cfg.get("steps", 2))
        if self.steps != 2:
            raise ValueError("v11-a9 is intentionally fixed to two shared steps")
        self.num_classes = gaussians._label.shape[1]
        self.scene_scale = max(float(gaussians.spatial_lr_scale), 1e-8)

        # Reference values are retained for verification.  As in official
        # GIFSplat refinement, the initializer is fixed and only the shared
        # residual head updates effective Gaussian parameters.
        self.initial_z = gaussians._z.detach().clone()
        self.initial_rgb = gaussians._rgb.detach().clone()
        self.initial_label = gaussians._label.detach().clone()
        self.initial_scaling = gaussians._scaling_xy.detach().clone()
        self.initial_rotation = gaussians._rotation.detach().clone()
        self.initial_opacity = torch.logit(
            gaussians.get_opacity.detach().clamp(1e-6, 1 - 1e-6)
        )
        self.fixed_xy = gaussians._xy.detach().clone()  # opt_xy=False in the inherited config.
        for parameter in (
            gaussians._xy, gaussians._z, gaussians._scaling_xy,
            gaussians._rotation, gaussians._opacity, gaussians._label, gaussians._rgb,
        ):
            parameter.requires_grad_(False)

        latent_dim = 2 + 1 + 2 + 4 + 1 + 3 + self.num_classes
        evidence_dim = 3 + self.num_classes + 1
        branch_dim = int(cfg.get("branch_dim", 32))
        geometric_prior_dim = int(cfg.get("geometric_prior_dim", 97))
        self.head = ConditionalResidualHead(
            latent_dim, evidence_dim, self.num_classes, branch_dim, self.steps,
            geometric_prior_dim=geometric_prior_dim,
        )
        if (geometric_prior is None) != (geometric_prior_valid is None):
            raise ValueError("geometry features and validity must be supplied together")
        if geometric_prior is not None:
            expected_shape = (self.initial_z.shape[0], geometric_prior_dim - 1)
            if tuple(geometric_prior.shape) != expected_shape:
                raise ValueError(
                    f"geometry features must be {expected_shape}, got {tuple(geometric_prior.shape)}"
                )
            if tuple(geometric_prior_valid.shape) != (self.initial_z.shape[0],):
                raise ValueError("geometry validity must have shape [N]")
            frozen_geometry = torch.cat(
                (geometric_prior.detach().to(device=self.initial_z.device, dtype=torch.float32),
                 self.initial_z), dim=-1,
            )
            if not torch.isfinite(frozen_geometry).all():
                raise FloatingPointError("non-finite v11-a9 geometry prior")
            self.register_buffer("geometry_prior", frozen_geometry, persistent=False)
            self.register_buffer(
                "geometry_prior_valid",
                geometric_prior_valid.detach().to(device=self.initial_z.device, dtype=torch.bool),
                persistent=False,
            )
        else:
            self.geometry_prior = None
            self.geometry_prior_valid = None
        if self.geometry_prior is None:
            raise ValueError("v11-a9 requires the frozen geometry prior")
        self.register_buffer(
            "delta_scale",
            torch.tensor(
                [float(cfg.get("z_delta_scale", 0.05))]
                + [float(cfg.get("scaling_delta_scale", 0.1))] * 2
                + [float(cfg.get("rotation_delta_scale", 0.1))] * 4
                + [float(cfg.get("opacity_delta_scale", 1.0))]
                + [float(cfg.get("rgb_delta_scale", 1.0))] * 3
                + [float(cfg.get("label_delta_scale", 1.0))] * self.num_classes
            ),
        )
        self.optimizer = torch.optim.Adam(
            self.head.parameters(), lr=float(cfg.get("residual_lr", 0.001)), eps=1e-15
        )
        self.to(gaussians._z.device)

    def initial_view(self):
        # Match GIFSplat: clamp activated opacity before the logit so RoGS's
        # initial opacity=1 remains finite and retains a nonzero derivative.
        return RefinedGaussianView(
            self.gaussians, self.initial_z, self.initial_rgb, self.initial_label,
            self.initial_scaling, self.initial_rotation, self.initial_opacity,
        )

    def tokens(self, view):
        return torch.cat(
            (
                self.fixed_xy / self.scene_scale,
                view._refined_z / self.scene_scale,
                view._refined_scaling,
                view._refined_rotation,
                view._refined_opacity / 30,
                view._refined_rgb,
                view._refined_label,
            ),
            dim=-1,
        )

    def apply_residual(self, view, evidence, step, priors=None, prior_valid=None):
        delta = torch.tanh(
            self.head(self.tokens(view), evidence, step, priors=priors, prior_valid=prior_valid)
        )
        delta = delta * self.delta_scale
        return RefinedGaussianView(
            self.gaussians,
            view._refined_z + delta[:, :1],
            view._refined_rgb + delta[:, 8:11],
            view._refined_label + delta[:, 11:],
            view._refined_scaling + delta[:, 1:3],
            view._refined_rotation + delta[:, 3:7],
            view._refined_opacity + delta[:, 7:8],
        )

    @staticmethod
    def render_stage(viewpoint_cam, view, pipe, bg):
        return (
            render(viewpoint_cam, view, pipe, bg),
            render(viewpoint_cam, view, pipe, bg, render_type="label"),
        )

    def _sample_map(self, feature_map, grid):
        count = grid.shape[0]
        return F.grid_sample(
            feature_map.unsqueeze(0),
            grid.reshape(1, count, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(feature_map.shape[0], count).transpose(0, 1)

    def observation_numerator_mass(
        self, view, sample, render_pkg, label_pkg, exposure_model, cam_idx
    ):
        """Project centers with the official c2w R/T/K and sample residuals."""
        xyz = view.get_xyz.detach()
        R = sample["R"].to(dtype=xyz.dtype)
        T = sample["T"].to(dtype=xyz.dtype)
        K = sample["K"].to(dtype=xyz.dtype)
        xyz_cam = (xyz - T) @ R
        projected = xyz_cam @ K.transpose(0, 1)
        depth = projected[:, 2]
        safe_depth = depth.clamp_min(1e-6)
        u = projected[:, 0] / safe_depth
        v = projected[:, 1] / safe_depth
        width, height = sample["W"], sample["H"]
        grid = torch.stack(
            (2 * u / (width - 1) - 1, 2 * v / (height - 1) - 1), dim=-1
        )
        valid = render_pkg["visibility_filter"] & (depth > 1.0) & (depth < 20.0)
        valid &= (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)

        src_rgb = render_pkg["render"]
        if exposure_model is not None:
            src_rgb = exposure_model(cam_idx, src_rgb)
        rgb_residual = sample["image"].permute(2, 0, 1) - src_rgb
        label_target = F.one_hot(
            sample["label"].long(), num_classes=self.num_classes
        ).permute(2, 0, 1).float()
        label_residual = label_target - label_pkg["render"]
        residual = torch.cat((rgb_residual, label_residual), dim=0)
        sampled = self._sample_map(residual.detach(), grid)
        if "mask" in sample:
            road_mask = self._sample_map(sample["mask"][None].detach(), grid)
        else:
            road_mask = torch.ones_like(depth[:, None])
        mass = valid.float()[:, None] * road_mask.clamp(0, 1)
        return sampled * mass, mass

    @staticmethod
    def pooled_evidence(numerator, mass):
        valid = mass > 1e-8
        evidence = numerator / mass.clamp_min(1e-8)
        evidence = evidence * valid
        return torch.cat((evidence, valid.float()), dim=-1)

    def prior_inputs(self):
        if self.geometry_prior is None:
            raise RuntimeError("v11-a9 geometry prior must be initialized")
        valid = self.geometry_prior_valid.unsqueeze(-1)
        return (self.geometry_prior,), valid

    def training_stages(self, viewpoint_cam, gaussians, sample, pipe, bg, exposure_model, cam_idx):
        current = self.initial_view()
        priors, prior_valid = self.prior_inputs()
        with torch.no_grad():
            render_pkg, label_pkg = self.render_stage(viewpoint_cam, current, pipe, bg)
        stages = []
        for step in range(self.steps):
            numerator, mass = self.observation_numerator_mass(
                current, sample, render_pkg, label_pkg, exposure_model, cam_idx
            )
            current = self.apply_residual(
                current, self.pooled_evidence(numerator, mass).detach(), step,
                priors=priors, prior_valid=prior_valid,
            )
            render_pkg, label_pkg = self.render_stage(viewpoint_cam, current, pipe, bg)
            stages.append(StageOutput(current, render_pkg, label_pkg))
            # Official GIFSplat defaults detach_prev_iter=True: each stage's
            # loss trains that stage, not the entire preceding unrolled graph.
            current = current.detached()
        return stages

    @torch.no_grad()
    def materialize(self, dataset, pipe, bg, exposure_model, device):
        """Pool evidence over all official training views before BEV evaluation."""
        current = self.initial_view()
        priors, prior_valid = self.prior_inputs()
        numerator = torch.zeros(
            self.initial_z.shape[0], 3 + self.num_classes, device=device
        )
        mass = torch.zeros(self.initial_z.shape[0], 1, device=device)
        loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, drop_last=False)
        for step in range(self.steps):
            numerator.zero_()
            mass.zero_()
            for sample in loader:
                for key, value in sample.items():
                    if key != "image_name":
                        sample[key] = value[0].to(device)
                    else:
                        sample[key] = value[0]
                viewpoint_cam = PerspectiveCamera(
                    sample["R"], sample["T"], sample["K"], sample["W"], sample["H"], 1, 20, device
                )
                render_pkg, label_pkg = self.render_stage(viewpoint_cam, current, pipe, bg)
                sample_numerator, sample_mass = self.observation_numerator_mass(
                    current, sample, render_pkg, label_pkg, exposure_model, sample["cam_idx"].item()
                )
                numerator += sample_numerator
                mass += sample_mass
            current = self.apply_residual(
                current, self.pooled_evidence(numerator, mass), step,
                priors=priors, prior_valid=prior_valid,
            )
        changes = {
            "z": (current._refined_z - self.initial_z).abs().max().item(),
            "rgb": (current._refined_rgb - self.initial_rgb).abs().max().item(),
            "label": (current._refined_label - self.initial_label).abs().max().item(),
            "scaling": (current._refined_scaling - self.initial_scaling).abs().max().item(),
            "rotation": (current._refined_rotation - self.initial_rotation).abs().max().item(),
            "opacity": (current._refined_opacity - self.initial_opacity).abs().max().item(),
        }
        bounds = {
            "z": self.steps * float(self.cfg.get("z_delta_scale", 0.05)),
            "rgb": self.steps * float(self.cfg.get("rgb_delta_scale", 1.0)),
            "label": self.steps * float(self.cfg.get("label_delta_scale", 1.0)),
            "scaling": self.steps * float(self.cfg.get("scaling_delta_scale", 0.1)),
            "rotation": self.steps * float(self.cfg.get("rotation_delta_scale", 0.1)),
            "opacity": self.steps * float(self.cfg.get("opacity_delta_scale", 1.0)),
        }
        for name, maximum in changes.items():
            if not torch.isfinite(torch.tensor(maximum)):
                raise FloatingPointError(f"v11-a9 non-finite residual field: {name}")
            if maximum > bounds[name] + 1e-5:
                raise RuntimeError(f"v11-a9 residual bound violated: {name}={maximum}")
        self.gaussians._z.data.copy_(current._refined_z)
        self.gaussians._rgb.data.copy_(current._refined_rgb)
        self.gaussians._label.data.copy_(current._refined_label)
        self.gaussians._scaling_xy.data.copy_(current._refined_scaling)
        self.gaussians._rotation.data.copy_(current._refined_rotation)
        self.gaussians._opacity.data.copy_(current._refined_opacity)
        return {
            "mean_views_per_gaussian": mass.mean().item(),
            "covered_gaussians": (mass[:, 0] > 0).sum().item(),
            "num_gaussians": mass.shape[0],
            "steps": self.steps,
            "max_abs_changes": changes,
            "mean_abs_z_change": (current._refined_z - self.initial_z).abs().mean().item(),
        }

    def capture(self):
        return {
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": dict(self.cfg),
            "initial_z": self.initial_z,
            "initial_rgb": self.initial_rgb,
            "initial_label": self.initial_label,
            "initial_scaling": self.initial_scaling,
            "initial_rotation": self.initial_rotation,
            "initial_opacity": self.initial_opacity,
        }
