#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapter: per-cell v8-A Gaussian outputs -> RoGS GaussianModel2D.

v8-A changes vs v7:
  * SH degree 1 is used.  Per-Gaussian features are (N, 16, 3) = (DC + 15 rest
    bands, but only the first 3 rest bands are meaningful for L=1, the rest is
    zero).  Colors are evaluated externally via eval_sh (the same path used by
    RoGS utils/render.py) and passed as colors_precomp.
  * A finite z-scale epsilon is appended (official hardcoded 0 -> NaN).
"""
import torch
import rogs_runtime  # noqa: F401
from models.gaussian_model import GaussianModel2D
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gs_label import GaussianRasterizationSettings as GaussianRasterizationSettingsLabel
from diff_gs_label import GaussianRasterizer as GaussianRasterizerLabel
from diff_gaussian_rasterization.scene.cameras import OrthographicCamera
from utils.render import get_orthographic_rect
from utils.sh_utils import eval_sh

Z_SCALE_EPS = 0.01
MAX_BANDS = 16  # (max_sh_degree 3 + 1) ** 2


class _Cfg:
    sh_degree = 1
    use_rgb = False
    opt_xy = True


class _PC2D(GaussianModel2D):
    @staticmethod
    def _append_z(scale_xy):
        z = torch.full((scale_xy.shape[0], 1), Z_SCALE_EPS,
                       device=scale_xy.device, dtype=scale_xy.dtype)
        return torch.cat([scale_xy, z], dim=-1)

    @property
    def get_scaling(self):
        return self._append_z(self.scaling_activation(self._scaling_xy))

    def get_scaling_mask(self, mask):
        return self._append_z(self.scaling_activation(self._scaling_xy[mask]))

    def get_features_mask(self, mask):
        return self._features[mask]


def _split_sh(sh_12: torch.Tensor) -> torch.Tensor:
    """(N,12) = (3 rgb) x (4 SH bands) -> (N,16,3) padded."""
    N = sh_12.shape[0]
    sh = sh_12.reshape(N, 4, 3)  # (N,4,3)
    padded = torch.zeros(N, MAX_BANDS, 3, device=sh_12.device, dtype=sh_12.dtype)
    padded[:, :4, :] = sh
    return padded


def build_gaussians(xy_world, z, sh_12, logits, scale_raw, quat, op_raw):
    pc = _PC2D(_Cfg())
    pc._xy = xy_world
    pc._z = z
    pc._label = logits
    pc._features = _split_sh(sh_12)
    # Standard render() reads get_features = cat(_features_dc, _features_rest, dim=1),
    # i.e. (N,3,16) RGB-major. _features is (N,16,3) band-major; transpose to match.
    F_rgb = pc._features.permute(0, 2, 1).contiguous()   # (N,3,16)
    pc._features_dc = F_rgb[:, 0:1, :].contiguous()
    pc._features_rest = F_rgb[:, 1:3, :].contiguous()
    pc.max_sh_degree = 3
    pc._scaling_xy = scale_raw
    pc._rotation = quat
    pc._opacity = op_raw
    pc.active_sh_degree = 1
    return pc


def make_ortho_camera(center_xy, extent_xy, znear, zfar, H, W, device,
                      top_down=False, camera_height=10.0):
    if top_down:
        R = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device))
        T = torch.tensor([float(center_xy[0]), float(center_xy[1]), camera_height], device=device)
        left, right = -float(extent_xy[0]) / 2, float(extent_xy[0]) / 2
        top, bottom = float(extent_xy[1]) / 2, -float(extent_xy[1]) / 2
    else:
        left = float(center_xy[0] - extent_xy[0]/2)
        right = float(center_xy[0] + extent_xy[0]/2)
        bottom = float(center_xy[1] - extent_xy[1]/2)
        top = float(center_xy[1] + extent_xy[1]/2)
        R = torch.eye(3, device=device)
        T = torch.tensor([0.0, 0.0, -10.0], device=device)
    return OrthographicCamera(R=R, T=T, W=W, H=H, znear=znear, zfar=zfar,
                              top=top, bottom=bottom, right=right, left=left, device=device)


def make_cell_center_ortho_camera(base_xy, size, resolution, device):
    points = base_xy.reshape(-1, 2)
    if points.shape[0] != size * size:
        raise ValueError(f"Expected {size * size} cells, got {tuple(base_xy.shape)}")
    center = (points[0] + points[-1]) * 0.5
    extent = float(size * resolution)
    return make_ortho_camera(
        center.detach().cpu().numpy(), [extent, extent], 0.1, 20.0,
        size, size, device, top_down=True, camera_height=10.0,
    )


def _sh_colors(pc, active, camera):
    """Evaluate SH L=1 to RGB for active Gaussians (RoGS convention)."""
    feats = pc.get_features_mask(active)  # (M,16,3)
    means3D = pc.get_xyz[active]
    dir_pp = means3D - camera.camera_center.repeat(means3D.shape[0], 1)
    dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True).clamp_min(1e-8)
    shs_view = feats.transpose(1, 2).contiguous()  # (M,3,16)
    rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp)
    return torch.clamp(rgb + 0.5, 0.0, 1.0)


def _render_bev(camera, pc, pipe, bg_color, render_type):
    if not isinstance(camera, OrthographicCamera):
        raise TypeError("BEV rendering requires an OrthographicCamera")
    settings_class = GaussianRasterizationSettings if render_type == "rgb" else GaussianRasterizationSettingsLabel
    settings = settings_class(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=0.0, tanfovy=0.0, bg=bg_color, scale_modifier=1.0,
        viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
        sh_degree=pc.active_sh_degree, campos=camera.camera_center,
        prefiltered=False, camera_type=1, debug=pipe.debug,
    )
    min_xy, max_xy = get_orthographic_rect(camera)
    means3D = pc.get_xyz
    active = ((means3D[:, 0] >= min_xy[0]) & (means3D[:, 0] <= max_xy[0]) &
              (means3D[:, 1] >= min_xy[1]) & (means3D[:, 1] <= max_xy[1]))
    means3D = means3D[active]
    opacity = pc.get_opacity_mask(active)
    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=means3D.device)
    scales = pc.get_scaling_mask(active)
    rotations = pc.get_rotation_mask(active)
    if render_type == "rgb":
        colors = _sh_colors(pc, active, camera)
        rasterizer = GaussianRasterizer(raster_settings=settings)
    elif render_type == "label":
        colors = pc.get_label_mask(active)
        rasterizer = GaussianRasterizerLabel(raster_settings=settings)
    else:
        raise ValueError(f"Unknown BEV render type: {render_type}")
    image, radii, depth, alpha = rasterizer(
        means3D=means3D, means2D=means2D, shs=None, colors_precomp=colors,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=None,
    )
    visibility = torch.zeros(pc.get_xyz.shape[0], dtype=torch.bool, device=means3D.device)
    visibility[active] = radii > 0
    return {
        "render": image,
        "depth": depth,
        "height": camera.cam2world[2, 3] - depth / alpha.clamp_min(1e-6),
        "height_rogs_native": camera.cam2world[2, 3] - depth,
        "alpha": alpha,
        "mask": depth[0] > 0,
        "visibility_filter": visibility,
    }


def render_bev_rgb(camera, pc, pipe, bg_color):
    return _render_bev(camera, pc, pipe, bg_color, "rgb")


def render_bev_label(camera, pc, pipe, bg_color):
    return _render_bev(camera, pc, pipe, bg_color, "label")
