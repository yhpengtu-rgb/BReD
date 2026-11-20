#!/usr/bin/env python3
"""v8-a4 RoGS head on top of the lifted BEV trunk/neck.

Differences from v8-a3:
  * The BEV grid is already the fine 256x256 @ 0.4 m lattice; there is NO
    learnable upsampling.  The elevation map (geom_z) is 1:1 with the BEV grid.
  * Geometry (z) and the regulariser heads run straight on the 256x256 lattice.
  * texture FPN / elevation query / SH decoder are still constructed but bypassed
    and frozen in geometry-only mode (no texture / query / SH / render in the
    forward or the optimisation).
"""
from __future__ import annotations

from typing import Dict, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones.resnet import ResNet
from mmdet3d.models.necks.second_fpn import SECONDFPN

from rogs_texture_fpn import TextureFPN
from elevation_query import ElevationGuidedQuery
from rogs_refiner import ElevationRefiner


class BEVStereoRoGSHead(nn.Module):
    def __init__(
        self,
        bev_channels: int = 160,
        hidden_channels: int = 128,
        fine_size: int = 256,
        texture_channels: int = 64,
        semantic_classes: int = 7,
        pc_range: Tuple[float, float, float, float, float, float] = (
            -51.2, -51.2, -5.0, 51.2, 51.2, 3.0
        ),
        query_detach: bool = True,
        geometry_only: bool = False,
        refiner_k: int = 1,
        refiner_res_max: float = 0.3,
        refiner_sz: float = 0.167,
        refiner_sigma: float = 0.5,
        refiner_cand_size: int = 128,
        refiner_cand_ch: int = 32,
        refiner_num_candidates: int = 6,
    ):
        super().__init__()
        self.geometry_only = bool(geometry_only)
        self.pc_range = tuple(float(v) for v in pc_range)
        self.bev_size = int(fine_size)
        self.fine_size = int(fine_size)
        self.query_detach = bool(query_detach)
        extent = self.pc_range[3] - self.pc_range[0]
        self.cell = extent / self.fine_size  # 0.4 m/cell

        register_range = torch.tensor(self.pc_range, dtype=torch.float32)
        self.register_buffer("pc_range_buf", register_range, persistent=False)

        # Unchanged official BEV encoder (pretrained trunk + neck).
        self.trunk = ResNet(
            depth=18, in_channels=bev_channels, num_stages=3,
            strides=(1, 2, 2), dilations=(1, 1, 1), out_indices=(0, 1, 2),
            norm_eval=False, base_channels=160, norm_cfg=dict(type="BN"),
            init_cfg=None,
        )
        self.neck = SECONDFPN(
            in_channels=[160, 160, 320, 640],
            upsample_strides=[1, 2, 4, 8],
            out_channels=[64, 64, 64, 64],
        )
        self.trunk.init_weights()
        self.neck.init_weights()
        del self.trunk.maxpool

        # v8-a4: no BEV upsample.  SECONDFPN emits the concatenated 256 ch
        # (4 x 64); that 256-channel map stays at the 256x256 lattice and is
        # projected straight to the hidden elevation feature.
        self.fine_project = nn.Sequential(
            nn.Conv2d(256, hidden_channels, 3, 1, 1),
            nn.SyncBatchNorm(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Fine-grid geometry + regularisers (output directly on the fine grid).
        self.geom_z = nn.Conv2d(hidden_channels, 1, 1)
        self.geom_xy = nn.Conv2d(hidden_channels, 2, 1)
        self.opacity = nn.Conv2d(hidden_channels, 1, 1)
        self.scale = nn.Conv2d(hidden_channels, 2, 1)
        self.rotation = nn.Conv2d(hidden_channels, 4, 1)
        self.semantic = nn.Conv2d(hidden_channels, semantic_classes, 1)

        # Independent perspective texture FPN + elevation-guided query + SH dec.
        self.texture_fpn = TextureFPN(256, 512, texture_channels)
        self.query = ElevationGuidedQuery(self.pc_range, self.fine_size)
        self.sh_decoder = nn.Sequential(
            nn.Conv2d(texture_channels, hidden_channels, 3, 1, 1),
            nn.SyncBatchNorm(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 12, 1),
        )

        # v8-a5 residual-diffusion elevation refiner (metric-space bounded
        # residual, lightweight UNet + bottleneck conv-GRU, K inner iters).
        # Strict DiffMVS: every iteration gets a multi-view candidate BEV and
        # predicts its own confidence.
        self.refiner = ElevationRefiner(
            fine_ch=hidden_channels,
            ctx_ch=256,
            hidden=64,
            k_iters=int(refiner_k),
            res_max_m=float(refiner_res_max),
            s_z=float(refiner_sz),
            sigma_z=float(refiner_sigma),
            num_candidates=int(refiner_num_candidates),
            cand_size=int(refiner_cand_size),
            cand_ch=int(refiner_cand_ch),
            z_min=float(self.pc_range[2]),
            z_max=float(self.pc_range[5]),
        )

        # Init: neutral elevation / XY offset, scale shrinks with fine cell.
        for layer in (self.geom_z, self.geom_xy):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.scale.bias, float(math.log(self.cell * 0.6)))
        with torch.no_grad():
            self.rotation.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    @classmethod
    def from_official_head(cls, official_head, **kwargs):
        head = cls(**kwargs)
        head.trunk.load_state_dict(official_head.trunk.state_dict(), strict=True)
        head.neck.load_state_dict(official_head.neck.state_dict(), strict=True)
        return head

    def encode_bev(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        trunk_outs = [x]
        if self.trunk.deep_stem:
            x = self.trunk.stem(x)
        else:
            x = self.trunk.conv1(x)
            x = self.trunk.norm1(x)
            x = self.trunk.relu(x)
        for index, layer_name in enumerate(self.trunk.res_layers):
            x = getattr(self.trunk, layer_name)(x)
            if index in self.trunk.out_indices:
                trunk_outs.append(x)
        return self.neck(trunk_outs)[0]

    def fine_centers(self, device, dtype):
        return self.query.fine_grid_centers(device, dtype)

    def forward(
        self,
        bev: torch.Tensor,
        c2,
        c3: torch.Tensor,
        mats: Dict[str, torch.Tensor],
        batch_size: Optional[int] = None,
        refiner_target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        refiner_use_noise: bool = True,
        refiner_outer_loops: int = 0,
        refiner_outer_seed: int = 0,
        refiner_loop_residual: str = "f_minus_c",
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        B = batch_size if batch_size is not None else bev.shape[0]
        enc = self.encode_bev(bev)                 # (B,256,256,256)
        fine = self.fine_project(enc)             # (B,H,256,256)

        z = self.geom_z(fine)                      # (B,1,256,256)
        xy_off = self.geom_xy(fine)                # (B,2,256,256)
        op = self.opacity(fine)                     # (B,1,256,256)
        scale = self.scale(fine)                    # (B,2,256,256)
        rot = self.rotation(fine)                   # (B,4,256,256)
        sem = self.semantic(fine)                   # (B,cls,256,256)

        if self.geometry_only:
            # v8-a5: residual refiner on the fine feature + lifted context.
            # Coarse z is the refiner condition (detached internally); the
            # refined elevation writes back to initial["z"] before the loss.
            # v8-a15: c2/c3/c4 multi-view features feed a lightweight FPN;
            # every refiner iteration samples and fuses all three scales.
            if c2 is None or len(c2) != 3 or any(feat is None for feat in c2):
                raise ValueError(
                    "v8-a15 multi-scale refiner requires c2/c3/c4 image features")
            z_refined, z_mid, conf_list = self.refiner(
                fine, enc, z, c2, mats,
                refiner_target=refiner_target,
                use_noise=refiner_use_noise)
            outer_loops = int(refiner_outer_loops)
            if outer_loops < 0:
                raise ValueError("refiner_outer_loops must be nonnegative")
            if outer_loops > 0 and self.training:
                raise ValueError("outer z_fine -> z_coarse loops are evaluation-only")
            z_outer = [z_refined]  # index 0 = original fine, before any feedback
            if refiner_loop_residual not in ("f_minus_c", "c_minus_f", "abs"):
                raise ValueError(f"unknown loop residual: {refiner_loop_residual}")
            z_previous_coarse = z
            for loop_idx in range(1, outer_loops + 1):
                # F is the latest fine; C is the coarse from the previous call.
                # First compute F-C (or its ablation), then set C <- F.  The
                # refiner therefore rebuilds FiLM and multi-scale candidate cond
                # from the latest fine while camdepth is no longer used.
                feedback = z_refined - z_previous_coarse
                if refiner_loop_residual == "c_minus_f":
                    feedback = -feedback
                elif refiner_loop_residual == "abs":
                    feedback = feedback.abs()
                feedback = (feedback / self.refiner.res_max_m).clamp(-1.0, 1.0)
                z_previous_coarse = z_refined
                devices = ([z_refined.device.index] if z_refined.is_cuda else [])
                with torch.random.fork_rng(devices=devices):
                    loop_seed = int(refiner_outer_seed) + loop_idx * 10007
                    torch.manual_seed(loop_seed)
                    if z_refined.is_cuda:
                        torch.cuda.manual_seed_all(loop_seed)
                    z_refined, z_mid, conf_list = self.refiner(
                        fine, enc, z_previous_coarse, c2, mats,
                        refiner_target=None, use_noise=False,
                        refiner_residual_override=feedback)
                z_outer.append(z_refined)
            # v8-a4: texture FPN / elevation query / SH decoder are bypassed.
            sh = z.new_zeros(z.shape[0], 12, z.shape[-2], z.shape[-1])
            diagnostics = {
                "enc_shape": tuple(enc.shape),
                "fine_shape": tuple(fine.shape),
                "c2_shape": tuple(c2[0].shape),
                "candidate_feature_shapes": [tuple(feat.shape) for feat in c2],
                "candidate_strides": (4, 8, 16),
                "fine_cell_m": self.cell,
                "geometry_only": True,
                "z_coarse": z.detach(),
                "z_mid": z_mid,
                "conf": conf_list,
            }
            if outer_loops > 0:
                diagnostics["z_outer"] = z_outer
                diagnostics["loop_residual_mode"] = refiner_loop_residual
            initial = {"z": z_refined, "xy": xy_off, "op": op, "scale": scale,
                       "rot": rot, "sem": sem, "sh": sh}
            return initial, diagnostics

        Nc = 6
        tex_feat = self.texture_fpn(c2, c3)
        Ct = tex_feat.shape[1]
        tex_feat = tex_feat.reshape(B, Nc, Ct, tex_feat.shape[2], tex_feat.shape[3])

        q = self.query(
            tex_feat, elevation=z, mats=mats,
            xy_offset=xy_off, detach=self.query_detach,
        )
        sh = self.sh_decoder(q["tex_bev"])

        initial = {
            "z": z, "xy": xy_off, "op": op, "scale": scale,
            "rot": rot, "sem": sem, "sh": sh,
        }
        diagnostics = {
            "enc_shape": tuple(enc.shape),
            "fine_shape": tuple(fine.shape),
            "fine_cell_m": self.cell,
            "valid_cams": q["valid_cams"],
            "query_coords": q["coords"],
        }
        return initial, diagnostics
