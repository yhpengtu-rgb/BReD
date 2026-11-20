#!/usr/bin/env python3
"""v8-A independent perspective TextureFPN.

Consumes ONLY the image-backbone mid/high-resolution stages (stride-4 C2 and
stride-8 C3) and fuses them into a single perspective feature map at the
stride-4 resolution (64 x 176 for 256x704 inputs).  The official depth/context
16x44 deep feature is never touched here.

This module is intentionally tiny: it exists to provide a high-resolution
perspective feature map that the elevation-guided query can sample from.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureFPN(nn.Module):
    """Fuse stride-4 (256ch) and stride-8 (512ch) ResNet stages.

    Args:
        c2_channels: channels of the stride-4 stage (ResNet50 = 256).
        c3_channels: channels of the stride-8 stage (ResNet50 = 512).
        out_channels: fused perspective feature channels.
        out_size: (H, W) of the fused map.  Default (64, 176) for 256x704.
    """

    def __init__(self, c2_channels: int = 256, c3_channels: int = 512,
                 out_channels: int = 64, out_size=(64, 176)):
        super().__init__()
        self.out_size = tuple(out_size)
        self.lateral_c2 = nn.Conv2d(c2_channels, out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(c3_channels, out_channels, kernel_size=1)
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        for m in (self.lateral_c2, self.lateral_c3):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, c2: torch.Tensor, c3: torch.Tensor) -> torch.Tensor:
        # c2: (N, C2, 64, 176); c3: (N, C3, 32, 88)
        f_c2 = self.lateral_c2(c2)
        f_c3 = self.lateral_c3(c3)
        f_c3 = F.interpolate(f_c3, size=f_c2.shape[-2:], mode="bilinear",
                             align_corners=False)
        fused = self.smooth(f_c2 + f_c3)
        if tuple(fused.shape[-2:]) != self.out_size:
            fused = F.interpolate(fused, size=self.out_size, mode="bilinear",
                                  align_corners=False)
        return fused
