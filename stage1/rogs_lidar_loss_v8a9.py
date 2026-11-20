#!/usr/bin/env python3
"""Strict DiffMVS-style geometry-only elevation loss.

The residual refiner's intermediate elevation maps ``z_{t,k}`` are supervised
with the DiffMVS confidence Laplace NLL and an exponential schedule
``beta^{J-k}`` on top of the final refined elevation.  All elevation terms live
inside the single elevation GradNorm task.

  * L_elev = RMSE(z_refined)
             + sum_{k<K-1} beta^{K-1-k} * mean[ |z_k - z_gt|/(1-C_k)
                                             + lambda_C * log(1-C_k) ]
  * The final confidence is not NLL-supervised, matching DiffMVS conf_flag.
  * C_k has no ground-truth; the depth/elevation error itself supervises it.
  * valid domain: target_valid & isfinite(target_z), full BEV lattice.
  * depth aux unchanged (official native BCE, already x3).
"""

from argparse import Namespace
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F

from rogs_gaussian_build import build_fine_gaussians


def _rmse_map(z_map: torch.Tensor, domain: torch.Tensor,
              safe_tz: torch.Tensor) -> torch.Tensor:
    flat = z_map.reshape(-1)
    resid = (flat - safe_tz) * domain.to(flat.dtype)
    denom = domain.to(flat.dtype).sum().clamp_min(1.0)
    return torch.sqrt((resid ** 2).sum() / denom + 1e-12)


def _confidence_nll(z_map: torch.Tensor, conf: torch.Tensor,
                    domain: torch.Tensor, safe_tz: torch.Tensor,
                    lambda_c: float) -> torch.Tensor:
    """Masked DiffMVS Laplace NLL for one intermediate elevation map."""
    z = z_map.reshape(-1).float()
    c = conf.reshape(-1).float().clamp(1e-3, 1.0 - 1e-3)
    dom = domain.to(dtype=z.dtype)
    err = (z - safe_tz.float()).abs()
    uncertainty = (1.0 - c).clamp_min(1e-3)
    nll = err / uncertainty + float(lambda_c) * torch.log(uncertainty)
    return (nll * dom).sum() / dom.sum().clamp_min(1.0)


def compute_v8a9_loss(initial, diagnostics, base_xy, target_z, target_valid,
                       resolution: float,
                       depth_loss: Optional[torch.Tensor] = None,
                       depth_weight: float = 1.0,
                       beta: float = 0.9,
                       lambda_c: float = 0.05) -> Dict[str, torch.Tensor]:
    device = initial["z"].device
    target_z = target_z.to(device=device, dtype=initial["z"].dtype)
    target_valid = target_valid.to(device=device).bool()
    cell = resolution
    point_cloud, world_xy, z = build_fine_gaussians(initial, base_xy, cell)
    flat_z = z.squeeze(-1)

    domain = (target_valid.reshape(-1) & torch.isfinite(target_z.reshape(-1)))
    safe_tz = torch.where(torch.isfinite(target_z), target_z,
                          flat_z.detach().reshape_as(target_z))
    safe_tz = safe_tz.reshape(-1)

    rmse_final = _rmse_map(flat_z.reshape_as(target_z), domain, safe_tz)

    # Intermediate refiner iterations, exponentially weighted.
    z_mid: List[torch.Tensor] = diagnostics.get("z_mid") or []
    confs: List[torch.Tensor] = diagnostics.get("conf") or []
    if len(confs) != len(z_mid):
        raise ValueError(
            f"confidence list ({len(confs)}) must match z_mid ({len(z_mid)})")

    elev_mid = target_z.new_zeros(())
    conf_term = target_z.new_zeros(())
    K = len(z_mid)
    # DiffMVS conf_flag: only non-final iterative confidences, which drive
    # the next candidate search, use confidence NLL.  Final elevation is
    # supervised by the plain RMSE above.
    for k, (zk, ck) in enumerate(zip(z_mid[:-1], confs[:-1])):
        w_k = beta ** (K - 1 - k)  # final plain loss has weight 1
        nll_k = _confidence_nll(zk, ck, domain, safe_tz, lambda_c)
        elev_mid = elev_mid + w_k * nll_k
        conf_term = conf_term + nll_k

    elevation_loss = rmse_final + elev_mid

    depth_term = depth_loss if depth_loss is not None else target_z.new_zeros(())
    total = elevation_loss + float(depth_weight) * depth_term

    if not bool(torch.isfinite(total)):
        raise FloatingPointError(
            f"Non-finite strict DiffMVS loss: total={float(total.detach())}")

    return {
        "total": total,
        "elevation_loss": elevation_loss,
        "elevation_rmse": rmse_final,
        "conf_term": conf_term.detach(),
        "depth_loss": depth_term.detach() if not depth_term.requires_grad else depth_term,
        "z_supervision_gaussians": int(domain.sum()),
    }
