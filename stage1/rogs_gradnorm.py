#!/usr/bin/env python3
"""GradNorm task-weight balancing for the v8-a4 geometry-only objectives.

Two independent learnable weights ``w_elev`` and ``w_depth`` (init 1.0) balance
the elevation RMSE and the perspective depth BCE.  The shared-parameter
reference ``W`` is the image neck (SECONDFPN) -- the last common layer reached
by both the elevation path (neck -> lift -> head) and the depth path
(neck -> depth_net).

Per step:
  1. forward produces ``elev`` (RMSE, metres) and ``depth`` (official native
     depth loss, already x3; do NOT rescale here).
  2. unweighted gradient norms ``g_i = ||grad(loss_i, W)||`` are computed with
     ``retain_graph=True`` and detached to constants.
  3. weighted norms ``G_i = w_i * g_i``; ``G_mean = mean(G)``.
  4. inverse training rate ``r_i = (L_i / L_i0) / mean_j(L_j / L_j0)`` where
     ``L_i0`` is the first-step loss; alpha = 0.5.
  5. ``L_grad = sum_i |G_i - G_mean * r_i**alpha|`` optimises ONLY ``w`` with a
     dedicated small optimizer (lr 5e-3).
  6. after the w step, renormalise ``w_elev + w_depth = 2`` to prevent overall
     weight drift.
  7. the model-update total is ``w_elev.detach()*elev + w_depth.detach()*depth``.

No EMA is maintained; the weights updated here are the live training weights.
"""
from __future__ import annotations

from typing import List, Optional, Sequence
import os

import torch
import torch.nn as nn


class GradNorm(nn.Module):
    def __init__(self, n_tasks: int = 2, lr: float = 5e-3,
                 alpha: float = 0.5, target_sum: float = 2.0,
                 weight_init: float = 1.0):
        super().__init__()
        self.n_tasks = int(n_tasks)
        self.alpha = float(alpha)
        self.target_sum = float(target_sum)
        # Independent learnable task weights (not tied to each other).
        self.weight = nn.Parameter(torch.full((self.n_tasks,), float(weight_init)))
        self.opt = torch.optim.AdamW([self.weight], lr=lr)
        # First-step losses L_i0 (buffers, set on the first call).
        self.register_buffer("loss0", torch.full((self.n_tasks,), float("nan")),
                             persistent=False)
        self.register_buffer("grad_norm", torch.zeros(self.n_tasks),
                             persistent=False)
        self.register_buffer("rate", torch.ones(self.n_tasks), persistent=False)
        self._initialized = False

    # ------------------------------------------------------------------ #
    @staticmethod
    def task_grad_norm(loss: torch.Tensor, shared_params: Sequence[torch.Tensor],
                       retain_graph: bool = False) -> torch.Tensor:
        """||grad(loss, W)|| on the reference parameters, detached.

        ``retain_graph`` must stay False here: the caller runs one fresh
        forward replica per task so its graph is released after this single
        backward.  A retained second backward over the DDP + SyncBN graph
        deadlocks on world_size>1.
        """
        grads = torch.autograd.grad(
            loss, list(shared_params), retain_graph=retain_graph,
            allow_unused=True)
        sq = loss.new_zeros(())
        for g in grads:
            if g is not None:
                sq = sq + (g.detach() ** 2).sum()
        return torch.sqrt(sq + 1e-12)

    def _balance(self, losses: List[torch.Tensor],
                 g: torch.Tensor) -> dict:
        g = g.detach()
        self.grad_norm = g

        L = torch.stack([loss.detach() for loss in losses])
        if not self._initialized:
            self.loss0 = L.clone()
            self._initialized = True

        ratio = L / self.loss0.abs().clamp_min(1e-12)
        r = ratio / ratio.mean().clamp_min(1e-12)
        self.rate = r.detach()

        G = self.weight * g
        G_mean = G.mean()
        target = G_mean * (r ** self.alpha)
        l_grad = (G - target).abs().sum()

        self.opt.zero_grad()
        l_grad.backward()
        self.opt.step()

        with torch.no_grad():
            self.weight.data = (
                self.weight.data
                / self.weight.data.sum().clamp_min(1e-12) * self.target_sum)

        w = self.weight.detach()
        return {
            "grad_norm": g.detach().cpu(),
            "rate": r.detach().cpu(),
            "weight": w.cpu(),
            "gradnorm_loss": float(l_grad.detach().cpu()),
            "weight_sum": float(w.sum().cpu()),
        }

    def update(self, losses: List[torch.Tensor],
               shared_params: Sequence[torch.Tensor]) -> dict:
        """Deprecated retained-graph path (kept for CPU unit tests only)."""
        g_list = []
        for i, loss in enumerate(losses):
            if os.environ.get("DEBUG_GN"):
                print(f"[gn] autograd.grad task {i} "
                      f"(loss={float(loss.detach()):.4f}) start", flush=True)
            g_i = self.task_grad_norm(loss, shared_params, retain_graph=True)
            if os.environ.get("DEBUG_GN"):
                print(f"[gn] task {i} g={float(g_i):.4f} done", flush=True)
            g_list.append(g_i)
        g = torch.stack(g_list)
        return self._balance(losses, g)

    def update_norms(self, losses: List[torch.Tensor],
                     g: torch.Tensor) -> dict:
        """Balance weights from externally supplied per-task gradient norms.

        The norms are expected to come from two independent forward replicas
        with single (non-retained) backwards; this module never triggers a
        retained second backward on the main DDP graph.
        """
        return self._balance(losses, g)

    def weighted_total(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """7) weighted total using detached task weights (model update path)."""
        w = self.weight.detach()
        total = losses[0] * w[0]
        for i in range(1, self.n_tasks):
            total = total + losses[i] * w[i]
        return total
