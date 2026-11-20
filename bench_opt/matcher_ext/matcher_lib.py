"""Direct CUDA uniform-grid matcher extension for scene-0523 validation.

Reuses the same static CSR uniform grid as GTBank.match_grid (origin =
reference min XY, h = 0.1 m, stable sort keeps ascending native order inside a
cell, duplicate XY retained).  The CUDA kernel scans the full 3x3 neighbor
cells per query without materializing candidate tensors / masks / gathers.

Selection semantics (unchanged): radius 0.1 m, FP32 squared XY distance, first
reference in native order within radius = smallest native index (argmin
native, never nearest distance); unmatched -> -1.
"""
import os
import torch

_H = 0.1
_R = 0.1

_MOD = None


def _load():
    global _MOD
    if _MOD is not None:
        return _MOD
    from torch.utils import cpp_extension
    here = os.path.dirname(os.path.abspath(__file__))
    build = os.path.join(here, "build_matcher")
    os.makedirs(build, exist_ok=True)
    _MOD = cpp_extension.load(
        name="matcher_ext",
        sources=[os.path.join(here, "src", "grid_matcher.cu")],
        build_directory=build,
        verbose=False,
        extra_cuda_cflags=["--expt-relaxed-constexpr"])
    return _MOD


class CudaGridMatcher:
    """GPU-resident static CSR index + direct kernel matcher."""

    def __init__(self, xy: torch.Tensor, z: torch.Tensor):
        self.device = xy.device
        self.xy = xy.contiguous()
        self.z = z.contiguous()
        h = _H
        origin = self.xy.min(dim=0).values
        cell = torch.floor((self.xy - origin) / h).long()
        nc = cell.max(dim=0).values + 1
        cid = cell[:, 0] * nc[1] + cell[:, 1]
        order = torch.argsort(cid, stable=True)
        scid = cid[order]
        uniq, cnt = torch.unique_consecutive(scid, return_counts=True)
        start = torch.cumsum(cnt, dim=0) - cnt
        U = uniq.shape[0]
        cell_slot = torch.full((nc[0].item() * nc[1].item(),), -1,
                               dtype=torch.int64, device=self.device)
        cell_slot[uniq] = torch.arange(U, dtype=torch.int64, device=self.device)
        self.grid = {
            "origin": origin, "nc": nc, "uniq": uniq, "start": start,
            "cnt": cnt, "order": order,
            "sorted_xy": self.xy[order], "cell_slot": cell_slot,
        }
        self.kcap = int(cnt.max().item())
        self.n_cells = int(nc[0].item() * nc[1].item())
        self.n_occupied = U
        # Load/verify the cached .so at bank construction, outside the eval
        # loop, so the one-time dlopen/ninja-verify cost does not enter the
        # per-frame matcher timing or eval_elapsed window.
        self._mod = _load()

    def match(self, q_xy: torch.Tensor) -> torch.Tensor:
        mod = _load()
        G = self.grid
        return mod.grid_matcher(
            q_xy.contiguous(), G["origin"], G["cell_slot"], G["start"],
            G["cnt"], G["order"], G["sorted_xy"],
            int(G["nc"][0].item()), int(G["nc"][1].item()),
            _H, _R)
