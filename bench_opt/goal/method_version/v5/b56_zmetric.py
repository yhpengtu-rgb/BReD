"""Exact, faster replacement for ``evalution.eval_z_metric`` (b56).

The original metric runs ``pytorch3d.ops.ball_query(K=1, radius=0.1)`` which is a
brute-force CUDA kernel: for every query (a Gaussian xy) it scans all LiDAR
points in index order and returns the *first* point whose squared planar
distance is ``< radius * radius`` (strict, float32).  It therefore yields, per
query, the minimum-index LiDAR point inside the disc, or -1.

This module reproduces that semantics with a uniform cell hash instead of an
O(Nq * Nl) scan.  The cell edge is set to ``sqrt(float32(radius)**2)`` (rounded
up), so any accepted pair is at most one cell apart on each axis and the 3x3
stencil is an exact superset of the brute-force candidate set.
Per-cell members are obtained with a single sort + ``searchsorted``; the
neighbour lists are expanded raggedly (no padding to the maximum occupancy)
and reduced with ``scatter_reduce_(reduce="amin")`` to keep the
minimum-index rule of ``ball_query``.

The candidate distances are evaluated in float32 with the same operation order
as the CUDA kernel (``dx*dx`` then ``+ dy*dy``) and compared against
``float32(radius) * float32(radius)`` rounded in float32, so the accept/reject
test is bit-identical to the kernel.
"""
import numpy as np
import torch

_OFFSETS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]


def _radius_sq(radius):
    """float32(radius * radius) computed exactly like the CUDA kernel."""
    r32 = np.float32(radius)
    return float(np.float32(r32 * r32))


def _cell_edge(radius):
    """Cell edge, kept just above ``sqrt(radius**2)``.

    The kernel accepts ``dist2 < float32(radius)**2``, whose true radius,
    ``sqrt(float32(radius)**2)``, is a hair larger than ``float32(radius)``.
    Taking the edge from that bound (rounded up) guarantees that any accepted
    pair satisfies ``|dx| < edge`` and ``|dy| < edge``, so its cells differ by
    at most 1 on each axis -- the 3x3 stencil is then an exact superset.
    """
    return float(np.nextafter(np.sqrt(_radius_sq(radius)), np.inf))


def _min_index_match(lidar_xyz, gaussian_xyz, radius=0.1, chunk_size=200000):
    """(Nq,) int64: min LiDAR index inside ``radius`` for each Gaussian, else -1."""
    device = lidar_xyz.device
    cell = _cell_edge(radius)
    r2 = torch.tensor(_radius_sq(radius), dtype=torch.float32, device=device)

    lidar_xy = lidar_xyz[:, :2]
    n_lidar = int(lidar_xy.shape[0])

    # Cell indices: floor(coord / edge) in float64.  Coordinates are float32
    # values, so the quotient is a small rational and the double-precision
    # division rounds far below the 1/2 ulp that a floor flip would need.
    cxl = torch.floor(lidar_xy[:, 0].double() / cell).to(torch.int64)
    cyl = torch.floor(lidar_xy[:, 1].double() / cell).to(torch.int64)
    min_x = int(cxl.min())
    min_y = int(cyl.min())
    ncx = int(cxl.max()) - min_x + 1
    ncy = int(cyl.max()) - min_y + 1
    keys = (cxl - min_x) * ncy + (cyl - min_y)

    order = torch.argsort(keys)
    keys_sorted = keys[order]

    off_x = torch.tensor([o[0] for o in _OFFSETS], dtype=torch.int64,
                         device=device).unsqueeze(1)
    off_y = torch.tensor([o[1] for o in _OFFSETS], dtype=torch.int64,
                         device=device).unsqueeze(1)

    n_query = int(gaussian_xyz.shape[0])
    best = torch.full((n_query,), n_lidar, dtype=torch.int64, device=device)
    qxy = gaussian_xyz[:, :2]

    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        qx = qxy[start:end, 0]
        qy = qxy[start:end, 1]
        qix = (torch.floor(qx.double() / cell).to(torch.int64) - min_x).unsqueeze(0) + off_x
        qiy = (torch.floor(qy.double() / cell).to(torch.int64) - min_y).unsqueeze(0) + off_y
        inside = (qix >= 0) & (qix < ncx) & (qiy >= 0) & (qiy < ncy)
        # Invalid cells get key -1, which searchsorted maps to an empty range.
        qkeys = (qix * ncy + qiy).masked_fill(~inside, -1)

        lo = torch.searchsorted(keys_sorted, qkeys, right=False)
        hi = torch.searchsorted(keys_sorted, qkeys, right=True)
        cnt = hi - lo

        n_chunk = end - start
        local = torch.arange(n_chunk, dtype=torch.int64, device=device)
        for oi in range(len(_OFFSETS)):
            cnt_i = cnt[oi]
            total = int(cnt_i.sum().item())
            if total == 0:
                continue
            cum = torch.cumsum(cnt_i, 0) - cnt_i
            pair_q = torch.repeat_interleave(local, cnt_i)
            pos = (torch.arange(total, dtype=torch.int64, device=device)
                   - cum[pair_q] + lo[oi][pair_q])
            pair_l = order[pos]

            ddx = qx[pair_q] - lidar_xy[pair_l, 0]
            ddy = qy[pair_q] - lidar_xy[pair_l, 1]
            d2 = ddx * ddx + ddy * ddy
            keep = d2 < r2
            if bool(keep.any()):
                best.scatter_reduce_(0, start + pair_q[keep], pair_l[keep],
                                     reduce="amin", include_self=True)

    best.masked_fill_(best == n_lidar, -1)
    return best


def fast_eval_z_metric(lidar_xyz, gaussian_xyz, radius=0.1):
    """Drop-in replacement for ``evalution.eval_z_metric``."""
    idx = _min_index_match(lidar_xyz, gaussian_xyz, radius=radius)
    valid = idx != -1
    # gt_z is gathered through the LiDAR hit, pre_z stays on the Gaussian row.
    gt_z = lidar_xyz[idx[valid], 2:3]
    pre_z = gaussian_xyz[valid.nonzero().squeeze(1), 2:3]
    loss = torch.sqrt(torch.mean((gt_z - pre_z) ** 2))
    return loss.item()
