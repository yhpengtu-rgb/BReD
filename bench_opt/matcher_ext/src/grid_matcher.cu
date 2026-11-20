// Direct CUDA uniform-grid matcher.
//
// Reuses the static CSR uniform grid built by the torch-side index
// (origin, occupied-cell slot -> [start, cnt), stable-sorted native order).
// One thread per query: scan the full 3x3 neighbor cells, walk each CSR
// range, judge FP32 squared distance against radius, and keep the smallest
// native RoadGT index among in-radius references (argmin native, NOT nearest
// distance).  No candidate tensor, no materialized mask, no gather on device.
//
// Semantics kept identical to GTBank.match_grid:
//   radius 0.1 m; FP32 XY squared distance; duplicates retained;
//   empty / out-of-bounds -> native -1; coarse/fine call independently.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <climits>

__global__ void grid_matcher_kernel(
    const float* __restrict__ q_xy,      // [Q, 2]
    const float* __restrict__ origin,    // [2]
    const long* __restrict__ cell_slot,   // [nc0*nc1], -1 if empty
    const long* __restrict__ start,       // [U]
    const long* __restrict__ cnt,         // [U]
    const long* __restrict__ order,       // [M] sorted slot -> native index
    const float* __restrict__ sorted_xy,  // [M, 2]
    const int nc0, const int nc1, const int Q,
    const float h, const float radius2,
    long* __restrict__ chosen) {           // [Q]
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= Q) return;

  const float qx = q_xy[2 * q];
  const float qy = q_xy[2 * q + 1];
  const float ox = origin[0];
  const float oy = origin[1];

  const int cx = (int)floorf((qx - ox) / h);
  const int cy = (int)floorf((qy - oy) / h);

  long best_native = LONG_MAX;

  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      const int cc = cx + dx;
      const int rr = cy + dy;
      if (cc < 0 || cc >= nc0 || rr < 0 || rr >= nc1) continue;
      const long c = (long)cc * nc1 + rr;
      const long slot = cell_slot[c];
      if (slot < 0) continue;
      const long st = start[slot];
      const long cn = cnt[slot];
      for (long j = 0; j < cn; ++j) {
        const long g = st + j;
        const float sx = sorted_xy[2 * g];
        const float sy = sorted_xy[2 * g + 1];
        const float ex = qx - sx;
        const float ey = qy - sy;
        const float d2 = ex * ex + ey * ey;
        if (d2 <= radius2) {
          const long nat = order[g];
          if (nat < best_native) best_native = nat;
        }
      }
    }
  }

  chosen[q] = (best_native == LONG_MAX) ? -1 : best_native;
}

torch::Tensor grid_matcher_forward(
    torch::Tensor q_xy,
    torch::Tensor origin,
    torch::Tensor cell_slot,
    torch::Tensor start,
    torch::Tensor cnt,
    torch::Tensor order,
    torch::Tensor sorted_xy,
    int64_t nc0,
    int64_t nc1,
    double h,
    double radius) {
  const int Q = q_xy.size(0);
  auto chosen = torch::full({Q}, -1, q_xy.options().dtype(torch::kInt64));

  const float radius2 = (float)(radius * radius);
  const int threads = 256;
  const int blocks = (Q + threads - 1) / threads;
  grid_matcher_kernel<<<blocks, threads>>>(
      q_xy.data_ptr<float>(),
      origin.data_ptr<float>(),
      cell_slot.data_ptr<long>(),
      start.data_ptr<long>(),
      cnt.data_ptr<long>(),
      order.data_ptr<long>(),
      sorted_xy.data_ptr<float>(),
      (int)nc0, (int)nc1, Q,
      (float)h, radius2,
      chosen.data_ptr<long>());
  return chosen;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grid_matcher", &grid_matcher_forward, "direct CSR grid matcher");
}
