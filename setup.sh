#!/usr/bin/env bash
# Prefer a local CUDA build; on the original ABI, bundled prebuilt extensions also work.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
export MAX_JOBS="${MAX_JOBS:-4}"
if ! command -v nvcc >/dev/null 2>&1; then
  "$PY" - "$ROOT" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
required=[
 'vendor/BEVDepth/bevdepth/ops/voxel_pooling_inference',
 'vendor/BEVDepth/bevdepth/ops/voxel_pooling_train',
 'vendor/gaussian_rasterization/diff_gaussian_rasterization',
 'vendor/ortho_gaussian_rasterization/diff_gaussian_rasterization',
 'vendor/diff_gs_label/diff_gs_label',
]
missing=[d for d in required if not list((root/d).glob('*.so'))]
if missing:
 raise SystemExit('nvcc not found and no bundled prebuilt extension for: '+', '.join(missing))
print('nvcc not found; using BReD-bundled prebuilt extensions.')
print('These binaries require the original Python 3.8 / PyTorch 1.13.1 ABI.')
PY
  exit 0
fi
for DIR in \
  "$ROOT/vendor/BEVDepth" \
  "$ROOT/vendor/gaussian_rasterization" \
  "$ROOT/vendor/ortho_gaussian_rasterization" \
  "$ROOT/vendor/diff_gs_label"; do
  echo "[build] $DIR"
  (cd "$DIR" && "$PY" setup.py build_ext --inplace)
done
echo "All BReD CUDA extensions built in place."
