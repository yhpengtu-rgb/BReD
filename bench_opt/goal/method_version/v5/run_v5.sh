#!/usr/bin/env bash
# v5 scene-0523 launcher: mono-only lift (+ optional fine-tuned depth branch).
#
# Usage: run_v5.sh <tag> [frames] [extra flags...]
#   extras go to eval_v5.py, e.g. --v5-depth-ckpt /abs/path.pt --split-timing
set -euo pipefail

BRED="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BENCH="$BRED/bench_opt"
V5DIR="$BENCH/goal/method_version/v5"
PY=/root/autodl-tmp/envs/bevdepth_rogs/bin/python

TAG="${1:?tag required}"
FRAMES="${2:-20}"
shift 2 || true
EXTRA="$*"

OUT="$V5DIR/out/$TAG"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$BRED/stage1:$BENCH"
export PATH="/root/autodl-tmp/envs/bevdepth_rogs/lib/python3.8/site-packages/ninja/data/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/root/autodl-tmp/envs/rogs/pkgs/cuda-toolkit}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
unset LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader > "$OUT/gpu_before.txt" 2>&1

START=$(date +%s.%N)
( cd "$BRED/stage1" && "$PY" -u "$V5DIR/eval_v5.py" \
    --info-path "$BRED/assets/metadata/scene0523_allframe_infos.pkl" \
    --checkpoint "$BRED/assets/checkpoints/v8a17_ms_last.pt" \
    --official-checkpoint "$BRED/assets/checkpoints/bevstereo_ema_cbgs.pth" \
    --roadgt-cache "$BRED/assets/metadata/scene0523_roadgt_rogs_cache.npz" \
    --native-road-ply "$BRED/assets/datasets/nuScenes_road_gt/scene-0523.ply" \
    --scene-name scene-0523 --allow-frozen-val-scene \
    --expected-frames 391 --max-frames "$FRAMES" \
    --out-json "$OUT/metrics.json" \
    --fine-size 256 --outer-loops 0 --workers 4 --fast \
    --matcher ball_grid_cuda \
    --v2-stereo --v1-slots 0,1,2 --v1-em-iters 1 \
    --v2-corr-channels 16 --v2-mask-batch $EXTRA ) \
  > "$OUT/stdout.log" 2> "$OUT/stderr.log"
END=$(date +%s.%N)

"$PY" - "$OUT" "$START" "$END" <<'PY'
import json, os, sys
out, start, end = sys.argv[1:4]
rec = {"wall_s": round(float(end) - float(start), 3)}
m = os.path.join(out, "metrics.json")
if os.path.isfile(m):
    mm = json.load(open(m))
    rec.update({
        "frames": mm.get("frames"),
        "eval_elapsed_s": mm.get("eval_elapsed_s"),
        "coarse_rmse_m": mm["coarse"]["Z_RMSE_m"],
        "fine_rmse_m": mm["fine"]["Z_RMSE_m"],
        "coarse_matched": mm["coarse"]["matched_points"],
        "fine_matched": mm["fine"]["matched_points"],
        "fine_mae_m": mm["fine"]["Z_MAE_m"],
    })
json.dump(rec, open(os.path.join(out, "summary.json"), "w"), indent=2)
print("V5_EVAL " + json.dumps(rec))
PY
grep -h "^SPLIT " "$OUT/stdout.log" || true
echo "v5 eval done: $OUT"
