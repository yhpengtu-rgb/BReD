#!/usr/bin/env bash
# v5 continuous Stage1 -> Stage2 E2E (scene-0523, physical GPU0 only).
# Same Stage2 path/cache as run_e2e_v2.sh; Stage1 runs the v5 mono-only lift
# with the 112-bin logit-bias calibration (see v5/meta.json).
# Usage: run_e2e_v5.sh <run_tag> [K] [extra v5 flags...]
set -euo pipefail

BRED="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BENCH="$BRED/bench_opt"
V5DIR="$BENCH/goal/method_version/v5"
PY=/root/autodl-tmp/envs/bevdepth_rogs/bin/python
RUN="${1:?run tag required}"
K="${2:-16}"
shift 2 || true
EXTRA="$*"
BIAS="$V5DIR/ckpt/bias_4x80_lr02_e2.pt"
OUT="$V5DIR/out/e2e_$RUN"
mkdir -p "$OUT/s1" "$OUT/s2"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$BRED/stage1:$BRED/stage2:$BENCH"
export PATH="/root/autodl-tmp/envs/bevdepth_rogs/lib/python3.8/site-packages/ninja/data/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/root/autodl-tmp/envs/rogs/pkgs/cuda-toolkit}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
unset LD_LIBRARY_PATH

CONFIG="$BRED/stage2/configs/scene0523.yaml"
RUN_DIR="$BRED/results/stage2/a9_reference"
CACHE="$BENCH/cache/scene0523_dataset.pkl"

{
  echo "physical_gpu=0 CUDA_VISIBLE_DEVICES=0 v5 e2e K=$K bias=$BIAS"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
} > "$OUT/gpu_before.txt" 2>&1

START=$(date +%s.%N)

S1_START=$(date +%s.%N)
( cd "$BRED/stage1" && "$PY" "$V5DIR/eval_v5.py" \
    --info-path "$BRED/assets/metadata/scene0523_allframe_infos.pkl" \
    --checkpoint "$BRED/assets/checkpoints/v8a17_ms_last.pt" \
    --official-checkpoint "$BRED/assets/checkpoints/bevstereo_ema_cbgs.pth" \
    --roadgt-cache "$BRED/assets/metadata/scene0523_roadgt_rogs_cache.npz" \
    --native-road-ply "$BRED/assets/datasets/nuScenes_road_gt/scene-0523.ply" \
    --scene-name scene-0523 --allow-frozen-val-scene --expected-frames 391 \
    --out-json "$OUT/s1/metrics.json" \
    --fine-size 256 --outer-loops 0 --workers 4 --fast \
    --matcher ball_grid_cuda \
    --v2-stereo --v1-slots 0,1,2 --v1-em-iters 1 \
    --v2-corr-channels "$K" --v2-mask-batch \
    --v5-bin-bias "$BIAS" $EXTRA ) \
  > "$OUT/s1/stdout.log" 2> "$OUT/s1/stderr.log"
S1_END=$(date +%s.%N)

S2_START=$(date +%s.%N)
( cd "$BRED/stage2" && "$PY" "$BENCH/evaluate_stage2_opt.py" \
    --config "$CONFIG" --run-dir "$RUN_DIR" --epoch 2 \
    --output-dir "$OUT/s2/standalone_eval" --cache-path "$CACHE" ) \
  > "$OUT/s2/stdout.log" 2> "$OUT/s2/stderr.log"
S2_END=$(date +%s.%N)
END=$(date +%s.%N)

"$PY" - "$OUT" "$START" "$S1_START" "$S1_END" "$S2_START" "$S2_END" "$END" \
        "$RUN" "$K" <<'PY'
import json, os, sys
out, start, s1s, s1e, s2s, s2e, end, run, k = sys.argv[1:11]
f = float
meta = {"run": run, "version": "v5", "corr_channels": k,
        "stage1_mode": "mono_only + 112-bin logit bias",
        "physical_gpu": 0, "cuda_visible_devices": "0",
        "e2e_wall_s": round(f(end) - f(start), 3),
        "stage1_wall_s": round(f(s1e) - f(s1s), 3),
        "stage2_wall_s": round(f(s2e) - f(s2s), 3),
        "gap_s": round(f(s2s) - f(s1e), 3)}
p1 = os.path.join(out, "s1", "metrics.json")
if os.path.isfile(p1):
    m = json.load(open(p1))
    meta.update({
        "s1_frames": m["frames"], "s1_eval_elapsed_s": m["eval_elapsed_s"],
        "s1_coarse": {"rmse": m["coarse"]["Z_RMSE_m"],
                      "mae": m["coarse"]["Z_MAE_m"],
                      "matched": m["coarse"]["matched_points"]},
        "s1_fine": {"rmse": m["fine"]["Z_RMSE_m"],
                    "mae": m["fine"]["Z_MAE_m"],
                    "matched": m["fine"]["matched_points"]},
    })
p2 = os.path.join(out, "s2", "standalone_eval", "epoch_002", "metrics.json")
if os.path.isfile(p2):
    m2 = json.load(open(p2))
    meta.update({"s2_psnr": m2["bev"]["PSNR"], "s2_miou": m2["bev"]["mIoU"][0],
                 "s2_z_rmse_m": m2["z_rmse_m"],
                 "s2_dataset_source": m2.get("dataset_source"),
                 "s2_views": m2.get("dataset_views")})
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
print(json.dumps(meta, indent=2))
PY
echo "v5 e2e done: $OUT"
