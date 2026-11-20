#!/usr/bin/env bash
# v7 E2E runner (Stage1 -> Stage2, physical GPU0).
# Same locked Stage1/Stage2 protocol as v5.6; every v7.x lever is an opt-in env
# (V5_HEAD_CAMBATCH, ...) or an eval_v5-side module, so this runner just forwards
# the environment and keeps artifacts under v7/out.
# Usage: [SCENE=scene-0035] V7VER=v7.1 V5_HEAD_CAMBATCH=1 run_v7.sh <run_tag> [K] [extra flags...]
# scene-0523 (default) uses the reference checkpoint / a9_reference epoch 2;
# every other scene uses its own stage1+stage2 trained weights.
set -euo pipefail

BRED="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BENCH="$BRED/bench_opt"
V5DIR="$BENCH/goal/method_version/v5"
V7DIR="$BENCH/goal/method_version/v7"
S2ENTRY="$V5DIR/evaluate_stage2_v56.py"
PY=/root/autodl-tmp/envs/bevdepth_rogs/bin/python
RUN="${1:?run tag required}"
K="${2:-16}"
shift 2 || true
EXTRA="$*"
BIAS="$V5DIR/ckpt/bias_4x80_lr02_e2.pt"
OUT="$V7DIR/out/e2e_$RUN"
VER="${V7VER:-v7.x}"
mkdir -p "$OUT/s1" "$OUT/s2"

SCENE="${SCENE:-scene-0523}"
SCENE_ID="${SCENE_ID:-${SCENE/-/}}"
INFO="$BRED/assets/metadata/${SCENE_ID}_allframe_infos.pkl"
ROADGT="$BRED/assets/metadata/${SCENE_ID}_roadgt_rogs_cache.npz"
PLY="$BRED/assets/datasets/nuScenes_road_gt/${SCENE}.ply"
FRAMES="$("$PY" - "$INFO" <<'PYEOF'
import pickle, sys
with open(sys.argv[1], 'rb') as f:
    print(len(pickle.load(f)))
PYEOF
)"
if [[ "$SCENE" == "scene-0523" ]]; then
  S1_CKPT="${S1_CKPT:-$BRED/assets/checkpoints/v8a17_ms_last.pt}"
  S2_RUN_DIR="${S2_RUN_DIR:-$BRED/results/stage2/a9_reference}"
  S2_EPOCH="${S2_EPOCH:-2}"
  S2_CACHE="${S2_CACHE:-$BENCH/cache/scene0523_dataset.pkl}"
  export V8_IP_CACHE_DIR="${V8_IP_CACHE_DIR:-$BRED/assets/cache/v8_img_u8}"
else
  S1_CKPT="${S1_CKPT:-$BRED/results/stage1/v8a17_ms_${SCENE_ID}_depth_prior_t0/last.pt}"
  S2_RUN_DIR="${S2_RUN_DIR:-$(ls -d "$BRED"/results/stage2/${SCENE_ID}_rogs_wolidar_6ep/*/ | head -1)}"
  S2_RUN_DIR="${S2_RUN_DIR%/}"
  S2_EPOCH="${S2_EPOCH:-6}"
  S2_CACHE="${S2_CACHE:-$BENCH/cache/${SCENE_ID}_dataset.pkl}"
  export V8_IP_CACHE_DIR="${V8_IP_CACHE_DIR:-$BRED/assets/cache/${SCENE_ID}/v8_img_u8}"
fi
echo "v7 runner scene=$SCENE frames=$FRAMES s1_ckpt=$S1_CKPT" >&2
echo "v7 runner stage2 run_dir=$S2_RUN_DIR epoch=$S2_EPOCH" >&2

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$BRED/stage1:$BRED/stage2:$BENCH:$V7DIR:$V7DIR/../v8"
export PATH="/root/autodl-tmp/envs/bevdepth_rogs/lib/python3.8/site-packages/ninja/data/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/root/autodl-tmp/envs/rogs/pkgs/cuda-toolkit}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
unset LD_LIBRARY_PATH

# locked v5.6 Stage1 flags, overridable by the caller
export V5_B54_POOL="${V5_B54_POOL:-shared}"
export V5_SKIP_STEREO_HEAD="${V5_SKIP_STEREO_HEAD:-1}"
export V5_HEAD_GEO="${V5_HEAD_GEO:-1}"

CONFIG="$BRED/stage2/configs/${SCENE_ID}.yaml"
RUN_DIR="$S2_RUN_DIR"
CACHE="$S2_CACHE"

{
  echo "physical_gpu=0 CUDA_VISIBLE_DEVICES=0 $VER K=$K cambatch=${V5_HEAD_CAMBATCH:-0}"
  echo "scene=$SCENE frames=$FRAMES epoch=$S2_EPOCH"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
} > "$OUT/gpu_before.txt" 2>&1

START=$(date +%s.%N)

S1_START=$(date +%s.%N)
( cd "$BRED/stage1" && "$PY" "$V5DIR/eval_v5.py" \
    --info-path "$INFO" \
    --checkpoint "$S1_CKPT" \
    --official-checkpoint "$BRED/assets/checkpoints/bevstereo_ema_cbgs.pth" \
    --roadgt-cache "$ROADGT" \
    --native-road-ply "$PLY" \
    --scene-name "$SCENE" --allow-frozen-val-scene --expected-frames "$FRAMES" \
    --out-json "$OUT/s1/metrics.json" \
    --fine-size 256 --outer-loops 0 --workers 4 --fast \
    --matcher ball_grid_cuda \
    --v2-stereo --v1-slots 0,1,2 --v1-em-iters 1 \
    --v2-corr-channels "$K" --v2-mask-batch \
    --v5-bin-bias "$BIAS" $EXTRA ) \
  > "$OUT/s1/stdout.log" 2> "$OUT/s1/stderr.log"
S1_END=$(date +%s.%N)

S2_START=$(date +%s.%N)
( cd "$BRED/stage2" && "$PY" "$S2ENTRY" \
    --config "$CONFIG" --run-dir "$RUN_DIR" --epoch "$S2_EPOCH" \
    --output-dir "$OUT/s2/standalone_eval" --cache-path "$CACHE" ) \
  > "$OUT/s2/stdout.log" 2> "$OUT/s2/stderr.log"
S2_END=$(date +%s.%N)
END=$(date +%s.%N)

"$PY" - "$OUT" "$START" "$S1_START" "$S1_END" "$S2_START" "$S2_END" "$END" \
        "$RUN" "$K" "$VER" "$SCENE" "$FRAMES" "$S2_EPOCH" <<'PY'
import json, os, sys
(out, start, s1s, s1e, s2s, s2e, end, run, k, ver,
 scene, frames, epoch) = sys.argv[1:15]
f = float
meta = {"run": run, "version": ver, "scene": scene, "frames": int(frames),
        "corr_channels": k, "stage2_epoch": int(epoch),
        "stage1_mode": "mono_only + 112-bin logit bias",
        "head_cambatch": os.environ.get("V5_HEAD_CAMBATCH", "0"),
        "physical_gpu": 0, "cuda_visible_devices": "0",
        "e2e_wall_s": round(f(end) - f(start), 3),
        "e2e_wall_min": round((f(end) - f(start)) / 60.0, 4),
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
p2 = os.path.join(out, "s2", "standalone_eval",
                  "epoch_%03d" % int(epoch), "metrics.json")
if os.path.isfile(p2):
    m2 = json.load(open(p2))
    meta.update({"s2_psnr": m2["bev"]["PSNR"], "s2_miou": m2["bev"]["mIoU"][0],
                 "s2_z_rmse_m": m2["z_rmse_m"],
                 "s2_dataset_source": m2.get("dataset_source"),
                 "s2_views": m2.get("dataset_views")})
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
print(json.dumps(meta, indent=2))
PY
echo "v7 e2e done: $OUT"
