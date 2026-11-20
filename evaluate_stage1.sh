#!/usr/bin/env bash
# BReD stage1: per-scene formal no-loop K=1 full-frame RoGS Z evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
SCENE="${SCENE:-scene-0523}"
SCENE_ID="${SCENE_ID:-${SCENE/-/}}"
RUN_NAME="${RUN_NAME:-v8a17_ms_${SCENE_ID}_depth_prior_t0}"
CKPT="${CKPT:-$ROOT/results/stage1/$RUN_NAME/last.pt}"
INFO="${INFO:-$ROOT/assets/metadata/${SCENE_ID}_allframe_infos.pkl}"
ROADGT="${ROADGT:-$ROOT/assets/metadata/${SCENE_ID}_roadgt_rogs_cache.npz}"
PLY="${PLY:-$ROOT/assets/datasets/nuScenes_road_gt/${SCENE}.ply}"
OUT="${OUT:-$ROOT/results/stage1/${RUN_NAME}_eval}"
FRAMES="$("$PY" - "$INFO" <<'PY'
import pickle,sys
with open(sys.argv[1],'rb') as f: print(len(pickle.load(f)))
PY
)"
mkdir -p "$OUT"
cd "$ROOT/stage1"
"$PY" eval_v8a17_ms_rogs_z.py \
  --info-path "$INFO" --checkpoint "$CKPT" \
  --official-checkpoint "$ROOT/assets/checkpoints/bevstereo_ema_cbgs.pth" \
  --roadgt-cache "$ROADGT" --native-road-ply "$PLY" \
  --scene-name "$SCENE" --allow-frozen-val-scene --expected-frames "$FRAMES" \
  --out-json "$OUT/metrics.json" --fine-size 256 --outer-loops 0 \
  --workers "${WORKERS:-4}" --fast
