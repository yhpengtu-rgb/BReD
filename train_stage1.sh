#!/usr/bin/env bash
# BReD stage1: per-scene v8-a17-ms formal 2-GPU, 6-epoch training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
SCENE="${SCENE:-scene-0523}"
SCENE_ID="${SCENE_ID:-${SCENE/-/}}"
RUN_NAME="${RUN_NAME:-v8a17_ms_${SCENE_ID}_depth_prior_t0}"
EPOCHS="${EPOCHS:-6}"
MAX_STEPS="${MAX_STEPS:-0}"
INFO="${INFO:-$ROOT/assets/metadata/${SCENE_ID}_allframe_infos.pkl}"
CKPT="${CKPT:-$ROOT/assets/checkpoints/bevstereo_ema_cbgs.pth}"
ROADGT="${ROADGT:-$ROOT/assets/metadata/${SCENE_ID}_roadgt_rogs_cache.npz}"
if [[ -z "${INIT:-}" ]]; then
  if [[ "$SCENE" == "scene-0523" ]]; then
    INIT="$ROOT/assets/checkpoints/v8a9_scene0523_last.pt"
  else
    INIT="$ROOT/results/stage1/v8a9_${SCENE_ID}_depth_prior_t0/last.pt"
  fi
fi
OUT="$ROOT/results/stage1/$RUN_NAME"
LOG="$ROOT/results/stage1/${RUN_NAME}.log"
for f in "$INFO" "$CKPT" "$INIT" "$ROADGT"; do
  [[ -f "$f" ]] || { echo "Missing required asset: $f (run train_stage1_warmstart.sh first for a new scene)" >&2; exit 1; }
done
[[ ! -e "$OUT" && ! -e "$LOG" ]] || { echo "Refusing to overwrite $OUT or $LOG" >&2; exit 1; }
mkdir -p "$ROOT/results/stage1"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export STEP_TIMEOUT="${STEP_TIMEOUT:-300}"
export USE_CHECKPOINT="${USE_CHECKPOINT:-0}"
export NCCL_ASYNC_ERROR_HANDLING=1
"$PY" - "$INFO" <<'PY'
import pickle, sys, torch
with open(sys.argv[1], 'rb') as f: infos = pickle.load(f)
assert infos, 'Empty scene info'
assert torch.cuda.device_count() >= 2, 'Two visible CUDA GPUs are required'
print(f'[preflight] {len(infos)} frames; two CUDA GPUs visible')
PY
cd "$ROOT/stage1"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 train_v8a17_ddp.py \
  --info-path "$INFO" --official-checkpoint "$CKPT" --init-checkpoint "$INIT" \
  --roadgt-cache "$ROADGT" --scene-name "$SCENE" --allow-frozen-val-scene \
  --run-name "$RUN_NAME" --fine-size 256 --epochs "$EPOCHS" --max-steps "$MAX_STEPS" \
  --workers "${WORKERS:-4}" --log-every "${LOG_EVERY:-20}" 2>&1 | tee "$LOG"
echo "[done] $OUT/last.pt" | tee -a "$LOG"
