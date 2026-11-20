#!/usr/bin/env bash
# Run the complete locked two-stage recipe for one scene, resuming completed stages.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE="${SCENE:?Set SCENE=scene-XXXX}"
SCENE_ID="${SCENE_ID:-${SCENE/-/}}"
export SCENE SCENE_ID
PY="${PYTHON:-python3}"
mkdir -p "$ROOT/results/multiscene"
exec 8>"$ROOT/results/multiscene/${SCENE_ID}.lock"
flock -n 8 || { echo "Scene pipeline is already running: $SCENE" >&2; exit 1; }
"$PY" "$ROOT/tools/prepare_scene_assets.py" --scene "$SCENE"
WARM="$ROOT/results/stage1/v8a9_${SCENE_ID}_depth_prior_t0/last.pt"
STAGE1="$ROOT/results/stage1/v8a17_ms_${SCENE_ID}_depth_prior_t0/last.pt"
METRICS="$ROOT/results/stage1/v8a17_ms_${SCENE_ID}_depth_prior_t0_eval/metrics.json"
archive_incomplete () {
  local run_name="$1" last="$2" stamp archive
  local out="$ROOT/results/stage1/$run_name"
  local log="$ROOT/results/stage1/${run_name}.log"
  if [[ ! -f "$last" && ( -d "$out" || -f "$log" ) ]]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    archive="$ROOT/results/stage1/archive/${run_name}_${stamp}"
    mkdir -p "$archive"
    [[ ! -d "$out" ]] || mv "$out" "$archive/output"
    [[ ! -f "$log" ]] || mv "$log" "$archive/train.log"
    echo "[resume] archived incomplete run to $archive"
  fi
}
archive_incomplete "v8a9_${SCENE_ID}_depth_prior_t0" "$WARM"
archive_incomplete "v8a17_ms_${SCENE_ID}_depth_prior_t0" "$STAGE1"
if [[ ! -f "$WARM" ]]; then
  PYTHON="$PY" bash "$ROOT/train_stage1_warmstart.sh"
fi
if [[ ! -f "$STAGE1" ]]; then
  PYTHON="$PY" bash "$ROOT/train_stage1.sh"
fi
if [[ ! -f "$METRICS" ]]; then
  PYTHON="$PY" bash "$ROOT/evaluate_stage1.sh"
fi
PYTHON="$PY" bash "$ROOT/prepare_stage2.sh"
PYTHON="$PY" bash "$ROOT/train_stage2.sh"
echo "[scene complete] $SCENE"
