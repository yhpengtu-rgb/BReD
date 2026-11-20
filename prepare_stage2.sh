#!/usr/bin/env bash
# Freeze one scene's v8-a17-ms and export both stage2 priors; no training occurs here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
SCENE="${SCENE:-scene-0523}"
SCENE_ID="${SCENE_ID:-${SCENE/-/}}"
RUN_NAME="${RUN_NAME:-v8a17_ms_${SCENE_ID}_depth_prior_t0}"
CKPT="${CKPT:-$ROOT/results/stage1/$RUN_NAME/last.pt}"
INFO="${INFO:-$ROOT/assets/metadata/${SCENE_ID}_allframe_infos.pkl}"
OFFICIAL="${OFFICIAL:-$ROOT/assets/checkpoints/bevstereo_ema_cbgs.pth}"
if [[ -z "${CACHE_ROOT:-}" ]]; then
  if [[ "$SCENE" == "scene-0523" ]]; then CACHE_ROOT="$ROOT/assets/cache"; else CACHE_ROOT="$ROOT/assets/cache/$SCENE_ID"; fi
fi
FRAMES="$("$PY" - "$INFO" <<'PY'
import pickle,sys
with open(sys.argv[1],'rb') as f: print(len(pickle.load(f)))
PY
)"
"$PY" "$ROOT/bridge/export_v10a1_elevation.py" \
  --checkpoint "$CKPT" --official-checkpoint "$OFFICIAL" --info-path "$INFO" --scene-name "$SCENE" \
  --out-dir "$CACHE_ROOT/elevation_export" --expected-frames "$FRAMES" \
  --workers "${WORKERS:-4}" --device "${DEVICE:-cuda:0}"
"$PY" "$ROOT/bridge/export_v11a4_geometry_prior.py" \
  --checkpoint "$CKPT" --official-checkpoint "$OFFICIAL" --info-path "$INFO" --scene-name "$SCENE" \
  --out-dir "$CACHE_ROOT/geometry_export" --expected-frames "$FRAMES" \
  --workers "${WORKERS:-4}" --device "${DEVICE:-cuda:0}"
echo "Stage2 priors for $SCENE are ready under $CACHE_ROOT"
