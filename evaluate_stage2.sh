#!/usr/bin/env bash
# Evaluate one epoch (EPOCH=N) or every saved epoch in one scene's stage2 run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
SCENE="${SCENE:-scene-0523}"
SCENE_ID="${SCENE_ID:-${SCENE/-/}}"
if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_DIR="$("$PY" - "$ROOT" "$SCENE" "$SCENE_ID" <<'PY'
from pathlib import Path
import sys
root,scene,scene_id=sys.argv[1:]
runs=sorted((Path(root)/f'results/stage2/{scene_id}_rogs_wolidar_6ep').glob(f'{scene}-no_z-8m-*'))
if not runs: raise SystemExit('No stage2 run found; set RUN_DIR explicitly')
print(runs[-1])
PY
)"
fi
ARGS=(--config "$ROOT/stage2/configs/${SCENE_ID}.yaml" --run-dir "$RUN_DIR")
if [[ -n "${EPOCH:-}" ]]; then ARGS+=(--epoch "$EPOCH"); fi
exec "$PY" "$ROOT/stage2/evaluate.py" "${ARGS[@]}"
