#!/usr/bin/env bash
# BReD stage2: one scene's v11-a9 formal six-epoch training with per-epoch evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
SCENE="${SCENE:-scene-0523}"
SCENE_ID="${SCENE_ID:-${SCENE/-/}}"
CONFIG="${CONFIG:-$ROOT/stage2/configs/${SCENE_ID}.yaml}"
[[ -f "$CONFIG" ]] || { echo "Missing scene config: $CONFIG (run tools/prepare_scene_assets.py)" >&2; exit 1; }
exec "$PY" "$ROOT/stage2/run_training.py" --config "$CONFIG"
