#!/usr/bin/env bash
# Sequential four-scene driver: one scene at a time on the same two GPUs.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
OUT="$ROOT/results/multiscene"
mkdir -p "$OUT"
exec 9>"$OUT/driver.lock"
flock -n 9 || { echo "Four-scene driver is already running" >&2; exit 1; }
STATUS="$OUT/status.jsonl"
SCENES=(${PIPELINE_SCENES:-scene-0035 scene-0519 scene-0905 scene-1062})
for SCENE in "${SCENES[@]}"; do
  SCENE_ID="${SCENE/-/}"
  LOG="$OUT/${SCENE_ID}.log"
  STARTED="$(date -Iseconds)"
  "$PY" - "$STATUS" "$SCENE" "$STARTED" <<'PY'
import json,sys
with open(sys.argv[1],'a') as f: f.write(json.dumps({'scene':sys.argv[2],'state':'running','started':sys.argv[3]})+'\n')
PY
  if SCENE="$SCENE" PYTHON="$PY" bash "$ROOT/run_scene_pipeline.sh" > "$LOG" 2>&1; then
    STATE=complete
  else
    STATE=failed
  fi
  FINISHED="$(date -Iseconds)"
  "$PY" - "$STATUS" "$SCENE" "$STATE" "$FINISHED" "$LOG" <<'PY'
import json,sys
with open(sys.argv[1],'a') as f: f.write(json.dumps({'scene':sys.argv[2],'state':sys.argv[3],'finished':sys.argv[4],'log':sys.argv[5]})+'\n')
PY
done
echo "Four-scene driver finished; see $STATUS"
