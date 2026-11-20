#!/usr/bin/env python3
"""Collect real multi-scene training and evaluation status without changing runs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

BRED_ROOT = Path(__file__).resolve().parents[1]
SCENES = ("scene-0035", "scene-0519", "scene-0905", "scene-1062")


def read_json(path):
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"error": "invalid_or_partial_json", "path": str(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    scenes = []
    for scene in SCENES:
        scene_id = scene.replace("-", "")
        warm = BRED_ROOT / f"results/stage1/v8a9_{scene_id}_depth_prior_t0"
        stage1 = BRED_ROOT / f"results/stage1/v8a17_ms_{scene_id}_depth_prior_t0"
        stage1_metrics = read_json(BRED_ROOT / f"results/stage1/v8a17_ms_{scene_id}_depth_prior_t0_eval/metrics.json")
        stage2_root = BRED_ROOT / f"results/stage2/{scene_id}_rogs_wolidar_6ep"
        stage2_state = read_json(stage2_root / "training_6ep_state.json")
        asset = read_json(BRED_ROOT / f"results/multiscene/assets/{scene}.json")
        runs = sorted(stage2_root.glob(f"{scene}-no_z-8m-*"))
        metrics = []
        if runs:
            for path in sorted(runs[-1].glob("metrics_epoch_*.json")):
                value = read_json(path)
                if value is not None:
                    metrics.append(value)
        scenes.append({
            "scene": scene, "asset": asset,
            "warmstart": {"last": (warm / "last.pt").is_file(),
                          "epoch_checkpoints": len(list(warm.glob("epoch_*.pt")))},
            "stage1": {"last": (stage1 / "last.pt").is_file(),
                       "epoch_checkpoints": len(list(stage1.glob("epoch_*.pt"))),
                       "metrics": stage1_metrics},
            "stage2": {"state": stage2_state, "run_dir": str(runs[-1]) if runs else None,
                       "metrics": metrics},
            "complete": bool(stage1_metrics and stage2_state and
                             stage2_state.get("state") == "complete" and
                             stage2_state.get("metric_count") == 6),
        })
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "complete_scenes": sum(item["complete"] for item in scenes), "scenes": scenes}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(output)
    print(text, end="")


if __name__ == "__main__":
    main()
