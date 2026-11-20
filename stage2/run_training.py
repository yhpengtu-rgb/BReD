"""Execute and monitor one scene's BReD stage2 six-epoch training."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parent
BRED_ROOT = ROOT.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    scene = config["dataset"]["clip_list"][0]
    output_root = Path(config["output"])
    if not output_root.is_absolute():
        output_root = BRED_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "training_6ep_state.json"
    log_path = output_root / "train_6ep.log"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("state") == "complete" and previous.get("metric_count") == 6:
            print(json.dumps(previous))
            return

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "train.py", "--config", str(config_path)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state_path.write_text(json.dumps({"state": "running", "pid": process.pid,
                                          "scene": scene}) + "\n")
        failures = ("Traceback (most recent call last)", "v11-a9 non-finite residual field",
                    "v11-a9 residual bound violated")
        while True:
            code = process.poll()
            text = log_path.read_text() if log_path.exists() else ""
            if any(marker in text for marker in failures):
                if code is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    code = process.wait(timeout=30)
                state_path.write_text(json.dumps({"state": "failed", "exit_code": code,
                                                  "scene": scene, "reason": "health_failure"}, indent=2) + "\n")
                raise RuntimeError(f"Stage2 health failure; see {log_path}")
            if code is not None:
                break
            time.sleep(15)
    runs = sorted(output_root.glob(f"{scene}-no_z-8m-*"))
    metrics = []
    if runs:
        for path in sorted(runs[-1].glob("metrics_epoch_*.json")):
            metrics.append(json.loads(path.read_text()))
    state = "complete" if code == 0 and len(metrics) == 6 else "failed"
    result = {"state": state, "exit_code": code, "scene": scene,
              "run_dir": str(runs[-1]) if runs else None, "metric_count": len(metrics)}
    state_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(result, metrics=metrics), ensure_ascii=False))
    if state != "complete":
        raise SystemExit(code or 1)


if __name__ == "__main__":
    main()
