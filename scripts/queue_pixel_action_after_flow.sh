#!/usr/bin/env bash
set -euo pipefail

flow_pid="${1:?usage: queue_pixel_action_after_flow.sh FLOW_PID [FLOW_OUTPUT] [ACTION_OUTPUT]}"
flow_output="${2:-reports/pixel-flow-human-keyroom-large}"
action_output="${3:-reports/pixel-flow-human-keyroom-action}"

echo "waiting for tile-pixel world-model process ${flow_pid}"
while kill -0 "${flow_pid}" 2>/dev/null; do
  sleep 30
done

feature="$(uv run python - "${flow_output}" <<'PY'
import json
from pathlib import Path
import sys
import torch

output = Path(sys.argv[1])
metrics = json.loads((output / "metrics.json").read_text())
checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
assert metrics["timeline"][-1]["step"] >= metrics["config"]["steps"], "world-model run incomplete"
assert checkpoint["config"]["observation_mode"] == "pixels", "refusing non-pixel checkpoint"
objective = checkpoint["config"]["prediction_objective"]
assert objective in {"mse", "flow"}, f"unsupported objective: {objective}"
print("flow_residual" if objective == "flow" else "predictor_hidden")
PY
)"

echo "world model complete; training weighted ${feature} action decoder"
uv run python -m pan1ni.minihack_action_train \
  --checkpoint "${flow_output}/checkpoint.pt" \
  --data data/minihack/keyroom-rgb-1600.hdf5 \
  --output "${action_output}" \
  --steps 2000 \
  --batch-size 128 \
  --context-length 8 \
  --goal-horizon 64 \
  --hidden-dim 1024 \
  --hidden-layers 2 \
  --feature "${feature}" \
  --eval-every 100 \
  --validation-samples 1024 \
  --num-workers 12 \
  --device cuda

echo "action decoder complete; rendering native-tile closed-loop evaluation"
uv run python -m pan1ni.minihack_closed_loop_video \
  --world-checkpoint "${flow_output}/checkpoint.pt" \
  --action-checkpoint "${action_output}/best_action_checkpoint.pt" \
  --output "${action_output}/closed-loop.mp4" \
  --max-steps 200 \
  --fps 8 \
  --seed 12345 \
  --device cuda

echo "tile-pixel world model, action probe, and closed-loop evaluation complete"
