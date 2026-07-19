#!/usr/bin/env bash
set -euo pipefail

world_pid="${1:?usage: queue_player_action_after_world.sh WORLD_PID [WORLD_OUTPUT] [ACTION_OUTPUT]}"
world_output="${2:-reports/pixel-mse-player-only}"
action_output="${3:-reports/pixel-mse-player-only-action}"

echo "waiting for player-only tile-pixel world-model process ${world_pid}"
while kill -0 "${world_pid}" 2>/dev/null; do
  sleep 30
done

uv run python - "${world_output}" <<'PY'
import json
from pathlib import Path
import sys
import torch

output = Path(sys.argv[1])
metrics = json.loads((output / "metrics.json").read_text())
checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
assert metrics["timeline"][-1]["step"] >= metrics["config"]["steps"], "world-model run incomplete"
assert metrics["config"]["training_sources"] == ["player"], "world model was not player-only"
assert checkpoint["config"]["observation_mode"] == "pixels", "refusing non-pixel checkpoint"
assert checkpoint["config"]["prediction_objective"] == "mse", "refusing non-MSE checkpoint"
PY

features=(current_latent predictor_layer1 predictor_layer2 predictor_hidden predicted_next)
mkdir -p "${action_output}"
uv run python - "${action_output}" "${world_output}" "${features[@]}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = {
    "status": "queued",
    "world_output": sys.argv[2],
    "action_label_sources": {
        "movement_0_7": "high-confidence one-cell cursor deltas from human NLD-NAO",
        "pickup_apply_8_9": "MiniHack KeyRoom oracle",
    },
    "features": sys.argv[3:],
    "action_steps_per_feature": 5000,
    "closed_loop_episodes_per_feature": 32,
    "representative_mp4_per_feature": True,
}
(root / "queue-manifest.json").write_text(json.dumps(manifest, indent=2))
PY

for feature in "${features[@]}"; do
  feature_output="${action_output}/${feature}"
  echo "training semantic action decoder from ${feature}"
  uv run python -m pan1ni.inferred_action_train \
    --checkpoint "${world_output}/checkpoint.pt" \
    --player-db data/nld/nld-nao.db \
    --player-dataset nld-nao-human-8shard \
    --simulator-data data/minihack/keyroom-rgb-1600.hdf5 \
    --output "${feature_output}" \
    --steps 5000 \
    --player-decode-batch-size 128 \
    --player-batch-size 48 \
    --special-batch-size 8 \
    --context-length 8 \
    --goal-horizon 64 \
    --hidden-dim 1024 \
    --hidden-layers 2 \
    --class-weight-samples 20000 \
    --feature "${feature}" \
    --eval-every 200 \
    --player-validation-samples 1024 \
    --special-validation-samples 512 \
    --num-workers 12 \
    --prefetch-depth 3 \
    --windows-per-block 4 \
    --window-stride 16 \
    --device cuda

  echo "evaluating ${feature} over 32 closed-loop episodes"
  uv run python -m pan1ni.minihack_closed_loop_eval \
    --world-checkpoint "${world_output}/checkpoint.pt" \
    --action-checkpoint "${feature_output}/best_action_checkpoint.pt" \
    --output "${feature_output}/closed-loop-eval.json" \
    --episodes 32 \
    --max-steps 200 \
    --seed 12345 \
    --device cuda

  echo "rendering representative ${feature} closed-loop MP4"
  uv run python -m pan1ni.minihack_closed_loop_video \
    --world-checkpoint "${world_output}/checkpoint.pt" \
    --action-checkpoint "${feature_output}/best_action_checkpoint.pt" \
    --output "${feature_output}/closed-loop.mp4" \
    --max-steps 200 \
    --fps 8 \
    --seed 12345 \
    --device cuda
done

uv run python -m pan1ni.action_ablation_report "${action_output}" \
  --output "${action_output}/ablation-summary.png"

echo "all inferred-human action ablations and closed-loop evaluations complete"
