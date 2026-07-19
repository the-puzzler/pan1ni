#!/usr/bin/env bash
# In-distribution action pipeline for a player-only tile-pixel MSE world model.
# For each goal-aware feature: train a movement-only (8-class) human-data head, then
# run the full-NLE closed-loop eval with both argmax and sampled policies. Adds a
# uniform-random control and an ablation summary. No MiniHack data is involved.
set -euo pipefail

world_output="${1:-reports/pixel-mse-player-only-sigreg02}"
action_output="${2:-reports/pixel-mse-player-only-sigreg02-action-indist}"
steps="${STEPS:-5000}"
episodes="${EPISODES:-32}"
max_steps="${MAX_STEPS:-128}"
seed="${SEED:-12345}"
checkpoint="${world_output}/checkpoint.pt"

uv run python - "${world_output}" <<'PY'
import json, sys
from pathlib import Path
import torch
output = Path(sys.argv[1])
metrics = json.loads((output / "metrics.json").read_text())
checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
assert metrics["timeline"][-1]["step"] >= metrics["config"]["steps"], "world-model run incomplete"
assert metrics["config"]["training_sources"] == ["player"], "world model was not player-only"
assert checkpoint["config"]["observation_mode"] == "pixels", "refusing non-pixel checkpoint"
assert checkpoint["config"]["prediction_objective"] == "mse", "refusing non-MSE checkpoint"
print("world checkpoint verified: player-only, pixels, MSE")
PY

features=(predictor_layer1 predictor_layer2 predictor_hidden predicted_next idm)
mkdir -p "${action_output}"

for feature in "${features[@]}"; do
  feature_output="${action_output}/${feature}"
  echo "=== training movement-only head from ${feature} ==="
  uv run python -m pan1ni.pixel_action_train \
    --checkpoint "${checkpoint}" \
    --player-db data/nld/nld-nao.db \
    --player-dataset nld-nao-human-8shard \
    --output "${feature_output}" \
    --steps "${steps}" \
    --feature "${feature}" \
    --context-length 8 \
    --goal-horizon 64 \
    --eval-every 200 \
    --device cuda

  echo "=== closed-loop (argmax) ${feature} ==="
  uv run python -m pan1ni.nle_closed_loop_eval \
    --world-checkpoint "${checkpoint}" \
    --action-checkpoint "${feature_output}/best_action_checkpoint.pt" \
    --output "${feature_output}/nle-closed-loop-eval.json" \
    --video-output "${feature_output}/nle-closed-loop.mp4" \
    --episodes "${episodes}" --max-steps "${max_steps}" --seed "${seed}" \
    --action-selection argmax --device cuda

  echo "=== closed-loop (sampled) ${feature} ==="
  uv run python -m pan1ni.nle_closed_loop_eval \
    --world-checkpoint "${checkpoint}" \
    --action-checkpoint "${feature_output}/best_action_checkpoint.pt" \
    --output "${feature_output}/nle-sampled-closed-loop-eval.json" \
    --video-output "${feature_output}/nle-sampled-closed-loop.mp4" \
    --episodes "${episodes}" --max-steps "${max_steps}" --seed "${seed}" \
    --action-selection sample --temperature 1.0 --device cuda
done

echo "=== uniform-random control ==="
mkdir -p "${action_output}/uniform_random"
uv run python -m pan1ni.nle_random_baseline \
  --output "${action_output}/uniform_random/nle-closed-loop-eval.json" \
  --episodes "${episodes}" --max-steps "${max_steps}" --seed "${seed}"

echo "=== ablation summaries ==="
uv run python -m pan1ni.nle_action_ablation_report "${action_output}" \
  --output "${action_output}/nle-ablation-summary.png" \
  --evaluation-name nle-closed-loop-eval.json
uv run python -m pan1ni.nle_action_ablation_report "${action_output}" \
  --output "${action_output}/nle-sampled-ablation-summary.png" \
  --evaluation-name nle-sampled-closed-loop-eval.json

echo "in-distribution action ablations and closed-loop evaluations complete"
