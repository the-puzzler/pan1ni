#!/usr/bin/env bash
set -euo pipefail

mse_pid="${1:?usage: queue_flow_after_mse.sh MSE_PID}"
mse_metrics="reports/nld-nao-human-rgb-encoder-heavy64-lewm-causal-goal/metrics.json"
mse_checkpoint="reports/nld-nao-human-rgb-encoder-heavy64-lewm-causal-goal/checkpoint.pt"
mse_action_output="reports/nld-nao-human-rgb-mse-action-weighted"
flow_output="reports/nld-nao-human-rgb-encoder-heavy64-flow-goal"
flow_action_output="reports/nld-nao-human-rgb-flow-action-weighted"

echo "waiting for MSE process ${mse_pid}"
while kill -0 "${mse_pid}" 2>/dev/null; do
  sleep 30
done

uv run python -c "import json; d=json.load(open('${mse_metrics}')); assert d['timeline'][-1]['step'] >= d['config']['steps'], 'MSE run did not complete; refusing to launch flow run'"

echo "MSE complete; training the MSE predictor-hidden action decoder"
uv run python -m pan1ni.nld_action_train \
  --checkpoint "${mse_checkpoint}" \
  --db data/nld/nld-aa-taster.db \
  --dataset nld-aa-taster \
  --output "${mse_action_output}" \
  --steps 2000 \
  --batch-size 64 \
  --context-length 8 \
  --goal-horizon 64 \
  --action-hidden-dim 1024 \
  --action-hidden-layers 2 \
  --class-weight-samples 100000 \
  --feature predictor_hidden \
  --eval-every 100 \
  --validation-samples 1024 \
  --num-workers 16 \
  --decoder-streams 2 \
  --prefetch-depth 8 \
  --windows-per-block 8 \
  --window-stride 16 \
  --device cuda

echo "MSE action decoder complete; rendering its closed-loop simulator evaluation"
uv run python -m pan1ni.minihack_closed_loop_video \
  --world-checkpoint "${mse_checkpoint}" \
  --action-checkpoint "${mse_action_output}/action_checkpoint.pt" \
  --output reports/nld-nao-human-rgb-mse-closed-loop.mp4 \
  --max-steps 200 \
  --fps 8 \
  --seed 12345 \
  --device cuda

echo "MSE action evaluation complete; running flow smoke test"
uv run python -m pan1ni.nld_ttyrec_train \
  --db data/nld/nld-nao.db \
  --dataset nld-nao-human-8shard \
  --output /tmp/pan1ni-flow-after-mse-smoke \
  --steps 2 \
  --batch-size 64 \
  --context-length 8 \
  --goal-horizon 64 \
  --sigreg-slices 32 \
  --eval-every 2 \
  --checkpoint-every 2 \
  --validation-samples 64 \
  --num-workers 8 \
  --prefetch-depth 2 \
  --windows-per-block 2 \
  --window-stride 16 \
  --model-scale large \
  --objective flow \
  --device cuda

echo "flow smoke test passed; starting full flow run"
uv run python -m pan1ni.nld_ttyrec_train \
  --db data/nld/nld-nao.db \
  --dataset nld-nao-human-8shard \
  --output "${flow_output}" \
  --steps 50000 \
  --batch-size 64 \
  --context-length 8 \
  --goal-horizon 64 \
  --sigreg-slices 256 \
  --eval-every 500 \
  --checkpoint-every 500 \
  --validation-samples 512 \
  --num-workers 16 \
  --decoder-streams 1 \
  --prefetch-depth 3 \
  --windows-per-block 4 \
  --window-stride 16 \
  --model-scale large \
  --objective flow \
  --device cuda

echo "flow pretraining complete; training residual-flow action decoder"
uv run python -m pan1ni.nld_action_train \
  --checkpoint "${flow_output}/checkpoint.pt" \
  --db data/nld/nld-aa-taster.db \
  --dataset nld-aa-taster \
  --output "${flow_action_output}" \
  --steps 2000 \
  --batch-size 64 \
  --context-length 8 \
  --goal-horizon 64 \
  --action-hidden-dim 1024 \
  --action-hidden-layers 2 \
  --class-weight-samples 100000 \
  --feature flow_residual \
  --eval-every 100 \
  --validation-samples 1024 \
  --num-workers 16 \
  --decoder-streams 2 \
  --prefetch-depth 8 \
  --windows-per-block 8 \
  --window-stride 16 \
  --device cuda

echo "flow action decoder complete; rendering its closed-loop simulator evaluation"
uv run python -m pan1ni.minihack_closed_loop_video \
  --world-checkpoint "${flow_output}/checkpoint.pt" \
  --action-checkpoint "${flow_action_output}/action_checkpoint.pt" \
  --output reports/nld-nao-human-rgb-flow-closed-loop.mp4 \
  --max-steps 200 \
  --fps 8 \
  --seed 12345 \
  --device cuda

echo "all queued MSE-action and flow experiments complete"
