#!/usr/bin/env bash
# Overnight: continue 300k -> 400k (lr 1e-4) + idm heads + 3-mode eval, then run the
# multi-level closed-loop eval on the 400k idm policy (numbers + matched random
# baseline) and render lots of game-left / goal-right videos across dungeon levels.
set -uo pipefail

# guard: wait until the GPU is free (no training / eval running)
while pgrep -f "pixel_flow_train" >/dev/null 2>&1 || pgrep -f "nle_multilevel_eval" >/dev/null 2>&1 \
   || pgrep -f "nle_closed_loop_eval" >/dev/null 2>&1 || pgrep -f "pixel_action_train" >/dev/null 2>&1; do
  echo "waiting for GPU to free up..."; sleep 60
done

# 1. round 4 pretrain + idm heads + 3-mode eval
echo "=== round 4: 300k -> 400k, lr 1e-4, idm variants ==="
OLD=reports/pixel-mse-player-only-sigreg02-300k \
NEW=reports/pixel-mse-player-only-sigreg02-400k \
ACT=reports/pixel-mse-player-only-sigreg02-400k-action \
STEPS=400000 LR=1e-4 FEATURES="idm idm_history" \
bash scripts/queue_continue_idm.sh

ck=reports/pixel-mse-player-only-sigreg02-400k/checkpoint.pt
head=reports/pixel-mse-player-only-sigreg02-400k-action/idm/best_action_checkpoint.pt
out=reports/pixel-mse-player-only-sigreg02-400k-action/multilevel
mkdir -p "$out"

# 2. multi-level eval — numbers (idm sampled)
echo "=== multilevel numbers: idm sampled (400k) ==="
uv run python -m pan1ni.nle_multilevel_eval --world-checkpoint "$ck" --action-checkpoint "$head" \
  --output "$out/idm-sample.json" --episodes 32 --max-steps 200 --min-goal-distance 8 \
  --levels 1 2 3 4 5 6 --action-selection sample --device cuda

# 3. matched uniform-random baseline
echo "=== multilevel baseline: uniform-random ==="
uv run python -m pan1ni.nle_multilevel_eval --world-checkpoint "$ck" --uniform-baseline \
  --output "$out/uniform.json" --episodes 32 --max-steps 200 --min-goal-distance 8 \
  --levels 1 2 3 4 5 6 --device cuda

# 4. lots of videos — best policy across dungeon levels 1-10 (game left / goal right)
echo "=== multilevel videos: 40 rollouts across levels 1-10 ==="
uv run python -m pan1ni.nle_multilevel_eval --world-checkpoint "$ck" --action-checkpoint "$head" \
  --output "$out/idm-video.json" --episodes 40 --max-steps 220 --min-goal-distance 6 \
  --levels 1 2 3 4 5 6 7 8 9 10 --action-selection sample \
  --video-dir "$out/videos" --num-videos 40 --device cuda

echo "round4 + multilevel + videos complete"
