#!/usr/bin/env bash
# After round 3 (300k) finishes, run the multi-level closed-loop eval (wizard-teleport
# to varied dungeon levels, far goals) on the 300k idm head plus a matched random baseline.
set -uo pipefail
r3log="${ROUND3_LOG:-/tmp/claude-0/-workspace-pan1ni/05ce2056-535d-456e-b5b0-441c15b27289/scratchpad/round3-run.log}"
ck=reports/pixel-mse-player-only-sigreg02-300k/checkpoint.pt
head=reports/pixel-mse-player-only-sigreg02-300k-action/idm/best_action_checkpoint.pt
out=reports/pixel-mse-player-only-sigreg02-300k-action/multilevel
echo "multilevel launcher: waiting for round 3..."
while true; do
  grep -q "3-mode in-distribution evaluation complete" "$r3log" 2>/dev/null && { echo "round 3 done."; break; }
  pgrep -f "queue_round3_after_stairs.sh" >/dev/null 2>&1 || pgrep -f "queue_continue_idm.sh" >/dev/null 2>&1 || { echo "round 3 gone; proceeding if 300k exists."; break; }
  sleep 60
done
[ -f "$ck" ] || { echo "ERROR: 300k checkpoint missing" >&2; exit 1; }
[ -f "$head" ] || { echo "ERROR: 300k idm head missing" >&2; exit 1; }
mkdir -p "$out"
echo "=== multilevel: idm sampled policy (300k) ==="
uv run python -m pan1ni.nle_multilevel_eval --world-checkpoint "$ck" --action-checkpoint "$head" \
  --output "$out/idm-sample.json" --episodes 32 --max-steps 200 --min-goal-distance 8 \
  --levels 1 2 3 4 5 6 --action-selection sample --device cuda
echo "=== multilevel: uniform-random baseline ==="
uv run python -m pan1ni.nle_multilevel_eval --world-checkpoint "$ck" --uniform-baseline \
  --output "$out/uniform.json" --episodes 32 --max-steps 200 --min-goal-distance 8 \
  --levels 1 2 3 4 5 6 --device cuda
echo "multilevel eval complete"
