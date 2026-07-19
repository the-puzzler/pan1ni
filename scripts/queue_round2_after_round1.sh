#!/usr/bin/env bash
# Round 2: wait for the current (round-1) continue-pretrain+eval driver to finish
# and free the GPU, then continue 100k -> 200k at a slightly lower LR and rerun the
# full head-training + 3-mode closed-loop evaluation on the 200k model.
set -uo pipefail

log1="${LOG1:-/tmp/claude-0/-workspace-pan1ni/05ce2056-535d-456e-b5b0-441c15b27289/scratchpad/continue-run.log}"

echo "round2 launcher: waiting for round1 to complete (frees the GPU)..."
while true; do
  if grep -q "3-mode evaluation complete" "${log1}" 2>/dev/null; then
    echo "round1 completed normally."; break
  fi
  if ! pgrep -f "queue_continue_pretrain_then_eval.sh" >/dev/null 2>&1; then
    echo "round1 driver is no longer running (no completion sentinel); the 100k checkpoint exists, proceeding."
    break
  fi
  sleep 60
done

echo "=== starting round2: 100k -> 200k, lr 2e-4 ==="
OLD=reports/pixel-mse-player-only-sigreg02-100k \
NEW=reports/pixel-mse-player-only-sigreg02-200k \
ACT=reports/pixel-mse-player-only-sigreg02-200k-action \
STEPS=200000 \
LR=2e-4 \
exec bash scripts/queue_continue_pretrain_then_eval.sh
