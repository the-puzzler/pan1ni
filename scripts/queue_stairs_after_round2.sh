#!/usr/bin/env bash
# Wait for round 2 (100k->200k continue + eval) to finish and free the GPU, then run
# the 10-class stairs-decoding experiment on the 200k model.
set -uo pipefail

log2="${LOG2:-/tmp/claude-0/-workspace-pan1ni/05ce2056-535d-456e-b5b0-441c15b27289/scratchpad/round2-run.log}"
checkpoint="${CHECKPOINT:-reports/pixel-mse-player-only-sigreg02-200k/checkpoint.pt}"
action_output="${ACT:-reports/pixel-mse-player-only-sigreg02-200k-stairs}"

echo "stairs launcher: waiting for round 2 to complete (frees the GPU)..."
while true; do
  if grep -q "3-mode in-distribution evaluation complete" "${log2}" 2>/dev/null; then
    echo "round 2 completed normally."; break
  fi
  if ! pgrep -f "queue_continue_pretrain_then_eval.sh" >/dev/null 2>&1 \
     && ! pgrep -f "queue_round2_after_round1.sh" >/dev/null 2>&1; then
    echo "round 2 driver no longer running (no sentinel); proceeding if the 200k checkpoint exists."
    break
  fi
  sleep 60
done

if [ ! -f "${checkpoint}" ]; then
  echo "ERROR: ${checkpoint} not found; round 2 pretrain did not finish. Aborting stairs run." >&2
  exit 1
fi
echo "=== starting stairs-decoding experiment on ${checkpoint} ==="
exec bash scripts/queue_stairs_decode.sh "${checkpoint}" "${action_output}"
