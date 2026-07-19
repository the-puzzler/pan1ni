#!/usr/bin/env bash
# Run the 10-class stairs-decoding experiment on the 300k model, once round 3
# (200k->300k continue + eval) finishes and frees the GPU. Mirrors the 200k stairs
# pass so descend/ascend recall can be compared across pretraining length.
set -uo pipefail

round3_log="${ROUND3_LOG:-/tmp/claude-0/-workspace-pan1ni/05ce2056-535d-456e-b5b0-441c15b27289/scratchpad/round3-run.log}"
checkpoint="${CHECKPOINT:-reports/pixel-mse-player-only-sigreg02-300k/checkpoint.pt}"
action_output="${ACT:-reports/pixel-mse-player-only-sigreg02-300k-stairs}"

echo "stairs-300k launcher: waiting for round 3 to complete (frees the GPU)..."
while true; do
  if grep -q "3-mode in-distribution evaluation complete" "${round3_log}" 2>/dev/null; then
    echo "round 3 completed normally."; break
  fi
  if ! pgrep -f "queue_round3_after_stairs.sh" >/dev/null 2>&1 \
     && ! pgrep -f "queue_continue_idm.sh" >/dev/null 2>&1; then
    echo "round 3 driver no longer running (no sentinel); proceeding if the 300k checkpoint exists."
    break
  fi
  sleep 60
done

if [ ! -f "${checkpoint}" ]; then
  echo "ERROR: ${checkpoint} not found; round 3 pretrain did not finish. Aborting 300k stairs run." >&2
  exit 1
fi
echo "=== starting stairs-decoding on 300k: ${checkpoint} ==="
exec bash scripts/queue_stairs_decode.sh "${checkpoint}" "${action_output}"
