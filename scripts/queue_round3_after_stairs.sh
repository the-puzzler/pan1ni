#!/usr/bin/env bash
# Round 3: wait for the stairs-decoding experiment (which itself waits for round 2)
# to finish and free the GPU, then continue 200k -> 300k at a further-reduced LR and
# rerun the full head-training + 3-mode closed-loop evaluation on the 300k model.
set -uo pipefail

stairs_log="${STAIRS_LOG:-/tmp/claude-0/-workspace-pan1ni/05ce2056-535d-456e-b5b0-441c15b27289/scratchpad/stairs-run.log}"

echo "round3 launcher: waiting for the stairs experiment to finish (frees the GPU)..."
while true; do
  if grep -q "stairs-decoding experiment complete" "${stairs_log}" 2>/dev/null; then
    echo "stairs experiment completed normally."; break
  fi
  if ! pgrep -f "queue_stairs_after_round2.sh" >/dev/null 2>&1 \
     && ! pgrep -f "queue_stairs_decode.sh" >/dev/null 2>&1; then
    echo "stairs driver no longer running (no sentinel); the 200k checkpoint exists, proceeding."
    break
  fi
  sleep 60
done

if [ ! -f reports/pixel-mse-player-only-sigreg02-200k/checkpoint.pt ]; then
  echo "ERROR: 200k checkpoint missing; aborting round 3." >&2
  exit 1
fi
echo "=== starting round3: 200k -> 300k, lr 1e-4 ==="
OLD=reports/pixel-mse-player-only-sigreg02-200k \
NEW=reports/pixel-mse-player-only-sigreg02-300k \
ACT=reports/pixel-mse-player-only-sigreg02-300k-action \
STEPS=300000 \
LR=1e-4 \
exec bash scripts/queue_continue_pretrain_then_eval.sh
