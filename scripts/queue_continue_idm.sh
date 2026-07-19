#!/usr/bin/env bash
# Continue the player-only tile-pixel MSE world model 50k -> 100k (preserving the
# original 50k checkpoint and archiving every 20k), then retrain the movement heads
# on the continued model and evaluate closed-loop in three goal-distance modes:
# short, far, and mixed. No MiniHack in the action pipeline.
set -euo pipefail

old_output="${OLD:-reports/pixel-mse-player-only-sigreg02}"
new_output="${NEW:-reports/pixel-mse-player-only-sigreg02-100k}"
action_output="${ACT:-reports/pixel-mse-player-only-sigreg02-100k-action}"
total_steps="${STEPS:-100000}"
archive_every="${ARCHIVE_EVERY:-20000}"
learning_rate="${LR:-3e-4}"
episodes="${EPISODES:-32}"
max_steps="${MAX_STEPS:-128}"
seed="${SEED:-12345}"

# ---------------------------------------------------------------------------
# 1. Continue pretraining. Resume from the 50k checkpoint; write to a NEW dir so
#    the original 50k checkpoint is never overwritten. Archive every 20k steps.
# ---------------------------------------------------------------------------
echo "=== continuing pretrain ${old_output} -> ${new_output} (to ${total_steps}, archive every ${archive_every}, lr ${learning_rate}) ==="
uv run python -m pan1ni.pixel_flow_train \
  --resume "${old_output}/checkpoint.pt" \
  --output "${new_output}" \
  --objective mse \
  --steps "${total_steps}" \
  --archive-every "${archive_every}" \
  --learning-rate "${learning_rate}" \
  --player-batch-size 64 \
  --player-decode-batch-size 128 \
  --simulator-batch-size 0 \
  --context-length 8 \
  --goal-horizon 64 \
  --sigreg-weight 0.2 \
  --sigreg-slices 256 \
  --eval-every 500 \
  --checkpoint-every 2000 \
  --num-workers 12 \
  --prefetch-depth 3 \
  --windows-per-block 4 \
  --window-stride 16 \
  --device cuda

checkpoint="${new_output}/checkpoint.pt"
uv run python - "${new_output}" <<'PY'
import json, sys
from pathlib import Path
import torch
out = Path(sys.argv[1])
metrics = json.loads((out / "metrics.json").read_text())
ck = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
assert metrics["timeline"][-1]["step"] >= metrics["config"]["steps"], "continued run incomplete"
assert metrics["config"]["training_sources"] == ["player"], "not player-only"
assert ck["config"]["observation_mode"] == "pixels" and ck["config"]["prediction_objective"] == "mse"
print(f"continued checkpoint verified @ step {ck['step']}; archives:",
      sorted(p.name for p in out.glob('checkpoint-step*.pt')))
PY

# ---------------------------------------------------------------------------
# 2. Retrain movement heads on the continued model.
# ---------------------------------------------------------------------------
read -ra features <<< "${FEATURES:-idm idm_history}"
mkdir -p "${action_output}"
for feature in "${features[@]}"; do
  echo "=== training movement head: ${feature} ==="
  uv run python -m pan1ni.pixel_action_train \
    --checkpoint "${checkpoint}" \
    --output "${action_output}/${feature}" \
    --steps 5000 --feature "${feature}" \
    --context-length 8 --goal-horizon 64 --eval-every 200 --device cuda
done

# ---------------------------------------------------------------------------
# 3. Closed-loop eval in three goal-distance modes. Each mode gets a matched
#    uniform-random baseline. Sampled policy renders a representative video.
# ---------------------------------------------------------------------------
mode_names=(short far mixed)
mode_min=(2 8 3)
mode_max=(6 1000000 1000000)
policies=(argmax sample)

mkdir -p "${action_output}/uniform_random"
for i in "${!mode_names[@]}"; do
  mode="${mode_names[$i]}"; mn="${mode_min[$i]}"; mx="${mode_max[$i]}"
  echo "=== uniform-random baseline · ${mode} (dist ${mn}-${mx}) ==="
  uv run python -m pan1ni.nle_random_baseline \
    --output "${action_output}/uniform_random/nle-sample-${mode}-eval.json" \
    --episodes "${episodes}" --max-steps "${max_steps}" --seed "${seed}" \
    --min-goal-distance "${mn}" --max-goal-distance "${mx}" --goal-distance-mode "${mode}"
  cp "${action_output}/uniform_random/nle-sample-${mode}-eval.json" \
     "${action_output}/uniform_random/nle-argmax-${mode}-eval.json"
done

for feature in "${features[@]}"; do
  fdir="${action_output}/${feature}"
  ckhead="${fdir}/best_action_checkpoint.pt"
  for i in "${!mode_names[@]}"; do
    mode="${mode_names[$i]}"; mn="${mode_min[$i]}"; mx="${mode_max[$i]}"
    for policy in "${policies[@]}"; do
      echo "=== closed-loop · ${feature} · ${mode} · ${policy} ==="
      video=()
      if [ "${policy}" = "sample" ]; then
        video=(--video-output "${fdir}/nle-sample-${mode}.mp4")
      fi
      uv run python -m pan1ni.nle_closed_loop_eval \
        --world-checkpoint "${checkpoint}" \
        --action-checkpoint "${ckhead}" \
        --output "${fdir}/nle-${policy}-${mode}-eval.json" \
        --episodes "${episodes}" --max-steps "${max_steps}" --seed "${seed}" \
        --goal-horizon 64 --min-goal-distance "${mn}" --max-goal-distance "${mx}" \
        --goal-distance-mode "${mode}" \
        --action-selection "${policy}" --temperature 1.0 \
        "${video[@]}" --device cuda
    done
  done
done

# ---------------------------------------------------------------------------
# 4. One ablation summary per (policy, mode).
# ---------------------------------------------------------------------------
for i in "${!mode_names[@]}"; do
  mode="${mode_names[$i]}"
  for policy in "${policies[@]}"; do
    echo "=== summary · ${policy} · ${mode} ==="
    uv run python -m pan1ni.nle_action_ablation_report "${action_output}" \
      --output "${action_output}/nle-${policy}-${mode}-summary.png" \
      --evaluation-name "nle-${policy}-${mode}-eval.json"
  done
done

echo "continued-pretrain + 3-mode in-distribution evaluation complete"
