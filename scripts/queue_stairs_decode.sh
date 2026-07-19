#!/usr/bin/env bash
# Stairs-decoding experiment: train 10-class heads (8 moves + descend + ascend,
# recovered from Dlvl changes) on frozen features of a tile-pixel world model, and
# report per-class descend/ascend recall. Tests whether the IDM / history features
# let the head recognise a descend from the (occluded-current-tile) observation.
set -euo pipefail

checkpoint="${1:?usage: queue_stairs_decode.sh WORLD_CHECKPOINT [ACTION_OUTPUT]}"
action_output="${2:-reports/stairs-decode}"
steps="${STEPS:-6000}"
val_samples="${VAL_SAMPLES:-4000}"

features=(predictor_layer1 predictor_layer2 predictor_hidden predicted_next idm idm_history)
mkdir -p "${action_output}"
for feature in "${features[@]}"; do
  echo "=== training 10-class stairs head: ${feature} ==="
  uv run python -m pan1ni.pixel_action_train \
    --checkpoint "${checkpoint}" \
    --output "${action_output}/${feature}" \
    --action-set movement_stairs --feature "${feature}" \
    --steps "${steps}" --context-length 8 --goal-horizon 64 --eval-every 500 \
    --validation-samples "${val_samples}" --device cuda
done

echo "=== per-class recall summary ==="
uv run python - "${action_output}" "${features[@]}" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); feats=sys.argv[2:]
names=["N","E","S","W","NE","SE","SW","NW","descend","ascend"]
print(f"{'feature':<16}{'bal.acc':>8}{'move.acc':>9}   per-class recall (support)")
rows=[]
for f in feats:
    p=root/f/'metrics.json'
    if not p.exists(): continue
    m=json.loads(p.read_text())
    best=max(m['timeline'][1:], key=lambda e:e.get('selection_balanced_accuracy',-1))
    pv=best['player_validation']; cc=pv['class_count']; ck=pv['class_correct']
    bal=pv['balanced_accuracy']*100; acc=pv['accuracy']*100
    def rec(i): return f"{(ck[i]/cc[i]*100 if cc[i] else 0):.0f}%({cc[i]})"
    print(f"{f:<16}{bal:>7.1f}%{acc:>8.1f}%   descend {rec(8):>9}  ascend {rec(9):>9}")
    rows.append((f, bal, ck[8], cc[8], ck[9], cc[9]))
tot_d=sum(r[3] for r in rows)//max(len(rows),1)
print(f"\n(descend support ~{tot_d} val frames/head; rare class — read recall with support in mind)")
PY
echo "stairs-decoding experiment complete"
