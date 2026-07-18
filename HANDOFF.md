# Agent handoff: real NLD training

## Goal

Train the action-free, goal-conditioned world model on real NetHack Learning
Dataset trajectories. MiniHack is a controlled causal sanity check, not the main
training corpus.

## Current implementation

- Official NLD-AA ttyrec streaming is implemented in
  `pan1ni/nld_data.py`.
- The raw-data training CLI is `pan1ni/nld_ttyrec_train.py`.
- Games are split 80/20 at the episode level for training and validation.
- Windows crossing episode boundaries are rejected.
- Ttyrecs are decoded concurrently and fed through bounded background
  prefetching.
- Host-to-GPU transfers use pinned memory where applicable and non-blocking
  CUDA copies.
- Long decoded blocks are reused for four goal windows at stride 16.
- Model and optimizer checkpoints plus metrics are written periodically.
- Validation batch size is independent of training batch size.
- All 10 unit tests pass.

## Recommended training command

```bash
uv run python -m pan1ni.nld_ttyrec_train \
  --db data/nld/nld-aa-taster.db \
  --dataset nld-aa-taster \
  --output reports/nld-aa-taster-medium \
  --steps 10000 \
  --batch-size 512 \
  --context-length 1 \
  --goal-horizon 64 \
  --sigreg-slices 256 \
  --eval-every 500 \
  --checkpoint-every 500 \
  --validation-samples 512 \
  --num-workers 16 \
  --prefetch-depth 3 \
  --windows-per-block 4 \
  --window-stride 16
```

Before starting the medium run:

```bash
uv run python -m unittest discover -s tests -v

uv run python -m pan1ni.nld_ttyrec_train \
  --output /tmp/nld-ttyrec-smoke \
  --steps 10 \
  --batch-size 32 \
  --goal-horizon 8 \
  --sigreg-slices 8 \
  --eval-every 10 \
  --checkpoint-every 10 \
  --validation-samples 32 \
  --num-workers 4 \
  --prefetch-depth 2 \
  --windows-per-block 4 \
  --window-stride 4
```

## Performance evidence

Benchmarks were collected on an RTX 5090 with 16 CPU cores.

- Original decoder: 4,239 samples/second.
- Block-reuse decoder: 7,777 samples/second.
- Batch 256: 5,427 samples/second, mostly 78–90% GPU, 2.7 GiB VRAM.
- Batch 512: 5,685 samples/second, 79–95% GPU, about 4.8 GiB VRAM.
- Batch 1024: 5,896 samples/second, mostly 84–99% GPU, about 9 GiB VRAM.

Batch 512 is the recommended balance. Batch 1024 gave only about 3.7% more
throughput and would introduce an unnecessary optimization/generalization
variable. Decoder capacity now exceeds end-to-end consumption, so the GPU is
the primary bottleneck.

## Local NLD data

The previous workspace contained:

- Archive: `data/downloads/nld-aa-taster.zip`
- Extracted data: `data/nld/nld-aa-taster/nle_data`
- SQLite database: `data/nld/nld-aa-taster.db`
- Registered dataset: `nld-aa-taster`
- 1,934 AutoAscend games
- 2,042 ttyrec files

The data directories are ignored by Git. They will not be included in a normal
push. Preserve the workspace volume or reconstruct them on the new machine.

### Reconstruct the taster

Download and extract:

```bash
mkdir -p data/downloads data/nld

curl -L --fail --continue-at - \
  --output data/downloads/nld-aa-taster.zip \
  https://dl.fbaipublicfiles.com/nld/nld-aa-taster/nld-aa-taster.zip

unzip data/downloads/nld-aa-taster.zip -d data/nld
```

Register it:

```bash
uv run python -c "
import nle.dataset as nld

db = 'data/nld/nld-aa-taster.db'
nld.db.create(db)
nld.add_nledata_directory(
    'data/nld/nld-aa-taster/nle_data',
    'nld-aa-taster',
    db,
)
"
```

The official taster archive is approximately 1.71 GB. Check free disk before
downloading full shards; one NLD-AA shard was measured at approximately 6.16 GB
compressed.

## Known limitations

- Official ttyrecs lack structured `blstats`, so the status input is
  zero-filled.
- Raw keypresses are not mapped to canonical NLE action IDs. This does not
  affect action-free pretraining, but it must be addressed before downstream
  action-head training.
- NLD-AA contains AutoAscend bot play. Human gameplay requires NLD-NAO.
- Four reused windows from a block overlap and are temporally correlated.
  Batches still span hundreds of games. Later experiments should compare fewer
  windows or a larger stride to quantify the effect.
- Do not interpret MiniHack results as evidence on real NetHack.

## Completed MiniHack control

The medium MiniHack control trained on 1.28 million sampled windows:

- Runtime: 664 seconds
- Throughput: 1,928 samples/second
- Held-out prediction MSE: 0.0938
- Shuffled-goal MSE: 0.2795, approximately 2.98 times worse
- Effective latent rank: 20.53/64

This strongly confirms that the model uses its goal input in the controlled
KeyRoom environment.

## Suggested new instance

For medium NLD-AA work:

- One RTX 5090, RTX 4090, or equivalent 24–32 GB GPU
- 16–24 modern vCPUs
- 64 GB RAM
- At least 500 GB local NVMe

For substantial NLD-NAO human data as well, prefer 1 TB local NVMe.

## First tasks after migration

1. Inspect the pushed diff and repository status.
2. Confirm the NLD archive, extracted ttyrecs, and database exist.
3. Run the complete unit suite.
4. Run the ten-step raw ttyrec smoke test.
5. Start the batch-512 medium run.
6. Monitor GPU, CPU, RAM, throughput, and checkpoint output.
7. Check disk capacity before downloading any full NLD shards.

## Planned experiment after the RGB MSE baseline

Do not replace or interrupt the current MSE experiment for this. After it has
finished, train a goal-conditioned flow-matching version on the same encoded
next-state targets:

- draw a source/noise latent and interpolate it with the true encoded next
  state at a sampled flow time;
- train the predictor as a conditional flow/denoising field using history and
  the randomly sampled future goal embedding;
- at inference, take exactly **one denoising/flow step** rather than integrating
  a full trajectory or producing many next-state samples;
- use the **predicted residual flow from that one step as the action-decoder
  features**;
- compare this action probe directly against the MSE predictor-feature baseline.

This one-step residual-flow feature strategy is intentional and user-validated
from prior experiments. Do not silently reinterpret it as full ODE sampling or
multi-step diffusion inference.
