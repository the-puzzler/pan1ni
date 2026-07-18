# Goal-conditioned LeWorldModel for NetHack

This repository is a runnable research scaffold for testing whether **action-free,
goal-conditioned predictive pretraining** improves the label efficiency of NetHack
action prediction. It supports both NLE's structured terminal observations and
MiniHack's tiled RGB observations.

Given a short history and the final observation of that episode, the model learns

```text
goal_z = E(o[T-1])
E(o[t-K+1:t]), goal_z -> predicted E(o[t+1])
```

with end-to-end latent MSE plus SIGReg. SIGReg is implemented as the sliced
Epps–Pulley normality statistic used to push projected embeddings toward an isotropic
Gaussian. No contrastive loss, reconstruction decoder, EMA encoder, or online RL is
included.

## What is implemented

- Small in-repo ViT over embedded terminal-cell patches, with message, status
  (`blstats`), and cursor data supplied as a metadata token.
- A standard tiny patch ViT for MiniHack's 144×144 tiled RGB crop.
- CLS projection through `Linear -> BatchNorm1d -> GELU -> Linear`; all temporal
  views are projected jointly so BatchNorm sees the same batch × views layout as
  the LeJEPA demo.
- Transformer predictor with 8–32-step temporal contexts and an explicit `goal_z`
  token produced by embedding the episode's final frame.
- A single goal-conditioned prediction path: history plus that fixed final-frame goal.
- SIGReg and end-to-end pretraining step.
- Inverse-dynamics `p(a_t | z_t, z_hat[t+1])` and direct-policy heads.
- Frozen-backbone or full-finetuning action training.
- Horizon-stratified MSE and nested reproducible label subsets.
- Goal-directed synthetic navigation where the final position causally determines
  each next movement, for fast end-to-end validation without NLE/NLD.

## Run the smoke experiment

```bash
uv sync
uv run pan1ni smoke --steps 200 --batch-size 32 --sigreg-slices 256
uv run python -m unittest discover -s tests -v
```

The smoke command performs real forward/backward updates and prints the prediction,
SIGReg, and total pretraining losses. It is a plumbing check, not evidence for the
research hypothesis.

The synthetic smoke task deliberately uses a one-frame context. Its predictor must
choose the next grid movement from the current embedding and the final-frame
`goal_z`; no previous movement direction is present in the input.

## Connecting NLD

NLD is intentionally an optional data source because its installation and storage are
environment-specific. Convert an episode returned by your NLD loader with:

```python
from pan1ni.data import GoalWindowDataset, trajectory_from_nle

trajectory = trajectory_from_nle(
    {
        "tty_chars": episode["tty_chars"],
        "tty_colors": episode["tty_colors"],
        "tty_cursor": episode["tty_cursor"],
        "message": episode["message"],
        "blstats": episode["blstats"],
        # optional: "tty_bg_colors"
    },
    actions=episode.get("actions"),
)

windows = GoalWindowDataset(
    [trajectory],
    context_length=8,
)
```

`Trajectory` is deliberately tensor-only, so an NLD/TorchBeast/HDF5 reader can feed
it without coupling model code to a particular dataset release. Split episodes—not
windows—into train/validation/test sets to avoid leakage.

### Local NLD-AA taster

The repository can also stream the compact NLD-AA HDF5 mirror directly, without
loading full episodes into memory. The checked file contains 45 AutoAscend episodes
and 1,592,977 transitions:

```bash
mkdir -p data/downloads
curl -L -o data/downloads/nld-aa-taster.hdf5 \
  https://huggingface.co/datasets/Howuhh/nld-aa-taster/resolve/main/data/data-cav-gno-neu-any.hdf5
sha256sum data/downloads/nld-aa-taster.hdf5
# dd404b6214a6c282a9f4e81b1acbfd353fdade505f09d1554093f6edce4b364b

uv run pan1ni nld-smoke \
  --steps 100 \
  --batch-size 4 \
  --sigreg-slices 256 \
  --goal-horizon 64
```

`NLDHDF5GoalDataset` performs an episode-level split and samples `t`, `t+1`, and
`t+goal_horizon` lazily. The last of those frames is the final frame of the sampled
sequence and becomes `goal_z`. This mirror contains terminal characters, colours,
cursor, actions, rewards, and done flags. Until raw ttyrec loading is enabled, the
message modality is copied from terminal row zero and the unavailable `blstats`
modality is zero-filled.

## Tiled MiniHack KeyRoom experiment

For the visually inspectable causal test, the repository defines 16 explicit
MiniHack layouts with varying key and goal positions. Every successful trajectory
requires the chain `find key -> unlock central door -> approach goal staircase`.
The post-termination NetHack buffer is excluded because it is not a valid gameplay
image.

```bash
uv run python -m pan1ni.minihack_data --episodes 1600
uv run python -m pan1ni.minihack_report \
  --steps 1000 \
  --batch-size 32 \
  --validation-samples 128
```

The report includes correct/shuffled/zero-goal controls, the copy-current baseline,
latent variance and effective rank, stage-stratified errors, tiled screenshots, and
an H.264 trajectory video. The pixel encoder retains the same
`CLS -> BatchNorm MLP -> z` projector required by SIGReg.

## Current experiment scope

The code currently trains only the goal-conditioned world model. Every pretraining
sample contains a history, the episode's final frame as its goal, and the immediate
next observation as its prediction target. The goal distance is therefore determined
by the sampled current timestep rather than sampled independently.

For each checkpoint, train both action heads on nested label fractions
`0.001, 0.01, 0.1, 1.0`, first with the backbone frozen and then with full fine-tuning.
Report mean and uncertainty over seeds, top-1 action accuracy/cross-entropy, and
metrics grouped by goal offset. Contrastive objectives and baseline/control variants
are intentionally deferred.

## Package map

- `pan1ni/data.py`: NLE adapter, episode filtering, and window sampling
- `pan1ni/nld_data.py`: lazy HDF5 NLD sequence sampler
- `pan1ni/minihack_data.py`: tiled KeyRoom environments, oracle collector, and HDF5 sampler
- `pan1ni/model.py`: terminal/pixel ViT encoders and goal-conditioned predictor
- `pan1ni/minihack_report.py`: tiled training diagnostics and HTML/video report
- `pan1ni/losses.py`: SIGReg and pretraining objective
- `pan1ni/action.py`: downstream action heads and freezing utility
- `pan1ni/train.py`: training and evaluation primitives
- `pan1ni/synthetic.py`: dependency-free synthetic episodes
