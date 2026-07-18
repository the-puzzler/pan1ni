from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .action import DirectPolicyHead, InverseDynamicsHead
from .losses import pretraining_loss
from .model import GoalConditionedLeWorldModel

Batch = Mapping[str, Tensor | Mapping[str, Tensor]]


def _observation(batch: Batch, key: str) -> Mapping[str, Tensor]:
    value = batch[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"batch[{key!r}] must be an observation mapping")
    return value


def move_batch(batch: Batch, device: torch.device | str) -> dict:
    return {
        key: ({name: tensor.to(device) for name, tensor in value.items()} if isinstance(value, Mapping) else value.to(device))
        for key, value in batch.items()
    }


def pretrain_step(
    model: GoalConditionedLeWorldModel,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    *,
    sigreg_weight: float = 0.1,
    sigreg_slices: int = 256,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # As in the LeJEPA multi-view encoder, all views pass through the CLS
    # projector together. BatchNorm therefore operates over batch × views.
    grouped_latents = model.encode_group(
        _observation(batch, "history"),
        _observation(batch, "goal"),
        _observation(batch, "target"),
    )
    history_latents = grouped_latents[:, :-2]
    goal_z = grouped_latents[:, -2]
    target_latent = grouped_latents[:, -1]
    output = model.predict_latents(history_latents, goal_z)
    losses = pretraining_loss(
        output.next_latent,
        target_latent,
        grouped_latents.transpose(0, 1),
        sigreg_weight=sigreg_weight,
        sigreg_slices=sigreg_slices,
    )
    losses["loss"].backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return {name: value.detach().item() for name, value in losses.items()}


def action_step(
    model: GoalConditionedLeWorldModel,
    head: InverseDynamicsHead | DirectPolicyHead,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    *,
    direct: bool = False,
    train_backbone: bool = False,
) -> dict[str, float]:
    action = batch.get("action")
    if not isinstance(action, Tensor):
        raise ValueError("action-labelled batch required")
    model.train(train_backbone)
    head.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(train_backbone):
        output = model(_observation(batch, "history"), _observation(batch, "goal"))
    if direct:
        if not isinstance(head, DirectPolicyHead):
            raise TypeError("direct=True requires DirectPolicyHead")
        logits = head(output.hidden)
    else:
        if not isinstance(head, InverseDynamicsHead):
            raise TypeError("direct=False requires InverseDynamicsHead")
        logits = head(output.history_latents[:, -1], output.next_latent)
    loss = F.cross_entropy(logits, action.long())
    loss.backward()
    optimizer.step()
    accuracy = logits.argmax(-1).eq(action).float().mean()
    return {"loss": loss.detach().item(), "accuracy": accuracy.item()}


def stratified_horizon_mse(
    prediction_errors: Tensor, offsets: Tensor, boundaries: tuple[int, ...] = (4, 16, 64, 256)
) -> dict[str, float]:
    if prediction_errors.shape[0] != offsets.shape[0]:
        raise ValueError("prediction_errors and offsets must share the batch dimension")
    result: dict[str, float] = {}
    lower = 0
    per_sample = prediction_errors.flatten(1).mean(1)
    for upper in boundaries:
        mask = (offsets > lower) & (offsets <= upper)
        if mask.any():
            result[f"{lower + 1}-{upper}"] = per_sample[mask].mean().item()
        lower = upper
    mask = offsets > lower
    if mask.any():
        result[f">{lower}"] = per_sample[mask].mean().item()
    return result


def label_subset_indices(size: int, fraction: float, seed: int = 0) -> Tensor:
    """Nested, reproducible labelled subsets for 0.1/1/10/100% studies."""

    if size < 1 or not 0 < fraction <= 1:
        raise ValueError("size must be positive and fraction must be in (0, 1]")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(size, generator=generator)[: max(1, round(size * fraction))]
