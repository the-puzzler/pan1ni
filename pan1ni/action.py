from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .model import GoalConditionedLeWorldModel


ACTION_FEATURES = (
    "current_latent",
    "predictor_layer1",
    "predictor_layer2",
    "predictor_hidden",
    "predicted_next",
    "flow_residual",
    "idm",
)

# Features in which the goal genuinely conditions the representation. current_latent
# is the pre-predictor encoder latent of the current frame and is goal-blind, so it
# is excluded from the movement-policy sweep even though predictor_features can still
# produce it. flow_residual needs a flow checkpoint and is likewise excluded here.
GOAL_AWARE_FEATURES = (
    "predictor_layer1",
    "predictor_layer2",
    "predictor_hidden",
    "predicted_next",
    "idm",
)


def feature_dim(feature: str, latent_dim: int) -> int:
    """Input width a policy head needs for a given predictor feature."""

    if feature == "idm":
        return latent_dim * 3
    if feature in ACTION_FEATURES:
        return latent_dim
    raise ValueError(f"feature must be one of {ACTION_FEATURES}")


@torch.no_grad()
def predictor_features(
    model: "GoalConditionedLeWorldModel",
    batch: dict,
    feature: str = "predictor_hidden",
    generator: torch.Generator | None = None,
) -> Tensor:
    model.eval()
    if feature in {
        "current_latent", "predictor_layer1", "predictor_layer2",
        "predictor_hidden", "predicted_next", "idm",
    }:
        grouped = model.encode_group(batch["history"], batch["goal"])
        history_z, goal_z = grouped[:, :-1], grouped[:, -1]
        current = history_z[:, -1]
        if feature == "current_latent":
            return current
        steps = history_z.size(1)
        value = model.predictor.dropout(
            history_z + model.predictor.position[:, :steps]
        )
        conditioning = goal_z[:, None].expand(-1, steps, -1)
        requested_layer = {
            "predictor_layer1": 1,
            "predictor_layer2": 2,
        }.get(feature)
        for layer_index, layer in enumerate(model.predictor.layers, start=1):
            value = layer(value, conditioning)
            if requested_layer == layer_index:
                return model.predictor.norm(value)[:, -1]
        hidden = model.predictor.norm(value)
        if feature == "predictor_hidden":
            return hidden[:, -1]
        prediction = model.pred_proj(hidden.flatten(0, 1)).unflatten(
            0, hidden.shape[:2]
        )
        predicted_next = prediction[:, -1]
        if feature == "predicted_next":
            return predicted_next
        # Inverse-dynamics feature: current state, the goal-conditioned predicted
        # next state, and their difference. Mirrors InverseDynamicsHead's inputs.
        return torch.cat((current, predicted_next, predicted_next - current), dim=-1)
    if feature == "flow_residual":
        grouped = model.encode_group(batch["history"], batch["goal"])
        history_z, goal_z = grouped[:, :-1], grouped[:, -1]
        source_noise = torch.randn(
            history_z.shape,
            device=history_z.device,
            dtype=history_z.dtype,
            generator=generator,
        )
        _, residual = model.one_step_flow(history_z, goal_z, source_noise)
        return residual[:, -1]
    raise ValueError(f"feature must be one of {ACTION_FEATURES}")


class InverseDynamicsHead(nn.Module):
    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, current: Tensor, desired_next: Tensor) -> Tensor:
        return self.network(torch.cat((current, desired_next, desired_next - current), dim=-1))


class DirectPolicyHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.GELU()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.GELU()))
        layers.append(nn.Linear(hidden_dim, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, predictor_hidden: Tensor) -> Tensor:
        return self.network(predictor_hidden)


def set_backbone_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)
