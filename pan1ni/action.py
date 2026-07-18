from __future__ import annotations

import torch
from torch import Tensor, nn


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
