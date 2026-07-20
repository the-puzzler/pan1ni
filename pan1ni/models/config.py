from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    observation_mode: str = "terminal"
    latent_dim: int = 256
    cell_dim: int = 64
    message_dim: int = 32
    status_dim: int = 27
    hidden_dim: int = 256
    vit_dim: int = 128
    vit_layers: int = 2
    vit_heads: int = 4
    terminal_patch_size: int = 2
    pixel_patch_size: int = 16
    max_patches: int = 2048
    projector_hidden_dim: int = 512
    predictor_layers: int = 3
    predictor_heads: int = 8
    max_context: int = 32
    num_actions: int = 121
    dropout: float = 0.1
    prediction_objective: str = "mse"

    def __post_init__(self) -> None:
        if self.observation_mode not in {"terminal", "terminal_rgb", "pixels"}:
            raise ValueError("observation_mode must be 'terminal', 'terminal_rgb', or 'pixels'")
        if self.latent_dim % self.predictor_heads:
            raise ValueError("latent_dim must be divisible by predictor_heads")
        if self.vit_dim % self.vit_heads:
            raise ValueError("vit_dim must be divisible by vit_heads")
        if self.cell_dim % 4:
            raise ValueError("cell_dim must be divisible by 4")
        if self.terminal_patch_size < 1 or self.pixel_patch_size < 1 or self.max_patches < 1:
            raise ValueError("patch sizes and max_patches must be positive")
        if self.max_context < 1:
            raise ValueError("max_context must be positive")
        if self.prediction_objective not in {"mse", "flow"}:
            raise ValueError("prediction_objective must be 'mse' or 'flow'")


@dataclass(frozen=True)
class SamplingConfig:
    context_length: int = 8
    samples_per_epoch: int = 100_000
    seed: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.context_length:
            raise ValueError("context_length must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    sigreg_weight: float = 0.1
    sigreg_slices: int = 256
    grad_clip_norm: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
