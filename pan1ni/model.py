from __future__ import annotations

from typing import Mapping, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


class Prediction(NamedTuple):
    next_latent: Tensor
    hidden: Tensor
    history_latents: Tensor
    goal_z: Tensor


class BatchNormProjectionMLP(nn.Module):
    """CLS projector following the Linear -> BatchNorm -> GELU -> Linear recipe."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, cls_embedding: Tensor) -> Tensor:
        return self.net(cls_embedding)


class TinyTerminalViT(nn.Module):
    """A small ViT over patches of embedded NetHack terminal cells.

    Message, status, and cursor features form one additional metadata token. The
    returned value is always the final CLS token, before the SIGReg projector.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        quarter = config.cell_dim // 4
        self.patch_size = config.terminal_patch_size
        self.max_patches = config.max_patches
        self.char_embedding = nn.Embedding(256, quarter * 2)
        self.fg_embedding = nn.Embedding(32, quarter)
        self.bg_embedding = nn.Embedding(32, quarter)
        self.patch_embedding = nn.Conv2d(
            config.cell_dim,
            config.vit_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.message_embedding = nn.Embedding(256, config.message_dim, padding_idx=0)
        self.metadata_projection = nn.Sequential(
            nn.Linear(config.message_dim + config.status_dim + 2, config.vit_dim),
            nn.LayerNorm(config.vit_dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.vit_dim) * 0.02)
        # CLS + metadata + terminal patch tokens. Slicing avoids image-size coupling.
        self.position = nn.Parameter(torch.randn(1, config.max_patches + 2, config.vit_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            config.vit_dim,
            config.vit_heads,
            config.vit_dim * 4,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            config.vit_layers,
            nn.LayerNorm(config.vit_dim),
        )

    def _metadata_token(self, observation: Mapping[str, Tensor], chars: Tensor) -> Tensor:
        message_tokens = self.message_embedding(observation["message"].long().clamp(0, 255))
        mask = observation["message"].ne(0).unsqueeze(-1)
        message = (message_tokens * mask).sum(-2) / mask.sum(-2).clamp_min(1)
        cursor = observation["cursor"].float()
        height, width = chars.shape[-2:]
        cursor_scale = cursor.new_tensor((max(height - 1, 1), max(width - 1, 1)))
        cursor = cursor / cursor_scale
        metadata = torch.cat((message, observation["status"].float(), cursor), dim=-1)
        return self.metadata_projection(metadata)

    def forward(self, observation: Mapping[str, Tensor]) -> Tensor:
        chars = observation["chars"].long().clamp(0, 255)
        colors = observation["colors"].long().clamp(0, 31)
        backgrounds = observation.get("bg_colors", torch.zeros_like(colors)).long().clamp(0, 31)
        cells = torch.cat(
            (self.char_embedding(chars), self.fg_embedding(colors), self.bg_embedding(backgrounds)),
            dim=-1,
        ).movedim(-1, -3)
        pad_height = (-cells.shape[-2]) % self.patch_size
        pad_width = (-cells.shape[-1]) % self.patch_size
        cells = F.pad(cells, (0, pad_width, 0, pad_height))
        patches = self.patch_embedding(cells).flatten(2).transpose(1, 2)
        if patches.shape[1] > self.max_patches:
            raise ValueError(
                f"terminal produces {patches.shape[1]} patches, exceeding max_patches={self.max_patches}"
            )
        batch = chars.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        metadata = self._metadata_token(observation, chars).unsqueeze(1)
        tokens = torch.cat((cls, metadata, patches), dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        return self.transformer(tokens)[:, 0]


class TinyPixelViT(nn.Module):
    """Small patch ViT for MiniHack's 144x144 RGB tile crop."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.max_patches = config.max_patches
        self.patch_embedding = nn.Conv2d(
            3,
            config.vit_dim,
            kernel_size=config.pixel_patch_size,
            stride=config.pixel_patch_size,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.vit_dim) * 0.02)
        self.position = nn.Parameter(torch.randn(1, config.max_patches + 1, config.vit_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            config.vit_dim,
            config.vit_heads,
            config.vit_dim * 4,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            config.vit_layers,
            nn.LayerNorm(config.vit_dim),
        )

    def forward(self, observation: Mapping[str, Tensor]) -> Tensor:
        pixels = observation["pixels"].float().div(255.0)
        patches = self.patch_embedding(pixels).flatten(2).transpose(1, 2)
        if patches.shape[1] > self.max_patches:
            raise ValueError(
                f"image produces {patches.shape[1]} patches, exceeding max_patches={self.max_patches}"
            )
        cls = self.cls_token.expand(pixels.shape[0], -1, -1)
        tokens = torch.cat((cls, patches), dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        return self.transformer(tokens)[:, 0]


class StructuredObservationEncoder(nn.Module):
    """Tiny ViT followed by the BatchNorm projection MLP used by SIGReg."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.backbone = TinyPixelViT(config) if config.observation_mode == "pixels" else TinyTerminalViT(config)
        self.proj = BatchNormProjectionMLP(
            config.vit_dim,
            config.projector_hidden_dim,
            config.latent_dim,
        )

    def forward_features(self, observation: Mapping[str, Tensor]) -> Tensor:
        return self.backbone(observation)

    def forward(self, observation: Mapping[str, Tensor]) -> Tensor:
        return self.proj(self.forward_features(observation))


class GoalPredictor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.randn(config.max_context + 2, config.latent_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            config.latent_dim,
            config.predictor_heads,
            config.hidden_dim * 4,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config.predictor_layers, nn.LayerNorm(config.latent_dim))
        self.query = nn.Parameter(torch.randn(1, 1, config.latent_dim) * 0.02)
        self.head = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

    def forward(self, history: Tensor, goal_z: Tensor) -> tuple[Tensor, Tensor]:
        batch, steps, _ = history.shape
        query = self.query.expand(batch, -1, -1)
        tokens = torch.cat((history, goal_z[:, None], query), dim=1)
        tokens = tokens + self.position[: steps + 2]
        hidden = self.transformer(tokens)[:, -1]
        return self.head(hidden), hidden


class GoalConditionedLeWorldModel(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = StructuredObservationEncoder(config)
        self.predictor = GoalPredictor(config)

    def encode_sequence(self, observation: Mapping[str, Tensor]) -> Tensor:
        first = next(iter(observation.values()))
        batch, steps = first.shape[:2]
        flattened = {key: value.flatten(0, 1) for key, value in observation.items()}
        return self.encoder(flattened).unflatten(0, (batch, steps))

    def encode_group(
        self,
        history: Mapping[str, Tensor],
        *single_observations: Mapping[str, Tensor],
    ) -> Tensor:
        """Project all temporal views together so BatchNorm sees batch × views."""

        combined = {
            key: torch.cat((value, *(observation[key].unsqueeze(1) for observation in single_observations)), dim=1)
            for key, value in history.items()
        }
        return self.encode_sequence(combined)

    def predict_latents(
        self,
        history_latents: Tensor,
        goal_z: Tensor,
    ) -> Prediction:
        prediction, hidden = self.predictor(history_latents, goal_z)
        return Prediction(prediction, hidden, history_latents, goal_z)

    def forward(
        self,
        history: Mapping[str, Tensor],
        goal: Mapping[str, Tensor],
    ) -> Prediction:
        grouped_latents = self.encode_group(history, goal)
        return self.predict_latents(grouped_latents[:, :-1], grouped_latents[:, -1])
