from __future__ import annotations

from typing import Mapping, NamedTuple

import torch
from PIL import Image, ImageDraw, ImageFont
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


class TerminalRGBPixelViT(TinyPixelViT):
    """Rasterize compact tty cells on GPU, then use the standard pixel ViT."""

    glyph_height = 16
    glyph_width = 8

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            14,
        )
        atlas = torch.zeros(256, self.glyph_height, self.glyph_width)
        for code in range(256):
            image = Image.new("L", (self.glyph_width, self.glyph_height), 0)
            glyph = bytes((code,)).decode("cp437", errors="replace") if code else " "
            ImageDraw.Draw(image).text((0, -1), glyph, font=font, fill=255)
            atlas[code] = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(
                self.glyph_height,
                self.glyph_width,
            ).div_(255)
        palette = torch.tensor(
            (
                (17, 19, 24), (185, 74, 72), (92, 171, 99), (181, 139, 69),
                (101, 119, 200), (169, 96, 184), (85, 167, 172), (200, 204, 212),
                (98, 103, 115), (255, 107, 104), (127, 224, 137), (245, 215, 110),
                (130, 152, 255), (223, 131, 239), (118, 229, 235), (255, 255, 255),
            ),
            dtype=torch.float32,
        ).div_(255)
        self.register_buffer("glyph_atlas", atlas, persistent=False)
        self.register_buffer("palette", palette, persistent=False)

    def render(self, observation: Mapping[str, Tensor]) -> Tensor:
        chars = observation["chars"].long().clamp(0, 255)
        foreground = observation["colors"].long().bitwise_and(15)
        background = observation.get("bg_colors", torch.zeros_like(chars)).long().bitwise_and(15)
        mask = self.glyph_atlas[chars].unsqueeze(-1)
        foreground_rgb = self.palette[foreground][..., None, None, :]
        background_rgb = self.palette[background][..., None, None, :]
        pixels = background_rgb + mask * (foreground_rgb - background_rgb)
        batch, rows, columns = chars.shape
        return pixels.permute(0, 1, 3, 2, 4, 5).reshape(
            batch,
            rows * self.glyph_height,
            columns * self.glyph_width,
            3,
        ).permute(0, 3, 1, 2).contiguous()

    def forward(self, observation: Mapping[str, Tensor]) -> Tensor:
        pixels = self.render(observation)
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
        if config.observation_mode == "pixels":
            self.backbone = TinyPixelViT(config)
        elif config.observation_mode == "terminal_rgb":
            self.backbone = TerminalRGBPixelViT(config)
        else:
            self.backbone = TinyTerminalViT(config)
        self.proj = BatchNormProjectionMLP(
            config.vit_dim,
            config.projector_hidden_dim,
            config.latent_dim,
        )

    def forward_features(self, observation: Mapping[str, Tensor]) -> Tensor:
        return self.backbone(observation)

    def forward(self, observation: Mapping[str, Tensor]) -> Tensor:
        return self.proj(self.forward_features(observation))


def _modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return value * (1 + scale) + shift


class PredictorFeedForward(nn.Module):
    """LeWorldModel predictor feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class CausalAttention(nn.Module):
    """LeWorldModel scaled-dot-product attention with its causal mask."""

    def __init__(self, dim: int, heads: int, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, value: Tensor) -> Tensor:
        value = self.norm(value)
        query, key, values = self.to_qkv(value).chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            batch, steps, _ = tensor.shape
            return tensor.view(batch, steps, self.heads, -1).transpose(1, 2)

        query, key, values = map(split_heads, (query, key, values))
        dropout = self.dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            query,
            key,
            values,
            dropout_p=dropout,
            is_causal=True,
        )
        output = output.transpose(1, 2).flatten(2)
        return self.to_out(output)


class GoalConditionalBlock(nn.Module):
    """LeWorldModel ConditionalBlock with goal embeddings replacing actions."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = CausalAttention(dim, heads, dim_head=64, dropout=dropout)
        self.mlp = PredictorFeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, value: Tensor, conditioning: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(conditioning).chunk(6, dim=-1)
        )
        value = value + gate_attn * self.attention(
            _modulate(self.norm1(value), shift_attn, scale_attn)
        )
        value = value + gate_mlp * self.mlp(
            _modulate(self.norm2(value), shift_mlp, scale_mlp)
        )
        return value


class GoalPredictor(nn.Module):
    """LeWorldModel ARPredictor with goal rather than action conditioning."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.position = nn.Parameter(
            torch.randn(1, config.max_context, config.latent_dim)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            GoalConditionalBlock(
                config.latent_dim,
                config.predictor_heads,
                config.hidden_dim * 4,
                config.dropout,
            )
            for _ in range(config.predictor_layers)
        )
        self.norm = nn.LayerNorm(config.latent_dim)

    def forward(
        self,
        history: Tensor,
        goal_z: Tensor,
        extra_conditioning: Tensor | None = None,
    ) -> Tensor:
        steps = history.size(1)
        value = self.dropout(history + self.position[:, :steps])
        conditioning = goal_z[:, None].expand(-1, steps, -1)
        if extra_conditioning is not None:
            conditioning = conditioning + extra_conditioning
        for layer in self.layers:
            value = layer(value, conditioning)
        return self.norm(value)


class GoalConditionedLeWorldModel(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = StructuredObservationEncoder(config)
        self.predictor = GoalPredictor(config)
        self.pred_proj = BatchNormProjectionMLP(
            config.latent_dim,
            config.projector_hidden_dim,
            config.latent_dim,
        )
        if config.prediction_objective == "flow":
            self.flow_state_projection = nn.Linear(config.latent_dim, config.latent_dim)
            self.flow_time_projection = nn.Sequential(
                nn.Linear(1, config.latent_dim),
                nn.SiLU(),
                nn.Linear(config.latent_dim, config.latent_dim),
            )

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
        """Encode temporal states jointly and a goal in a separate call.

        History and target form the real temporal sequence and are flattened
        to B*T before projection. The non-temporal goal is encoded separately.
        """

        if len(single_observations) not in (1, 2):
            raise ValueError("encode_group expects history plus goal, optionally followed by target")
        goal = single_observations[0]
        if len(single_observations) == 2:
            target = single_observations[1]
            temporal = {
                key: torch.cat((value, target[key].unsqueeze(1)), dim=1)
                for key, value in history.items()
            }
            temporal_z = self.encode_sequence(temporal)
            history_z, target_z = temporal_z[:, :-1], temporal_z[:, -1:]
        else:
            history_z = self.encode_sequence(history)
            target_z = None
        goal_z = self.encoder(goal).unsqueeze(1)
        pieces = (history_z, goal_z) if target_z is None else (history_z, goal_z, target_z)
        return torch.cat(pieces, dim=1)

    def predict_latents(
        self,
        history_latents: Tensor,
        goal_z: Tensor,
    ) -> Prediction:
        predictions, hidden_sequence = self.predict_sequence_latents(history_latents, goal_z)
        return Prediction(predictions[:, -1], hidden_sequence[:, -1], history_latents, goal_z)

    def predict_sequence_latents(
        self,
        history_latents: Tensor,
        goal_z: Tensor,
    ) -> tuple[Tensor, Tensor]:
        predictor_output = self.predictor(history_latents, goal_z)
        batch, steps, _ = predictor_output.shape
        predictions = self.pred_proj(predictor_output.flatten(0, 1)).unflatten(0, (batch, steps))
        return predictions, predictor_output

    def predict_flow_sequence(
        self,
        history_latents: Tensor,
        goal_z: Tensor,
        noisy_next_latents: Tensor,
        flow_times: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.config.prediction_objective != "flow":
            raise RuntimeError("flow prediction requires a flow-configured model")
        extra_conditioning = (
            self.flow_state_projection(noisy_next_latents)
            + self.flow_time_projection(flow_times)
        )
        predictor_output = self.predictor(
            history_latents,
            goal_z,
            extra_conditioning=extra_conditioning,
        )
        batch, steps, _ = predictor_output.shape
        residual_flow = self.pred_proj(predictor_output.flatten(0, 1)).unflatten(
            0,
            (batch, steps),
        )
        return residual_flow, predictor_output

    def one_step_flow(
        self,
        history_latents: Tensor,
        goal_z: Tensor,
        source_noise: Tensor,
    ) -> tuple[Tensor, Tensor]:
        flow_times = source_noise.new_zeros(*source_noise.shape[:-1], 1)
        residual_flow, _ = self.predict_flow_sequence(
            history_latents,
            goal_z,
            source_noise,
            flow_times,
        )
        return source_noise + residual_flow, residual_flow

    def forward(
        self,
        history: Mapping[str, Tensor],
        goal: Mapping[str, Tensor],
    ) -> Prediction:
        grouped_latents = self.encode_group(history, goal)
        return self.predict_latents(grouped_latents[:, :-1], grouped_latents[:, -1])
