from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SIGReg(nn.Module):
    """LeJEPA's sketched isotropic Gaussian regularizer.

    The preferred input shape is ``(views, batch, dimensions)``. A two-dimensional
    ``(batch, dimensions)`` input is also supported. Gaussianity is measured across
    the batch independently for every view, matching the LeJEPA demo.
    """

    def __init__(self, knots: int = 17, num_projections: int = 256) -> None:
        super().__init__()
        if knots < 2 or num_projections < 1:
            raise ValueError("knots must be >= 2 and num_projections must be positive")
        self.num_projections = num_projections
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, projected_embeddings: Tensor) -> Tensor:
        directions = torch.randn(
            projected_embeddings.size(-1),
            self.num_projections,
            device=projected_embeddings.device,
            dtype=projected_embeddings.dtype,
        )
        directions = directions.div_(directions.norm(p=2, dim=0).clamp_min(1e-12))
        x_t = (projected_embeddings @ directions).unsqueeze(-1) * self.t
        error = (
            (x_t.cos().mean(-3) - self.phi).square()
            + x_t.sin().mean(-3).square()
        )
        statistic = (error @ self.weights) * projected_embeddings.size(-2)
        return statistic.mean()


def sigreg(embeddings: Tensor, num_slices: int = 256, num_points: int = 17) -> Tensor:
    """Functional convenience wrapper around :class:`SIGReg`."""

    regularizer = SIGReg(knots=num_points, num_projections=num_slices).to(embeddings.device)
    return regularizer(embeddings)


def pretraining_loss(
    prediction: Tensor,
    target: Tensor,
    regularized_embeddings: Tensor,
    *,
    sigreg_weight: float = 0.1,
    sigreg_slices: int = 256,
) -> dict[str, Tensor]:
    prediction_loss = F.mse_loss(prediction, target)
    regularization = sigreg(regularized_embeddings, sigreg_slices)
    return {
        "loss": prediction_loss + sigreg_weight * regularization,
        "prediction_loss": prediction_loss,
        "sigreg_loss": regularization,
    }
