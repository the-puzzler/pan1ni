from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pan1ni.models.config import ModelConfig
from pan1ni.data.windows import GoalWindowDataset
from pan1ni.models.model import GoalConditionedLeWorldModel
from pan1ni.data.nld import NLDHDF5GoalDataset, nld_episode_keys
from pan1ni.data.synthetic import make_goal_directed_trajectories
from pan1ni.training.primitives import move_batch, pretrain_step


@torch.no_grad()
def _validation_metrics(model: GoalConditionedLeWorldModel, batch: dict) -> dict[str, float]:
    model.eval()
    grouped = model.encode_group(batch["history"], batch["goal"], batch["target"])
    output = model.predict_latents(grouped[:, :-2], grouped[:, -2])
    target_z = grouped[:, -1]
    flat_z = grouped.flatten(0, 1)
    centered = flat_z - flat_z.mean(0)
    covariance = centered.T @ centered / max(flat_z.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    prediction_loss = F.mse_loss(output.next_latent, target_z)
    copy_current_loss = F.mse_loss(grouped[:, -3], target_z)
    target_variance = target_z.var(0).mean().clamp_min(1e-12)
    return {
        "validation_prediction_loss": prediction_loss.item(),
        "validation_copy_current_loss": copy_current_loss.item(),
        "validation_normalized_prediction_loss": (prediction_loss / target_variance).item(),
        "validation_prediction_vs_copy": (prediction_loss / copy_current_loss.clamp_min(1e-12)).item(),
        "validation_cosine_similarity": F.cosine_similarity(output.next_latent, target_z).mean().item(),
        "latent_feature_std": flat_z.std(0).mean().item(),
        "latent_effective_rank": effective_rank.item(),
    }


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        latent_dim=64,
        cell_dim=32,
        message_dim=16,
        hidden_dim=64,
        vit_dim=32,
        vit_layers=1,
        vit_heads=4,
        terminal_patch_size=2,
        projector_hidden_dim=128,
        predictor_layers=1,
        predictor_heads=4,
        max_context=8,
        num_actions=8,
        dropout=0.0,
    )


def _summarize_run(
    history: list[dict[str, float]],
    initial: dict[str, float],
    final: dict[str, float],
    *,
    steps: int,
    batch_size: int,
    sigreg_slices: int,
    elapsed_seconds: float,
) -> dict[str, float]:
    window = min(10, len(history))
    return {
        "steps": float(steps),
        "batch_size": float(batch_size),
        "sigreg_slices": float(sigreg_slices),
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": steps * batch_size / max(elapsed_seconds, 1e-12),
        "early_prediction_loss": sum(item["prediction_loss"] for item in history[:window]) / window,
        "late_prediction_loss": sum(item["prediction_loss"] for item in history[-window:]) / window,
        "early_total_loss": sum(item["loss"] for item in history[:window]) / window,
        "late_total_loss": sum(item["loss"] for item in history[-window:]) / window,
        **{f"initial_{name}": value for name, value in initial.items()},
        **{f"final_{name}": value for name, value in final.items()},
        "final_sigreg_loss": history[-1]["sigreg_loss"],
    }


def smoke_test(steps: int, device: str, batch_size: int = 32, sigreg_slices: int = 256) -> dict[str, float]:
    if steps < 1:
        raise ValueError("steps must be positive")
    if batch_size < 2 or sigreg_slices < 1:
        raise ValueError("batch_size must be >= 2 and sigreg_slices must be positive")
    torch.manual_seed(0)
    config = _tiny_config()
    model = GoalConditionedLeWorldModel(config).to(device)
    trajectories = make_goal_directed_trajectories(
        count=max(64, batch_size * 2),
        min_distance=8,
    )
    dataset = GoalWindowDataset(
        trajectories,
        context_length=1,
        samples_per_epoch=steps * batch_size,
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    validation = GoalWindowDataset(
        make_goal_directed_trajectories(count=32, min_distance=8, seed=10_000),
        context_length=1,
        samples_per_epoch=64,
        seed=10_000,
    )
    validation_batch = move_batch(next(iter(DataLoader(validation, batch_size=64))), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    iterator = iter(loader)
    initial = _validation_metrics(model, validation_batch)
    history: list[dict[str, float]] = []
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        batch = move_batch(next(iterator), device)
        history.append(pretrain_step(model, batch, optimizer, sigreg_slices=sigreg_slices))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final = _validation_metrics(model, validation_batch)
    return _summarize_run(
        history,
        initial,
        final,
        steps=steps,
        batch_size=batch_size,
        sigreg_slices=sigreg_slices,
        elapsed_seconds=elapsed,
    )


def nld_smoke_test(
    path: str,
    steps: int,
    device: str,
    batch_size: int = 4,
    sigreg_slices: int = 256,
    goal_horizon: int = 64,
) -> dict[str, float]:
    if steps < 1 or batch_size < 2:
        raise ValueError("steps must be positive and batch_size must be >= 2")
    torch.manual_seed(0)
    keys = nld_episode_keys(path)
    validation_count = max(1, len(keys) // 5)
    train_keys, validation_keys = keys[:-validation_count], keys[-validation_count:]
    if not train_keys:
        raise ValueError("NLD smoke test requires at least two episodes")
    train_data = NLDHDF5GoalDataset(
        path,
        episode_keys=train_keys,
        context_length=1,
        goal_horizon=goal_horizon,
        samples_per_epoch=steps * batch_size,
    )
    validation_data = NLDHDF5GoalDataset(
        path,
        episode_keys=validation_keys,
        context_length=1,
        goal_horizon=goal_horizon,
        samples_per_epoch=max(8, batch_size),
        seed=10_000,
    )
    loader = DataLoader(train_data, batch_size=batch_size)
    validation_batch = move_batch(
        next(iter(DataLoader(validation_data, batch_size=min(8, len(validation_data))))),
        device,
    )
    model = GoalConditionedLeWorldModel(_tiny_config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    initial = _validation_metrics(model, validation_batch)
    history: list[dict[str, float]] = []
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for batch in loader:
        history.append(pretrain_step(model, move_batch(batch, device), optimizer, sigreg_slices=sigreg_slices))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final = _validation_metrics(model, validation_batch)
    summary = _summarize_run(
        history,
        initial,
        final,
        steps=steps,
        batch_size=batch_size,
        sigreg_slices=sigreg_slices,
        elapsed_seconds=elapsed,
    )
    summary.update(
        {
            "episodes": float(len(keys)),
            "train_episodes": float(len(train_keys)),
            "validation_episodes": float(len(validation_keys)),
            "available_windows": float(train_data.total_windows + validation_data.total_windows),
            "goal_horizon": float(goal_horizon),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal-conditioned NetHack LeWorldModel")
    parser.add_argument("command", nargs="?", choices=("smoke", "nld-smoke"), default="smoke")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sigreg-slices", type=int, default=256)
    parser.add_argument("--goal-horizon", type=int, default=64)
    parser.add_argument(
        "--data",
        default="data/downloads/nld-aa-taster.hdf5",
        help="Path to the NLD-AA HDF5 file for nld-smoke",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.command == "nld-smoke":
        if not Path(args.data).is_file():
            parser.error(f"NLD file not found: {args.data}")
        result = nld_smoke_test(
            args.data,
            args.steps,
            args.device,
            args.batch_size,
            args.sigreg_slices,
            args.goal_horizon,
        )
    else:
        result = smoke_test(args.steps, args.device, args.batch_size, args.sigreg_slices)
    print(json.dumps(result, indent=2))
