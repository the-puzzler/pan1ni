from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .action import DirectPolicyHead, set_backbone_trainable
from .config import ModelConfig
from .minihack_data import MiniHackSpecialActionDataset
from .model import GoalConditionedLeWorldModel
from .nld_action_train import ACTION_FEATURES, predictor_features
from .nld_data import prefetch_batches
from .player_tile_data import NLDPlayerTileGoalBatchStream, SEMANTIC_ACTION_NAMES
from .train import move_batch


def _slice_batch(batch: dict, size: int) -> dict:
    return {
        key: (
            {name: tensor[:size] for name, tensor in value.items()}
            if isinstance(value, dict)
            else value[:size]
        )
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(model, head, batches: list[dict], feature: str) -> dict[str, float | list[int]]:
    model.eval()
    head.eval()
    count = 0
    loss_sum = 0.0
    class_count = torch.zeros(10, dtype=torch.long)
    class_correct = torch.zeros(10, dtype=torch.long)
    for batch in batches:
        targets = batch["action"].long()
        logits = head(predictor_features(model, batch, feature))
        loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        predictions = logits.argmax(-1)
        matches = predictions.eq(targets)
        count += targets.numel()
        class_count += torch.bincount(targets.cpu(), minlength=10)
        class_correct += torch.bincount(targets[matches].cpu(), minlength=10)
    supported = class_count > 0
    return {
        "loss": loss_sum / max(count, 1),
        "accuracy": int(class_correct.sum()) / max(count, 1),
        "balanced_accuracy": (
            class_correct[supported].float().div(class_count[supported]).mean().item()
        ),
        "samples": count,
        "class_count": class_count.tolist(),
        "class_correct": class_correct.tolist(),
    }


def run(
    checkpoint_path: Path,
    player_db: Path,
    player_dataset: str,
    simulator_data: Path,
    output: Path,
    *,
    steps: int,
    player_decode_batch_size: int,
    player_batch_size: int,
    special_batch_size: int,
    context_length: int,
    goal_horizon: int,
    hidden_dim: int,
    hidden_layers: int,
    feature: str,
    class_weight_samples: int,
    eval_every: int,
    player_validation_samples: int,
    special_validation_samples: int,
    num_workers: int,
    prefetch_depth: int,
    windows_per_block: int,
    window_stride: int,
    device: str,
) -> Path:
    if min(
        steps, player_decode_batch_size, player_batch_size, special_batch_size,
        context_length, goal_horizon, hidden_dim, hidden_layers, class_weight_samples,
        eval_every, player_validation_samples, special_validation_samples, num_workers,
        prefetch_depth, windows_per_block, window_stride,
    ) < 1:
        raise ValueError("training and stream sizes must be positive")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    use_cuda = device.startswith("cuda")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GoalConditionedLeWorldModel(ModelConfig(**checkpoint["config"])).to(device)
    if model.config.observation_mode != "pixels":
        raise ValueError("inferred action training requires a tile-pixel checkpoint")
    if model.config.prediction_objective != "mse":
        raise ValueError("inferred action training requires an MSE checkpoint")
    model.load_state_dict(checkpoint["model"])
    set_backbone_trainable(model, False)
    model.eval()
    if feature not in ACTION_FEATURES or feature == "flow_residual":
        raise ValueError("MSE action feature must be an encoder or predictor feature")
    head = DirectPolicyHead(
        model.config.latent_dim,
        10,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)

    catalog = NLDPlayerTileGoalBatchStream(
        player_db, player_dataset, batch_size=1, num_workers=1
    )
    gameids = catalog.available_gameids()
    validation_count = max(1, len(gameids) // 5)
    train_ids, validation_ids = gameids[:-validation_count], gameids[-validation_count:]

    with h5py.File(simulator_data, "r") as handle:
        simulator_keys = sorted(handle.keys(), key=int)
    simulator_validation_keys = simulator_keys[::5]
    simulator_validation_set = set(simulator_validation_keys)
    simulator_train_keys = [key for key in simulator_keys if key not in simulator_validation_set]

    weight_stream = NLDPlayerTileGoalBatchStream(
        player_db,
        player_dataset,
        batch_size=player_decode_batch_size,
        context_length=context_length,
        goal_horizon=goal_horizon,
        gameids=train_ids,
        num_workers=num_workers,
        windows_per_block=windows_per_block,
        window_stride=window_stride,
        shuffle=True,
        loop_forever=True,
        seed=30_000,
        action_mode="inferred_movement",
    )
    movement_counts = torch.zeros(10, dtype=torch.long)
    counted = 0
    for batch in weight_stream:
        actions = batch["action"].long()
        movement_counts += torch.bincount(actions, minlength=10)
        counted += actions.numel()
        if counted >= class_weight_samples:
            break

    special_train = MiniHackSpecialActionDataset(
        simulator_data,
        episode_keys=simulator_train_keys,
        context_length=context_length,
        goal_horizon=goal_horizon,
        samples_per_epoch=steps * special_batch_size,
        seed=40_000,
    )
    special_counts = torch.zeros(10, dtype=torch.long)
    with h5py.File(simulator_data, "r") as handle:
        for key, timestep in special_train.examples:
            special_counts[int(handle[key]["actions"][timestep])] += 1
    expected = movement_counts.float().div(max(counted, 1)) * player_batch_size
    expected += special_counts.float().div(max(int(special_counts.sum()), 1)) * special_batch_size
    supported = expected > 0
    class_weights = torch.zeros(10, dtype=torch.float32)
    class_weights[supported] = expected.sum() / (supported.sum() * expected[supported])
    class_weights.clamp_(max=10.0)
    class_weights[supported] /= class_weights[supported].mean()
    class_weights = class_weights.to(device)

    player_validation_stream = NLDPlayerTileGoalBatchStream(
        player_db,
        player_dataset,
        batch_size=min(64, len(validation_ids)),
        context_length=context_length,
        goal_horizon=goal_horizon,
        gameids=validation_ids,
        num_workers=min(num_workers, 8),
        windows_per_block=2,
        window_stride=window_stride,
        shuffle=False,
        loop_forever=True,
        seed=20_000,
        action_mode="inferred_movement",
    )
    player_validation_host = []
    player_seen = 0
    for batch in player_validation_stream:
        player_validation_host.append(batch)
        player_seen += batch["action"].numel()
        if player_seen >= player_validation_samples:
            break

    loader_options = {"num_workers": num_workers, "pin_memory": use_cuda}
    if num_workers:
        loader_options.update(prefetch_factor=2, persistent_workers=True)
    special_validation = MiniHackSpecialActionDataset(
        simulator_data,
        episode_keys=simulator_validation_keys,
        context_length=context_length,
        goal_horizon=goal_horizon,
        samples_per_epoch=special_validation_samples,
        seed=50_000,
    )
    special_validation_host = list(
        DataLoader(special_validation, batch_size=special_batch_size, **loader_options)
    )
    player_validation_batches = [
        move_batch(batch, device, non_blocking=use_cuda) for batch in player_validation_host
    ]
    special_validation_batches = [
        move_batch(batch, device, non_blocking=use_cuda) for batch in special_validation_host
    ]

    player_stream = NLDPlayerTileGoalBatchStream(
        player_db,
        player_dataset,
        batch_size=player_decode_batch_size,
        context_length=context_length,
        goal_horizon=goal_horizon,
        gameids=train_ids,
        num_workers=num_workers,
        windows_per_block=windows_per_block,
        window_stride=window_stride,
        shuffle=True,
        loop_forever=True,
        seed=0,
        action_mode="inferred_movement",
    )
    player_iterator = prefetch_batches(iter(player_stream), prefetch_depth)
    special_iterator = iter(
        DataLoader(special_train, batch_size=special_batch_size, **loader_options)
    )

    initial_player = evaluate(model, head, player_validation_batches, feature)
    initial_special = evaluate(model, head, special_validation_batches, feature)
    timeline = [{"step": 0, "player_validation": initial_player, "special_validation": initial_special}]
    best_balanced = 0.0
    trained_player_samples = 0
    trained_special_samples = 0
    started = time.perf_counter()
    try:
        for step in range(1, steps + 1):
            player_host = next(player_iterator)
            if player_host["action"].numel() > player_batch_size:
                player_host = _slice_batch(player_host, player_batch_size)
            player_batch = move_batch(player_host, device, non_blocking=use_cuda)
            special_batch = move_batch(next(special_iterator), device, non_blocking=use_cuda)
            trained_player_samples += player_batch["action"].numel()
            trained_special_samples += special_batch["action"].numel()
            with torch.no_grad():
                features = torch.cat(
                    (
                        predictor_features(model, player_batch, feature),
                        predictor_features(model, special_batch, feature),
                    )
                )
            targets = torch.cat((player_batch["action"], special_batch["action"]))
            head.train()
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = F.cross_entropy(logits, targets, weight=class_weights)
            loss.backward()
            optimizer.step()
            if step % eval_every == 0 or step == steps:
                player_metrics = evaluate(model, head, player_validation_batches, feature)
                special_metrics = evaluate(model, head, special_validation_batches, feature)
                balanced = (
                    float(player_metrics["balanced_accuracy"]) * 0.8
                    + float(special_metrics["balanced_accuracy"]) * 0.2
                )
                timeline.append(
                    {
                        "step": step,
                        "train_loss": loss.detach().item(),
                        "player_validation": player_metrics,
                        "special_validation": special_metrics,
                        "selection_balanced_accuracy": balanced,
                    }
                )
                if balanced > best_balanced:
                    best_balanced = balanced
                    torch.save(
                        {
                            "head": head.state_dict(),
                            "config": {
                                "feature": feature,
                                "target": "semantic_10",
                                "num_classes": 10,
                                "action_hidden_dim": hidden_dim,
                                "action_hidden_layers": hidden_layers,
                            },
                        },
                        output / "best_action_checkpoint.pt",
                    )
                print(
                    f"step {step:5d}/{steps} | train {loss.item():.4f} | "
                    f"human move acc {float(player_metrics['accuracy']):.2%} | "
                    f"special acc {float(special_metrics['accuracy']):.2%} | "
                    f"balanced {balanced:.2%}",
                    flush=True,
                )
    finally:
        close = getattr(player_iterator, "close", None)
        if close is not None:
            close()

    config = {
        "world_checkpoint": str(checkpoint_path),
        "world_step": int(checkpoint["step"]),
        "feature": feature,
        "target": "semantic_10",
        "action_names": list(SEMANTIC_ACTION_NAMES),
        "num_classes": 10,
        "human_source": player_dataset,
        "human_label_method": "successful one-cell @ cursor delta",
        "human_train_games": len(train_ids),
        "human_validation_games": len(validation_ids),
        "special_source": str(simulator_data),
        "special_classes": ["pickup", "apply"],
        "steps": steps,
        "player_batch_size": player_batch_size,
        "special_batch_size": special_batch_size,
        "trained_player_samples": trained_player_samples,
        "trained_special_samples": trained_special_samples,
        "class_weight_samples": counted,
        "movement_class_counts": movement_counts.tolist(),
        "special_example_counts": special_counts.tolist(),
        "class_weights": class_weights.cpu().tolist(),
        "action_hidden_dim": hidden_dim,
        "action_hidden_layers": hidden_layers,
        "action_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "best_balanced_accuracy": best_balanced,
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save(
        {"head": head.state_dict(), "config": config}, output / "action_checkpoint.pt"
    )
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps({"config": config, "timeline": timeline}, indent=2))
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train semantic actions from inferred human moves plus MiniHack specials"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--player-db", type=Path, default=Path("data/nld/nld-nao.db"))
    parser.add_argument("--player-dataset", default="nld-nao-human-8shard")
    parser.add_argument("--simulator-data", type=Path, default=Path("data/minihack/keyroom-rgb-1600.hdf5"))
    parser.add_argument("--output", type=Path, default=Path("reports/inferred-human-action"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--player-decode-batch-size", type=int, default=128)
    parser.add_argument("--player-batch-size", type=int, default=48)
    parser.add_argument("--special-batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--goal-horizon", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--feature", choices=ACTION_FEATURES, default="predictor_hidden")
    parser.add_argument("--class-weight-samples", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--player-validation-samples", type=int, default=1024)
    parser.add_argument("--special-validation-samples", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-depth", type=int, default=3)
    parser.add_argument("--windows-per-block", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(
        run(
            args.checkpoint,
            args.player_db,
            args.player_dataset,
            args.simulator_data,
            args.output,
            steps=args.steps,
            player_decode_batch_size=args.player_decode_batch_size,
            player_batch_size=args.player_batch_size,
            special_batch_size=args.special_batch_size,
            context_length=args.context_length,
            goal_horizon=args.goal_horizon,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            feature=args.feature,
            class_weight_samples=args.class_weight_samples,
            eval_every=args.eval_every,
            player_validation_samples=args.player_validation_samples,
            special_validation_samples=args.special_validation_samples,
            num_workers=args.num_workers,
            prefetch_depth=args.prefetch_depth,
            windows_per_block=args.windows_per_block,
            window_stride=args.window_stride,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
