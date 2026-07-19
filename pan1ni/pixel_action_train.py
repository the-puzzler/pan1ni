"""Train a movement-only, in-distribution action head on frozen world-model features.

The pretrained tile-pixel world model was trained purely on human NLD-NAO NetHack
trajectories. This trains an 8-class semantic movement head (north .. northwest) on
exactly that distribution, using only goal-aware predictor features (the goal enters
through the predictor; the pre-predictor ``current_latent`` is deliberately excluded).
No MiniHack data is involved, so the head and its validation are fully in-distribution.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from .action import (
    GOAL_AWARE_FEATURES,
    DirectPolicyHead,
    feature_dim,
    predictor_features,
    set_backbone_trainable,
)
from .config import ModelConfig
from .model import GoalConditionedLeWorldModel
from .nld_data import prefetch_batches
from .player_tile_data import SEMANTIC_ACTION_NAMES, NLDPlayerTileGoalBatchStream
from .train import move_batch

MOVEMENT_CLASSES = 8
MOVEMENT_ACTION_NAMES = SEMANTIC_ACTION_NAMES[:MOVEMENT_CLASSES]


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
def evaluate(model, head, batches: list[dict], feature: str) -> dict:
    model.eval()
    head.eval()
    count = 0
    loss_sum = 0.0
    class_count = torch.zeros(MOVEMENT_CLASSES, dtype=torch.long)
    class_correct = torch.zeros(MOVEMENT_CLASSES, dtype=torch.long)
    for batch in batches:
        targets = batch["action"].long()
        logits = head(predictor_features(model, batch, feature))
        loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        matches = logits.argmax(-1).eq(targets)
        count += targets.numel()
        class_count += torch.bincount(targets.cpu(), minlength=MOVEMENT_CLASSES)
        class_correct += torch.bincount(targets[matches].cpu(), minlength=MOVEMENT_CLASSES)
    supported = class_count > 0
    return {
        "loss": loss_sum / max(count, 1),
        "accuracy": int(class_correct.sum()) / max(count, 1),
        "balanced_accuracy": (
            class_correct[supported].float().div(class_count[supported]).mean().item()
            if supported.any()
            else 0.0
        ),
        "samples": count,
        "class_count": class_count.tolist(),
        "class_correct": class_correct.tolist(),
    }


def _render_curves(payload: dict, output_path: Path) -> None:
    timeline = payload["timeline"]
    eval_steps = [entry["step"] for entry in timeline]
    accuracy = [entry["player_validation"]["accuracy"] for entry in timeline]
    balanced = [entry["player_validation"]["balanced_accuracy"] for entry in timeline]
    trained = [entry for entry in timeline if "train_loss" in entry]
    train_steps = [entry["step"] for entry in trained]
    train_loss = [entry["train_loss"] for entry in trained]
    val_loss = [entry["player_validation"]["loss"] for entry in timeline]

    plt.style.use("dark_background")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    feature = payload["config"]["feature"]
    figure.suptitle(
        f"Movement action head — feature: {feature} "
        f"(chance balanced acc = {1 / MOVEMENT_CLASSES:.1%})",
        fontsize=13,
    )
    axes[0].plot(train_steps, train_loss, marker="o", markersize=3, color="#22d3ee", label="train")
    axes[0].plot(eval_steps, val_loss, marker="o", markersize=3, color="#f0b35a", label="val")
    axes[0].set(title="Cross-entropy loss", xlabel="step", ylabel="loss")
    axes[0].legend(frameon=False)
    axes[1].plot(eval_steps, accuracy, marker="o", markersize=3, color="#70d6a5", label="accuracy")
    axes[1].plot(eval_steps, balanced, marker="o", markersize=3, color="#c084fc", label="balanced")
    axes[1].axhline(1 / MOVEMENT_CLASSES, color="#64748b", linestyle="--", linewidth=1, label="chance")
    axes[1].set(title="Held-out human-move accuracy", xlabel="step", ylabel="accuracy", ylim=(0, 1))
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def run(
    checkpoint_path: Path,
    player_db: Path,
    player_dataset: str,
    output: Path,
    *,
    steps: int,
    decode_batch_size: int,
    batch_size: int,
    context_length: int,
    goal_horizon: int,
    hidden_dim: int,
    hidden_layers: int,
    feature: str,
    class_weight_samples: int,
    eval_every: int,
    validation_samples: int,
    num_workers: int,
    prefetch_depth: int,
    windows_per_block: int,
    window_stride: int,
    device: str,
) -> Path:
    if min(
        steps, decode_batch_size, batch_size, context_length, goal_horizon, hidden_dim,
        hidden_layers, class_weight_samples, eval_every, validation_samples, num_workers,
        prefetch_depth, windows_per_block, window_stride,
    ) < 1:
        raise ValueError("training and stream sizes must be positive")
    if feature not in GOAL_AWARE_FEATURES:
        raise ValueError(f"feature must be one of {GOAL_AWARE_FEATURES}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    use_cuda = device.startswith("cuda")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GoalConditionedLeWorldModel(ModelConfig(**checkpoint["config"])).to(device)
    if model.config.observation_mode != "pixels":
        raise ValueError("movement action training requires a tile-pixel checkpoint")
    if model.config.prediction_objective != "mse":
        raise ValueError("movement action training requires an MSE checkpoint")
    if context_length > model.config.max_context:
        raise ValueError("context length exceeds the pretrained model's maximum context")
    model.load_state_dict(checkpoint["model"])
    set_backbone_trainable(model, False)
    model.eval()

    head = DirectPolicyHead(
        feature_dim(feature, model.config.latent_dim),
        MOVEMENT_CLASSES,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)

    catalog = NLDPlayerTileGoalBatchStream(player_db, player_dataset, batch_size=1, num_workers=1)
    gameids = catalog.available_gameids()
    validation_count = max(1, len(gameids) // 5)
    train_ids, validation_ids = gameids[:-validation_count], gameids[-validation_count:]

    def make_stream(*, seed: int, ids, bsz: int, windows: int, shuffle: bool) -> NLDPlayerTileGoalBatchStream:
        return NLDPlayerTileGoalBatchStream(
            player_db,
            player_dataset,
            batch_size=bsz,
            context_length=context_length,
            goal_horizon=goal_horizon,
            gameids=ids,
            num_workers=num_workers,
            windows_per_block=windows,
            window_stride=window_stride,
            shuffle=shuffle,
            loop_forever=True,
            seed=seed,
            action_mode="inferred_movement",
        )

    # Inverse-frequency class weights estimated on the training split only. Movement is
    # heavily imbalanced (cardinal moves ~5x more frequent than diagonals).
    weight_stream = make_stream(seed=30_000, ids=train_ids, bsz=decode_batch_size, windows=windows_per_block, shuffle=True)
    movement_counts = torch.zeros(MOVEMENT_CLASSES, dtype=torch.long)
    counted = 0
    for batch in weight_stream:
        movement_counts += torch.bincount(batch["action"].long(), minlength=MOVEMENT_CLASSES)
        counted += batch["action"].numel()
        if counted >= class_weight_samples:
            break
    supported = movement_counts > 0
    class_weights = torch.zeros(MOVEMENT_CLASSES, dtype=torch.float32)
    class_weights[supported] = counted / (supported.sum() * movement_counts[supported].float())
    class_weights.clamp_(max=10.0)
    class_weights[supported] /= class_weights[supported].mean()
    class_weights = class_weights.to(device)

    validation_stream = make_stream(
        seed=20_000, ids=validation_ids, bsz=min(64, len(validation_ids)), windows=2, shuffle=False
    )
    validation_batches = []
    seen = 0
    for batch in validation_stream:
        validation_batches.append(move_batch(batch, device, non_blocking=use_cuda))
        seen += batch["action"].numel()
        if seen >= validation_samples:
            break

    player_stream = make_stream(seed=0, ids=train_ids, bsz=decode_batch_size, windows=windows_per_block, shuffle=True)
    iterator = prefetch_batches(iter(player_stream), prefetch_depth)

    initial = evaluate(model, head, validation_batches, feature)
    timeline = [{"step": 0, "player_validation": initial, "selection_balanced_accuracy": initial["balanced_accuracy"]}]
    best_balanced = 0.0
    trained_samples = 0
    started = time.perf_counter()
    try:
        for step in range(1, steps + 1):
            host = next(iterator)
            if host["action"].numel() > batch_size:
                host = _slice_batch(host, batch_size)
            batch = move_batch(host, device, non_blocking=use_cuda)
            trained_samples += batch["action"].numel()
            with torch.no_grad():
                features = predictor_features(model, batch, feature)
            head.train()
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = F.cross_entropy(logits, batch["action"].long(), weight=class_weights)
            loss.backward()
            optimizer.step()
            if step % eval_every == 0 or step == steps:
                validation = evaluate(model, head, validation_batches, feature)
                balanced = float(validation["balanced_accuracy"])
                timeline.append(
                    {
                        "step": step,
                        "train_loss": loss.detach().item(),
                        "player_validation": validation,
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
                                "target": "movement_8",
                                "num_classes": MOVEMENT_CLASSES,
                                "action_hidden_dim": hidden_dim,
                                "action_hidden_layers": hidden_layers,
                            },
                        },
                        output / "best_action_checkpoint.pt",
                    )
                print(
                    f"step {step:5d}/{steps} | feature {feature} | train {loss.item():.4f} | "
                    f"val {validation['loss']:.4f} | move acc {validation['accuracy']:.2%} | "
                    f"balanced {balanced:.2%}",
                    flush=True,
                )
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()

    config = {
        "world_checkpoint": str(checkpoint_path),
        "world_step": int(checkpoint["step"]),
        "feature": feature,
        "target": "movement_8",
        "action_names": list(MOVEMENT_ACTION_NAMES),
        "num_classes": MOVEMENT_CLASSES,
        "human_source": player_dataset,
        "human_label_method": "successful one-cell @ cursor delta",
        "human_train_games": len(train_ids),
        "human_validation_games": len(validation_ids),
        "steps": steps,
        "batch_size": batch_size,
        "trained_samples": trained_samples,
        "class_weight_samples": counted,
        "movement_class_counts": movement_counts.tolist(),
        "class_weights": class_weights.cpu().tolist(),
        "action_hidden_dim": hidden_dim,
        "action_hidden_layers": hidden_layers,
        "action_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "feature_dim": feature_dim(feature, model.config.latent_dim),
        "best_balanced_accuracy": best_balanced,
        "backbone_frozen": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload = {"config": config, "timeline": timeline}
    torch.save({"head": head.state_dict(), "config": config}, output / "action_checkpoint.pt")
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _render_curves(payload, output / "learning-curves.png")
    return output / "metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a movement-only in-distribution action head on human tile features"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--player-db", type=Path, default=Path("data/nld/nld-nao.db"))
    parser.add_argument("--player-dataset", default="nld-nao-human-8shard")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--decode-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--goal-horizon", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--feature", choices=GOAL_AWARE_FEATURES, default="predictor_hidden")
    parser.add_argument("--class-weight-samples", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--validation-samples", type=int, default=1024)
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
            args.output,
            steps=args.steps,
            decode_batch_size=args.decode_batch_size,
            batch_size=args.batch_size,
            context_length=args.context_length,
            goal_horizon=args.goal_horizon,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            feature=args.feature,
            class_weight_samples=args.class_weight_samples,
            eval_every=args.eval_every,
            validation_samples=args.validation_samples,
            num_workers=args.num_workers,
            prefetch_depth=args.prefetch_depth,
            windows_per_block=args.windows_per_block,
            window_stride=args.window_stride,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
