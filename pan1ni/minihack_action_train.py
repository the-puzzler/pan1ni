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
from .minihack_data import MiniHackPixelGoalDataset
from .model import GoalConditionedLeWorldModel
from .nld_action_train import predictor_features
from .train import move_batch


@torch.no_grad()
def evaluate(model, head, batches: list[dict], feature: str) -> dict[str, float]:
    model.eval()
    head.eval()
    loss_sum = 0.0
    correct = 0
    top5_correct = 0
    count = 0
    class_count = torch.zeros(10, dtype=torch.long)
    class_correct = torch.zeros(10, dtype=torch.long)
    for batch_index, batch in enumerate(batches):
        target = batch["action"].long()
        generator = torch.Generator(device=target.device).manual_seed(70_000 + batch_index)
        logits = head(predictor_features(model, batch, feature, generator))
        loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
        prediction = logits.argmax(-1)
        matches = prediction.eq(target)
        correct += int(matches.sum())
        top5_correct += int(logits.topk(5, dim=-1).indices.eq(target[:, None]).any(-1).sum())
        count += target.numel()
        class_count += torch.bincount(target.cpu(), minlength=10)
        class_correct += torch.bincount(target[matches].cpu(), minlength=10)
    supported = class_count > 0
    return {
        "loss": loss_sum / max(count, 1),
        "accuracy": correct / max(count, 1),
        "top5_accuracy": top5_correct / max(count, 1),
        "balanced_accuracy": (
            class_correct[supported].float().div(class_count[supported]).mean().item()
        ),
        "samples": count,
    }


def run(
    checkpoint_path: Path,
    data_path: Path,
    output: Path,
    *,
    steps: int,
    batch_size: int,
    context_length: int,
    goal_horizon: int,
    hidden_dim: int,
    hidden_layers: int,
    eval_every: int,
    validation_samples: int,
    num_workers: int,
    device: str,
    feature: str = "flow_residual",
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GoalConditionedLeWorldModel(ModelConfig(**payload["config"])).to(device)
    if model.config.observation_mode != "pixels":
        raise ValueError("action training requires a tile-pixel checkpoint")
    if feature == "flow_residual" and model.config.prediction_objective != "flow":
        raise ValueError("flow residual features require a flow checkpoint")
    model.load_state_dict(payload["model"])
    set_backbone_trainable(model, False)
    model.eval()
    head = DirectPolicyHead(
        model.config.latent_dim,
        10,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)

    with h5py.File(data_path, "r") as handle:
        keys = sorted(handle.keys(), key=int)
        class_counts = torch.zeros(10, dtype=torch.long)
        validation_keys = keys[::5]
        validation_set = set(validation_keys)
        training_keys = [key for key in keys if key not in validation_set]
        for key in training_keys:
            class_counts += torch.bincount(
                torch.from_numpy(handle[key]["actions"][:]).long(), minlength=10
            )
    supported = class_counts > 0
    class_weights = torch.zeros(10, dtype=torch.float32)
    class_weights[supported] = class_counts.sum() / (
        supported.sum() * class_counts[supported].float()
    )
    class_weights.clamp_(max=10.0)
    class_weights[supported] /= class_weights[supported].mean()
    class_weights = class_weights.to(device)

    training_data = MiniHackPixelGoalDataset(
        data_path,
        episode_keys=training_keys,
        context_length=context_length,
        goal_horizon=goal_horizon,
        random_future_goal=True,
        pad_initial_context=True,
        samples_per_epoch=steps * batch_size,
    )
    validation_data = MiniHackPixelGoalDataset(
        data_path,
        episode_keys=validation_keys,
        context_length=context_length,
        goal_horizon=goal_horizon,
        random_future_goal=True,
        pad_initial_context=True,
        samples_per_epoch=validation_samples,
        seed=20_000,
    )
    use_cuda = device.startswith("cuda")
    options = {"num_workers": num_workers, "pin_memory": use_cuda}
    if num_workers:
        options.update(prefetch_factor=2, persistent_workers=True)
    validation_batches = [
        move_batch(batch, device, non_blocking=use_cuda)
        for batch in DataLoader(validation_data, batch_size=batch_size, **options)
    ]
    loader = DataLoader(training_data, batch_size=batch_size, **options)
    initial = evaluate(model, head, validation_batches, feature)
    timeline = [{"step": 0, "validation": initial}]
    best_balanced = initial["balanced_accuracy"]
    best_step = 0
    started = time.perf_counter()

    for step, host_batch in enumerate(loader, start=1):
        batch = move_batch(host_batch, device, non_blocking=use_cuda)
        with torch.no_grad():
            features = predictor_features(model, batch, feature)
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = head(features)
        loss = F.cross_entropy(logits, batch["action"].long(), weight=class_weights)
        loss.backward()
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            metrics = evaluate(model, head, validation_batches, feature)
            timeline.append(
                {"step": step, "train_loss": loss.detach().item(), "validation": metrics}
            )
            if metrics["balanced_accuracy"] > best_balanced:
                best_balanced = metrics["balanced_accuracy"]
                best_step = step
                torch.save(
                    {
                        "head": head.state_dict(),
                        "config": {
                            "feature": feature,
                            "action_hidden_dim": hidden_dim,
                            "action_hidden_layers": hidden_layers,
                            "world_checkpoint": str(checkpoint_path),
                        },
                    },
                    output / "best_action_checkpoint.pt",
                )
            print(
                f"step {step:5d}/{steps} | train {loss.item():.4f} | "
                f"val {metrics['loss']:.4f} | acc {metrics['accuracy']:.2%} | "
                f"top5 {metrics['top5_accuracy']:.2%} | "
                f"balanced {metrics['balanced_accuracy']:.2%}",
                flush=True,
            )
        if step >= steps:
            break

    elapsed = time.perf_counter() - started
    config = {
        "world_checkpoint": str(checkpoint_path),
        "world_step": int(payload["step"]),
        "data": str(data_path),
        "representation": "native 144x144 MiniHack pixel_crop",
        "feature": feature,
        "backbone_frozen": True,
        "num_actions": 10,
        "steps": steps,
        "batch_size": batch_size,
        "context_length": context_length,
        "goal_horizon": goal_horizon,
        "random_future_goal": True,
        "action_hidden_dim": hidden_dim,
        "action_hidden_layers": hidden_layers,
        "action_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "class_counts": class_counts.tolist(),
        "class_weights": class_weights.cpu().tolist(),
        "training_episodes": len(training_keys),
        "validation_episodes": len(validation_keys),
        "best_step": best_step,
        "best_balanced_accuracy": best_balanced,
        "elapsed_seconds": elapsed,
        "samples_per_second": steps * batch_size / max(elapsed, 1e-12),
    }
    torch.save(
        {"head": head.state_dict(), "config": config}, output / "action_checkpoint.pt"
    )
    if not (output / "best_action_checkpoint.pt").exists():
        torch.save(
            {"head": head.state_dict(), "config": config},
            output / "best_action_checkpoint.pt",
        )
    metrics_path = output / "metrics.json"
    metrics_path.write_text(
        json.dumps({"config": config, "timeline": timeline}, indent=2),
        encoding="utf-8",
    )
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a weighted residual-flow action head on native MiniHack pixels"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", default="reports/pixel-flow-action", type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--goal-horizon", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--feature", choices=("predictor_hidden", "flow_residual"), default="flow_residual")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(
        run(
            args.checkpoint,
            args.data,
            Path(args.output),
            steps=args.steps,
            batch_size=args.batch_size,
            context_length=args.context_length,
            goal_horizon=args.goal_horizon,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            feature=args.feature,
            eval_every=args.eval_every,
            validation_samples=args.validation_samples,
            num_workers=args.num_workers,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
