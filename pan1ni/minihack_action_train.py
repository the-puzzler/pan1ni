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
from .train import move_batch


@torch.no_grad()
def evaluate(model, head, batches: list[dict]) -> dict[str, float]:
    model.eval()
    head.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    class_count = torch.zeros(10, dtype=torch.long)
    class_correct = torch.zeros(10, dtype=torch.long)
    for batch in batches:
        target = batch["action"].long()
        logits = head(model(batch["history"], batch["goal"]).hidden)
        loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
        prediction = logits.argmax(-1)
        matches = prediction.eq(target)
        correct += int(matches.sum())
        count += target.numel()
        class_count += torch.bincount(target.cpu(), minlength=10)
        class_correct += torch.bincount(target[matches].cpu(), minlength=10)
    supported = class_count > 0
    return {
        "loss": loss_sum / max(count, 1),
        "accuracy": correct / max(count, 1),
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
    eval_every: int,
    validation_samples: int,
    num_workers: int,
    device: str,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GoalConditionedLeWorldModel(ModelConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    set_backbone_trainable(model, False)
    model.eval()
    head = DirectPolicyHead(model.config.latent_dim, 10).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)

    with h5py.File(data_path, "r") as handle:
        keys = sorted(handle.keys(), key=int)
    validation_keys = keys[::5]
    validation_set = set(validation_keys)
    training_keys = [key for key in keys if key not in validation_set]
    training_data = MiniHackPixelGoalDataset(
        data_path,
        episode_keys=training_keys,
        context_length=1,
        samples_per_epoch=steps * batch_size,
    )
    validation_data = MiniHackPixelGoalDataset(
        data_path,
        episode_keys=validation_keys,
        context_length=1,
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
    timeline = [{"step": 0, "validation": evaluate(model, head, validation_batches)}]
    started = time.perf_counter()

    for step, host_batch in enumerate(loader, start=1):
        batch = move_batch(host_batch, device, non_blocking=use_cuda)
        with torch.no_grad():
            hidden = model(batch["history"], batch["goal"]).hidden
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = head(hidden)
        loss = F.cross_entropy(logits, batch["action"].long())
        loss.backward()
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            metrics = evaluate(model, head, validation_batches)
            timeline.append(
                {"step": step, "train_loss": loss.detach().item(), "validation": metrics}
            )
            print(
                f"step {step:5d}/{steps} | train {loss.item():.4f} | "
                f"val {metrics['loss']:.4f} | acc {metrics['accuracy']:.2%} | "
                f"balanced {metrics['balanced_accuracy']:.2%}",
                flush=True,
            )
        if step >= steps:
            break

    elapsed = time.perf_counter() - started
    config = {
        "world_checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "feature": "predictor_hidden",
        "backbone_frozen": True,
        "num_actions": 10,
        "steps": steps,
        "batch_size": batch_size,
        "training_episodes": len(training_keys),
        "validation_episodes": len(validation_keys),
        "elapsed_seconds": elapsed,
        "samples_per_second": steps * batch_size / max(elapsed, 1e-12),
    }
    torch.save({"head": head.state_dict(), "config": config}, output / "checkpoint.pt")
    metrics_path = output / "metrics.json"
    metrics_path.write_text(
        json.dumps({"config": config, "timeline": timeline}, indent=2),
        encoding="utf-8",
    )
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a frozen predictor-hidden MiniHack pixel action head"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", default="reports/minihack-rgb-action", type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    print(
        run(
            args.checkpoint,
            args.data,
            args.output,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            validation_samples=args.validation_samples,
            num_workers=args.num_workers,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
