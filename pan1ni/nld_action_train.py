from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .action import DirectPolicyHead, set_backbone_trainable
from .config import ModelConfig
from .model import GoalConditionedLeWorldModel
from .nld_data import NLDTtyrecGoalBatchStream, prefetch_streams
from .train import move_batch


@torch.no_grad()
def predictor_features(
    model: GoalConditionedLeWorldModel,
    batch: dict,
    feature: str = "predictor_hidden",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    model.eval()
    if feature == "predictor_hidden":
        return model(batch["history"], batch["goal"]).hidden
    if feature == "flow_residual":
        grouped = model.encode_group(batch["history"], batch["goal"])
        history_z, goal_z = grouped[:, :-1], grouped[:, -1]
        source_noise = torch.randn(
            history_z.shape,
            device=history_z.device,
            dtype=history_z.dtype,
            generator=generator,
        )
        _, residual = model.one_step_flow(history_z, goal_z, source_noise)
        return residual[:, -1]
    raise ValueError("feature must be 'predictor_hidden' or 'flow_residual'")


@torch.no_grad()
def evaluate(
    model: GoalConditionedLeWorldModel,
    head: DirectPolicyHead,
    batches: list[dict],
    feature: str = "predictor_hidden",
) -> dict[str, float]:
    head.eval()
    loss_sum = 0.0
    correct = 0
    top5_correct = 0
    count = 0
    class_count = torch.zeros(256, dtype=torch.long)
    class_correct = torch.zeros(256, dtype=torch.long)
    for batch_index, batch in enumerate(batches):
        target = batch["action"].long()
        generator = torch.Generator(device=target.device).manual_seed(60_000 + batch_index)
        logits = head(predictor_features(model, batch, feature, generator))
        loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
        prediction = logits.argmax(-1)
        matches = prediction.eq(target)
        correct += int(matches.sum())
        top5_correct += int(logits.topk(5, dim=-1).indices.eq(target[:, None]).any(-1).sum())
        count += target.numel()
        class_count += torch.bincount(target.cpu(), minlength=256)
        class_correct += torch.bincount(target[matches].cpu(), minlength=256)
    supported = class_count > 0
    balanced_accuracy = (
        (class_correct[supported].float() / class_count[supported]).mean().item()
        if supported.any()
        else 0.0
    )
    return {
        "loss": loss_sum / max(count, 1),
        "accuracy": correct / max(count, 1),
        "top5_accuracy": top5_correct / max(count, 1),
        "balanced_accuracy": balanced_accuracy,
        "samples": count,
        "supported_keycodes": int(supported.sum()),
    }


def run(
    checkpoint_path: Path,
    db_path: Path,
    dataset_name: str,
    output: Path,
    *,
    steps: int,
    batch_size: int,
    context_length: int,
    goal_horizon: int,
    action_hidden_dim: int,
    action_hidden_layers: int,
    class_weight_samples: int,
    eval_every: int,
    validation_samples: int,
    num_workers: int,
    decoder_streams: int,
    prefetch_depth: int,
    windows_per_block: int,
    window_stride: int,
    device: str,
    feature: str = "predictor_hidden",
) -> Path:
    if min(
        steps, batch_size, context_length, goal_horizon, action_hidden_dim,
        action_hidden_layers, class_weight_samples, eval_every, validation_samples,
        num_workers, decoder_streams, prefetch_depth,
    ) < 1:
        raise ValueError("training and stream sizes must be positive")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    use_cuda = device.startswith("cuda")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GoalConditionedLeWorldModel(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    if feature == "flow_residual" and model.config.prediction_objective != "flow":
        raise ValueError("flow_residual features require a flow checkpoint")
    if context_length > model.config.max_context:
        raise ValueError("context length exceeds the pretrained model's maximum context")
    set_backbone_trainable(model, False)
    model.eval()

    head = DirectPolicyHead(
        model.config.latent_dim,
        256,
        hidden_dim=action_hidden_dim,
        hidden_layers=action_hidden_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)

    catalog = NLDTtyrecGoalBatchStream(
        db_path, dataset_name, batch_size=1, num_workers=1
    )
    gameids = catalog.available_gameids()
    validation_count = max(1, len(gameids) // 5)
    train_ids, validation_ids = gameids[:-validation_count], gameids[-validation_count:]
    if batch_size > len(train_ids):
        raise ValueError("batch size cannot exceed the training game count")

    # Estimate inverse-frequency weights on the training split only. Clipping
    # prevents a key observed once from overwhelming an entire minibatch.
    weight_stream = NLDTtyrecGoalBatchStream(
        db_path,
        dataset_name,
        batch_size=batch_size,
        context_length=context_length,
        goal_horizon=goal_horizon,
        gameids=train_ids,
        num_workers=num_workers,
        windows_per_block=windows_per_block,
        window_stride=window_stride,
        shuffle=True,
        loop_forever=True,
        seed=30_000,
    )
    class_counts = torch.zeros(256, dtype=torch.long)
    counted = 0
    for weight_batch in weight_stream:
        actions = weight_batch.get("action")
        if actions is None:
            raise ValueError("selected dataset does not contain action keypresses")
        class_counts += torch.bincount(actions.long(), minlength=256)
        counted += actions.numel()
        if counted >= class_weight_samples:
            break
    supported = class_counts > 0
    class_weights = torch.zeros(256, dtype=torch.float32)
    class_weights[supported] = counted / (supported.sum() * class_counts[supported].float())
    class_weights.clamp_(max=10.0)
    class_weights[supported] /= class_weights[supported].mean()
    class_weights = class_weights.to(device)

    validation_batch_size = min(batch_size, len(validation_ids))
    validation_stream = NLDTtyrecGoalBatchStream(
        db_path,
        dataset_name,
        batch_size=validation_batch_size,
        context_length=context_length,
        goal_horizon=goal_horizon,
        gameids=validation_ids,
        num_workers=num_workers,
        shuffle=False,
        loop_forever=True,
        seed=20_000,
    )
    validation_batches = []
    validation_seen = 0
    for batch in validation_stream:
        validation_batches.append(move_batch(batch, device, non_blocking=use_cuda))
        validation_seen += int(batch["action"].shape[0])
        if validation_seen >= validation_samples:
            break

    streams = [
        NLDTtyrecGoalBatchStream(
            db_path,
            dataset_name,
            batch_size=batch_size,
            context_length=context_length,
            goal_horizon=goal_horizon,
            gameids=train_ids,
            num_workers=num_workers,
            windows_per_block=windows_per_block,
            window_stride=window_stride,
            shuffle=True,
            loop_forever=True,
            seed=index,
        )
        for index in range(decoder_streams)
    ]
    iterator = prefetch_streams([iter(stream) for stream in streams], prefetch_depth)
    timeline = [{"step": 0, "validation": evaluate(model, head, validation_batches, feature)}]
    started = time.perf_counter()
    data_wait = 0.0

    try:
        for step in range(1, steps + 1):
            wait_started = time.perf_counter()
            batch = move_batch(next(iterator), device, non_blocking=use_cuda)
            data_wait += time.perf_counter() - wait_started
            head.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                features = predictor_features(model, batch, feature)
            logits = head(features)
            loss = F.cross_entropy(logits, batch["action"].long(), weight=class_weights)
            loss.backward()
            optimizer.step()
            if step % eval_every == 0 or step == steps:
                validation = evaluate(model, head, validation_batches, feature)
                entry = {
                    "step": step,
                    "train_loss": loss.detach().item(),
                    "validation": validation,
                }
                timeline.append(entry)
                elapsed = time.perf_counter() - started
                print(
                    f"step {step:6d}/{steps} | train {entry['train_loss']:.4f} | "
                    f"val {validation['loss']:.4f} | acc {validation['accuracy']:.3%} | "
                    f"top5 {validation['top5_accuracy']:.3%} | "
                    f"balanced {validation['balanced_accuracy']:.3%} | "
                    f"{step * batch_size / max(elapsed, 1e-12):.0f} samples/s",
                    flush=True,
                )
    finally:
        iterator.close()

    elapsed = time.perf_counter() - started
    payload = {
        "config": {
            "checkpoint": str(checkpoint_path),
            "pretraining_step": int(checkpoint["step"]),
            "dataset": dataset_name,
            "steps": steps,
            "batch_size": batch_size,
            "context_length": context_length,
            "goal_horizon": goal_horizon,
            "action_hidden_dim": action_hidden_dim,
            "action_hidden_layers": action_hidden_layers,
            "action_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "class_weight_samples": counted,
            "class_counts": class_counts.tolist(),
            "class_weights": class_weights.cpu().tolist(),
            "train_games": len(train_ids),
            "validation_games": len(validation_ids),
            "validation_samples": validation_seen,
            "target": "raw_keycode",
            "num_classes": 256,
            "feature": feature,
            "backbone_frozen": True,
            "elapsed_seconds": elapsed,
            "samples_per_second": steps * batch_size / max(elapsed, 1e-12),
            "data_wait_seconds": data_wait,
        },
        "timeline": timeline,
    }
    torch.save(
        {"head": head.state_dict(), "config": payload["config"]},
        output / "action_checkpoint.pt",
    )
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a frozen predictor-hidden action decoder on NLD ttyrecs"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--db", default="data/nld/nld-aa-taster.db", type=Path)
    parser.add_argument("--dataset", default="nld-aa-taster")
    parser.add_argument("--output", default="reports/nld-action-probe", type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--goal-horizon", type=int, default=64)
    parser.add_argument("--action-hidden-dim", type=int, default=1024)
    parser.add_argument("--action-hidden-layers", type=int, default=2)
    parser.add_argument("--class-weight-samples", type=int, default=100000)
    parser.add_argument(
        "--feature",
        choices=("predictor_hidden", "flow_residual"),
        default="predictor_hidden",
    )
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--validation-samples", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--decoder-streams", type=int, default=2)
    parser.add_argument("--prefetch-depth", type=int, default=16)
    parser.add_argument("--windows-per-block", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    result = run(
        args.checkpoint,
        args.db,
        args.dataset,
        args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        goal_horizon=args.goal_horizon,
        action_hidden_dim=args.action_hidden_dim,
        action_hidden_layers=args.action_hidden_layers,
        class_weight_samples=args.class_weight_samples,
        feature=args.feature,
        eval_every=args.eval_every,
        validation_samples=args.validation_samples,
        num_workers=args.num_workers,
        decoder_streams=args.decoder_streams,
        prefetch_depth=args.prefetch_depth,
        windows_per_block=args.windows_per_block,
        window_stride=args.window_stride,
        device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
