from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .model import GoalConditionedLeWorldModel
from .nld_data import NLDTtyrecGoalBatchStream, prefetch_batches
from .report import diagnose, report_model_config
from .train import move_batch, pretrain_step


def _save_checkpoint(
    output: Path,
    model: GoalConditionedLeWorldModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    timeline: list[dict],
    config: dict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": asdict(report_model_config()),
        },
        output / "checkpoint.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps({"config": config, "timeline": timeline}, indent=2),
        encoding="utf-8",
    )


def run(
    db_path: Path,
    dataset_name: str,
    output: Path,
    *,
    steps: int,
    batch_size: int,
    context_length: int,
    goal_horizon: int,
    sigreg_slices: int,
    eval_every: int,
    checkpoint_every: int,
    validation_samples: int,
    num_workers: int,
    prefetch_depth: int,
    windows_per_block: int,
    window_stride: int,
    device: str,
) -> Path:
    if min(steps, batch_size, eval_every, checkpoint_every, validation_samples) < 1:
        raise ValueError("steps, batch size, intervals, and validation samples must be positive")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    use_cuda = device.startswith("cuda")

    catalog = NLDTtyrecGoalBatchStream(
        db_path, dataset_name, batch_size=1, num_workers=1
    )
    gameids = catalog.available_gameids()
    validation_count = max(1, len(gameids) // 5)
    train_ids, validation_ids = gameids[:-validation_count], gameids[-validation_count:]
    if batch_size > len(train_ids):
        raise ValueError("batch size cannot exceed the training game count")
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
        validation_batches.append(
            move_batch(batch, device, non_blocking=use_cuda)
        )
        validation_seen += int(batch["trajectory_id"].shape[0])
        if validation_seen >= validation_samples:
            break

    train_stream = NLDTtyrecGoalBatchStream(
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
    )
    model = GoalConditionedLeWorldModel(report_model_config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    timeline = [{"step": 0, "diagnostics": diagnose(model, validation_batches)}]
    config = {
        "db": str(db_path),
        "dataset": dataset_name,
        "steps": steps,
        "batch_size": batch_size,
        "context_length": context_length,
        "goal_horizon": goal_horizon,
        "sigreg_slices": sigreg_slices,
        "eval_every": eval_every,
        "checkpoint_every": checkpoint_every,
        "validation_samples": validation_seen,
        "num_workers": num_workers,
        "prefetch_depth": prefetch_depth,
        "windows_per_block": windows_per_block,
        "window_stride": window_stride,
        "device": device,
        "train_games": len(train_ids),
        "validation_games": len(validation_ids),
    }

    started = time.perf_counter()
    train_iterator = prefetch_batches(iter(train_stream), prefetch_depth)
    for step in range(1, steps + 1):
        batch = move_batch(next(train_iterator), device, non_blocking=use_cuda)
        metrics = pretrain_step(
            model, batch, optimizer, sigreg_slices=sigreg_slices
        )
        if step % eval_every == 0 or step == steps:
            entry = {
                "step": step,
                "train_prediction_loss": metrics["prediction_loss"],
                "train_sigreg_loss": metrics["sigreg_loss"],
                "train_total_loss": metrics["loss"],
                "diagnostics": diagnose(model, validation_batches),
            }
            timeline.append(entry)
            diagnostic = entry["diagnostics"]
            print(
                f"step {step:6d}/{steps} | train {metrics['prediction_loss']:.4f} | "
                f"held-out {diagnostic['prediction_loss']:.4f} | "
                f"goal/shuffle {diagnostic['correct_vs_shuffled']:.3f} | "
                f"rank {diagnostic['latent_effective_rank']:.2f}",
                flush=True,
            )
        if step % checkpoint_every == 0 or step == steps:
            _save_checkpoint(output, model, optimizer, step, timeline, config)
    train_iterator.close()

    if use_cuda:
        torch.cuda.synchronize()
    config["elapsed_seconds"] = time.perf_counter() - started
    config["samples_per_second"] = (
        steps * batch_size / max(config["elapsed_seconds"], 1e-12)
    )
    _save_checkpoint(output, model, optimizer, steps, timeline, config)
    return output / "metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train directly from official NLD ttyrec archives"
    )
    parser.add_argument("--db", default="data/nld/nld-aa-taster.db")
    parser.add_argument("--dataset", default="nld-aa-taster")
    parser.add_argument("--output", default="reports/nld-aa-ttyrec")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=1)
    parser.add_argument("--goal-horizon", type=int, default=64)
    parser.add_argument("--sigreg-slices", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument("--windows-per-block", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    result = run(
        Path(args.db),
        args.dataset,
        Path(args.output),
        steps=args.steps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        goal_horizon=args.goal_horizon,
        sigreg_slices=args.sigreg_slices,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        validation_samples=args.validation_samples,
        num_workers=args.num_workers,
        prefetch_depth=args.prefetch_depth,
        windows_per_block=args.windows_per_block,
        window_stride=args.window_stride,
        device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
