from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from pan1ni.models.config import ModelConfig
from pan1ni.models.model import GoalConditionedLeWorldModel
from pan1ni.data.nld import NLDTtyrecGoalBatchStream, prefetch_streams
from pan1ni.reporting.report import diagnose, report_model_config
from pan1ni.training.primitives import move_batch, pretrain_step


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
            "config": asdict(model.config),
        },
        output / "checkpoint.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps({"config": config, "timeline": timeline}, indent=2),
        encoding="utf-8",
    )


def nld_model_config(scale: str) -> ModelConfig:
    if scale == "small":
        return report_model_config()
    if scale == "medium":
        return ModelConfig(
            latent_dim=128, cell_dim=64, message_dim=32, hidden_dim=128,
            vit_dim=64, vit_layers=2, vit_heads=8, terminal_patch_size=2,
            projector_hidden_dim=256, predictor_layers=2, predictor_heads=8,
            max_context=8, num_actions=256, dropout=0.0,
        )
    if scale == "large":
        return ModelConfig(
            observation_mode="terminal_rgb",
            latent_dim=64, cell_dim=64, message_dim=64, hidden_dim=64,
            vit_dim=320, vit_layers=8, vit_heads=8, terminal_patch_size=2,
            pixel_patch_size=32, projector_hidden_dim=512,
            predictor_layers=4, predictor_heads=8,
            max_context=8, num_actions=256, dropout=0.0,
        )
    raise ValueError(f"unknown NLD model scale: {scale}")


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
    decoder_streams: int,
    prefetch_depth: int,
    windows_per_block: int,
    window_stride: int,
    device: str,
    model_scale: str = "small",
    objective: str = "mse",
    resume: Path | None = None,
) -> Path:
    if min(steps, batch_size, eval_every, checkpoint_every, validation_samples, decoder_streams) < 1:
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

    train_streams = [
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
            seed=stream_index,
        )
        for stream_index in range(decoder_streams)
    ]
    model_config = replace(nld_model_config(model_scale), prediction_objective=objective)
    model = GoalConditionedLeWorldModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    start_step = 0
    timeline = [{"step": 0, "diagnostics": diagnose(model, validation_batches, objective=objective)}]
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        metrics_path = resume.with_name("metrics.json")
        if metrics_path.exists():
            timeline = json.loads(metrics_path.read_text(encoding="utf-8"))["timeline"]
        if start_step >= steps:
            raise ValueError("resume checkpoint is already at or beyond requested steps")
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
        "decoder_streams": decoder_streams,
        "prefetch_depth": prefetch_depth,
        "windows_per_block": windows_per_block,
        "window_stride": window_stride,
        "device": device,
        "model_scale": model_scale,
        "objective": objective,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_games": len(train_ids),
        "validation_games": len(validation_ids),
    }

    started = time.perf_counter()
    data_wait_seconds = 0.0
    train_seconds = 0.0
    train_iterator = prefetch_streams(
        [iter(stream) for stream in train_streams], prefetch_depth
    )
    for step in range(start_step + 1, steps + 1):
        wait_started = time.perf_counter()
        host_batch = next(train_iterator)
        data_wait_seconds += time.perf_counter() - wait_started
        train_started = time.perf_counter()
        batch = move_batch(host_batch, device, non_blocking=use_cuda)
        metrics = pretrain_step(
            model,
            batch,
            optimizer,
            sigreg_slices=sigreg_slices,
            objective=objective,
        )
        train_seconds += time.perf_counter() - train_started
        if step % eval_every == 0 or step == steps:
            entry = {
                "step": step,
                "train_prediction_loss": metrics["prediction_loss"],
                "train_sigreg_loss": metrics["sigreg_loss"],
                "train_total_loss": metrics["loss"],
                "diagnostics": diagnose(model, validation_batches, objective=objective),
            }
            timeline.append(entry)
            diagnostic = entry["diagnostics"]
            print(
                f"step {step:6d}/{steps} | train {metrics['prediction_loss']:.4f} | "
                f"held-out {diagnostic['prediction_loss']:.4f} | "
                f"goal/shuffle {diagnostic['correct_vs_shuffled']:.3f} | "
                f"rank {diagnostic['latent_effective_rank']:.2f} | "
                f"wait {data_wait_seconds / max(data_wait_seconds + train_seconds, 1e-12):.1%} | "
                f"{(step - start_step) * batch_size / max(time.perf_counter() - started, 1e-12):.0f} samples/s",
                flush=True,
            )
        if step % checkpoint_every == 0 or step == steps:
            _save_checkpoint(output, model, optimizer, step, timeline, config)
    train_iterator.close()

    if use_cuda:
        torch.cuda.synchronize()
    config["elapsed_seconds"] = time.perf_counter() - started
    config["samples_per_second"] = (
        (steps - start_step) * batch_size / max(config["elapsed_seconds"], 1e-12)
    )
    config["data_wait_seconds"] = data_wait_seconds
    config["train_seconds"] = train_seconds
    config["resumed_from_step"] = start_step
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
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--goal-horizon", type=int, default=64)
    parser.add_argument("--sigreg-slices", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--decoder-streams", type=int, default=1)
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument("--windows-per-block", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--model-scale", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--objective", choices=("mse", "flow"), default="mse")
    parser.add_argument("--resume", type=Path)
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
        decoder_streams=args.decoder_streams,
        prefetch_depth=args.prefetch_depth,
        windows_per_block=args.windows_per_block,
        window_stride=args.window_stride,
        device=args.device,
        model_scale=args.model_scale,
        objective=args.objective,
        resume=args.resume,
    )
    print(result)


if __name__ == "__main__":
    main()
