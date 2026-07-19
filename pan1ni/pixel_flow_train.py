from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
from minihack.tiles.glyph_mapper import GlyphMapper
from torch.utils.data import DataLoader

from .config import ModelConfig
from .minihack_data import MiniHackPixelGoalDataset
from .model import GoalConditionedLeWorldModel
from .nld_data import prefetch_batches
from .player_tile_converter import build_canonical_lookup
from .player_tile_data import NLDPlayerTileGoalBatchStream
from .report import diagnose
from .train import move_batch, pretrain_step


def pixel_model_config(objective: str = "flow") -> ModelConfig:
    return ModelConfig(
        observation_mode="pixels",
        latent_dim=64,
        hidden_dim=64,
        vit_dim=320,
        vit_layers=8,
        vit_heads=8,
        pixel_patch_size=16,
        max_patches=128,
        projector_hidden_dim=512,
        predictor_layers=4,
        predictor_heads=8,
        max_context=8,
        num_actions=10,
        dropout=0.0,
        prediction_objective=objective,
    )


def _combine(player: dict, simulator: dict) -> dict:
    result = {
        name: {
            "pixels": torch.cat(
                (player[name]["pixels"], simulator[name]["pixels"]), dim=0
            )
        }
        for name in ("history", "target", "goal")
    }
    result["goal_offset"] = torch.cat(
        (player["goal_offset"], simulator["goal_offset"]), dim=0
    )
    result["source"] = torch.cat(
        (
            torch.zeros(player["goal_offset"].shape[0], dtype=torch.long),
            torch.ones(simulator["goal_offset"].shape[0], dtype=torch.long),
        )
    )
    return result


def _slice_batch(batch: dict, size: int) -> dict:
    return {
        key: (
            {name: tensor[:size] for name, tensor in value.items()}
            if isinstance(value, dict)
            else value[:size]
        )
        for key, value in batch.items()
    }


def _save(
    output: Path,
    model: GoalConditionedLeWorldModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    timeline: list[dict],
    config: dict,
) -> None:
    if model.config.observation_mode != "pixels":
        raise RuntimeError("refusing to checkpoint a non-pixel model")
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


def run(
    player_db: Path,
    player_dataset: str,
    simulator_data: Path,
    output: Path,
    *,
    steps: int,
    player_batch_size: int,
    player_decode_batch_size: int,
    simulator_batch_size: int,
    context_length: int,
    goal_horizon: int,
    objective: str,
    sigreg_weight: float,
    sigreg_slices: int,
    eval_every: int,
    checkpoint_every: int,
    player_validation_samples: int,
    simulator_validation_samples: int,
    num_workers: int,
    prefetch_depth: int,
    windows_per_block: int,
    window_stride: int,
    device: str,
    resume: Path | None = None,
) -> Path:
    if context_length != 8:
        raise ValueError("the approved pixel experiment uses exactly eight context frames")
    if objective not in {"mse", "flow"}:
        raise ValueError("objective must be 'mse' or 'flow'")
    if sigreg_weight <= 0:
        raise ValueError("SIGReg weight must be positive")
    if min(
        steps,
        player_batch_size,
        player_decode_batch_size,
        sigreg_slices,
        eval_every,
        checkpoint_every,
        num_workers,
        prefetch_depth,
    ) < 1:
        raise ValueError("training sizes and intervals must be positive")
    if simulator_batch_size < 0:
        raise ValueError("simulator batch size cannot be negative")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    use_cuda = device.startswith("cuda")

    lookup, conversion_metadata = build_canonical_lookup()
    mapper = GlyphMapper()
    tile_atlas = np.stack(
        [mapper.tiles[index] for index in range(max(mapper.tiles) + 1)]
    )
    catalog = NLDPlayerTileGoalBatchStream(
        player_db,
        player_dataset,
        batch_size=1,
        lookup=lookup,
        tile_atlas=tile_atlas,
    )
    player_gameids = catalog.available_gameids()
    player_validation_count = max(1, len(player_gameids) // 5)
    player_train_ids = player_gameids[:-player_validation_count]
    player_validation_ids = player_gameids[-player_validation_count:]

    with h5py.File(simulator_data, "r") as handle:
        simulator_keys = sorted(handle.keys(), key=int)
    simulator_validation_keys = simulator_keys[::5]
    simulator_validation_set = set(simulator_validation_keys)
    simulator_train_keys = [
        key for key in simulator_keys if key not in simulator_validation_set
    ]

    player_validation_stream = NLDPlayerTileGoalBatchStream(
        player_db,
        player_dataset,
        batch_size=min(64, len(player_validation_ids)),
        context_length=context_length,
        goal_horizon=goal_horizon,
        gameids=player_validation_ids,
        num_workers=min(num_workers, 8),
        windows_per_block=2,
        window_stride=window_stride,
        shuffle=False,
        loop_forever=True,
        seed=20_000,
        lookup=lookup,
        tile_atlas=tile_atlas,
    )
    player_validation_host = []
    player_seen = 0
    for batch in player_validation_stream:
        player_validation_host.append(batch)
        player_seen += batch["goal_offset"].numel()
        if player_seen >= player_validation_samples:
            break

    simulator_validation_data = MiniHackPixelGoalDataset(
        simulator_data,
        episode_keys=simulator_validation_keys,
        context_length=context_length,
        goal_horizon=goal_horizon,
        random_future_goal=True,
        pad_initial_context=True,
        samples_per_epoch=simulator_validation_samples,
        seed=30_000,
    )
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers:
        loader_options.update(prefetch_factor=2, persistent_workers=True)
    simulator_validation_host = list(
        DataLoader(
            simulator_validation_data,
            batch_size=16,
            **loader_options,
        )
    )

    model_config = pixel_model_config(objective)
    if model_config.observation_mode != "pixels":
        raise RuntimeError("pixel-flow training requires native tile pixels")
    model = GoalConditionedLeWorldModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    start_step = 0
    timeline: list[dict] = []
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if checkpoint["config"].get("observation_mode") != "pixels":
            raise ValueError("refusing to resume a non-pixel checkpoint")
        if checkpoint["config"].get("prediction_objective") != objective:
            raise ValueError(f"refusing to resume a non-{objective} checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        metrics_path = resume.with_name("metrics.json")
        if metrics_path.exists():
            timeline = json.loads(metrics_path.read_text(encoding="utf-8"))["timeline"]

    player_validation_batches = [
        move_batch(batch, device, non_blocking=use_cuda)
        for batch in player_validation_host
    ]
    simulator_validation_batches = [
        move_batch(batch, device, non_blocking=use_cuda)
        for batch in simulator_validation_host
    ]
    if not timeline:
        timeline = [
            {
                "step": 0,
                "player_diagnostics": diagnose(
                    model, player_validation_batches, objective=objective
                ),
                "simulator_diagnostics": diagnose(
                    model, simulator_validation_batches, objective=objective
                ),
            }
        ]

    player_train_stream = NLDPlayerTileGoalBatchStream(
        player_db,
        player_dataset,
        batch_size=player_decode_batch_size,
        context_length=context_length,
        goal_horizon=goal_horizon,
        gameids=player_train_ids,
        num_workers=num_workers,
        windows_per_block=windows_per_block,
        window_stride=window_stride,
        shuffle=True,
        loop_forever=True,
        seed=0,
        lookup=lookup,
        tile_atlas=tile_atlas,
    )
    player_iterator = prefetch_batches(iter(player_train_stream), prefetch_depth)
    simulator_iterator = None
    if simulator_batch_size:
        simulator_train_data = MiniHackPixelGoalDataset(
            simulator_data,
            episode_keys=simulator_train_keys,
            context_length=context_length,
            goal_horizon=goal_horizon,
            random_future_goal=True,
            pad_initial_context=True,
            samples_per_epoch=max(steps - start_step, 1) * simulator_batch_size,
            seed=start_step * simulator_batch_size,
        )
        simulator_iterator = iter(
            DataLoader(
                simulator_train_data,
                batch_size=simulator_batch_size,
                **loader_options,
            )
        )
    config = {
        "player_db": str(player_db),
        "player_dataset": player_dataset,
        "simulator_data": str(simulator_data),
        "representation": "native/canonical MiniHack 16x16 tile atlas; 9x9 crop; uint8 RGB",
        "observation_shape": [3, 144, 144],
        "objective": objective,
        "one_step_inference": True,
        "steps": steps,
        "player_batch_size": player_batch_size,
        "player_decode_batch_size": player_decode_batch_size,
        "simulator_batch_size": simulator_batch_size,
        "training_sources": ["player"] if not simulator_batch_size else ["player", "simulator"],
        "context_length": context_length,
        "goal_horizon": goal_horizon,
        "random_future_goal": True,
        "sigreg_weight": sigreg_weight,
        "sigreg_slices": sigreg_slices,
        "learning_rate": 3e-4,
        "eval_every": eval_every,
        "checkpoint_every": checkpoint_every,
        "player_train_games": len(player_train_ids),
        "player_validation_games": len(player_validation_ids),
        "simulator_train_episodes": len(simulator_train_keys) if simulator_batch_size else 0,
        "simulator_validation_episodes": len(simulator_validation_keys),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "encoder_parameters": sum(parameter.numel() for parameter in model.encoder.parameters()),
        "predictor_parameters": sum(parameter.numel() for parameter in model.predictor.parameters()),
        "conversion_empirical_pairs": conversion_metadata["empirical_pairs"],
        "conversion_keyroom_pairs": conversion_metadata["keyroom_calibration_pairs"],
    }

    started = time.perf_counter()
    data_wait_seconds = 0.0
    trained_samples = 0
    try:
        for step in range(start_step + 1, steps + 1):
            wait_started = time.perf_counter()
            player_batch = next(player_iterator)
            if player_batch["goal_offset"].numel() > player_batch_size:
                player_batch = _slice_batch(player_batch, player_batch_size)
            if simulator_iterator is None:
                host_batch = player_batch
            else:
                simulator_batch = next(simulator_iterator)
                host_batch = _combine(player_batch, simulator_batch)
            data_wait_seconds += time.perf_counter() - wait_started
            trained_samples += host_batch["goal_offset"].numel()
            batch = move_batch(host_batch, device, non_blocking=use_cuda)
            metrics = pretrain_step(
                model,
                batch,
                optimizer,
                sigreg_weight=sigreg_weight,
                sigreg_slices=sigreg_slices,
                objective=objective,
            )
            if step % eval_every == 0 or step == steps:
                player_diagnostics = diagnose(
                    model, player_validation_batches, objective=objective
                )
                simulator_diagnostics = diagnose(
                    model, simulator_validation_batches, objective=objective
                )
                entry = {
                    "step": step,
                    "train_prediction_loss": metrics["prediction_loss"],
                    f"train_{objective}_loss": metrics["prediction_loss"],
                    "train_sigreg_loss": metrics["sigreg_loss"],
                    "train_total_loss": metrics["loss"],
                    "player_diagnostics": player_diagnostics,
                    "simulator_diagnostics": simulator_diagnostics,
                }
                timeline.append(entry)
                elapsed = time.perf_counter() - started
                print(
                    f"step {step:6d}/{steps} | {objective} {metrics['prediction_loss']:.4f} | "
                    f"player {player_diagnostics['prediction_loss']:.4f} | "
                    f"sim {simulator_diagnostics['prediction_loss']:.4f} | "
                    f"sim goal/shuffle {simulator_diagnostics['correct_vs_shuffled']:.3f} | "
                    f"rank {simulator_diagnostics['latent_effective_rank']:.2f} | "
                    f"{trained_samples / max(elapsed, 1e-12):.0f} samples/s",
                    flush=True,
                )
            if step % checkpoint_every == 0 or step == steps:
                _save(output, model, optimizer, step, timeline, config)
    finally:
        close = getattr(player_iterator, "close", None)
        if close is not None:
            close()

    config["elapsed_seconds"] = time.perf_counter() - started
    config["trained_samples"] = trained_samples
    config["samples_per_second"] = trained_samples / max(config["elapsed_seconds"], 1e-12)
    config["data_wait_seconds"] = data_wait_seconds
    config["resumed_from_step"] = start_step
    _save(output, model, optimizer, steps, timeline, config)
    return output / "metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the approved human+MiniHack native tile-pixel world model"
    )
    parser.add_argument("--player-db", type=Path, default=Path("data/nld/nld-nao.db"))
    parser.add_argument("--player-dataset", default="nld-nao-human-8shard")
    parser.add_argument(
        "--simulator-data",
        type=Path,
        default=Path("data/minihack/keyroom-rgb-1600.hdf5"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/pixel-flow-human-keyroom-large"))
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--player-batch-size", type=int, default=48)
    parser.add_argument("--player-decode-batch-size", type=int, default=128)
    parser.add_argument("--simulator-batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--goal-horizon", type=int, default=64)
    parser.add_argument("--objective", choices=("mse", "flow"), default="flow")
    parser.add_argument("--sigreg-weight", type=float, default=0.1)
    parser.add_argument("--sigreg-slices", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--player-validation-samples", type=int, default=128)
    parser.add_argument("--simulator-validation-samples", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-depth", type=int, default=3)
    parser.add_argument("--windows-per-block", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(
        run(
            args.player_db,
            args.player_dataset,
            args.simulator_data,
            args.output,
            steps=args.steps,
            player_batch_size=args.player_batch_size,
            player_decode_batch_size=args.player_decode_batch_size,
            simulator_batch_size=args.simulator_batch_size,
            context_length=args.context_length,
            goal_horizon=args.goal_horizon,
            objective=args.objective,
            sigreg_weight=args.sigreg_weight,
            sigreg_slices=args.sigreg_slices,
            eval_every=args.eval_every,
            checkpoint_every=args.checkpoint_every,
            player_validation_samples=args.player_validation_samples,
            simulator_validation_samples=args.simulator_validation_samples,
            num_workers=args.num_workers,
            prefetch_depth=args.prefetch_depth,
            windows_per_block=args.windows_per_block,
            window_stride=args.window_stride,
            device=args.device,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
