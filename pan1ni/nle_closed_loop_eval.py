from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import tempfile
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import nle.env  # noqa: F401 - importing registers the NetHack environments
import numpy as np
import torch
from minihack.tiles.glyph_mapper import GlyphMapper
from nle import nethack
from PIL import Image, ImageDraw, ImageFont

from .action import DirectPolicyHead, feature_dim, predictor_features
from .config import ModelConfig
from .minihack_report import _render_frame
from .model import GoalConditionedLeWorldModel
from .player_tile_converter import build_canonical_lookup, player_centered_tile_crop
from .player_tile_data import SEMANTIC_ACTION_NAMES, _pixel_observation


MOVEMENT_ACTIONS = tuple(nethack.CompassDirection)
MOVEMENT_DELTAS = (
    (-1, 0), (0, 1), (1, 0), (0, -1),
    (-1, 1), (1, 1), (1, -1), (-1, -1),
)
OBSERVATION_KEYS = ("tty_chars", "tty_colors", "tty_cursor", "blstats", "message")
BLOCKED_TTY_CHARS = frozenset(map(ord, " |-+"))


@dataclass
class TileConverter:
    lookup: np.ndarray
    atlas: np.ndarray

    @classmethod
    def create(cls) -> "TileConverter":
        lookup = build_canonical_lookup()[0]
        mapper = GlyphMapper()
        atlas = np.stack([mapper.tiles[index] for index in range(max(mapper.tiles) + 1)])
        return cls(lookup, atlas)

    def __call__(self, observation: dict) -> np.ndarray:
        return player_centered_tile_crop(
            observation["tty_chars"],
            observation["tty_colors"],
            observation["tty_cursor"],
            self.lookup,
            self.atlas,
        )[0]


@dataclass
class ReferenceGoal:
    seed: int
    initial_tty_chars: np.ndarray
    initial_tty_colors: np.ndarray
    initial_cursor: np.ndarray
    initial_frame: np.ndarray
    initial_position: tuple[int, int, int, int]
    goal_frame: np.ndarray
    goal_position: tuple[int, int, int, int]
    reference_step: int
    reference_action: int
    initial_distance: int


def _position(observation: dict) -> tuple[int, int, int, int]:
    stats = observation["blstats"]
    return int(stats[0]), int(stats[1]), int(stats[23]), int(stats[24])


def _distance(
    position: tuple[int, int, int, int],
    goal: tuple[int, int, int, int],
) -> int:
    if position[2:] != goal[2:]:
        return 1_000
    return max(abs(position[0] - goal[0]), abs(position[1] - goal[1]))


def _make_env(env_id: str, seed: int, max_episode_steps: int):
    env = gym.make(
        env_id,
        observation_keys=OBSERVATION_KEYS,
        actions=MOVEMENT_ACTIONS,
        max_episode_steps=max_episode_steps,
        spawn_monsters=False,
    )
    # NLE 1.3 uses its native core/display seed API rather than Gymnasium's
    # reset(seed=...). Both seeds are required for exact tty reproduction.
    env.unwrapped.seed(core=seed, disp=seed, reseed=True)
    return env


def _reference_action_candidates(
    observation: dict,
    visited: set[tuple[int, int, int, int]],
    rng: random.Random,
) -> list[int]:
    row, column = map(int, observation["tty_cursor"])
    base = _position(observation)
    candidates = []
    for action, (delta_row, delta_column) in enumerate(MOVEMENT_DELTAS):
        target_row = row + delta_row
        target_column = column + delta_column
        if not (1 <= target_row <= 21 and 0 <= target_column < 79):
            continue
        if int(observation["tty_chars"][target_row, target_column]) in BLOCKED_TTY_CHARS:
            continue
        target = (
            base[0] + delta_column,
            base[1] + delta_row,
            base[2],
            base[3],
        )
        candidates.append((target in visited, rng.random(), action))
    candidates.sort()
    return [action for _, _, action in candidates]


def collect_reference_goal(
    env_id: str,
    seed: int,
    converter: TileConverter,
    *,
    goal_horizon: int,
    min_goal_distance: int,
) -> ReferenceGoal | None:
    rng = random.Random(seed + 71_003)
    env = _make_env(env_id, seed, max(goal_horizon + 8, 80))
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray, int, int]] = []
    try:
        observation, _ = env.reset()
        initial_chars = observation["tty_chars"].copy()
        initial_colors = observation["tty_colors"].copy()
        initial_cursor = observation["tty_cursor"].copy()
        initial_frame = converter(observation)
        initial_position = _position(observation)
        visited = {initial_position}
        for step in range(1, goal_horizon + 1):
            actions = _reference_action_candidates(observation, visited, rng)
            action = actions[0] if actions else rng.randrange(8)
            previous = _position(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            position = _position(observation)
            if _distance(previous, position) == 1:
                visited.add(position)
                distance = _distance(initial_position, position)
                if step >= 2 and distance >= min_goal_distance:
                    candidates.append((position, converter(observation), step, action))
            if terminated or truncated:
                break
        if not candidates:
            return None
        goal_position, goal_frame, reference_step, reference_action = rng.choice(candidates)
        return ReferenceGoal(
            seed=seed,
            initial_tty_chars=initial_chars,
            initial_tty_colors=initial_colors,
            initial_cursor=initial_cursor,
            initial_frame=initial_frame,
            initial_position=initial_position,
            goal_frame=goal_frame,
            goal_position=goal_position,
            reference_step=reference_step,
            reference_action=reference_action,
            initial_distance=_distance(initial_position, goal_position),
        )
    finally:
        env.close()


def _compose_video_frame(
    current: np.ndarray,
    goal: np.ndarray,
    *,
    step: int,
    action_name: str,
    confidence: float,
    position: tuple[int, int, int, int],
    goal_position: tuple[int, int, int, int],
    distance: int,
    feature: str,
) -> Image.Image:
    left = _render_frame(current, "Live full-NLE tile observation")
    right = _render_frame(goal, "Reachable reference goal frame")
    margin, header, footer = 16, 62, 86
    canvas = Image.new(
        "RGB",
        (left.width + right.width + margin * 3, header + left.height + footer),
        "#080b11",
    )
    canvas.paste(left, (margin, header))
    canvas.paste(right, (left.width + margin * 2, header))
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    title = ImageFont.truetype(font_path, 17)
    body = ImageFont.truetype(font_path, 14)
    draw.text((margin, 12), "Closed loop · full procedural NLE", font=title, fill="#e7ebf2")
    draw.text(
        (margin, 36),
        f"same tty→tile conversion as human pretraining · feature: {feature}",
        font=body,
        fill="#98a3b5",
    )
    y = header + left.height + 12
    draw.text(
        (margin, y),
        f"step {step:03d}   action {action_name:<10} confidence {confidence:6.2%}",
        font=body,
        fill="#70d6a5",
    )
    draw.text(
        (margin, y + 25),
        f"position {position[:2]!s:<12} goal {goal_position[:2]!s:<12} distance {distance}",
        font=body,
        fill="#e7ebf2",
    )
    return canvas


def _verify_same_reset(observation: dict, reference: ReferenceGoal) -> None:
    if not (
        np.array_equal(observation["tty_chars"], reference.initial_tty_chars)
        and np.array_equal(observation["tty_colors"], reference.initial_tty_colors)
        and np.array_equal(observation["tty_cursor"], reference.initial_cursor)
    ):
        raise RuntimeError(f"NLE seed {reference.seed} did not reproduce the reference reset")


@torch.no_grad()
def evaluate_episode(
    model: GoalConditionedLeWorldModel,
    head: DirectPolicyHead,
    feature: str,
    reference: ReferenceGoal,
    converter: TileConverter,
    *,
    env_id: str,
    max_steps: int,
    device: str,
    action_selection: str,
    temperature: float,
    video_output: Path | None = None,
    fps: int = 8,
) -> dict:
    env = _make_env(env_id, reference.seed, max_steps + 8)
    history: deque[np.ndarray] = deque(maxlen=model.config.max_context)
    positions: list[tuple[int, int, int, int]] = []
    action_counts: Counter[int] = Counter()
    collisions = 0
    best_distance = reference.initial_distance
    terminated = truncated = False
    frames_dir: tempfile.TemporaryDirectory[str] | None = None
    generator = torch.Generator(device=device).manual_seed(reference.seed + 91_117)
    try:
        observation, _ = env.reset()
        _verify_same_reset(observation, reference)
        history.extend(reference.initial_frame.copy() for _ in range(model.config.max_context))
        positions.append(_position(observation))
        if video_output is not None:
            frames_dir = tempfile.TemporaryDirectory()
        success = False
        for step in range(max_steps):
            current = converter(observation)
            if step:
                history.append(current)
            position = _position(observation)
            distance = _distance(position, reference.goal_position)
            best_distance = min(best_distance, distance)
            batch = {
                "history": _pixel_observation(list(history), history=True, device=device),
                "goal": _pixel_observation([reference.goal_frame], history=False, device=device),
            }
            logits = head(predictor_features(model, batch, feature))[:, :8]
            probabilities = (logits / temperature).softmax(-1)
            if action_selection == "sample":
                prediction = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
                confidence = probabilities.gather(-1, prediction[:, None]).squeeze(-1)
            else:
                confidence, prediction = probabilities.max(-1)
            action = int(prediction.item())
            action_counts[action] += 1
            if frames_dir is not None:
                _compose_video_frame(
                    current,
                    reference.goal_frame,
                    step=step,
                    action_name=SEMANTIC_ACTION_NAMES[action],
                    confidence=float(confidence.item()),
                    position=position,
                    goal_position=reference.goal_position,
                    distance=distance,
                    feature=feature,
                ).save(Path(frames_dir.name) / f"frame-{step:05d}.png")
            previous = position
            observation, _, terminated, truncated, _ = env.step(action)
            position = _position(observation)
            positions.append(position)
            if position == previous:
                collisions += 1
            distance = _distance(position, reference.goal_position)
            best_distance = min(best_distance, distance)
            if position == reference.goal_position:
                success = True
                break
            if terminated or truncated:
                break
        steps_taken = step + 1
        if frames_dir is not None and video_output is not None:
            video_output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(Path(frames_dir.name) / "frame-%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    str(video_output),
                ],
                check=True,
            )
        initial_distance = reference.initial_distance
        final_distance = _distance(positions[-1], reference.goal_position)
        revisits = max(0, len(positions) - len(set(positions)))
        return {
            "seed": reference.seed,
            "success": success,
            "steps": steps_taken,
            "initial_position": reference.initial_position,
            "goal_position": reference.goal_position,
            "final_position": positions[-1],
            "reference_step": reference.reference_step,
            "initial_distance": initial_distance,
            "best_distance": best_distance,
            "final_distance": final_distance,
            "best_progress": (initial_distance - best_distance) / max(initial_distance, 1),
            "final_progress": (initial_distance - final_distance) / max(initial_distance, 1),
            "unique_positions": len(set(positions)),
            "revisit_rate": revisits / max(len(positions) - 1, 1),
            "collisions": collisions,
            "collision_rate": collisions / max(steps_taken, 1),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "action_counts": {str(key): value for key, value in sorted(action_counts.items())},
        }
    finally:
        env.close()
        if frames_dir is not None:
            frames_dir.cleanup()


@torch.no_grad()
def run(
    world_checkpoint: Path,
    action_checkpoint: Path,
    output: Path,
    *,
    env_id: str,
    episodes: int,
    max_steps: int,
    goal_horizon: int,
    min_goal_distance: int,
    seed: int,
    device: str,
    video_output: Path | None,
    fps: int,
    action_selection: str,
    temperature: float,
) -> Path:
    if min(episodes, max_steps, goal_horizon, min_goal_distance, fps) < 1:
        raise ValueError("evaluation sizes must be positive")
    if action_selection not in {"argmax", "sample"}:
        raise ValueError("action_selection must be 'argmax' or 'sample'")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    world_payload = torch.load(world_checkpoint, map_location=device, weights_only=False)
    model = GoalConditionedLeWorldModel(ModelConfig(**world_payload["config"])).to(device)
    if model.config.observation_mode != "pixels":
        raise ValueError("full-NLE evaluation requires the tile-pixel world model")
    model.load_state_dict(world_payload["model"])
    model.eval()
    action_payload = torch.load(action_checkpoint, map_location=device, weights_only=False)
    action_config = action_payload["config"]
    feature = action_config["feature"]
    if int(action_config.get("num_classes", 10)) < 8:
        raise ValueError("action head must expose all eight semantic movement classes")
    head = DirectPolicyHead(
        feature_dim(feature, model.config.latent_dim),
        int(action_config.get("num_classes", 10)),
        hidden_dim=int(action_config.get("action_hidden_dim", 1024)),
        hidden_layers=int(action_config.get("action_hidden_layers", 2)),
    ).to(device)
    head.load_state_dict(action_payload["head"])
    head.eval()

    converter = TileConverter.create()
    references: list[ReferenceGoal] = []
    attempted_seed = seed
    while len(references) < episodes:
        reference = collect_reference_goal(
            env_id,
            attempted_seed,
            converter,
            goal_horizon=goal_horizon,
            min_goal_distance=min_goal_distance,
        )
        if reference is not None:
            references.append(reference)
        attempted_seed += 1
        if attempted_seed - seed > episodes * 20:
            raise RuntimeError("could not generate enough reachable procedural NLE goals")

    records = []
    for episode, reference in enumerate(references):
        record = evaluate_episode(
            model,
            head,
            feature,
            reference,
            converter,
            env_id=env_id,
            max_steps=max_steps,
            device=device,
            action_selection=action_selection,
            temperature=temperature,
            video_output=video_output if episode == 0 else None,
            fps=fps,
        )
        record["episode"] = episode
        records.append(record)
        print(
            f"episode {episode + 1:3d}/{episodes} | seed {reference.seed} | "
            f"success {record['success']} | best progress {record['best_progress']:.1%} | "
            f"unique {record['unique_positions']}",
            flush=True,
        )

    successful_steps = [record["steps"] for record in records if record["success"]]
    total_steps = sum(record["steps"] for record in records)
    summary = {
        "mode": "closed_loop",
        "environment": env_id,
        "representation": "full NLE tty converted by the human-pretraining tile pipeline",
        "spawn_monsters": False,
        "world_checkpoint": str(world_checkpoint),
        "action_checkpoint": str(action_checkpoint),
        "feature": feature,
        "action_policy": (
            f"{action_selection} over eight human-derived semantic movement classes; "
            f"pickup/apply masked; temperature {temperature:g}"
        ),
        "action_selection": action_selection,
        "temperature": temperature,
        "episodes": episodes,
        "successes": len(successful_steps),
        "success_rate": len(successful_steps) / episodes,
        "median_success_steps": statistics.median(successful_steps) if successful_steps else None,
        "mean_success_steps": statistics.mean(successful_steps) if successful_steps else None,
        "mean_best_progress": statistics.mean(record["best_progress"] for record in records),
        "mean_final_progress": statistics.mean(record["final_progress"] for record in records),
        "mean_unique_positions": statistics.mean(record["unique_positions"] for record in records),
        "collision_rate": sum(record["collisions"] for record in records) / max(total_steps, 1),
        "mean_revisit_rate": statistics.mean(record["revisit_rate"] for record in records),
        "max_steps": max_steps,
        "goal_horizon": goal_horizon,
        "min_goal_distance": min_goal_distance,
        "requested_seed": seed,
        "generated_through_seed": attempted_seed - 1,
        "video": str(video_output) if video_output is not None else None,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a human-movement head in full procedural NLE"
    )
    parser.add_argument("--world-checkpoint", required=True, type=Path)
    parser.add_argument("--action-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--env-id", default="NetHackScore-v0")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--goal-horizon", type=int, default=64)
    parser.add_argument("--min-goal-distance", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--action-selection", choices=("argmax", "sample"), default="argmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(
        run(
            args.world_checkpoint,
            args.action_checkpoint,
            args.output,
            env_id=args.env_id,
            episodes=args.episodes,
            max_steps=args.max_steps,
            goal_horizon=args.goal_horizon,
            min_goal_distance=args.min_goal_distance,
            seed=args.seed,
            device=args.device,
            video_output=args.video_output,
            fps=args.fps,
            action_selection=args.action_selection,
            temperature=args.temperature,
        )
    )


if __name__ == "__main__":
    main()
