from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .action import DirectPolicyHead
from .config import ModelConfig
from .minihack_data import KeyRoomOracle, make_keyroom_env
from .model import GoalConditionedLeWorldModel
from .nld_action_train import predictor_features
from .report import render_terminal


def _terminal_observation(
    frames: list[tuple[np.ndarray, np.ndarray]],
    *,
    history: bool,
    device: str,
) -> dict[str, torch.Tensor]:
    chars = torch.from_numpy(np.stack([frame[0] for frame in frames])).long()
    colors = torch.from_numpy(np.stack([frame[1] for frame in frames])).long()
    if not history:
        chars = chars[0]
        colors = colors[0]
    return {
        "chars": chars.unsqueeze(0).to(device),
        "colors": colors.unsqueeze(0).to(device),
        "bg_colors": torch.zeros_like(colors).unsqueeze(0).to(device),
    }


def _compose(
    current: np.ndarray,
    current_colors: np.ndarray,
    goal: np.ndarray,
    goal_colors: np.ndarray,
    *,
    step: int,
    action_name: str,
    keycode: int,
    confidence: float,
    reward: float,
    position: tuple[int, int],
    feature: str,
) -> Image.Image:
    left = render_terminal(
        torch.from_numpy(current),
        torch.from_numpy(current_colors),
        "Live simulator observation",
    )
    right = render_terminal(
        torch.from_numpy(goal),
        torch.from_numpy(goal_colors),
        "Fixed goal observation",
    )
    margin = 16
    header = 58
    footer = 82
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
    body = ImageFont.truetype(font_path, 15)
    draw.text(
        (margin, 12),
        "Closed loop · predicted key applied to MiniHack",
        font=title,
        fill="#e7ebf2",
    )
    draw.text(
        (margin, 35),
        f"true observation fed back each step · feature: {feature}",
        font=body,
        fill="#98a3b5",
    )
    y = header + left.height + 12
    draw.text(
        (margin, y),
        f"step {step:03d}   action {action_name:<6} key {keycode:3d}   confidence {confidence:6.2%}",
        font=body,
        fill="#70d6a5",
    )
    draw.text(
        (margin, y + 26),
        f"position {position!s:<12} cumulative reward {reward:+.3f}",
        font=body,
        fill="#e7ebf2",
    )
    return canvas


@torch.no_grad()
def run(
    world_checkpoint: Path,
    action_checkpoint: Path,
    output: Path,
    *,
    max_steps: int,
    fps: int,
    seed: int,
    device: str,
) -> Path:
    if min(max_steps, fps) < 1:
        raise ValueError("max_steps and fps must be positive")

    world_payload = torch.load(world_checkpoint, map_location=device, weights_only=False)
    model = GoalConditionedLeWorldModel(ModelConfig(**world_payload["config"])).to(device)
    model.load_state_dict(world_payload["model"])
    model.eval()
    if model.config.observation_mode != "terminal_rgb":
        raise ValueError("this closed-loop evaluator requires a terminal_rgb world model")

    action_payload = torch.load(action_checkpoint, map_location=device, weights_only=False)
    action_config = action_payload["config"]
    feature = action_config.get("feature", "predictor_hidden")
    head = DirectPolicyHead(
        model.config.latent_dim,
        256,
        hidden_dim=action_config.get("action_hidden_dim", 256),
        hidden_layers=action_config.get("action_hidden_layers", 1),
    ).to(device)
    head.load_state_dict(action_payload["head"])
    head.eval()

    # Generate the goal from a successful oracle rollout in the identical fixed
    # level. The policy receives only its terminal frame, never oracle actions.
    reference_env = make_keyroom_env(seed=seed)
    try:
        reference = KeyRoomOracle(reference_env).collect()
        goal_chars = reference.tty_chars[-1]
        goal_colors = reference.tty_colors[-1]
    finally:
        reference_env.close()

    env = make_keyroom_env(seed=seed)
    action_values = list(env.unwrapped.actions)
    legal_keycodes = torch.tensor(
        [int(value) for value in action_values],
        dtype=torch.long,
        device=device,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    history_frames: deque[tuple[np.ndarray, np.ndarray]] = deque(
        maxlen=model.config.max_context
    )
    noise_generator = torch.Generator(device=device).manual_seed(seed + 90_000)
    total_reward = 0.0
    success = False
    try:
        observation, _ = env.reset()
        with tempfile.TemporaryDirectory() as temporary:
            frame_dir = Path(temporary)
            for step in range(max_steps):
                current_chars = observation["tty_chars"].copy()
                current_colors = observation["tty_colors"].copy()
                history_frames.append((current_chars, current_colors))
                batch = {
                    "history": _terminal_observation(
                        list(history_frames), history=True, device=device
                    ),
                    "goal": _terminal_observation(
                        [(goal_chars, goal_colors)], history=False, device=device
                    ),
                }
                features = predictor_features(
                    model,
                    batch,
                    feature=feature,
                    generator=noise_generator,
                )
                logits = head(features)
                legal_logits = logits.index_select(-1, legal_keycodes)
                legal_probabilities = legal_logits.softmax(-1)
                confidence, prediction = legal_probabilities.max(-1)
                action = int(prediction.item())
                action_value = action_values[action]
                action_name = getattr(action_value, "name", str(action_value))
                keycode = int(action_value)
                position = tuple(map(int, observation["blstats"][:2]))
                image = _compose(
                    current_chars,
                    current_colors,
                    goal_chars,
                    goal_colors,
                    step=step,
                    action_name=action_name,
                    keycode=keycode,
                    confidence=float(confidence.item()),
                    reward=total_reward,
                    position=position,
                    feature=feature,
                )
                image.save(frame_dir / f"frame-{step:05d}.png")
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                records.append(
                    {
                        "step": step,
                        "action_index": action,
                        "action_name": action_name,
                        "raw_keycode": keycode,
                        "legal_action_confidence": float(confidence.item()),
                        "position": position,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                if terminated or truncated:
                    success = bool(total_reward > 0)
                    break
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(frame_dir / "frame-%05d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    str(output),
                ],
                check=True,
            )
    finally:
        env.close()

    summary = {
        "mode": "closed_loop",
        "world_checkpoint": str(world_checkpoint),
        "action_checkpoint": str(action_checkpoint),
        "feature": feature,
        "seed": seed,
        "steps": len(records),
        "total_reward": total_reward,
        "success": success,
        "goal_source": "successful oracle final gameplay frame",
        "legal_raw_keycodes": legal_keycodes.cpu().tolist(),
        "records": records,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an NLD terminal-RGB action head closed-loop in MiniHack"
    )
    parser.add_argument("--world-checkpoint", required=True, type=Path)
    parser.add_argument("--action-checkpoint", required=True, type=Path)
    parser.add_argument("--output", default="reports/nld-minihack-closed-loop.mp4", type=Path)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    print(
        run(
            args.world_checkpoint,
            args.action_checkpoint,
            args.output,
            max_steps=args.max_steps,
            fps=args.fps,
            seed=args.seed,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
