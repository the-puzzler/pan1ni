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
from .minihack_report import _render_frame
from .model import GoalConditionedLeWorldModel
from .nld_action_train import predictor_features


def _pixel_observation(
    frames: list[np.ndarray], *, history: bool, device: str
) -> dict[str, torch.Tensor]:
    value = torch.from_numpy(np.stack(frames)).movedim(-1, -3).contiguous()
    if not history:
        value = value[0]
    return {"pixels": value.unsqueeze(0).to(device)}


def _compose(
    current: np.ndarray,
    goal: np.ndarray,
    *,
    step: int,
    action_name: str,
    confidence: float,
    reward: float,
    position: tuple[int, int],
    feature: str,
) -> Image.Image:
    left = _render_frame(current, "Live native pixel_crop observation")
    right = _render_frame(goal, "Fixed native pixel_crop goal")
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
        "Closed loop · predicted action applied to MiniHack",
        font=title,
        fill="#e7ebf2",
    )
    draw.text(
        (margin, 35),
        f"native tile pixels fed back each step · feature: {feature}",
        font=body,
        fill="#98a3b5",
    )
    y = header + left.height + 12
    draw.text(
        (margin, y),
        f"step {step:03d}   action {action_name:<7} confidence {confidence:6.2%}",
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
    world_payload = torch.load(world_checkpoint, map_location=device, weights_only=False)
    model = GoalConditionedLeWorldModel(ModelConfig(**world_payload["config"])).to(device)
    if model.config.observation_mode != "pixels":
        raise ValueError("closed-loop evaluation requires a native tile-pixel model")
    model.load_state_dict(world_payload["model"])
    model.eval()
    action_payload = torch.load(action_checkpoint, map_location=device, weights_only=False)
    action_config = action_payload["config"]
    feature = action_config.get("feature", "flow_residual")
    num_actions = int(action_config.get("num_classes", 10))
    raw_keycode_actions = action_config.get("target") == "raw_keycode"
    head = DirectPolicyHead(
        model.config.latent_dim,
        num_actions,
        hidden_dim=action_config.get("action_hidden_dim", 1024),
        hidden_layers=action_config.get("action_hidden_layers", 2),
    ).to(device)
    head.load_state_dict(action_payload["head"])
    head.eval()

    reference_env = make_keyroom_env(seed=seed)
    try:
        reference = KeyRoomOracle(reference_env).collect()
        goal = reference.pixels[-1]
    finally:
        reference_env.close()

    env = make_keyroom_env(
        seed=seed,
        actions=tuple(range(num_actions)) if raw_keycode_actions else None,
    )
    action_values = list(env.unwrapped.actions)
    history: deque[np.ndarray] = deque(maxlen=model.config.max_context)
    generator = torch.Generator(device=device).manual_seed(seed + 90_000)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    total_reward = 0.0
    success = False
    try:
        observation, _ = env.reset()
        first = observation["pixel_crop"].copy()
        history.extend(first.copy() for _ in range(model.config.max_context))
        with tempfile.TemporaryDirectory() as temporary:
            frame_dir = Path(temporary)
            for step in range(max_steps):
                current = observation["pixel_crop"].copy()
                if step:
                    history.append(current)
                batch = {
                    "history": _pixel_observation(
                        list(history), history=True, device=device
                    ),
                    "goal": _pixel_observation([goal], history=False, device=device),
                }
                features = predictor_features(
                    model, batch, feature=feature, generator=generator
                )
                probabilities = head(features).softmax(-1)
                confidence, prediction = probabilities.max(-1)
                action = int(prediction.item())
                action_value = action_values[action]
                action_name = getattr(action_value, "name", str(action_value))
                position = tuple(map(int, observation["blstats"][:2]))
                _compose(
                    current,
                    goal,
                    step=step,
                    action_name=action_name,
                    confidence=float(confidence.item()),
                    reward=total_reward,
                    position=position,
                    feature=feature,
                ).save(frame_dir / f"frame-{step:05d}.png")
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                records.append(
                    {
                        "step": step,
                        "action": action,
                        "action_name": action_name,
                        "confidence": float(confidence.item()),
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
                    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(frame_dir / "frame-%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", str(output),
                ],
                check=True,
            )
    finally:
        env.close()
    summary = {
        "mode": "closed_loop",
        "representation": "native MiniHack pixel_crop",
        "world_checkpoint": str(world_checkpoint),
        "action_checkpoint": str(action_checkpoint),
        "feature": feature,
        "seed": seed,
        "steps": len(records),
        "total_reward": total_reward,
        "success": success,
        "records": records,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tile-pixel flow policy closed-loop")
    parser.add_argument("--world-checkpoint", required=True, type=Path)
    parser.add_argument("--action-checkpoint", required=True, type=Path)
    parser.add_argument("--output", default="reports/pixel-flow-closed-loop.mp4", type=Path)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
