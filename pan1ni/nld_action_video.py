from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .action import DirectPolicyHead
from .config import ModelConfig
from .model import GoalConditionedLeWorldModel
from .action import predictor_features
from .nld_data import NLDTtyrecGoalBatchStream
from .report import render_terminal
from .train import move_batch


def keycode_label(value: int) -> str:
    if value == 0:
        return "NUL"
    if value == 13:
        return "ENTER"
    if value == 27:
        return "ESC"
    if value == 32:
        return "SPACE"
    if 33 <= value <= 126:
        return repr(chr(value))
    return f"0x{value:02x}"


def compose_frame(
    current: Image.Image,
    goal: Image.Image,
    *,
    frame_index: int,
    goal_horizon: int,
    predicted: int,
    actual: int,
    confidence: float,
    correct: bool,
) -> Image.Image:
    margin = 16
    header = 58
    footer = 76
    width = current.width + goal.width + margin * 3
    height = header + max(current.height, goal.height) + footer
    canvas = Image.new("RGB", (width, height), "#080b11")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    title_font = ImageFont.truetype(font_path, 17)
    body_font = ImageFont.truetype(font_path, 15)
    draw.text(
        (margin, 12),
        f"Teacher-forced action probe · held-out frame {frame_index}",
        font=title_font,
        fill="#e7ebf2",
    )
    draw.text(
        (margin, 35),
        "True observation is supplied after every prediction",
        font=body_font,
        fill="#98a3b5",
    )
    canvas.paste(current, (margin, header))
    canvas.paste(goal, (current.width + margin * 2, header))
    y = header + max(current.height, goal.height) + 12
    result_color = "#70d6a5" if correct else "#ff7b72"
    draw.text(
        (margin, y),
        f"predicted key: {keycode_label(predicted):>8}   confidence: {confidence:6.2%}",
        font=body_font,
        fill=result_color,
    )
    draw.text(
        (margin, y + 25),
        f"recorded key:  {keycode_label(actual):>8}   goal offset: +{goal_horizon}",
        font=body_font,
        fill="#e7ebf2",
    )
    return canvas


@torch.no_grad()
def run(
    world_checkpoint: Path,
    action_checkpoint: Path,
    db_path: Path,
    dataset_name: str,
    output: Path,
    *,
    frames: int,
    goal_horizon: int,
    fps: int,
    gameid: int | None,
    skip: int,
    device: str,
) -> Path:
    if min(frames, goal_horizon, fps) < 1 or skip < 0:
        raise ValueError("frames, horizon, and fps must be positive; skip cannot be negative")

    world_payload = torch.load(world_checkpoint, map_location=device, weights_only=False)
    model = GoalConditionedLeWorldModel(ModelConfig(**world_payload["config"])).to(device)
    model.load_state_dict(world_payload["model"])
    model.eval()

    action_payload = torch.load(action_checkpoint, map_location=device, weights_only=False)
    config = action_payload["config"]
    head = DirectPolicyHead(
        model.config.latent_dim,
        256,
        hidden_dim=config.get("action_hidden_dim", 256),
        hidden_layers=config.get("action_hidden_layers", 1),
    ).to(device)
    head.load_state_dict(action_payload["head"])
    head.eval()

    catalog = NLDTtyrecGoalBatchStream(
        db_path, dataset_name, batch_size=1, num_workers=1
    )
    gameids = catalog.available_gameids()
    validation_ids = gameids[-max(1, len(gameids) // 5):]
    selected_gameid = gameid if gameid is not None else validation_ids[0]
    if selected_gameid not in validation_ids:
        raise ValueError("video gameid must belong to the held-out episode split")

    stream = NLDTtyrecGoalBatchStream(
        db_path,
        dataset_name,
        batch_size=1,
        context_length=1,
        goal_horizon=goal_horizon,
        gameids=[selected_gameid],
        num_workers=1,
        windows_per_block=frames + skip,
        window_stride=1,
        shuffle=False,
        loop_forever=True,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    use_cuda = device.startswith("cuda")
    with tempfile.TemporaryDirectory() as temporary:
        frame_dir = Path(temporary)
        generated = 0
        for index, host_batch in enumerate(stream):
            if index < skip:
                continue
            batch = move_batch(host_batch, device, non_blocking=use_cuda)
            features = predictor_features(
                model,
                batch,
                action_payload["config"].get("feature", "predictor_hidden"),
            )
            probabilities = head(features).softmax(-1)
            confidence, prediction = probabilities.max(-1)
            predicted = int(prediction.item())
            actual = int(batch["action"].item())
            current_image = render_terminal(
                host_batch["history"]["chars"][0, -1],
                host_batch["history"]["colors"][0, -1],
                "Current true observation",
            )
            goal_image = render_terminal(
                host_batch["goal"]["chars"][0],
                host_batch["goal"]["colors"][0],
                f"Current goal observation (+{goal_horizon})",
            )
            image = compose_frame(
                current_image,
                goal_image,
                frame_index=generated,
                goal_horizon=goal_horizon,
                predicted=predicted,
                actual=actual,
                confidence=float(confidence.item()),
                correct=predicted == actual,
            )
            image.save(frame_dir / f"frame-{generated:05d}.png")
            records.append(
                {
                    "frame": generated,
                    "predicted_keycode": predicted,
                    "recorded_keycode": actual,
                    "confidence": float(confidence.item()),
                    "correct": predicted == actual,
                }
            )
            generated += 1
            if generated >= frames:
                break
        if generated < frames:
            raise RuntimeError(f"episode produced only {generated} valid frames")
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
                "20",
                str(output),
            ],
            check=True,
        )

    summary = {
        "mode": "teacher_forced",
        "gameid": selected_gameid,
        "frames": frames,
        "goal_horizon": goal_horizon,
        "fps": fps,
        "accuracy": sum(record["correct"] for record in records) / len(records),
        "records": records,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a teacher-forced predictor-hidden action video"
    )
    parser.add_argument("--world-checkpoint", required=True, type=Path)
    parser.add_argument("--action-checkpoint", required=True, type=Path)
    parser.add_argument("--db", default="data/nld/nld-aa-taster.db", type=Path)
    parser.add_argument("--dataset", default="nld-aa-taster")
    parser.add_argument("--output", default="reports/nld-action-video.mp4", type=Path)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--goal-horizon", type=int, default=64)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--gameid", type=int)
    parser.add_argument("--skip", type=int, default=64)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    print(
        run(
            args.world_checkpoint,
            args.action_checkpoint,
            args.db,
            args.dataset,
            args.output,
            frames=args.frames,
            goal_horizon=args.goal_horizon,
            fps=args.fps,
            gameid=args.gameid,
            skip=args.skip,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
