from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from .action import DirectPolicyHead
from .config import ModelConfig
from .minihack_closed_loop_video import _pixel_observation
from .minihack_data import GOAL_POSITIONS, KEY_POSITIONS, KeyRoomOracle, make_keyroom_env
from .model import GoalConditionedLeWorldModel
from .nld_action_train import predictor_features


@torch.no_grad()
def run(
    world_checkpoint: Path,
    action_checkpoint: Path,
    output: Path,
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    device: str,
) -> Path:
    world_payload = torch.load(world_checkpoint, map_location=device, weights_only=False)
    model = GoalConditionedLeWorldModel(ModelConfig(**world_payload["config"])).to(device)
    model.load_state_dict(world_payload["model"])
    model.eval()
    action_payload = torch.load(action_checkpoint, map_location=device, weights_only=False)
    config = action_payload["config"]
    feature = config["feature"]
    num_classes = int(config.get("num_classes", 10))
    if num_classes != 10 or config.get("target") != "semantic_10":
        raise ValueError("KeyRoom success evaluation requires the semantic 10-action head")
    head = DirectPolicyHead(
        model.config.latent_dim,
        num_classes,
        hidden_dim=int(config.get("action_hidden_dim", 1024)),
        hidden_layers=int(config.get("action_hidden_layers", 2)),
    ).to(device)
    head.load_state_dict(action_payload["head"])
    head.eval()

    variants = [(key, goal) for key in KEY_POSITIONS for goal in GOAL_POSITIONS]
    records = []
    for episode in range(episodes):
        episode_seed = seed + episode
        key_position, goal_position = variants[episode % len(variants)]
        reference_env = make_keyroom_env(
            seed=episode_seed,
            key_position=key_position,
            goal_position=goal_position,
        )
        try:
            goal = KeyRoomOracle(reference_env).collect().pixels[-1]
        finally:
            reference_env.close()

        env = make_keyroom_env(
            seed=episode_seed,
            key_position=key_position,
            goal_position=goal_position,
        )
        history: deque[np.ndarray] = deque(maxlen=model.config.max_context)
        total_reward = 0.0
        action_counts: Counter[int] = Counter()
        terminated = truncated = False
        try:
            observation, _ = env.reset()
            first = observation["pixel_crop"].copy()
            history.extend(first.copy() for _ in range(model.config.max_context))
            for step in range(max_steps):
                if step:
                    history.append(observation["pixel_crop"].copy())
                batch = {
                    "history": _pixel_observation(list(history), history=True, device=device),
                    "goal": _pixel_observation([goal], history=False, device=device),
                }
                logits = head(predictor_features(model, batch, feature))
                action = int(logits.argmax(-1).item())
                action_counts[action] += 1
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            steps_taken = step + 1
            final_position = tuple(map(int, observation["blstats"][:2]))
        finally:
            env.close()
        records.append(
            {
                "episode": episode,
                "seed": episode_seed,
                "key_position": key_position,
                "goal_position": goal_position,
                "success": bool(total_reward > 0),
                "reward": total_reward,
                "steps": steps_taken,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "final_position": final_position,
                "action_counts": {str(key): value for key, value in sorted(action_counts.items())},
            }
        )
        print(
            f"episode {episode + 1:3d}/{episodes} | seed {episode_seed} | "
            f"success {records[-1]['success']} | steps {steps_taken}",
            flush=True,
        )

    successful_steps = [record["steps"] for record in records if record["success"]]
    summary = {
        "world_checkpoint": str(world_checkpoint),
        "action_checkpoint": str(action_checkpoint),
        "feature": feature,
        "episodes": episodes,
        "successes": len(successful_steps),
        "success_rate": len(successful_steps) / episodes,
        "median_success_steps": statistics.median(successful_steps) if successful_steps else None,
        "mean_success_steps": statistics.mean(successful_steps) if successful_steps else None,
        "max_steps": max_steps,
        "seed": seed,
        "variants": len(variants),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic action heads over KeyRoom seeds")
    parser.add_argument("--world-checkpoint", required=True, type=Path)
    parser.add_argument("--action-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(
        run(
            args.world_checkpoint,
            args.action_checkpoint,
            args.output,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
