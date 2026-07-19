from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from .nle_closed_loop_eval import (
    TileConverter,
    _distance,
    _make_env,
    _position,
    _verify_same_reset,
    collect_reference_goal,
)


def run(
    output: Path,
    *,
    env_id: str,
    episodes: int,
    max_steps: int,
    goal_horizon: int,
    min_goal_distance: int,
    seed: int,
    max_goal_distance: int = 1_000_000,
    goal_distance_mode: str = "mixed",
) -> Path:
    converter = TileConverter.create()
    references = []
    attempted_seed = seed
    attempt_budget = episodes * 60
    while len(references) < episodes:
        reference = collect_reference_goal(
            env_id,
            attempted_seed,
            converter,
            goal_horizon=goal_horizon,
            min_goal_distance=min_goal_distance,
            max_goal_distance=max_goal_distance,
        )
        if reference is not None:
            references.append(reference)
        attempted_seed += 1
        if attempted_seed - seed > attempt_budget:
            break
    if not references:
        raise RuntimeError(
            f"could not generate any reachable goals in mode '{goal_distance_mode}'"
        )
    episodes = len(references)

    records = []
    for episode, reference in enumerate(references):
        rng = random.Random(reference.seed + 91_117)
        env = _make_env(env_id, reference.seed, max_steps + 8)
        try:
            observation, _ = env.reset()
            _verify_same_reset(observation, reference)
            positions = [_position(observation)]
            counts: Counter[int] = Counter()
            collisions = 0
            best_distance = reference.initial_distance
            terminated = truncated = success = False
            for step in range(max_steps):
                action = rng.randrange(8)
                counts[action] += 1
                previous = _position(observation)
                observation, _, terminated, truncated, _ = env.step(action)
                position = _position(observation)
                positions.append(position)
                collisions += int(position == previous)
                best_distance = min(best_distance, _distance(position, reference.goal_position))
                if position == reference.goal_position:
                    success = True
                    break
                if terminated or truncated:
                    break
            steps_taken = step + 1
            final_distance = _distance(positions[-1], reference.goal_position)
            revisits = max(0, len(positions) - len(set(positions)))
            records.append(
                {
                    "episode": episode,
                    "seed": reference.seed,
                    "success": success,
                    "steps": steps_taken,
                    "initial_position": reference.initial_position,
                    "goal_position": reference.goal_position,
                    "final_position": positions[-1],
                    "reference_step": reference.reference_step,
                    "initial_distance": reference.initial_distance,
                    "best_distance": best_distance,
                    "final_distance": final_distance,
                    "best_progress": (reference.initial_distance - best_distance)
                    / reference.initial_distance,
                    "final_progress": (reference.initial_distance - final_distance)
                    / reference.initial_distance,
                    "unique_positions": len(set(positions)),
                    "revisit_rate": revisits / max(steps_taken, 1),
                    "collisions": collisions,
                    "collision_rate": collisions / max(steps_taken, 1),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "action_counts": {str(key): value for key, value in sorted(counts.items())},
                }
            )
        finally:
            env.close()
        print(
            f"episode {episode + 1:3d}/{episodes} | seed {reference.seed} | "
            f"success {success} | best progress {records[-1]['best_progress']:.1%}",
            flush=True,
        )

    successes = [record["steps"] for record in records if record["success"]]
    total_steps = sum(record["steps"] for record in records)
    summary = {
        "mode": "closed_loop_random_baseline",
        "environment": env_id,
        "representation": "full NLE tty converted by the human-pretraining tile pipeline",
        "spawn_monsters": False,
        "feature": "uniform_random",
        "action_policy": "uniform random over eight semantic movement classes",
        "goal_distance_mode": goal_distance_mode,
        "min_goal_distance": min_goal_distance,
        "max_goal_distance": max_goal_distance,
        "episodes": episodes,
        "successes": len(successes),
        "success_rate": len(successes) / episodes,
        "median_success_steps": statistics.median(successes) if successes else None,
        "mean_success_steps": statistics.mean(successes) if successes else None,
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
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Uniform-random control for full-NLE evaluation")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--env-id", default="NetHackScore-v0")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--goal-horizon", type=int, default=64)
    parser.add_argument("--min-goal-distance", type=int, default=3)
    parser.add_argument("--max-goal-distance", type=int, default=1_000_000)
    parser.add_argument("--goal-distance-mode", default="mixed")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    print(
        run(
            args.output,
            env_id=args.env_id,
            episodes=args.episodes,
            max_steps=args.max_steps,
            goal_horizon=args.goal_horizon,
            min_goal_distance=args.min_goal_distance,
            seed=args.seed,
            max_goal_distance=args.max_goal_distance,
            goal_distance_mode=args.goal_distance_mode,
        )
    )


if __name__ == "__main__":
    main()
