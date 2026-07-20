from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

OBSERVATION_KEYS = ("chars", "colors", "bg_colors", "message", "status", "cursor")


@dataclass
class Trajectory:
    """One episode. Every observation field has time as its first dimension."""

    observations: dict[str, Tensor]
    actions: Tensor | None = None
    role: str | None = None
    max_dungeon_level: int | None = None

    def __post_init__(self) -> None:
        missing = {"chars", "colors", "message", "status", "cursor"} - self.observations.keys()
        if missing:
            raise ValueError(f"missing observation fields: {sorted(missing)}")
        lengths = {value.shape[0] for value in self.observations.values()}
        if len(lengths) != 1:
            raise ValueError("all observation fields must have the same time length")
        if self.actions is not None and self.actions.shape[0] < self.length - 1:
            raise ValueError("actions must cover every transition")

    @property
    def length(self) -> int:
        return self.observations["chars"].shape[0]


def trajectory_from_nle(mapping: Mapping[str, Tensor], actions: Tensor | None = None) -> Trajectory:
    """Convert common NLE/NLD field names without importing their heavy packages."""

    aliases = {
        "chars": "tty_chars",
        "colors": "tty_colors",
        "bg_colors": "tty_bg_colors",
        "message": "message",
        "status": "blstats",
        "cursor": "tty_cursor",
    }
    obs: dict[str, Tensor] = {}
    for destination, source in aliases.items():
        if source in mapping:
            obs[destination] = torch.as_tensor(mapping[source])
    if "bg_colors" not in obs:
        obs["bg_colors"] = torch.zeros_like(obs["colors"])
    return Trajectory(obs, actions)


def filter_trajectories(
    trajectories: Sequence[Trajectory],
    *,
    roles: set[str] | None = None,
    max_dungeon_level: int | None = None,
    max_horizon: int | None = None,
) -> list[Trajectory]:
    return [
        trajectory
        for trajectory in trajectories
        if (roles is None or trajectory.role in roles)
        and (max_dungeon_level is None or trajectory.max_dungeon_level is None or trajectory.max_dungeon_level <= max_dungeon_level)
        and (max_horizon is None or trajectory.length <= max_horizon)
    ]


def _slice_observation(observation: Mapping[str, Tensor], index: int | slice) -> dict[str, Tensor]:
    return {key: value[index] for key, value in observation.items()}


class GoalWindowDataset(Dataset[dict[str, Tensor | dict[str, Tensor]]]):
    """Random history windows whose goal is always the episode's final frame."""

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        *,
        context_length: int = 8,
        samples_per_epoch: int = 100_000,
        seed: int = 0,
    ) -> None:
        self.trajectories = [
            trajectory for trajectory in trajectories if trajectory.length >= context_length + 1
        ]
        if not self.trajectories:
            raise ValueError("no trajectory has room for the context and a next-frame target")
        self.context_length = context_length
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, Tensor | dict[str, Tensor]]:
        rng = random.Random(self.seed + index)
        trajectory_index = rng.randrange(len(self.trajectories))
        trajectory = self.trajectories[trajectory_index]
        t_min = self.context_length - 1
        t_max = trajectory.length - 2
        t = rng.randint(t_min, t_max)
        final_timestep = trajectory.length - 1
        offset = final_timestep - t
        start = t - self.context_length + 1
        item: dict[str, Tensor | dict[str, Tensor]] = {
            "history": _slice_observation(trajectory.observations, slice(start, t + 1)),
            "goal": _slice_observation(trajectory.observations, final_timestep),
            "target": _slice_observation(trajectory.observations, t + 1),
            "goal_offset": torch.tensor(offset, dtype=torch.long),
            "timestep": torch.tensor(t, dtype=torch.long),
            "trajectory_id": torch.tensor(trajectory_index, dtype=torch.long),
        }
        if trajectory.actions is not None:
            item["action"] = trajectory.actions[t].long()
        return item
