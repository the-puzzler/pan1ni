from __future__ import annotations

import bisect
import random
from pathlib import Path
from typing import Sequence

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset


def nld_episode_keys(path: str | Path) -> list[str]:
    """Return episode keys in numeric order without loading trajectory arrays."""

    with h5py.File(path, "r") as handle:
        return sorted(handle.keys(), key=int)


class NLDHDF5GoalDataset(Dataset[dict[str, Tensor | dict[str, Tensor]]]):
    """Lazy final-frame-goal windows over the compact NLD-AA HDF5 mirror.

    A sampled sequence begins at current timestep ``t`` and ends after
    ``goal_horizon`` transitions. Only history, ``t+1``, and the sequence's final
    frame are decoded. Episode selection is weighted by its number of valid windows.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        episode_keys: Sequence[str] | None = None,
        context_length: int = 1,
        goal_horizon: int = 64,
        samples_per_epoch: int = 100_000,
        status_dim: int = 27,
        message_length: int = 256,
        seed: int = 0,
    ) -> None:
        self.path = str(Path(path))
        self.context_length = context_length
        self.goal_horizon = goal_horizon
        self.samples_per_epoch = samples_per_epoch
        self.status_dim = status_dim
        self.message_length = message_length
        self.seed = seed
        self._handle: h5py.File | None = None
        if context_length < 1 or goal_horizon < 1 or samples_per_epoch < 1:
            raise ValueError("context_length, goal_horizon, and samples_per_epoch must be positive")

        requested = set(episode_keys) if episode_keys is not None else None
        keys: list[str] = []
        window_counts: list[int] = []
        lengths: list[int] = []
        with h5py.File(self.path, "r") as handle:
            for key in sorted(handle.keys(), key=int):
                if requested is not None and key not in requested:
                    continue
                length = int(handle[key]["tty_chars"].shape[0])
                count = length - goal_horizon - context_length + 1
                if count > 0:
                    keys.append(key)
                    lengths.append(length)
                    window_counts.append(count)
        if not keys:
            raise ValueError("no selected episode is long enough for this context and goal horizon")
        if requested is not None and set(keys) != requested:
            missing = requested - set(keys)
            if missing:
                raise ValueError(f"unknown or too-short episode keys: {sorted(missing)}")
        self.episode_keys = keys
        self.episode_lengths = lengths
        self.window_counts = window_counts
        cumulative = 0
        self.cumulative_windows: list[int] = []
        for count in window_counts:
            cumulative += count
            self.cumulative_windows.append(cumulative)
        self.total_windows = cumulative

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    @property
    def handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r", swmr=True)
        return self._handle

    def _observation(self, group: h5py.Group, index: int | slice) -> dict[str, Tensor]:
        chars = torch.from_numpy(group["tty_chars"][index])
        colors = torch.from_numpy(group["tty_colors"][index]).long()
        cursor = torch.from_numpy(group["tty_cursor"][index]).long()
        leading_shape = chars.shape[:-2]
        message = torch.zeros(*leading_shape, self.message_length, dtype=torch.long)
        # The first terminal row is NetHack's message line. Preserve it as the
        # dedicated message modality while it also remains in the terminal grid.
        message_width = min(chars.shape[-1], self.message_length)
        message[..., :message_width] = chars[..., 0, :message_width].long()
        return {
            "chars": chars.long(),
            "colors": colors,
            "bg_colors": torch.zeros_like(colors),
            "message": message,
            "status": torch.zeros(*leading_shape, self.status_dim),
            "cursor": cursor,
        }

    def __getitem__(self, index: int) -> dict[str, Tensor | dict[str, Tensor]]:
        rng = random.Random(self.seed + index)
        flat_window = rng.randrange(self.total_windows)
        episode_index = bisect.bisect_right(self.cumulative_windows, flat_window)
        previous = self.cumulative_windows[episode_index - 1] if episode_index else 0
        offset = flat_window - previous
        t = self.context_length - 1 + offset
        goal_t = t + self.goal_horizon
        key = self.episode_keys[episode_index]
        group = self.handle[key]
        start = t - self.context_length + 1
        return {
            "history": self._observation(group, slice(start, t + 1)),
            "target": self._observation(group, t + 1),
            "goal": self._observation(group, goal_t),
            "action": torch.tensor(group["actions"][t], dtype=torch.long),
            "goal_offset": torch.tensor(self.goal_horizon, dtype=torch.long),
            "timestep": torch.tensor(t, dtype=torch.long),
            "trajectory_id": torch.tensor(int(key), dtype=torch.long),
        }
