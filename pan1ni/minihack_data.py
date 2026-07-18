from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import h5py
import minihack  # noqa: F401 - importing registers the MiniHack environments
import numpy as np
import torch
from minihack.envs.keyroom import MiniHackKeyDoor
from torch.utils.data import Dataset

KEYROOM_ENV = "Pan1ni-KeyRoom-S5-v0"
KEY_POSITIONS = ((2, 2), (4, 2), (2, 4), (4, 4))
GOAL_POSITIONS = ((8, 2), (10, 2), (8, 4), (10, 4))


def keyroom_des(key_position: tuple[int, int], goal_position: tuple[int, int]) -> str:
    return f"""MAZE: "mylevel",' '
INIT_MAP:solidfill,' '
GEOMETRY:center,center
MAP
-------------
|.....|.....|
|.....|.....|
|.....+.....|
|.....|.....|
|.....|.....|
-------------
ENDMAP
REGION:(0,0,12,6),lit,"ordinary"
BRANCH:(1,1,5,5),(0,0,0,0)
DOOR:locked,(6,3)
STAIR:({goal_position[0]},{goal_position[1]}),down
OBJECT:('(',"skeleton key"),({key_position[0]},{key_position[1]}),blessed,0
"""


KEYROOM_DES = keyroom_des(KEY_POSITIONS[0], GOAL_POSITIONS[0])
MOVES = ((0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1))
CARDINAL_MOVES = MOVES[:4]
OBSERVATION_KEYS = (
    "screen_descriptions", "blstats", "message", "pixel_crop",
    "tty_chars", "tty_colors", "tty_cursor",
)


@dataclass
class KeyRoomEpisode:
    pixels: np.ndarray
    tty_chars: np.ndarray
    tty_colors: np.ndarray
    actions: np.ndarray
    positions: np.ndarray
    stages: np.ndarray
    reward: float

    @property
    def length(self) -> int:
        return int(self.pixels.shape[0])


def make_keyroom_env(
    env_id: str = KEYROOM_ENV,
    *,
    key_position: tuple[int, int] = KEY_POSITIONS[0],
    goal_position: tuple[int, int] = GOAL_POSITIONS[0],
    seed: int | None = None,
):
    if env_id == KEYROOM_ENV:
        env = MiniHackKeyDoor(
            des_file=keyroom_des(key_position, goal_position),
            observation_keys=OBSERVATION_KEYS,
            obs_crop_h=9,
            obs_crop_w=9,
        )
    else:
        env = gym.make(
            env_id,
            observation_keys=OBSERVATION_KEYS,
            obs_crop_h=9,
            obs_crop_w=9,
        )
    if seed is not None:
        env.unwrapped.seed(core=seed, disp=seed, reseed=True)
    return env


def _descriptions(observation: dict) -> list[list[str]]:
    return [
        [bytes(value).split(b"\0", 1)[0].decode(errors="replace") for value in row]
        for row in observation["screen_descriptions"]
    ]


def _find(observation: dict, needle: str) -> list[tuple[int, int]]:
    descriptions = _descriptions(observation)
    return [
        (x, y)
        for y, row in enumerate(descriptions)
        for x, value in enumerate(row)
        if needle in value
    ]


def _position(observation: dict) -> tuple[int, int]:
    return tuple(map(int, observation["blstats"][:2]))


def _route(
    observation: dict,
    start: tuple[int, int],
    goals: Iterable[tuple[int, int]],
    blocked: set[tuple[tuple[int, int], tuple[int, int]]],
) -> list[int] | None:
    descriptions = _descriptions(observation)
    goals = set(goals)
    queue = deque((start,))
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    actions: dict[tuple[int, int], int] = {}

    def passable(point: tuple[int, int]) -> bool:
        value = descriptions[point[1]][point[0]]
        return any(
            label in value
            for label in ("floor", "human rogue", "key", "staircase", "open door", "corridor")
        )

    while queue:
        point = queue.popleft()
        if point in goals:
            path: list[int] = []
            while previous[point] is not None:
                path.append(actions[point])
                point = previous[point]  # type: ignore[assignment]
            return path[::-1]
        for action, (dx, dy) in enumerate(MOVES):
            neighbor = point[0] + dx, point[1] + dy
            if not (0 <= neighbor[0] < 79 and 0 <= neighbor[1] < 21):
                continue
            if neighbor in previous or (point, neighbor) in blocked or not passable(neighbor):
                continue
            previous[neighbor] = point
            actions[neighbor] = action
            queue.append(neighbor)
    return None


class KeyRoomOracle:
    """Collect successful key -> locked door -> staircase trajectories."""

    def __init__(self, env) -> None:
        self.env = env
        self.observation: dict = {}
        self.pixels: list[np.ndarray] = []
        self.tty_chars: list[np.ndarray] = []
        self.tty_colors: list[np.ndarray] = []
        self.actions: list[int] = []
        self.positions: list[tuple[int, int]] = []
        self.stages: list[int] = []
        self.blocked: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self.total_reward = 0.0
        self.done = False

    def _record_initial(self, observation: dict) -> None:
        self.observation = observation
        self.pixels = [observation["pixel_crop"].copy()]
        self.tty_chars = [observation["tty_chars"].copy()]
        self.tty_colors = [observation["tty_colors"].copy()]
        self.actions = []
        self.positions = [_position(observation)]
        self.stages = [0]
        self.blocked = set()
        self.total_reward = 0.0
        self.done = False

    def _step(self, action: int, stage: int) -> None:
        before = _position(self.observation)
        observation, reward, terminated, truncated, _ = self.env.step(action)
        self.observation = observation
        self.actions.append(action)
        self.pixels.append(observation["pixel_crop"].copy())
        self.tty_chars.append(observation["tty_chars"].copy())
        self.tty_colors.append(observation["tty_colors"].copy())
        self.positions.append(_position(observation))
        self.stages.append(stage)
        self.total_reward += float(reward)
        self.done = bool(terminated or truncated)
        if action < len(MOVES) and _position(observation) == before:
            dx, dy = MOVES[action]
            self.blocked.add((before, (before[0] + dx, before[1] + dy)))

    def _walk_to(self, goals: Iterable[tuple[int, int]], stage: int) -> None:
        goals = set(goals)
        for _ in range(200):
            if _position(self.observation) in goals or self.done:
                return
            path = _route(self.observation, _position(self.observation), goals, self.blocked)
            if not path:
                raise RuntimeError(f"no route from {_position(self.observation)} to {sorted(goals)}")
            self._step(path[0], stage)
        raise RuntimeError("oracle movement exceeded 200 steps")

    def _reveal(self, needle: str, stage: int) -> None:
        visited = {_position(self.observation)}
        for _ in range(30):
            if _find(self.observation, needle):
                return
            position = _position(self.observation)
            candidates = [
                point
                for point in _find(self.observation, "floor of a room")
                if point not in visited
                and _route(self.observation, position, (point,), self.blocked) is not None
            ]
            if not candidates:
                break
            target = max(candidates, key=lambda point: abs(point[0] - position[0]) + abs(point[1] - position[1]))
            self._walk_to((target,), stage)
            visited.add(target)
        if not _find(self.observation, needle):
            raise RuntimeError(f"could not reveal {needle!r}")

    def collect(self, seed: int | None = None) -> KeyRoomEpisode:
        # MiniHack 1.0.2 predates Gymnasium's reset(seed=...) contract. Passing
        # the seed through reset can bypass the compiled custom level in NLE
        # 1.3, so use NLE's native seed method first and then reset normally.
        observation, _ = self.env.reset()
        self._record_initial(observation)

        if not _find(self.observation, "key"):
            # The random object placement may put the key under the player;
            # moving away reveals it without guessing an inventory action.
            self._reveal("key", 0)
        self._walk_to(_find(self.observation, "key"), 0)
        self._step(8, 1)
        if self.env.unwrapped.key_in_inventory("key") is None:
            raise RuntimeError("oracle failed to pick up the key")

        if not _find(self.observation, "closed door"):
            self._reveal("closed door", 1)
        door = _find(self.observation, "closed door")[0]
        cardinal_neighbors = ((door[0] + dx, door[1] + dy) for dx, dy in CARDINAL_MOVES)
        self._walk_to(cardinal_neighbors, 1)
        self._step(9, 2)
        if _find(self.observation, "closed door"):
            raise RuntimeError("oracle failed to unlock the door")

        if not self.done:
            if not _find(self.observation, "staircase down"):
                self._reveal("staircase down", 2)
            self._walk_to(_find(self.observation, "staircase down"), 3)
        if not self.done or self.total_reward <= 0:
            raise RuntimeError("oracle did not finish KeyRoom successfully")

        # After termination MiniHack exposes NetHack's end-screen buffer. Its
        # glyph-to-tile conversion is not a gameplay observation (it appears as
        # repeated player sprites), so end each demonstration on the last valid
        # frame immediately before descending the staircase.
        self.pixels.pop()
        self.tty_chars.pop()
        self.tty_colors.pop()
        self.positions.pop()
        self.stages.pop()
        self.actions.pop()
        self.stages[-1] = 3

        return KeyRoomEpisode(
            pixels=np.stack(self.pixels),
            tty_chars=np.stack(self.tty_chars),
            tty_colors=np.stack(self.tty_colors),
            actions=np.asarray(self.actions, dtype=np.int16),
            positions=np.asarray(self.positions, dtype=np.int16),
            stages=np.asarray(self.stages, dtype=np.int8),
            reward=self.total_reward,
        )


def collect_keyroom_dataset(
    path: str | Path,
    *,
    episodes: int,
    seed: int = 0,
    env_id: str = KEYROOM_ENV,
) -> dict[str, float | int | str]:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths: list[int] = []
    variants = [(key, goal) for key in KEY_POSITIONS for goal in GOAL_POSITIONS]
    envs = [
        make_keyroom_env(
            env_id,
            key_position=key,
            goal_position=goal,
            seed=seed + variant_index,
        )
        for variant_index, (key, goal) in enumerate(variants)
    ]
    try:
        with h5py.File(path, "w") as handle:
            handle.attrs["environment"] = env_id
            handle.attrs["seed"] = seed
            for episode_index in range(episodes):
                variant_index = episode_index % len(variants)
                episode = KeyRoomOracle(envs[variant_index]).collect()
                group = handle.create_group(str(episode_index))
                group.create_dataset("pixels", data=episode.pixels, compression="gzip", compression_opts=1)
                group.create_dataset("tty_chars", data=episode.tty_chars, compression="gzip", compression_opts=1)
                group.create_dataset("tty_colors", data=episode.tty_colors, compression="gzip", compression_opts=1)
                group.create_dataset("actions", data=episode.actions)
                group.create_dataset("positions", data=episode.positions)
                group.create_dataset("stages", data=episode.stages)
                group.attrs["reward"] = episode.reward
                group.attrs["key_position"] = variants[variant_index][0]
                group.attrs["goal_position"] = variants[variant_index][1]
                lengths.append(episode.length)
    finally:
        for env in envs:
            env.close()
    return {
        "path": str(path),
        "episodes": episodes,
        "frames": sum(lengths),
        "min_length": min(lengths),
        "mean_length": sum(lengths) / len(lengths),
        "max_length": max(lengths),
    }


class MiniHackPixelGoalDataset(Dataset[dict]):
    def __init__(
        self,
        path: str | Path,
        *,
        episode_keys: Iterable[str] | None = None,
        context_length: int = 1,
        samples_per_epoch: int = 10_000,
        seed: int = 0,
    ) -> None:
        self.path = Path(path)
        with h5py.File(self.path, "r") as handle:
            available = sorted(handle.keys(), key=int)
            requested = available if episode_keys is None else list(episode_keys)
            self.episode_keys = [
                key for key in requested if key in handle and handle[key]["pixels"].shape[0] >= context_length + 1
            ]
        if not self.episode_keys:
            raise ValueError("no MiniHack episodes are long enough")
        self.context_length = context_length
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return self.samples_per_epoch

    @property
    def handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    @staticmethod
    def _pixels(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(value).movedim(-1, -3).contiguous()

    def __getitem__(self, index: int) -> dict:
        rng = random.Random(self.seed + index)
        key = self.episode_keys[rng.randrange(len(self.episode_keys))]
        group = self.handle[key]
        length = group["pixels"].shape[0]
        t = rng.randint(self.context_length - 1, length - 2)
        start = t - self.context_length + 1
        final = length - 1
        return {
            "history": {"pixels": self._pixels(group["pixels"][start : t + 1])},
            "goal": {"pixels": self._pixels(group["pixels"][final])},
            "target": {"pixels": self._pixels(group["pixels"][t + 1])},
            "action": torch.tensor(group["actions"][t], dtype=torch.long),
            "goal_offset": torch.tensor(final - t, dtype=torch.long),
            "timestep": torch.tensor(t, dtype=torch.long),
            "trajectory_id": torch.tensor(int(key), dtype=torch.long),
            "stage": torch.tensor(group["stages"][t], dtype=torch.long),
        }

    def __del__(self) -> None:
        if self._handle is not None:
            self._handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect successful tiled MiniHack KeyRoom trajectories")
    parser.add_argument("--output", default="data/minihack/keyroom-s5-16v-v2.hdf5")
    parser.add_argument("--episodes", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(collect_keyroom_dataset(args.output, episodes=args.episodes, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
