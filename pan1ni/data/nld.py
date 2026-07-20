from __future__ import annotations

import bisect
import random
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from typing import Sequence

import h5py
import numpy as np
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
        # Sample a genuinely future goal rather than fixing every example at
        # the maximum horizon. Keep it beyond t+1 so the goal cannot equal the
        # one-step prediction target.
        goal_offset = rng.randint(2, self.goal_horizon) if self.goal_horizon > 1 else 1
        goal_t = t + goal_offset
        key = self.episode_keys[episode_index]
        group = self.handle[key]
        start = t - self.context_length + 1
        return {
            "history": self._observation(group, slice(start, t + 1)),
            "target": self._observation(group, t + 1),
            "goal": self._observation(group, goal_t),
            "action": torch.tensor(group["actions"][t], dtype=torch.long),
            "goal_offset": torch.tensor(goal_offset, dtype=torch.long),
            "timestep": torch.tensor(t, dtype=torch.long),
            "trajectory_id": torch.tensor(int(key), dtype=torch.long),
        }


class NLDTtyrecGoalBatchStream:
    """Sequential, pre-batched goal windows decoded directly from NLD ttyrecs."""

    def __init__(
        self,
        db_path: str | Path,
        dataset_name: str,
        *,
        batch_size: int,
        context_length: int = 1,
        goal_horizon: int = 64,
        gameids: Sequence[int] | None = None,
        num_workers: int = 4,
        windows_per_block: int = 1,
        window_stride: int = 16,
        shuffle: bool = True,
        loop_forever: bool = False,
        status_dim: int = 27,
        message_length: int = 256,
        seed: int = 0,
    ) -> None:
        if min(
            batch_size, context_length, goal_horizon, num_workers,
            windows_per_block, window_stride,
        ) < 1:
            raise ValueError("stream sizes, worker count, and stride must be positive")
        self.db_path = str(db_path)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.context_length = context_length
        self.goal_horizon = goal_horizon
        self.windows_per_block = windows_per_block
        self.window_stride = window_stride
        self.gameids = list(gameids) if gameids is not None else None
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.loop_forever = loop_forever
        self.status_dim = status_dim
        self.message_length = message_length
        self.seed = seed

    def available_gameids(self) -> list[int]:
        import nle.dataset as nld

        dataset = nld.TtyrecDataset(
            self.dataset_name,
            batch_size=1,
            seq_length=1,
            dbfilename=self.db_path,
            shuffle=False,
        )
        return sorted(dataset._gameids)

    def _observation(self, raw: dict, index: int | slice | np.ndarray) -> dict[str, Tensor]:
        # Keep ttyrec fields in their compact on-disk dtypes. The encoder casts
        # them on the GPU immediately before embedding. Expanding chars/colors
        # to int64 here inflated both host work and PCIe traffic by up to 8x.
        def take(name: str) -> np.ndarray:
            values = raw[name]
            if isinstance(index, np.ndarray):
                return values[np.arange(values.shape[0]), index].copy()
            return values[:, index].copy()

        chars = torch.from_numpy(take("tty_chars"))
        colors = torch.from_numpy(take("tty_colors"))
        cursor = torch.from_numpy(take("tty_cursor"))
        leading_shape = chars.shape[:-2]
        message = torch.zeros(*leading_shape, self.message_length, dtype=chars.dtype)
        width = min(chars.shape[-1], self.message_length)
        message[..., :width] = chars[..., 0, :width]
        return {
            "chars": chars,
            "colors": colors,
            "bg_colors": torch.zeros_like(colors),
            "message": message,
            "status": torch.zeros(*leading_shape, self.status_dim),
            "cursor": cursor,
        }

    def __iter__(self) -> Iterator[dict[str, Tensor | dict[str, Tensor]]]:
        import nle.dataset as nld

        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        window_length = self.context_length + self.goal_horizon
        sequence_length = (
            window_length + (self.windows_per_block - 1) * self.window_stride
        )
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            dataset = nld.TtyrecDataset(
                self.dataset_name,
                batch_size=self.batch_size,
                seq_length=sequence_length,
                dbfilename=self.db_path,
                threadpool=pool,
                gameids=self.gameids,
                shuffle=self.shuffle,
                loop_forever=self.loop_forever,
            )
            for raw in dataset:
                offsets = [
                    index * self.window_stride
                    for index in range(self.windows_per_block)
                ]
                if self.shuffle:
                    rng.shuffle(offsets)
                for offset in offsets:
                    stop = offset + window_length
                    gameids = raw["gameids"][:, offset:stop]
                    valid = (
                        (gameids[:, 0] != 0)
                        & np.all(gameids == gameids[:, :1], axis=1)
                        & ~np.any(raw["done"][:, offset : stop - 1], axis=1)
                    )
                    if not valid.any():
                        continue
                    batch = {name: value[valid] for name, value in raw.items()}
                    current = offset + self.context_length - 1
                    batch_size = int(valid.sum())
                    minimum_goal_offset = 2 if self.goal_horizon > 1 else 1
                    goal_offsets = rng.integers(
                        minimum_goal_offset,
                        self.goal_horizon + 1,
                        size=batch_size,
                    )
                    yield {
                        **(
                            {
                                "action": torch.from_numpy(
                                    batch["keypresses"][:, current].copy()
                                ).long()
                            }
                            if "keypresses" in batch
                            else {}
                        ),
                        "history": self._observation(
                            batch, slice(offset, offset + self.context_length)
                        ),
                        "target": self._observation(batch, current + 1),
                        "goal": self._observation(batch, current + goal_offsets),
                        "goal_offset": torch.from_numpy(goal_offsets.copy()).long(),
                        "trajectory_id": torch.from_numpy(
                            batch["gameids"][:, current].copy()
                        ).long(),
                        "timestep": torch.from_numpy(
                            batch["timestamps"][:, current].copy()
                        ).long(),
                    }


def prefetch_streams(
    sources: Sequence[Iterator[dict[str, Tensor | dict[str, Tensor]]]],
    depth: int,
) -> Iterator[dict[str, Tensor | dict[str, Tensor]]]:
    """Merge independently decoded streams through one bounded queue."""

    if not sources or depth < 1:
        raise ValueError("sources and depth must be non-empty and positive")
    queue: Queue = Queue(maxsize=depth)
    sentinel = object()
    stopped = Event()

    def put(item: object) -> bool:
        while not stopped.is_set():
            try:
                queue.put(item, timeout=0.1)
                return True
            except Full:
                pass
        return False

    def produce(source: Iterator) -> None:
        try:
            for batch in source:
                if not put(batch):
                    break
        except BaseException as error:
            put(error)
        finally:
            put(sentinel)

    workers = [
        Thread(target=produce, args=(source,), name=f"nld-ttyrec-prefetch-{index}", daemon=True)
        for index, source in enumerate(sources)
    ]
    for worker in workers:
        worker.start()
    remaining = len(workers)
    try:
        while remaining:
            item = queue.get()
            if item is sentinel:
                remaining -= 1
            elif isinstance(item, BaseException):
                raise item
            else:
                yield item
    finally:
        stopped.set()
        for worker in workers:
            worker.join(timeout=5)


def prefetch_batches(
    source: Iterator[dict[str, Tensor | dict[str, Tensor]]],
    depth: int = 2,
) -> Iterator[dict[str, Tensor | dict[str, Tensor]]]:
    """Decode batches on a background thread into a bounded host-memory queue."""

    if depth < 1:
        yield from source
        return
    queue: Queue = Queue(maxsize=depth)
    sentinel = object()
    stopped = Event()

    def put(item: object) -> bool:
        while not stopped.is_set():
            try:
                queue.put(item, timeout=0.1)
                return True
            except Full:
                pass
        return False

    def produce() -> None:
        try:
            for batch in source:
                if not put(batch):
                    break
        except BaseException as error:
            put(error)
        finally:
            put(sentinel)

    worker = Thread(target=produce, name="nld-ttyrec-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            item = queue.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stopped.set()
        worker.join(timeout=5)
