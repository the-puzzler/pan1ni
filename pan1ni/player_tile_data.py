from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nle.dataset as nld
import numpy as np
import torch
from minihack.tiles.glyph_mapper import GlyphMapper

from .player_tile_converter import (
    batch_player_centered_tile_crops,
    build_canonical_lookup,
)


SEMANTIC_ACTION_NAMES = ("north", "east", "south", "west", "northeast", "southeast", "southwest", "northwest", "pickup", "apply")
CURSOR_DELTA_TO_ACTION = {
    (-1, 0): 0,
    (0, 1): 1,
    (1, 0): 2,
    (0, -1): 3,
    (-1, 1): 4,
    (1, 1): 5,
    (1, -1): 6,
    (-1, -1): 7,
}
RAW_MOVEMENT_KEY_TO_ACTION = {
    ord("k"): 0,
    ord("l"): 1,
    ord("j"): 2,
    ord("h"): 3,
    ord("u"): 4,
    ord("n"): 5,
    ord("b"): 6,
    ord("y"): 7,
}


def infer_movement_actions(current: np.ndarray, following: np.ndarray) -> np.ndarray:
    """Infer successful one-cell semantic moves from absolute tty cursors."""

    delta = following.astype(np.int64) - current.astype(np.int64)
    actions = np.full(delta.shape[:-1], -1, dtype=np.int16)
    for movement, action in CURSOR_DELTA_TO_ACTION.items():
        actions[(delta[..., 0] == movement[0]) & (delta[..., 1] == movement[1])] = action
    return actions


class NLDPlayerTileGoalBatchStream:
    """Decode human ttyrecs and emit canonical 144x144 MiniHack tile batches."""

    def __init__(
        self,
        db_path: str | Path,
        dataset_name: str,
        *,
        batch_size: int,
        context_length: int = 8,
        goal_horizon: int = 64,
        gameids: Sequence[int] | None = None,
        num_workers: int = 8,
        windows_per_block: int = 2,
        window_stride: int = 16,
        shuffle: bool = True,
        loop_forever: bool = False,
        seed: int = 0,
        lookup: np.ndarray | None = None,
        tile_atlas: np.ndarray | None = None,
        action_mode: str = "recorded",
    ) -> None:
        if min(
            batch_size,
            context_length,
            goal_horizon,
            num_workers,
            windows_per_block,
            window_stride,
        ) < 1:
            raise ValueError("stream sizes and worker count must be positive")
        self.db_path = str(db_path)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.context_length = context_length
        self.goal_horizon = goal_horizon
        self.gameids = list(gameids) if gameids is not None else None
        self.num_workers = num_workers
        self.windows_per_block = windows_per_block
        self.window_stride = window_stride
        self.shuffle = shuffle
        self.loop_forever = loop_forever
        self.seed = seed
        self.lookup = build_canonical_lookup()[0] if lookup is None else lookup
        if tile_atlas is None:
            mapper = GlyphMapper()
            tile_atlas = np.stack(
                [mapper.tiles[index] for index in range(max(mapper.tiles) + 1)]
            )
        self.tile_atlas = tile_atlas
        if action_mode not in {"recorded", "inferred_movement"}:
            raise ValueError("action_mode must be 'recorded' or 'inferred_movement'")
        self.action_mode = action_mode

    def available_gameids(self) -> list[int]:
        dataset = nld.TtyrecDataset(
            self.dataset_name,
            batch_size=1,
            seq_length=1,
            dbfilename=self.db_path,
            shuffle=False,
        )
        return sorted(dataset._gameids)

    @staticmethod
    def _pixels(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(value).movedim(-1, -3).contiguous()

    @staticmethod
    def _gameplay_mask(raw: dict, indices: np.ndarray) -> np.ndarray:
        batch = np.arange(indices.shape[0])[:, None]
        cursors = raw["tty_cursor"][batch, indices]
        rows = cursors[..., 0].astype(np.int64)
        columns = cursors[..., 1].astype(np.int64)
        valid = (rows >= 1) & (rows <= 21) & (columns >= 0) & (columns < 79)
        chars = raw["tty_chars"][
            batch,
            indices,
            rows.clip(0, 23),
            columns.clip(0, 79),
        ]
        return valid & (chars == ord("@"))

    def __iter__(self) -> Iterator[dict]:
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        window_length = self.context_length + self.goal_horizon
        sequence_length = window_length + (self.windows_per_block - 1) * self.window_stride
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
                offsets = np.arange(self.windows_per_block) * self.window_stride
                if self.shuffle:
                    rng.shuffle(offsets)
                for offset in offsets:
                    current = int(offset) + self.context_length - 1
                    batch_count = raw["tty_chars"].shape[0]
                    goal_offsets = rng.integers(2, self.goal_horizon + 1, size=batch_count)
                    goal_indices = current + goal_offsets
                    history_indices = np.broadcast_to(
                        np.arange(int(offset), current + 1),
                        (batch_count, self.context_length),
                    )
                    selected_indices = np.concatenate(
                        (
                            history_indices,
                            np.full((batch_count, 1), current + 1),
                            goal_indices[:, None],
                        ),
                        axis=1,
                    )
                    batch_indices = np.arange(batch_count)[:, None]
                    selected_gameids = raw["gameids"][batch_indices, selected_indices]
                    valid = (
                        (selected_gameids[:, 0] != 0)
                        & np.all(selected_gameids == selected_gameids[:, :1], axis=1)
                        & self._gameplay_mask(raw, selected_indices).all(axis=1)
                    )
                    inferred_actions = None
                    if self.action_mode == "inferred_movement":
                        inferred_actions = infer_movement_actions(
                            raw["tty_cursor"][:, current],
                            raw["tty_cursor"][:, current + 1],
                        )
                        valid &= inferred_actions >= 0
                    if not valid.any():
                        continue
                    rows = np.flatnonzero(valid)
                    histories = batch_player_centered_tile_crops(
                        raw["tty_chars"][rows, int(offset) : current + 1],
                        raw["tty_colors"][rows, int(offset) : current + 1],
                        raw["tty_cursor"][rows, int(offset) : current + 1],
                        self.lookup,
                        self.tile_atlas,
                    )
                    targets = batch_player_centered_tile_crops(
                        raw["tty_chars"][rows, current + 1],
                        raw["tty_colors"][rows, current + 1],
                        raw["tty_cursor"][rows, current + 1],
                        self.lookup,
                        self.tile_atlas,
                    )
                    goals = batch_player_centered_tile_crops(
                        raw["tty_chars"][rows, goal_indices[rows]],
                        raw["tty_colors"][rows, goal_indices[rows]],
                        raw["tty_cursor"][rows, goal_indices[rows]],
                        self.lookup,
                        self.tile_atlas,
                    )
                    result = {
                        "history": {"pixels": self._pixels(histories)},
                        "target": {"pixels": self._pixels(targets)},
                        "goal": {"pixels": self._pixels(goals)},
                        "goal_offset": torch.from_numpy(goal_offsets[rows].copy()).long(),
                        "trajectory_id": torch.from_numpy(
                            selected_gameids[rows, 0].copy()
                        ).long(),
                        "source": torch.zeros(len(rows), dtype=torch.long),
                    }
                    if inferred_actions is not None:
                        result["action"] = torch.from_numpy(
                            inferred_actions[rows].copy()
                        ).long()
                    elif "keypresses" in raw:
                        result["action"] = torch.from_numpy(
                            raw["keypresses"][rows, current].copy()
                        ).long()
                    yield result
