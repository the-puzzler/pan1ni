from __future__ import annotations

import torch

from .data import Trajectory


def make_synthetic_trajectories(
    count: int = 8,
    length: int = 80,
    height: int = 8,
    width: int = 16,
    status_dim: int = 27,
    num_actions: int = 8,
    seed: int = 0,
) -> list[Trajectory]:
    """Small deterministic random-walk episodes for tests and plumbing checks."""

    generator = torch.Generator().manual_seed(seed)
    trajectories: list[Trajectory] = []
    for episode in range(count):
        actions = torch.randint(0, num_actions, (length - 1,), generator=generator)
        cursor = torch.zeros(length, 2, dtype=torch.long)
        cursor[0] = torch.tensor((height // 2, width // 2))
        deltas = torch.tensor(((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (0, 0), (0, 0), (0, 0)))
        for step in range(1, length):
            cursor[step] = cursor[step - 1] + deltas[actions[step - 1]]
            cursor[step, 0].clamp_(0, height - 1)
            cursor[step, 1].clamp_(0, width - 1)
        chars = torch.full((length, height, width), ord("."), dtype=torch.long)
        colors = torch.zeros_like(chars)
        for step, (row, column) in enumerate(cursor.tolist()):
            chars[step, row, column] = ord("@")
            colors[step, row, column] = 15
        message = torch.zeros(length, 32, dtype=torch.long)
        message[:, :7] = torch.tensor(list(b"explore"))
        status = torch.randn(length, status_dim, generator=generator) * 0.01
        status[:, 0] = cursor[:, 1]
        status[:, 1] = cursor[:, 0]
        trajectories.append(
            Trajectory(
                {
                    "chars": chars,
                    "colors": colors,
                    "bg_colors": torch.zeros_like(colors),
                    "message": message,
                    "status": status,
                    "cursor": cursor,
                },
                actions,
                role="synthetic",
                max_dungeon_level=1,
            )
        )
    return trajectories


def make_goal_directed_trajectories(
    count: int = 128,
    height: int = 8,
    width: int = 16,
    status_dim: int = 27,
    min_distance: int = 8,
    seed: int = 0,
) -> list[Trajectory]:
    """Create causal trajectories that deterministically navigate to a hidden goal.

    The current observation contains the agent but no goal marker. The goal becomes
    observable only through the episode's final frame. At every transition the agent
    moves along the largest remaining coordinate difference (horizontal on ties), so
    ``(current state, final state)`` uniquely determines the next movement.
    """

    if min_distance < 1 or min_distance > height + width - 2:
        raise ValueError("min_distance must be reachable on the requested grid")
    generator = torch.Generator().manual_seed(seed)
    action_deltas = torch.tensor(((-1, 0), (1, 0), (0, -1), (0, 1)))
    trajectories: list[Trajectory] = []
    for _ in range(count):
        while True:
            start = torch.tensor(
                (
                    torch.randint(height, (), generator=generator).item(),
                    torch.randint(width, (), generator=generator).item(),
                ),
                dtype=torch.long,
            )
            goal = torch.tensor(
                (
                    torch.randint(height, (), generator=generator).item(),
                    torch.randint(width, (), generator=generator).item(),
                ),
                dtype=torch.long,
            )
            if (goal - start).abs().sum().item() >= min_distance:
                break

        positions = [start]
        actions: list[int] = []
        current = start.clone()
        while not torch.equal(current, goal):
            row_delta, column_delta = (goal - current).tolist()
            if column_delta and abs(column_delta) >= abs(row_delta):
                action = 3 if column_delta > 0 else 2
            else:
                action = 1 if row_delta > 0 else 0
            current = current + action_deltas[action]
            actions.append(action)
            positions.append(current.clone())

        cursor = torch.stack(positions)
        length = cursor.shape[0]
        chars = torch.full((length, height, width), ord("."), dtype=torch.long)
        colors = torch.zeros_like(chars)
        for step, (row, column) in enumerate(cursor.tolist()):
            chars[step, row, column] = ord("@")
            colors[step, row, column] = 15
        message = torch.zeros(length, 32, dtype=torch.long)
        message[:, :9] = torch.tensor(list(b"find exit"))
        status = torch.zeros(length, status_dim)
        status[:, 0] = cursor[:, 1]
        status[:, 1] = cursor[:, 0]
        status[:, 2] = torch.arange(length)
        trajectories.append(
            Trajectory(
                {
                    "chars": chars,
                    "colors": colors,
                    "bg_colors": torch.zeros_like(colors),
                    "message": message,
                    "status": status,
                    "cursor": cursor,
                },
                torch.tensor(actions, dtype=torch.long),
                role="goal_directed",
                max_dungeon_level=1,
            )
        )
    return trajectories
