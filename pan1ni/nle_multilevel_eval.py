"""Multi-level closed-loop navigation eval.

Uses NetHack wizard mode to teleport the setup agent to a random dungeon level
(varied starting level + map), a frontier-BFS explorer to map that level and pick a
genuinely far goal, then hands the current position to the frozen movement policy,
which must navigate same-level to the goal. Setups that fail to produce a far-enough
goal are rejected, so the highly-variable NetHack setup never blocks the eval.

Only the setup phase uses the full action set (teleport, prompt-clearing); the policy
uses the eight movement classes exactly as in nle_closed_loop_eval.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import tempfile
from collections import Counter, deque
from pathlib import Path

import gymnasium as gym
import nle.env  # noqa: F401 - registers the NetHack environments
import numpy as np
import torch
from minihack.tiles.glyph_mapper import GlyphMapper
from nle import nethack
from PIL import Image, ImageDraw, ImageFont

from .action import DirectPolicyHead, feature_dim, predictor_features
from .config import ModelConfig
from .minihack_report import _render_frame
from .model import GoalConditionedLeWorldModel
from .player_tile_converter import build_canonical_lookup, player_centered_tile_crop
from .player_tile_data import SEMANTIC_ACTION_NAMES, _pixel_observation


COMBAT_KW = ("You hit", "You miss", "You kill", "You destroy", "You swing",
             "You bite", "You attack", "You smite", "You force")


def _is_attack(message: str) -> bool:
    return any(k in message for k in COMBAT_KW)


def _compose(current, goal, *, step, level, action_name, confidence, distance, feature, message=""):
    left = _render_frame(current, "Live full-NLE tile observation")
    right = _render_frame(goal, "Reachable goal frame")
    margin, header, footer = 16, 60, 108
    canvas = Image.new("RGB", (left.width + right.width + margin * 3, header + left.height + footer), "#0c0f0d")
    canvas.paste(left, (margin, header))
    canvas.paste(right, (left.width + margin * 2, header))
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    title = ImageFont.truetype(font_path, 17)
    body = ImageFont.truetype(font_path, 14)
    draw.text((margin, 12), f"Multi-level closed loop · Dlvl {level}", font=title, fill="#e7ebf2")
    draw.text((margin, 36), f"feature: {feature}   same tty->tile conversion as human pretraining", font=body, fill="#98a3b5")
    y = header + left.height + 12
    draw.text((margin, y), f"step {step:03d}   action {action_name:<10} conf {confidence:5.0%}", font=body, fill="#70d6a5")
    draw.text((margin, y + 24), f"chebyshev distance to goal: {distance}", font=body, fill="#e7ebf2")
    if message:
        draw.text((margin, y + 48), message[:72], font=body,
                  fill="#ff6b6b" if _is_attack(message) else "#98a3b5")
    return canvas

COMPASS = tuple(nethack.CompassDirection)          # policy actions (indices 0-7)
CTRL_V, ENTER, ESC, SPACE = 22, 13, 27, 32
SETUP_ACTIONS = COMPASS + (CTRL_V, ENTER, ESC, SPACE) + tuple(range(48, 58))
AIDX = {value: index for index, value in enumerate(SETUP_ACTIONS)}
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1))
D2A = {delta: index for index, delta in enumerate(DELTAS)}
BLOCK = frozenset(map(ord, "|-+ "))                # walls, closed doors, unknown


def _make_env(seed: int, max_steps: int, spawn_monsters: bool = False):
    env = gym.make(
        "NetHackScore-v0",
        wizard=True,
        allow_all_modes=True,
        allow_all_yn_questions=True,
        actions=SETUP_ACTIONS,
        observation_keys=("tty_chars", "tty_colors", "tty_cursor", "blstats", "message"),
        max_episode_steps=max_steps,
        spawn_monsters=spawn_monsters,
    )
    env.unwrapped.seed(core=seed, disp=seed, reseed=True)
    return env


def _msg(o) -> str:
    return "".join(chr(c) if 32 <= c < 127 else " " for c in o["tty_chars"][0])


def _pos(o):
    return int(o["blstats"][0]), int(o["blstats"][1])   # (x, y)


def _dlvl(o) -> int:
    return int(o["blstats"][12])


def _cheb(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _clear_more(env, o):
    for _ in range(16):
        if "--More--" not in _msg(o):
            break
        o, _, term, trunc, _ = env.step(AIDX[SPACE])
        if term or trunc:
            break
    return o


def _teleport(env, o, level: int):
    o, _, _, _, _ = env.step(AIDX[CTRL_V])
    o = _clear_more(env, o)
    for digit in str(level):
        o, _, _, _, _ = env.step(AIDX[ord(digit)])
    o, _, _, _, _ = env.step(AIDX[ENTER])
    return _clear_more(env, o)


def _passable(ch, r, c) -> bool:
    return 1 <= r <= 21 and 0 <= c < 80 and int(ch[r, c]) not in BLOCK


_WALL_DOOR = frozenset(map(ord, "|-+"))


def _passable_mask(o):
    """Which of the eight compass moves are not into a known wall / closed door / edge.
    Unexplored ' ' is left available so the policy can still search into the fog."""

    ch = o["tty_chars"]
    r, c = map(int, o["tty_cursor"])
    mask = []
    for dr, dc in DELTAS:
        tr, tc = r + dr, c + dc
        mask.append(1 <= tr <= 21 and 0 <= tc < 80 and int(ch[tr, tc]) not in _WALL_DOOR)
    if not any(mask):
        mask = [True] * 8
    return mask


def _frontiers(ch):
    out = set()
    for r in range(1, 22):
        for c in range(80):
            if _passable(ch, r, c) and any(
                0 <= r + dr < 24 and 0 <= c + dc < 80 and int(ch[r + dr, c + dc]) == ord(" ")
                for dr, dc in DELTAS
            ):
                out.add((r, c))
    return out


def _bfs_first_step(ch, sr, sc, targets):
    if not targets:
        return None
    queue = deque([(sr, sc)])
    prev = {(sr, sc): None}
    while queue:
        r, c = queue.popleft()
        if (r, c) in targets and (r, c) != (sr, sc):
            cur = (r, c)
            while prev[cur] != (sr, sc):
                cur = prev[cur]
            return cur
        for dr, dc in DELTAS:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in prev and _passable(ch, nr, nc):
                prev[(nr, nc)] = (r, c)
                queue.append((nr, nc))
    return None


class TileConverter:
    def __init__(self):
        self.lookup = build_canonical_lookup()[0]
        mapper = GlyphMapper()
        self.atlas = np.stack([mapper.tiles[i] for i in range(max(mapper.tiles) + 1)])

    def __call__(self, o):
        return player_centered_tile_crop(
            o["tty_chars"], o["tty_colors"], o["tty_cursor"], self.lookup, self.atlas
        )[0]


def _setup_level(env, converter, level, explore_budget):
    """Teleport to `level`, frontier-explore, and return the reached cells with the
    player-centered tile crop captured at each. Returns (frames, positions_order) or
    None if the setup stalls immediately."""

    o, _ = env.reset()
    o = _teleport(env, o, level)
    if _dlvl(o) != level:
        return None
    arrival = _pos(o)            # the open upstairs landing cell -> the episode START
    frame = {}
    order = []
    stuck = 0
    for _ in range(explore_budget):
        o = _clear_more(env, o)
        p = _pos(o)
        if p not in frame:
            frame[p] = converter(o)
            order.append(p)
        ch = o["tty_chars"]
        r, c = map(int, o["tty_cursor"])
        step = _bfs_first_step(ch, r, c, _frontiers(ch))
        if step is None:
            break
        o, _, term, trunc, _ = env.step(D2A[(step[0] - r, step[1] - c)])
        stuck = stuck + 1 if _pos(o) == p else 0
        if stuck > 18 or term or trunc:
            break
    return (frame, order, arrival, o)


def _navigate_to(env, o, target, *, budget=400):
    """Walk the agent to `target` (a mapped, reachable cell) via BFS, clearing prompts.
    Returns the observation; caller checks whether the target was actually reached."""

    stuck = 0
    for _ in range(budget):
        o = _clear_more(env, o)
        if _pos(o) == target:
            break
        r, c = map(int, o["tty_cursor"])
        step = _bfs_first_step(o["tty_chars"], r, c, {target})
        if step is None:
            break
        p = _pos(o)
        o, _, term, trunc, _ = env.step(D2A[(step[0] - r, step[1] - c)])
        stuck = stuck + 1 if _pos(o) == p else 0
        if stuck > 12 or term or trunc:
            break
    return o


@torch.no_grad()
def _policy_episode(env, o, model, head, feature, converter, goal_pos, goal_frame,
                    start_pos, *, max_steps, device, temperature, generator, action_selection,
                    uniform_rng=None, video_path=None, level=1, fps=8):
    history = deque(maxlen=model.config.max_context)
    history.extend(converter(o).copy() for _ in range(model.config.max_context))
    positions = [start_pos]
    collisions = 0
    attacks = 0
    best = _cheb(start_pos, goal_pos)
    success = False
    counts = Counter()
    frames_dir = tempfile.TemporaryDirectory() if video_path is not None else None
    try:
        for step in range(max_steps):
            o = _clear_more(env, o)
            current = converter(o)
            game_msg = _msg(o).strip()
            if step:
                history.append(current)
            position = _pos(o)
            best = min(best, _cheb(position, goal_pos))
            mask = _passable_mask(o)
            if uniform_rng is not None:
                choices = [i for i in range(8) if mask[i]]
                action = uniform_rng.choice(choices)
                confidence = 1.0 / len(choices)
            else:
                batch = {
                    "history": _pixel_observation(list(history), history=True, device=device),
                    "goal": _pixel_observation([goal_frame], history=False, device=device),
                }
                logits = head(predictor_features(model, batch, feature))[:, :8].clone()
                blocked = torch.tensor([not m for m in mask], device=logits.device)
                logits[:, blocked] = float("-inf")
                probs = (logits / temperature).softmax(-1)
                if action_selection == "sample":
                    action = int(torch.multinomial(probs, 1, generator=generator).item())
                else:
                    action = int(probs.argmax(-1).item())
                confidence = float(probs[0, action].item())
            counts[action] += 1
            if frames_dir is not None:
                _compose(current, goal_frame, step=step, level=level,
                         action_name=SEMANTIC_ACTION_NAMES[action], confidence=confidence,
                         distance=_cheb(position, goal_pos), feature=feature or "uniform",
                         message=game_msg
                         ).save(Path(frames_dir.name) / f"f{step:05d}.png")
            previous = position
            o, _, term, trunc, _ = env.step(action)
            if _is_attack(_msg(o).strip()):
                attacks += 1
            position = _pos(o)
            positions.append(position)
            if position == previous:
                collisions += 1
            best = min(best, _cheb(position, goal_pos))
            if position == goal_pos:
                success = True
                break
            if term or trunc:
                break
        steps = step + 1
        if frames_dir is not None:
            try:  # append the final post-move frame so a success shows the agent landing on the goal
                _compose(converter(o), goal_frame, step=steps, level=level,
                         action_name="arrived" if success else "end", confidence=1.0,
                         distance=_cheb(_pos(o), goal_pos), feature=feature or "uniform",
                         message=_msg(o).strip()
                         ).save(Path(frames_dir.name) / f"f{steps:05d}.png")
            except Exception:
                pass
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                 "-i", str(Path(frames_dir.name) / "f%05d.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(video_path)],
                check=True,
            )
    finally:
        if frames_dir is not None:
            frames_dir.cleanup()
    revisits = max(0, len(positions) - len(set(positions)))
    initial = _cheb(start_pos, goal_pos)
    final = _cheb(positions[-1], goal_pos)
    return {
        "success": success,
        "steps": steps,
        "initial_distance": initial,
        "best_distance": best,
        "final_distance": final,
        "best_progress": (initial - best) / max(initial, 1),
        "final_progress": (initial - final) / max(initial, 1),
        "unique_positions": len(set(positions)),
        "revisit_rate": revisits / max(len(positions) - 1, 1),
        "collisions": collisions,
        "collision_rate": collisions / max(steps, 1),
        "attacks": attacks,
    }


@torch.no_grad()
def run(world_checkpoint, action_checkpoint, output, *, episodes, max_steps,
        min_goal_distance, explore_budget, levels, seed, device, action_selection,
        temperature, uniform_baseline, video_dir=None, num_videos=0,
        target_distance=None, distance_tol=2, spawn_monsters=False):
    world = torch.load(world_checkpoint, map_location=device, weights_only=False)
    model = GoalConditionedLeWorldModel(ModelConfig(**world["config"])).to(device)
    if model.config.observation_mode != "pixels":
        raise ValueError("multilevel eval requires the tile-pixel world model")
    model.load_state_dict(world["model"])
    model.eval()
    feature = None
    head = None
    if not uniform_baseline:
        payload = torch.load(action_checkpoint, map_location=device, weights_only=False)
        cfg = payload["config"]
        feature = cfg["feature"]
        head = DirectPolicyHead(
            int(cfg.get("feature_dim") or feature_dim(feature, model.config.latent_dim)),
            int(cfg.get("num_classes", 8)),
            hidden_dim=int(cfg.get("action_hidden_dim", 1024)),
            hidden_layers=int(cfg.get("action_hidden_layers", 2)),
        ).to(device)
        head.load_state_dict(payload["head"])
        head.eval()

    converter = TileConverter()
    records = []
    attempted = 0
    attempt_seed = seed
    while len(records) < episodes and attempted < episodes * 40:
        level = levels[len(records) % len(levels)]
        attempted += 1
        eseed = attempt_seed
        attempt_seed += 1
        try:
            # PASS 1: map the level and choose a goal at the target distance from arrival
            env = _make_env(eseed, explore_budget + 64)
            try:
                setup = _setup_level(env, converter, level, explore_budget)
            finally:
                env.close()
            if setup is None:
                continue
            frame, order, start, _ = setup   # start = open arrival cell
            if start not in frame:
                continue
            if target_distance is not None:
                goal = min(frame, key=lambda q: abs(_cheb(q, start) - target_distance))
                if abs(_cheb(goal, start) - target_distance) > distance_tol:
                    continue
            else:
                goal = max(frame, key=lambda q: _cheb(q, start))
                if _cheb(goal, start) < min_goal_distance:
                    continue
            # PASS 2: fresh env, same seed -> same level + same arrival; run policy from arrival
            env = _make_env(eseed, max_steps + 64, spawn_monsters=spawn_monsters)
            try:
                o, _ = env.reset()
                o = _teleport(env, o, level)
                o = _clear_more(env, o)
                if _dlvl(o) != level or _pos(o) != start:
                    continue
                gen = torch.Generator(device=device).manual_seed(eseed + 91_117)
                urng = random.Random(eseed + 5) if uniform_baseline else None
                vpath = None
                if video_dir and len(records) < num_videos:
                    vpath = Path(video_dir) / f"ep{len(records):02d}-Dlvl{level}-d{_cheb(goal, start)}.mp4"
                rec = _policy_episode(
                    env, o, model, head, feature, converter, goal, frame[goal], start,
                    max_steps=max_steps, device=device, temperature=temperature,
                    generator=gen, action_selection=action_selection, uniform_rng=urng,
                    video_path=vpath, level=level,
                )
            finally:
                env.close()
            rec.update(episode=len(records), seed=eseed, level=level,
                       goal_distance=_cheb(goal, start))
            records.append(rec)
            print(f"ep {len(records):3d}/{episodes} | Dlvl {level} | dist {rec['goal_distance']:2d} | "
                  f"success {rec['success']} | best {rec['best_distance']} | atk {rec.get('attacks', 0)}", flush=True)
        except Exception as exc:  # NetHack states occasionally break the tile crop; skip that attempt
            print(f"  attempt {attempted} (Dlvl {level}) skipped: {type(exc).__name__}: {exc}", flush=True)
            continue

    def near(rs): return sum(r["best_distance"] <= 1 for r in rs) / max(len(rs), 1)
    successes = [r for r in records if r["success"]]
    summary = {
        "mode": "multilevel_closed_loop",
        "environment": "NetHackScore-v0 (wizard teleport)",
        "policy": "uniform_random" if uniform_baseline else f"{action_selection} · feature {feature}",
        "levels": list(levels),
        "min_goal_distance": min_goal_distance,
        "episodes": len(records),
        "successes": len(successes),
        "success_rate": len(successes) / max(len(records), 1),
        "reached_rate": near(records),
        "mean_best_distance": statistics.mean(r["best_distance"] for r in records) if records else None,
        "mean_goal_distance": statistics.mean(r["goal_distance"] for r in records) if records else None,
        "level_distribution": dict(Counter(r["level"] for r in records)),
        "collision_rate": statistics.mean(r["collision_rate"] for r in records) if records else None,
        "total_attacks": sum(r.get("attacks", 0) for r in records),
        "episodes_with_attacks": sum(1 for r in records if r.get("attacks", 0) > 0),
        "spawn_monsters": bool(spawn_monsters),
        "attempts": attempted,
        "records": records,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def main():
    p = argparse.ArgumentParser(description="Multi-level closed-loop navigation eval (wizard teleport)")
    p.add_argument("--world-checkpoint", required=True, type=Path)
    p.add_argument("--action-checkpoint", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--min-goal-distance", type=int, default=8)
    p.add_argument("--explore-budget", type=int, default=300)
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--action-selection", choices=("argmax", "sample"), default="sample")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--uniform-baseline", action="store_true")
    p.add_argument("--video-dir", type=Path)
    p.add_argument("--num-videos", type=int, default=0)
    p.add_argument("--target-distance", type=int)
    p.add_argument("--distance-tol", type=int, default=2)
    p.add_argument("--spawn-monsters", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if not a.uniform_baseline and a.action_checkpoint is None:
        p.error("--action-checkpoint required unless --uniform-baseline")
    print(run(a.world_checkpoint, a.action_checkpoint, a.output, episodes=a.episodes,
              max_steps=a.max_steps, min_goal_distance=a.min_goal_distance,
              explore_budget=a.explore_budget, levels=a.levels, seed=a.seed, device=a.device,
              action_selection=a.action_selection, temperature=a.temperature,
              uniform_baseline=a.uniform_baseline, video_dir=a.video_dir, num_videos=a.num_videos,
              target_distance=a.target_distance, distance_tol=a.distance_tol,
              spawn_monsters=a.spawn_monsters))


if __name__ == "__main__":
    main()
