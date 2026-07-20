"""Export NLE levels for the browser demo using REAL captured crops.

Validated finding (diagnose2): the model navigates at eval-level accuracy only when
fed the *real* NLE converter crop at each cell; a reconstructed-from-terrain crop is
subtly off and stalls. So we capture the exact converter crop at every reachable cell
(frontier explore + a coverage tour), decompose each 144x144 crop into a 9x9 grid of
atlas tile indices, and ship those. The browser just blits stored tiles -- no lookup,
wall-resolution, or fog logic needed, and the crops are byte-identical to training.

Per level we emit: start cell, goals (varied distance), and every reachable cell's 9x9
tile-index grid. Movement/masking use the captured-cell set. A compact atlas holds only
the referenced tiles. Each candidate level is solve-tested with the GPU torch policy
(sampling, same masking the browser uses) and kept only if it mostly solves.
"""
from __future__ import annotations
import argparse, json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from minihack.tiles.glyph_mapper import GlyphMapper
from PIL import Image

from pan1ni.data.tile_converter import build_canonical_lookup
from pan1ni.models.model import GoalConditionedLeWorldModel
from pan1ni.models.config import ModelConfig
from pan1ni.models.action import DirectPolicyHead, feature_dim
from pan1ni.eval.multilevel import (
    _make_env, _teleport, _clear_more, _frontiers, _bfs_first_step,
    _pos, _dlvl, _cheb, DELTAS, D2A, TileConverter,
)

BLOCK = frozenset(map(ord, "|-+ "))
_CACHE = Path("/tmp/claude-0/-workspace-pan1ni/05ce2056-535d-456e-b5b0-441c15b27289/scratchpad/lookup_atlas.npz")


def load_atlas():
    if _CACHE.exists():
        return np.load(_CACHE)["atlas"]
    build_canonical_lookup()  # warms nothing we need here, but keep parity
    mapper = GlyphMapper()
    return np.stack([mapper.tiles[i] for i in range(max(mapper.tiles) + 1)])


class BrowserPolicy(torch.nn.Module):
    def __init__(self, model, head):
        super().__init__(); self.model = model; self.head = head

    def forward(self, history_pixels, goal_pixels):
        grouped = self.model.encode_group({"pixels": history_pixels}, {"pixels": goal_pixels})
        history_z, goal_z = grouped[:, :-1], grouped[:, -1]
        current = history_z[:, -1]
        steps = history_z.size(1)
        value = self.model.predictor.dropout(history_z + self.model.predictor.position[:, :steps])
        conditioning = goal_z[:, None].expand(-1, steps, -1)
        for layer in self.model.predictor.layers:
            value = layer(value, conditioning)
        hidden = self.model.predictor.norm(value)
        prediction = self.model.pred_proj(hidden.flatten(0, 1)).unflatten(0, hidden.shape[:2])
        predicted_next = prediction[:, -1]
        feat = torch.cat((current, predicted_next, predicted_next - current), dim=-1)
        return self.head(feat)


def build_torch_policy(world_ckpt, head_ckpt, device):
    w = torch.load(world_ckpt, map_location=device, weights_only=False)
    model = GoalConditionedLeWorldModel(ModelConfig(**w["config"])).to(device)
    model.load_state_dict(w["model"]); model.eval()
    hp = torch.load(head_ckpt, map_location=device, weights_only=False)
    c = hp["config"]
    head = DirectPolicyHead(int(c.get("feature_dim") or feature_dim(c["feature"], w["config"]["latent_dim"])),
                            int(c.get("num_classes", 8)), hidden_dim=int(c.get("action_hidden_dim", 1024)),
                            hidden_layers=int(c.get("action_hidden_layers", 2))).to(device)
    head.load_state_dict(hp["head"]); head.eval()
    return BrowserPolicy(model, head).to(device).eval()


def tiles_of(pixels, tile_bytes_to_id):
    """Decompose a 144x144x3 crop into a 9x9 grid of atlas tile ids (lossless)."""
    grid = np.zeros((9, 9), np.int32)
    for r in range(9):
        for c in range(9):
            block = pixels[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
            grid[r, c] = tile_bytes_to_id[block.tobytes()]
    return grid


def passable(terr_ch, r, c):
    return 1 <= r <= 21 and 0 <= c < 80 and int(terr_ch[r, c]) not in BLOCK


def explore_and_cover(env, level, conv, budget):
    """Frontier-explore, then tour every reachable cell, capturing the real crop at each."""
    o, _ = env.reset()
    o = _teleport(env, o, level)
    if _dlvl(o) != level:
        return None
    start = tuple(map(int, o["tty_cursor"]))
    terr_ch = np.full((24, 80), ord(" "), np.uint8)
    real = {}

    def learn(o):
        ch = o["tty_chars"]
        cr, cc = map(int, o["tty_cursor"])
        for r in range(1, 22):
            for c in range(80):
                if (r, c) != (cr, cc):
                    terr_ch[r, c] = ch[r, c]
        if (cr, cc) not in real:
            try:
                real[(cr, cc)] = conv(o).copy()
            except Exception:
                pass

    # phase 1: frontier exploration to reveal the level
    stuck = 0
    for _ in range(budget):
        o = _clear_more(env, o)
        learn(o)
        r, c = map(int, o["tty_cursor"])
        step = _bfs_first_step(o["tty_chars"], r, c, _frontiers(o["tty_chars"]))
        if step is None:
            break
        p = _pos(o)
        o, _, term, trunc, _ = env.step(D2A[(step[0] - r, step[1] - c)])
        stuck = stuck + 1 if _pos(o) == p else 0
        if stuck > 18 or term or trunc:
            break
    learn(o)

    # reachable set from revealed terrain
    reach = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DELTAS:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in reach and passable(terr_ch, nr, nc):
                reach.add((nr, nc))
                q.append((nr, nc))

    # phase 2: coverage tour -- walk to every uncaptured reachable cell
    guard = 0
    while (set(reach) - set(real)) and guard < len(reach) * 8:
        guard += 1
        r, c = map(int, o["tty_cursor"])
        step = _bfs_first_step(o["tty_chars"], r, c, set(reach) - set(real))
        if step is None:
            break
        o, _, term, trunc, _ = env.step(D2A[(step[0] - r, step[1] - c)])
        o = _clear_more(env, o)
        learn(o)
        if term or trunc:
            break

    return terr_ch, start, reach, real


@torch.no_grad()
def solve(policy, device, cells, start, goal, max_steps, seed, ctx=8):
    """Grid-sim identical to the browser: blit stored crops, sample, mask to cell set."""
    rng = np.random.default_rng(seed)
    goal_in = torch.from_numpy(cells[goal].transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    cur = start
    hist = deque([cells[cur]] * ctx, maxlen=ctx)
    best = _cheb(cur, goal)
    for _ in range(max_steps):
        hist.append(cells[cur])
        h = torch.from_numpy(np.stack(hist).transpose(0, 3, 1, 2)[None].astype(np.float32)).to(device)
        lg = policy(h, goal_in)[0].cpu().numpy()
        mask = [(cur[0] + dr, cur[1] + dc) in cells for dr, dc in DELTAS]
        if not any(mask):
            mask = [True] * 8
        lg = np.where(mask, lg, -1e9)
        p = np.exp(lg - lg.max()); p /= p.sum()
        a = int(rng.choice(8, p=p))
        nxt = (cur[0] + DELTAS[a][0], cur[1] + DELTAS[a][1])
        if nxt in cells:
            cur = nxt
        best = min(best, _cheb(cur, goal))
        if cur == goal:
            return True, best
    return False, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--need", type=int, default=4)
    ap.add_argument("--seed", type=int, default=500000)
    ap.add_argument("--world", default="reports/pixel-mse-player-only-sigreg02-400k/checkpoint.pt")
    ap.add_argument("--head", default="reports/pixel-mse-player-only-sigreg02-400k-action/idm/best_action_checkpoint.pt")
    ap.add_argument("--out", default="nethack-navigator")
    ap.add_argument("--explore-budget", type=int, default=500)
    ap.add_argument("--min-span", type=int, default=14)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    print("loading atlas + policy ...", flush=True)
    atlas = load_atlas()
    tile_bytes_to_id = {atlas[i].tobytes(): i for i in range(len(atlas))}
    policy = build_torch_policy(a.world, a.head, a.device)

    out = Path(a.out)
    for sub in ("levels", "goals", "assets"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    conv = TileConverter()
    used_tiles = set()
    kept = []
    seed = a.seed
    attempts = 0
    while len(kept) < a.need and attempts < a.need * 60:
        level = a.levels[len(kept) % len(a.levels)]
        attempts += 1
        s = seed
        seed += 1
        try:
            env = _make_env(s, a.explore_budget + 200)
            try:
                res = explore_and_cover(env, level, conv, a.explore_budget)
            finally:
                env.close()
            if res is None:
                continue
            terr_ch, start, reach, real = res
            coverage = len(real) / max(len(reach), 1)
            if start not in real or coverage < 0.9 or len(real) < 30:
                continue
            cells = real  # captured cells only (guaranteed crops)
            dists = {rc: _cheb(rc, start) for rc in cells if rc != start}
            span = max(dists.values())
            if span < a.min_span:
                continue
            targets = np.linspace(5, span, 4).round().astype(int)
            goals = []
            for t in targets:
                cand = min(dists, key=lambda rc: (abs(dists[rc] - t), rc))
                if cand not in [g[0] for g in goals]:
                    goals.append((cand, int(dists[cand])))
            if len(goals) < 3:
                continue
            gstats = []
            for gc, gd in goals:
                r = [solve(policy, a.device, cells, start, gc, gd * 14 + 40, sd) for sd in range(6)]
                gstats.append((gc, gd, sum(x[0] for x in r), 6))
            solved_goals = sum(1 for _, _, sc, _ in gstats if sc >= 3)
            print(f"Dlvl{level} seed{s}: cover {coverage:.0%} span {span} "
                  f"goals {[(gd, f'{sc}/6') for _, gd, sc, _ in gstats]} ok{solved_goals}/{len(gstats)}",
                  flush=True)
            if solved_goals < max(2, len(gstats) - 1):
                continue

            idx = len(kept) + 1
            cell_grids = {}
            for rc, px in cells.items():
                grid = tiles_of(px, tile_bytes_to_id)
                used_tiles.update(int(x) for x in grid.flat)
                cell_grids[f"{rc[0]},{rc[1]}"] = grid.tolist()
            level_obj = {
                "id": idx, "dlvl": level, "seed": s,
                "start": list(start),
                "goals": [{"rc": list(gc), "dist": gd,
                           "solve_rate": f"{sc}/6"} for (gc, gd), (_, _, sc, _) in zip(goals, gstats)],
                "cells": cell_grids,
            }
            (out / "levels" / f"level{idx}.json").write_text(json.dumps(level_obj))
            for gi, (gc, gd) in enumerate(goals, 1):
                Image.fromarray(cells[gc]).resize((144, 144), Image.NEAREST).save(
                    out / "goals" / f"level{idx}_goal{gi}_d{gd}.png")
            kept.append(level_obj)
            print(f"  KEPT level{idx} (Dlvl{level}, {len(cells)} cells)", flush=True)
            if a.smoke:
                break
        except Exception as exc:
            print(f"  attempt {attempts} (Dlvl{level}) skipped: {type(exc).__name__}: {exc}", flush=True)
            continue

    # compact atlas
    uniq = sorted(used_tiles)
    remap = {t: i for i, t in enumerate(uniq)}
    cols = 16
    rows = (len(uniq) + cols - 1) // cols
    sheet = np.zeros((rows * 16, cols * 16, 3), np.uint8)
    for i, t in enumerate(uniq):
        rr, cc = divmod(i, cols)
        sheet[rr * 16:(rr + 1) * 16, cc * 16:(cc + 1) * 16] = atlas[t]
    Image.fromarray(sheet).save(out / "assets" / "atlas.png")
    # remap every level's tile ids into compact indices
    for lf in (out / "levels").glob("level*.json"):
        obj = json.loads(lf.read_text())
        for k, grid in obj["cells"].items():
            obj["cells"][k] = [[remap[t] for t in row] for row in grid]
        lf.write_text(json.dumps(obj))
    (out / "assets" / "atlas.json").write_text(json.dumps({"tile_size": 16, "cols": cols, "count": len(uniq)}))
    print(f"EXPORT DONE: {len(kept)} levels, {len(uniq)} atlas tiles", flush=True)


if __name__ == "__main__":
    main()
