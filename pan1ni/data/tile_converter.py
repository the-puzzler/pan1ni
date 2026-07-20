from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gymnasium as gym
import h5py
import nle.dataset as nld
import nle.env  # noqa: F401 - registers the NetHack environments
import numpy as np
from minihack.tiles import glyph2tile
from minihack.tiles.glyph_mapper import GlyphMapper
from nle import nethack
from PIL import Image, ImageDraw, ImageFont

from pan1ni.data.minihack import (
    GOAL_POSITIONS,
    KEY_POSITIONS,
    KeyRoomOracle,
    make_keyroom_env,
)


NLE_COLORS = (
    "#111318", "#b94a48", "#5cab63", "#b58b45",
    "#6577c8", "#a960b8", "#55a7ac", "#c8ccd4",
    "#626773", "#ff6b68", "#7fe089", "#f5d76e",
    "#8298ff", "#df83ef", "#76e5eb", "#ffffff",
)


def _add_full_observation(
    counts: dict[tuple[int, int], Counter[int]], observation: dict
) -> None:
    # NLE's 21x79 glyph map aligns with tty rows 1:22 and columns 0:79.
    chars = observation["tty_chars"][1:22, :79]
    colors = observation["tty_colors"][1:22, :79].astype(np.int16) & 15
    glyphs = observation["glyphs"]
    for char, color, glyph in zip(chars.flat, colors.flat, glyphs.flat):
        counts[(int(char), int(color))][int(glyph)] += 1


def build_canonical_lookup(
    *, simulator_episodes: int = 64, simulator_steps: int = 32, seed: int = 0
) -> tuple[np.ndarray, dict]:
    """Build a deterministic tty character/color -> canonical glyph table.

    Live paired NLE observations establish the common terrain mappings. Monster
    and object tables fill rare symbols that were not observed. Where tty
    rendering discarded identity, the most frequent canonical glyph is used.
    """

    counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    env = gym.make(
        "NetHackScore-v0",
        observation_keys=("glyphs", "tty_chars", "tty_colors", "blstats"),
    )
    rng = np.random.default_rng(seed)
    try:
        for episode in range(simulator_episodes):
            observation, _ = env.reset(seed=seed + episode)
            _add_full_observation(counts, observation)
            for _ in range(simulator_steps):
                # The first eight actions are compass movements. Restricting
                # sampling to them avoids menus whose tty no longer aligns with
                # the still-live glyph map.
                observation, _, terminated, truncated, _ = env.step(
                    int(rng.integers(0, 8))
                )
                x, y = map(int, observation["blstats"][:2])
                if (
                    0 <= y < 21
                    and 0 <= x < 79
                    and observation["tty_chars"][y + 1, x] == ord("@")
                ):
                    _add_full_observation(counts, observation)
                if terminated or truncated:
                    break
    finally:
        env.close()

    empirical_pairs = set(counts)
    # Calibrate against the exact MiniHack representation used by the target
    # task. These paired tty/pixel observations resolve task-critical aliases
    # such as the player sprite, staircase, key, and locked/open door.
    mapper = GlyphMapper()
    tile_atlas = np.stack([mapper.tiles[index] for index in range(max(mapper.tiles) + 1)])
    tile_bytes_to_id = {tile.tobytes(): index for index, tile in enumerate(tile_atlas)}
    glyph_by_tile: dict[int, int] = {}
    for glyph, tile_id in enumerate(np.asarray(glyph2tile)):
        glyph_by_tile.setdefault(int(tile_id), glyph)
    keyroom_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    variants = [(key, goal) for key in KEY_POSITIONS for goal in GOAL_POSITIONS]
    for variant_index, (key_position, goal_position) in enumerate(variants):
        keyroom_env = make_keyroom_env(
            key_position=key_position,
            goal_position=goal_position,
            seed=10_000 + variant_index,
        )
        try:
            episode = KeyRoomOracle(keyroom_env).collect()
        finally:
            keyroom_env.close()
        for frame_index, (x_position, y_position) in enumerate(episode.positions):
            center_row, center_column = int(y_position) + 1, int(x_position)
            for crop_row, tty_row in enumerate(range(center_row - 4, center_row + 5)):
                for crop_column, tty_column in enumerate(
                    range(center_column - 4, center_column + 5)
                ):
                    if not (1 <= tty_row <= 21 and 0 <= tty_column < 79):
                        continue
                    char = int(episode.tty_chars[frame_index, tty_row, tty_column])
                    color = int(episode.tty_colors[frame_index, tty_row, tty_column]) & 15
                    tile = episode.pixels[
                        frame_index,
                        crop_row * 16 : (crop_row + 1) * 16,
                        crop_column * 16 : (crop_column + 1) * 16,
                    ]
                    tile_id = tile_bytes_to_id.get(tile.tobytes())
                    if tile_id is not None:
                        keyroom_counts[(char, color)][glyph_by_tile[tile_id]] += 1
    # Target-task pairs take precedence over generic NetHack frequency.
    for key, candidates in keyroom_counts.items():
        counts[key] = candidates
        empirical_pairs.add(key)

    # Add normal monster and object identities only for pairs unseen in live
    # observations. This broadens coverage without overriding terrain frequency.
    static: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for index in range(nethack.NUMMONS):
        monster = nethack.permonst(index)
        symbol = nethack.class_sym.from_mlet(monster.mlet).sym
        static[(ord(symbol), int(monster.mcolor) & 15)][
            nethack.GLYPH_MON_OFF + index
        ] += 1
    for index in range(nethack.NUM_OBJECTS):
        obj = nethack.objclass(index)
        symbol = nethack.class_sym.from_oc_class(obj.oc_class).sym
        static[(ord(symbol), int(obj.oc_color) & 15)][
            nethack.GLYPH_OBJ_OFF + index
        ] += 1
    for key, candidates in static.items():
        if key not in counts:
            counts[key].update(candidates)

    # NAO server tty options render boulders as '0' rather than the default '`'.
    boulder_index = next(
        index
        for index in range(nethack.NUM_OBJECTS)
        if (nethack.objdescr.from_idx(index).oc_name or "") == "boulder"
    )
    aliases = {(ord("0"), 7): nethack.GLYPH_OBJ_OFF + boulder_index}

    by_char: dict[int, Counter[int]] = defaultdict(Counter)
    for (char, _), candidates in counts.items():
        by_char[char].update(candidates)
    default_glyph = counts.get((ord(" "), 0), Counter({nethack.GLYPH_CMAP_OFF: 1})).most_common(1)[0][0]
    lookup = np.full((256, 16), default_glyph, dtype=np.int16)
    source = np.full((256, 16), "default", dtype="U9")
    candidate_counts = np.zeros((256, 16), dtype=np.int16)
    for char in range(256):
        char_fallback = by_char.get(char)
        for color in range(16):
            key = (char, color)
            if key in aliases:
                lookup[char, color] = aliases[key]
                source[char, color] = "alias"
                candidate_counts[char, color] = 1
            elif key in counts:
                lookup[char, color] = counts[key].most_common(1)[0][0]
                source[char, color] = "empirical" if key in empirical_pairs else "static"
                candidate_counts[char, color] = len(counts[key])
            elif char_fallback:
                lookup[char, color] = char_fallback.most_common(1)[0][0]
                source[char, color] = "char"
                candidate_counts[char, color] = len(char_fallback)
    metadata = {
        "simulator_episodes": simulator_episodes,
        "simulator_steps": simulator_steps,
        "keyroom_calibration_pairs": len(keyroom_counts),
        "empirical_pairs": len(empirical_pairs),
        "all_exact_pairs": len(counts),
        "ambiguous_exact_pairs": sum(len(value) > 1 for value in counts.values()),
        "lookup_source": source,
        "candidate_counts": candidate_counts,
    }
    return lookup, metadata


def player_centered_tile_crop(
    chars: np.ndarray,
    colors: np.ndarray,
    cursor: np.ndarray,
    lookup: np.ndarray,
    tiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one tty frame to the same 9x9 / 144x144 MiniHack tile crop."""

    center_row, center_column = map(int, cursor)
    if not (1 <= center_row <= 21 and 0 <= center_column < 79):
        raise ValueError("tty cursor is outside the playable map")
    if chars[center_row, center_column] != ord("@"):
        raise ValueError("tty cursor is not on the player glyph")
    crop_chars = np.full((9, 9), ord(" "), dtype=np.uint8)
    crop_colors = np.zeros((9, 9), dtype=np.int16)
    for crop_row, tty_row in enumerate(range(center_row - 4, center_row + 5)):
        for crop_column, tty_column in enumerate(
            range(center_column - 4, center_column + 5)
        ):
            if 1 <= tty_row <= 21 and 0 <= tty_column < 79:
                crop_chars[crop_row, crop_column] = chars[tty_row, tty_column]
                crop_colors[crop_row, crop_column] = int(colors[tty_row, tty_column]) & 15
    glyphs = lookup[crop_chars, crop_colors].copy()
    # TTY collapses several wall-junction cmap glyphs to the same '-' symbol.
    # Their orientation is recoverable from the four neighboring wall cells.
    # Bits are north/east/south/west; values are NetHack cmap indices learned
    # from the paired native KeyRoom calibration trajectories above.
    horizontal_wall_cmap = {
        2: 2, 3: 8, 5: 13, 6: 11, 8: 2,
        9: 8, 10: 2, 11: 8, 12: 11, 14: 11,
    }
    wall_codes = (ord("-"), ord("|"), ord("+"))
    for row in range(9):
        for column in range(9):
            char = int(crop_chars[row, column])
            if char not in (ord("-"), ord("|")):
                continue
            mask = 0
            for bit, (delta_row, delta_column) in enumerate(
                ((-1, 0), (0, 1), (1, 0), (0, -1))
            ):
                neighbor_row = row + delta_row
                neighbor_column = column + delta_column
                if (
                    0 <= neighbor_row < 9
                    and 0 <= neighbor_column < 9
                    and int(crop_chars[neighbor_row, neighbor_column]) in wall_codes
                ):
                    mask |= 1 << bit
            if char == ord("|"):
                glyphs[row, column] = nethack.GLYPH_CMAP_OFF + 1
            elif mask in horizontal_wall_cmap:
                glyphs[row, column] = (
                    nethack.GLYPH_CMAP_OFF + horizontal_wall_cmap[mask]
                )
    tile_ids = np.asarray(glyph2tile)[glyphs]
    selected_tiles = tiles[tile_ids]
    pixels = selected_tiles.transpose(0, 2, 1, 3, 4).reshape(144, 144, 3)
    return pixels.astype(np.uint8, copy=False), np.stack((crop_chars, crop_colors))


def batch_player_centered_tile_crops(
    chars: np.ndarray,
    colors: np.ndarray,
    cursors: np.ndarray,
    lookup: np.ndarray,
    tiles: np.ndarray,
) -> np.ndarray:
    """Vectorized conversion for arrays shaped ``[..., 24, 80]``."""

    leading_shape = chars.shape[:-2]
    flat_chars = chars.reshape(-1, 24, 80)
    flat_colors = colors.reshape(-1, 24, 80)
    flat_cursors = cursors.reshape(-1, 2).astype(np.int64)
    count = flat_chars.shape[0]
    offsets = np.arange(-4, 5)
    rows = flat_cursors[:, 0, None, None] + offsets[None, :, None]
    columns = flat_cursors[:, 1, None, None] + offsets[None, None, :]
    rows = np.broadcast_to(rows, (count, 9, 9))
    columns = np.broadcast_to(columns, (count, 9, 9))
    valid = (rows >= 1) & (rows <= 21) & (columns >= 0) & (columns < 79)
    batch_indices = np.arange(count)[:, None, None]
    clipped_rows = rows.clip(0, 23)
    clipped_columns = columns.clip(0, 79)
    crop_chars = np.where(
        valid,
        flat_chars[batch_indices, clipped_rows, clipped_columns],
        ord(" "),
    ).astype(np.uint8)
    crop_colors = np.where(
        valid,
        flat_colors[batch_indices, clipped_rows, clipped_columns].astype(np.int16) & 15,
        0,
    )
    glyphs = lookup[crop_chars, crop_colors].copy()

    wall = np.isin(crop_chars, (ord("-"), ord("|"), ord("+")))
    padded = np.pad(wall, ((0, 0), (1, 1), (1, 1)))
    mask = (
        padded[:, :-2, 1:-1].astype(np.uint8)
        | (padded[:, 1:-1, 2:].astype(np.uint8) << 1)
        | (padded[:, 2:, 1:-1].astype(np.uint8) << 2)
        | (padded[:, 1:-1, :-2].astype(np.uint8) << 3)
    )
    cmap_by_mask = np.full(16, -1, dtype=np.int16)
    for wall_mask, cmap in {
        2: 2, 3: 8, 5: 13, 6: 11, 8: 2,
        9: 8, 10: 2, 11: 8, 12: 11, 14: 11,
    }.items():
        cmap_by_mask[wall_mask] = cmap
    horizontal = crop_chars == ord("-")
    resolved = cmap_by_mask[mask]
    resolve_horizontal = horizontal & (resolved >= 0)
    glyphs[resolve_horizontal] = nethack.GLYPH_CMAP_OFF + resolved[resolve_horizontal]
    glyphs[crop_chars == ord("|")] = nethack.GLYPH_CMAP_OFF + 1

    tile_ids = np.asarray(glyph2tile)[glyphs]
    selected = tiles[tile_ids]
    pixels = selected.transpose(0, 1, 3, 2, 4, 5).reshape(count, 144, 144, 3)
    return pixels.reshape(*leading_shape, 144, 144, 3).astype(np.uint8, copy=False)


def _valid_player_frames(raw: dict, batch_index: int) -> list[int]:
    result = []
    for timestep in range(raw["tty_chars"].shape[1]):
        if raw["gameids"][batch_index, timestep] == 0:
            continue
        row, column = map(int, raw["tty_cursor"][batch_index, timestep])
        if (
            1 <= row <= 21
            and 0 <= column < 79
            and raw["tty_chars"][batch_index, timestep, row, column] == ord("@")
        ):
            result.append(timestep)
    return result


def _render_tty_crop(crop: np.ndarray, *, scale: int = 2) -> Image.Image:
    """Render the exact 9x9 tty cells used by the converter."""

    chars, colors = crop
    cell = 16 * scale
    image = Image.new("RGB", (9 * cell, 9 * cell), "#080b11")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14 * scale
    )
    for row in range(9):
        for column in range(9):
            code = int(chars[row, column])
            glyph = bytes((code,)).decode("cp437", errors="replace") if code else " "
            draw.text(
                (column * cell + 3 * scale, row * cell),
                glyph,
                font=font,
                fill=NLE_COLORS[int(colors[row, column]) & 15],
            )
    return image


def create_side_by_side_collage(
    db_path: Path,
    dataset_name: str,
    output: Path,
    *,
    batch_size: int = 4,
    sequence_length: int = 1024,
) -> Path:
    lookup, metadata = build_canonical_lookup()
    mapper = GlyphMapper()
    tile_atlas = np.stack([mapper.tiles[index] for index in range(max(mapper.tiles) + 1)])
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        dataset = nld.TtyrecDataset(
            dataset_name,
            batch_size=batch_size,
            seq_length=sequence_length,
            dbfilename=str(db_path),
            threadpool=pool,
            shuffle=False,
        )
        raw = next(iter(dataset))

    panel = 288
    pair_gap = 10
    column_gap = 28
    left = 142
    top = 112
    row_height = panel + 68
    pair_width = panel * 2 + pair_gap
    width = left + pair_width * 2 + column_gap + 18
    height = top + row_height * batch_size + 18
    canvas = Image.new("RGB", (width, height), "#080b11")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    title_font = ImageFont.truetype(font_path, 21)
    body_font = ImageFont.truetype(font_path, 14)
    small_font = ImageFont.truetype(font_path, 12)
    draw.text(
        (18, 14),
        "REAL PLAYER SOURCE vs CANONICAL TILE CONVERSION",
        font=title_font,
        fill="#f2f5fa",
    )
    draw.text(
        (18, 47),
        "Each pair uses the identical 9×9 player-centered cells from the same game and tty frame.",
        font=small_font,
        fill="#9aa6b7",
    )
    draw.text((left, 80), "ORIGINAL TTY CROP", font=body_font, fill="#79a8ff")
    draw.text((left + panel + pair_gap, 80), "CONVERTED TILE CROP", font=body_font, fill="#70d6a5")
    second_x = left + pair_width + column_gap
    draw.text((second_x, 80), "ORIGINAL TTY CROP", font=body_font, fill="#79a8ff")
    draw.text((second_x + panel + pair_gap, 80), "CONVERTED TILE CROP", font=body_font, fill="#70d6a5")

    source_table = metadata["lookup_source"]
    candidate_counts = metadata["candidate_counts"]
    for row_index in range(batch_size):
        valid = _valid_player_frames(raw, row_index)
        if len(valid) < 3:
            raise RuntimeError(f"game batch {row_index} has only {len(valid)} valid player frames")
        chosen = [valid[len(valid) // 3], valid[(2 * len(valid)) // 3]]
        gameid = int(raw["gameids"][row_index, chosen[0]])
        y = top + row_index * row_height
        draw.text((18, y + 8), f"game {gameid}", font=body_font, fill="#f2f5fa")
        draw.text((18, y + 31), "human ttyrec", font=small_font, fill="#9aa6b7")
        for pair_index, timestep in enumerate(chosen):
            pixels, crop = player_centered_tile_crop(
                raw["tty_chars"][row_index, timestep],
                raw["tty_colors"][row_index, timestep],
                raw["tty_cursor"][row_index, timestep],
                lookup,
                tile_atlas,
            )
            chars, colors = crop
            sources = source_table[chars, colors]
            candidates = candidate_counts[chars, colors]
            fallback = int(np.isin(sources, ("char", "default")).sum())
            ambiguous = int((candidates > 1).sum())
            x = left + pair_index * (pair_width + column_gap)
            original = _render_tty_crop(crop)
            converted = Image.fromarray(pixels).resize(
                (panel, panel), Image.Resampling.NEAREST
            )
            canvas.paste(original, (x, y))
            canvas.paste(converted, (x + panel + pair_gap, y))
            draw.rectangle((x - 1, y - 1, x + panel, y + panel), outline="#364155")
            draw.rectangle(
                (
                    x + panel + pair_gap - 1,
                    y - 1,
                    x + panel * 2 + pair_gap,
                    y + panel,
                ),
                outline="#364155",
            )
            draw.text(
                (x + 4, y + panel + 8),
                f"tty frame {timestep:04d}",
                font=small_font,
                fill="#9aa6b7",
            )
            draw.text(
                (x + panel + pair_gap + 4, y + panel + 8),
                f"fallback {fallback}/81 · ambiguous {ambiguous}/81",
                font=small_font,
                fill="#f0b35a" if fallback or ambiguous else "#70d6a5",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def create_ground_truth_comparison(
    data_path: Path,
    output: Path,
    *,
    episode_keys: tuple[str, ...] = ("0", "5", "10", "15"),
) -> tuple[Path, Path]:
    """Compare TTY reconstruction against stored native pixel_crop ground truth."""

    lookup, metadata = build_canonical_lookup()
    mapper = GlyphMapper()
    tile_atlas = np.stack([mapper.tiles[index] for index in range(max(mapper.tiles) + 1)])
    panel = 288
    pair_gap = 10
    column_gap = 28
    left = 142
    top = 112
    row_height = panel + 70
    pair_width = panel * 2 + pair_gap
    width = left + pair_width * 2 + column_gap + 18
    height = top + row_height * len(episode_keys) + 18
    canvas = Image.new("RGB", (width, height), "#080b11")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    title_font = ImageFont.truetype(font_path, 21)
    body_font = ImageFont.truetype(font_path, 14)
    small_font = ImageFont.truetype(font_path, 12)
    draw.text(
        (18, 14),
        "KNOWN-CORRECT pixel_crop vs TTY-ONLY RECONSTRUCTION",
        font=title_font,
        fill="#f2f5fa",
    )
    draw.text(
        (18, 47),
        "Same simulator episode, state, crop, and frame. Left is stored native data; right never sees glyph IDs.",
        font=small_font,
        fill="#9aa6b7",
    )
    for x in (left, left + pair_width + column_gap):
        draw.text((x, 80), "TRUE NATIVE pixel_crop", font=body_font, fill="#79a8ff")
        draw.text((x + panel + pair_gap, 80), "TTY → CANONICAL TILES", font=body_font, fill="#70d6a5")

    records = []
    with h5py.File(data_path, "r") as handle:
        dataset_seed = int(handle.attrs["seed"])
        for row_index, episode_key in enumerate(episode_keys):
            group = handle[episode_key]
            stored_pixels = group["pixels"][:]
            key_position = tuple(map(int, group.attrs["key_position"]))
            goal_position = tuple(map(int, group.attrs["goal_position"]))
            variant_seed = dataset_seed + int(episode_key) % 16
            env = make_keyroom_env(
                key_position=key_position,
                goal_position=goal_position,
                seed=variant_seed,
            )
            try:
                recreated = KeyRoomOracle(env).collect()
            finally:
                env.close()
            if not np.array_equal(stored_pixels, recreated.pixels):
                raise RuntimeError(
                    f"episode {episode_key} recreation does not match stored native pixels"
                )
            chosen = (0, len(stored_pixels) - 1)
            y = top + row_index * row_height
            draw.text((18, y + 8), f"episode {episode_key}", font=body_font, fill="#f2f5fa")
            draw.text((18, y + 31), "stored dataset", font=small_font, fill="#9aa6b7")
            for pair_index, frame_index in enumerate(chosen):
                x_position, y_position = map(int, recreated.positions[frame_index])
                cursor = np.asarray((y_position + 1, x_position), dtype=np.int16)
                reconstructed, crop = player_centered_tile_crop(
                    recreated.tty_chars[frame_index],
                    recreated.tty_colors[frame_index],
                    cursor,
                    lookup,
                    tile_atlas,
                )
                truth = stored_pixels[frame_index]
                exact_tiles = sum(
                    np.array_equal(
                        truth[row * 16 : (row + 1) * 16, column * 16 : (column + 1) * 16],
                        reconstructed[row * 16 : (row + 1) * 16, column * 16 : (column + 1) * 16],
                    )
                    for row in range(9)
                    for column in range(9)
                )
                pixel_channel_agreement = float((truth == reconstructed).mean())
                mean_absolute_error = float(
                    np.abs(truth.astype(np.int16) - reconstructed.astype(np.int16)).mean()
                )
                chars, colors = crop
                sources = metadata["lookup_source"][chars, colors]
                fallback = int(np.isin(sources, ("char", "default")).sum())
                x = left + pair_index * (pair_width + column_gap)
                canvas.paste(
                    Image.fromarray(truth).resize((panel, panel), Image.Resampling.NEAREST),
                    (x, y),
                )
                canvas.paste(
                    Image.fromarray(reconstructed).resize(
                        (panel, panel), Image.Resampling.NEAREST
                    ),
                    (x + panel + pair_gap, y),
                )
                draw.rectangle((x - 1, y - 1, x + panel, y + panel), outline="#364155")
                draw.rectangle(
                    (x + panel + pair_gap - 1, y - 1, x + panel * 2 + pair_gap, y + panel),
                    outline="#364155",
                )
                draw.text(
                    (x + 4, y + panel + 8),
                    f"frame {frame_index:02d} · verified stored",
                    font=small_font,
                    fill="#9aa6b7",
                )
                draw.text(
                    (x + panel + pair_gap + 4, y + panel + 8),
                    f"exact tiles {exact_tiles}/81 · fallback {fallback}/81",
                    font=small_font,
                    fill="#70d6a5" if exact_tiles == 81 else "#f0b35a",
                )
                records.append(
                    {
                        "episode": int(episode_key),
                        "frame": frame_index,
                        "exact_tiles": exact_tiles,
                        "total_tiles": 81,
                        "pixel_channel_agreement": pixel_channel_agreement,
                        "mean_absolute_error": mean_absolute_error,
                        "fallback_cells": fallback,
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    summary = {
        "data": str(data_path),
        "stored_recreation_verified": True,
        "comparisons": len(records),
        "exact_tiles": sum(record["exact_tiles"] for record in records),
        "total_tiles": 81 * len(records),
        "exact_tile_fraction": sum(record["exact_tiles"] for record in records) / (81 * len(records)),
        "mean_pixel_channel_agreement": sum(record["pixel_channel_agreement"] for record in records) / len(records),
        "mean_absolute_error": sum(record["mean_absolute_error"] for record in records) / len(records),
        "records": records,
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output, summary_path


def create_player_collage(
    db_path: Path,
    dataset_name: str,
    output: Path,
    *,
    batch_size: int = 4,
    sequence_length: int = 1024,
) -> tuple[Path, Path]:
    lookup, metadata = build_canonical_lookup()
    mapper = GlyphMapper()
    tile_atlas = np.stack([mapper.tiles[index] for index in range(max(mapper.tiles) + 1)])
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        dataset = nld.TtyrecDataset(
            dataset_name,
            batch_size=batch_size,
            seq_length=sequence_length,
            dbfilename=str(db_path),
            threadpool=pool,
            shuffle=False,
        )
        raw = next(iter(dataset))

    scale = 2
    panel = 144 * scale
    gap = 16
    left = 150
    title_height = 86
    row_height = panel + 58
    width = left + 4 * (panel + gap) + gap
    height = title_height + batch_size * row_height + gap
    canvas = Image.new("RGB", (width, height), "#080b11")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    title_font = ImageFont.truetype(font_path, 21)
    body_font = ImageFont.truetype(font_path, 14)
    small_font = ImageFont.truetype(font_path, 12)
    draw.text(
        (18, 14),
        "REAL PLAYER TTYRECS → canonical MiniHack tile pixels",
        font=title_font,
        fill="#f2f5fa",
    )
    draw.text(
        (18, 47),
        "9×9 player-centered crops · exact MiniHack 16×16 tile atlas · output 144×144×3 RGB",
        font=small_font,
        fill="#9aa6b7",
    )

    source_table = metadata.pop("lookup_source")
    candidate_counts = metadata.pop("candidate_counts")
    coverage = Counter()
    panels = 0
    for row_index in range(batch_size):
        valid = _valid_player_frames(raw, row_index)
        if len(valid) < 4:
            raise RuntimeError(f"game batch {row_index} has only {len(valid)} valid player frames")
        chosen = [valid[index] for index in np.linspace(0, len(valid) - 1, 4, dtype=int)]
        gameid = int(raw["gameids"][row_index, chosen[0]])
        y = title_height + row_index * row_height
        draw.text((18, y + 8), f"game {gameid}", font=body_font, fill="#f2f5fa")
        draw.text((18, y + 31), "human ttyrec", font=small_font, fill="#9aa6b7")
        for column, timestep in enumerate(chosen):
            chars = raw["tty_chars"][row_index, timestep]
            colors = raw["tty_colors"][row_index, timestep]
            cursor = raw["tty_cursor"][row_index, timestep]
            pixels, crop = player_centered_tile_crop(
                chars, colors, cursor, lookup, tile_atlas
            )
            crop_chars, crop_colors = crop
            sources = source_table[crop_chars, crop_colors]
            candidates = candidate_counts[crop_chars, crop_colors]
            coverage.update(sources.flat)
            coverage["ambiguous_cells"] += int((candidates > 1).sum())
            coverage["total_cells"] += 81
            image = Image.fromarray(pixels).resize(
                (panel, panel), Image.Resampling.NEAREST
            )
            x = left + column * (panel + gap)
            canvas.paste(image, (x, y))
            draw.rectangle((x - 1, y - 1, x + panel, y + panel), outline="#364155")
            draw.text(
                (x + 5, y + panel + 8),
                f"tty frame {timestep:04d} · converted tile crop",
                font=small_font,
                fill="#9aa6b7",
            )
            panels += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    summary = {
        **metadata,
        "dataset": dataset_name,
        "panels": panels,
        "cells": coverage["total_cells"],
        "exact_pair_cells": coverage["empirical"] + coverage["static"] + coverage["alias"],
        "char_fallback_cells": coverage["char"],
        "default_cells": coverage["default"],
        "ambiguous_cells": coverage["ambiguous_cells"],
        "exact_pair_fraction": (
            coverage["empirical"] + coverage["static"] + coverage["alias"]
        ) / coverage["total_cells"],
        "char_fallback_fraction": coverage["char"] / coverage["total_cells"],
        "default_fraction": coverage["default"] / coverage["total_cells"],
        "ambiguous_fraction": coverage["ambiguous_cells"] / coverage["total_cells"],
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert real-player ttyrecs to a MiniHack tile collage")
    parser.add_argument("--db", type=Path, default=Path("data/nld/nld-nao.db"))
    parser.add_argument("--dataset", default="nld-nao-human-8shard")
    parser.add_argument("--output", type=Path, default=Path("reports/player-tile-confirmation.png"))
    parser.add_argument("--side-by-side", action="store_true")
    parser.add_argument("--ground-truth", action="store_true")
    parser.add_argument(
        "--pixel-data",
        type=Path,
        default=Path("data/minihack/keyroom-rgb-1600.hdf5"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=1024)
    args = parser.parse_args()
    if args.ground_truth:
        print(create_ground_truth_comparison(args.pixel_data, args.output))
    else:
        function = create_side_by_side_collage if args.side_by_side else create_player_collage
        print(
            function(
                args.db,
                args.dataset,
                args.output,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
            )
        )


if __name__ == "__main__":
    main()
