from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nle.dataset as nld
import numpy as np
from minihack.tiles.glyph_mapper import GlyphMapper
from PIL import Image, ImageDraw, ImageFont

from .player_tile_converter import batch_player_centered_tile_crops, build_canonical_lookup
from .player_tile_data import (
    RAW_MOVEMENT_KEY_TO_ACTION,
    SEMANTIC_ACTION_NAMES,
    infer_movement_actions,
)


def _gameplay(raw: dict, timestep: int) -> np.ndarray:
    cursors = raw["tty_cursor"][:, timestep]
    rows = cursors[:, 0].astype(np.int64)
    columns = cursors[:, 1].astype(np.int64)
    valid = (rows >= 1) & (rows <= 21) & (columns >= 0) & (columns < 79)
    batch = np.arange(len(rows))
    chars = raw["tty_chars"][batch, timestep, rows.clip(0, 23), columns.clip(0, 79)]
    return valid & (chars == ord("@"))


def _collect(
    db: Path,
    dataset_name: str,
    *,
    labelled: bool,
    batches: int,
) -> tuple[list[dict], dict]:
    lookup, _ = build_canonical_lookup()
    mapper = GlyphMapper()
    atlas = np.stack([mapper.tiles[index] for index in range(max(mapper.tiles) + 1)])
    examples: dict[int, dict] = {}
    inferred_counts = Counter()
    true_matches = Counter()
    total_pairs = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        stream = nld.TtyrecDataset(
            dataset_name,
            batch_size=32,
            seq_length=512,
            dbfilename=str(db),
            threadpool=pool,
            shuffle=True,
        )
        for batch_index, raw in enumerate(stream):
            for timestep in range(raw["tty_chars"].shape[1] - 1):
                same_game = (
                    (raw["gameids"][:, timestep] != 0)
                    & (raw["gameids"][:, timestep] == raw["gameids"][:, timestep + 1])
                )
                valid = same_game & _gameplay(raw, timestep) & _gameplay(raw, timestep + 1)
                inferred = infer_movement_actions(
                    raw["tty_cursor"][:, timestep], raw["tty_cursor"][:, timestep + 1]
                )
                valid &= inferred >= 0
                rows = np.flatnonzero(valid)
                total_pairs += int(valid.sum())
                for row in rows:
                    action = int(inferred[row])
                    inferred_counts[action] += 1
                    true_action = None
                    if labelled:
                        raw_key = int(raw["keypresses"][row, timestep])
                        true_action = RAW_MOVEMENT_KEY_TO_ACTION.get(raw_key)
                        if true_action == action:
                            true_matches[action] += 1
                        else:
                            continue
                    if action in examples:
                        continue
                    frame_chars = raw["tty_chars"][row, timestep : timestep + 2]
                    frame_colors = raw["tty_colors"][row, timestep : timestep + 2]
                    frame_cursors = raw["tty_cursor"][row, timestep : timestep + 2]
                    pixels = batch_player_centered_tile_crops(
                        frame_chars, frame_colors, frame_cursors, lookup, atlas
                    )
                    examples[action] = {
                        "action": action,
                        "gameid": int(raw["gameids"][row, timestep]),
                        "timestep": timestep,
                        "before_cursor": frame_cursors[0].astype(int).tolist(),
                        "after_cursor": frame_cursors[1].astype(int).tolist(),
                        "raw_key": int(raw["keypresses"][row, timestep]) if labelled else None,
                        "pixels": pixels,
                    }
            if batch_index + 1 >= batches:
                break
    ordered = [examples[action] for action in range(8) if action in examples]
    stats = {
        "dataset": dataset_name,
        "labelled": labelled,
        "scanned_batches": batches,
        "inferred_unit_moves": total_pairs,
        "counts_by_action": {
            SEMANTIC_ACTION_NAMES[action]: inferred_counts[action] for action in range(8)
        },
    }
    if labelled:
        matches = sum(true_matches.values())
        stats.update(
            exact_matches=matches,
            semantic_precision=matches / max(total_pairs, 1),
            precision_by_action={
                SEMANTIC_ACTION_NAMES[action]: true_matches[action]
                / max(inferred_counts[action], 1)
                for action in range(8)
            },
        )
    return ordered, stats


def _render_example(example: dict, *, labelled: bool, font_path: str) -> Image.Image:
    panel = 288
    gap = 38
    footer = 70
    image = Image.new("RGB", (panel * 2 + gap, panel + footer), "#080b11")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, 14)
    small = ImageFont.truetype(font_path, 12)
    before = Image.fromarray(example["pixels"][0]).resize((panel, panel), Image.Resampling.NEAREST)
    after = Image.fromarray(example["pixels"][1]).resize((panel, panel), Image.Resampling.NEAREST)
    image.paste(before, (0, 0))
    image.paste(after, (panel + gap, 0))
    draw.rectangle((0, 0, panel - 1, panel - 1), outline="#364155", width=2)
    draw.rectangle((panel + gap, 0, panel * 2 + gap - 1, panel - 1), outline="#364155", width=2)
    draw.text((panel + 8, panel // 2 - 12), "→", font=ImageFont.truetype(font_path, 22), fill="#70d6a5")
    name = SEMANTIC_ACTION_NAMES[example["action"]]
    cursor = f"cursor {tuple(example['before_cursor'])} → {tuple(example['after_cursor'])}"
    draw.text((4, panel + 8), f"inferred: {name.upper()} · {cursor}", font=font, fill="#70d6a5")
    source = f"game {example['gameid']}"
    if labelled:
        source += f" · true key {chr(example['raw_key'])!r} · MATCH"
    else:
        source += " · human ttyrec (no stored key)"
    draw.text((4, panel + 34), source, font=small, fill="#a7b0c0")
    return image


def create_collage(output: Path, *, batches: int = 40) -> tuple[Path, Path]:
    labelled, labelled_stats = _collect(
        Path("data/nld/nld-aa-taster.db"), "nld-aa-taster", labelled=True, batches=batches
    )
    human, human_stats = _collect(
        Path("data/nld/nld-nao.db"), "nld-nao-human-8shard", labelled=False, batches=batches
    )
    if len(labelled) != 8 or len(human) != 8:
        raise RuntimeError("did not find all eight movement directions")

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    cell_width = 614
    cell_height = 370
    margin = 18
    section_header = 72
    title_height = 105
    width = margin * 3 + cell_width * 2
    height = title_height + 2 * (section_header + 4 * cell_height) + margin
    canvas = Image.new("RGB", (width, height), "#080b11")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(font_path, 21)
    body = ImageFont.truetype(font_path, 14)
    small = ImageFont.truetype(font_path, 12)
    draw.text((margin, 14), "REVERSE-ENGINEERED HUMAN MOVEMENT ACTIONS", font=title, fill="#f2f5fa")
    draw.text(
        (margin, 48),
        "One-cell @ cursor deltas → 8 semantic moves · tiles are the exact 144×144 RGB model inputs",
        font=small,
        fill="#9aa6b7",
    )
    draw.text(
        (margin, 70),
        "The crop recenters @, so the surrounding world shifts opposite the inferred movement.",
        font=small,
        fill="#9aa6b7",
    )

    y = title_height
    sections = (
        ("VALIDATION: action-labelled AutoAscend ttyrec3", labelled, labelled_stats),
        ("APPLICATION: human NAO ttyrec without stored keys", human, human_stats),
    )
    for heading, examples, stats in sections:
        subtitle = f"{stats['inferred_unit_moves']:,} inferred moves scanned"
        if stats["labelled"]:
            subtitle += f" · semantic precision {stats['semantic_precision']:.3%}"
        draw.text((margin, y + 8), heading, font=body, fill="#79a8ff")
        draw.text((margin, y + 34), subtitle, font=small, fill="#a7b0c0")
        y += section_header
        for index, example in enumerate(examples):
            row, column = divmod(index, 2)
            rendered = _render_example(example, labelled=stats["labelled"], font_path=font_path)
            canvas.paste(rendered, (margin + column * (cell_width + margin), y + row * cell_height))
        y += 4 * cell_height

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    summary = {
        "method": "successful one-cell movement inferred from consecutive absolute @ cursor coordinates",
        "classes": list(SEMANTIC_ACTION_NAMES[:8]),
        "labelled_validation": labelled_stats,
        "human_application": human_stats,
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and visualize inferred human actions")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/inferred-human-actions.png")
    )
    parser.add_argument("--batches", type=int, default=40)
    args = parser.parse_args()
    print(create_collage(args.output, batches=args.batches))


if __name__ == "__main__":
    main()
