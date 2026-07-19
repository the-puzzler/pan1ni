from __future__ import annotations

import argparse
from pathlib import Path

import h5py
from PIL import Image, ImageDraw, ImageFont


STAGES = (
    ("START", lambda stages: 0),
    ("KEY ACQUIRED", lambda stages: next(i for i, value in enumerate(stages) if value >= 1)),
    ("DOOR UNLOCKED", lambda stages: next(i for i, value in enumerate(stages) if value >= 2)),
    ("GOAL APPROACH", lambda stages: len(stages) - 1),
)


def create_collage(
    data_path: Path,
    output: Path,
    *,
    episode_keys: tuple[str, ...] = ("0", "5", "10", "15"),
) -> Path:
    scale = 2
    tile_size = 144 * scale
    gap = 16
    left_label = 142
    title_height = 76
    column_header = 42
    row_height = tile_size + 62
    width = left_label + len(STAGES) * (tile_size + gap) + gap
    height = title_height + column_header + len(episode_keys) * row_height + gap
    canvas = Image.new("RGB", (width, height), "#080b11")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    title_font = ImageFont.truetype(font_path, 21)
    body_font = ImageFont.truetype(font_path, 15)
    small_font = ImageFont.truetype(font_path, 13)
    draw.text(
        (18, 14),
        "EXACT MODEL INPUT · MiniHack native pixel_crop tiles",
        font=title_font,
        fill="#f2f5fa",
    )
    draw.text(
        (18, 44),
        "Each panel is an unmodified 144×144×3 uint8 dataset frame, enlarged 2× with nearest-neighbor.",
        font=small_font,
        fill="#9aa6b7",
    )
    for column, (label, _) in enumerate(STAGES):
        x = left_label + column * (tile_size + gap)
        draw.text((x + 6, title_height + 10), label, font=body_font, fill="#70d6a5")

    with h5py.File(data_path, "r") as handle:
        for row, episode_key in enumerate(episode_keys):
            group = handle[episode_key]
            pixels = group["pixels"]
            stages = group["stages"][:]
            key_position = tuple(map(int, group.attrs["key_position"]))
            goal_position = tuple(map(int, group.attrs["goal_position"]))
            y = title_height + column_header + row * row_height
            draw.text((18, y + 8), f"episode {episode_key}", font=body_font, fill="#f2f5fa")
            draw.text((18, y + 32), f"key  {key_position}", font=small_font, fill="#9aa6b7")
            draw.text((18, y + 52), f"goal {goal_position}", font=small_font, fill="#9aa6b7")
            for column, (_, choose_index) in enumerate(STAGES):
                index = choose_index(stages)
                frame = Image.fromarray(pixels[index]).resize(
                    (tile_size, tile_size), Image.Resampling.NEAREST
                )
                x = left_label + column * (tile_size + gap)
                canvas.paste(frame, (x, y))
                draw.rectangle(
                    (x - 1, y - 1, x + tile_size, y + tile_size),
                    outline="#364155",
                    width=1,
                )
                draw.text(
                    (x + 6, y + tile_size + 8),
                    f"raw frame {index:02d} · shape 144×144×3",
                    font=small_font,
                    fill="#9aa6b7",
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an exact MiniHack pixel-input contact sheet")
    parser.add_argument("--data", type=Path, default=Path("data/minihack/keyroom-rgb-1600.hdf5"))
    parser.add_argument("--output", type=Path, default=Path("reports/pixel-input-confirmation.png"))
    parser.add_argument("--episodes", nargs="+", default=("0", "5", "10", "15"))
    args = parser.parse_args()
    print(create_collage(args.data, args.output, episode_keys=tuple(args.episodes)))


if __name__ == "__main__":
    main()
