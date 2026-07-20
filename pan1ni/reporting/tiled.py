from __future__ import annotations

import argparse
import html
import json
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from pan1ni.models.config import ModelConfig
from pan1ni.data.minihack import MiniHackPixelGoalDataset
from pan1ni.models.model import GoalConditionedLeWorldModel
from pan1ni.reporting.report import _svg_chart, diagnose
from pan1ni.training.primitives import move_batch, pretrain_step

STAGE_NAMES = {0: "find key", 1: "reach locked door", 2: "enter goal room", 3: "reach staircase"}


def pixel_model_config(scale: str = "small") -> ModelConfig:
    if scale == "small":
        return ModelConfig(
            observation_mode="pixels", latent_dim=64, hidden_dim=64,
            vit_dim=64, vit_layers=1, vit_heads=4, pixel_patch_size=16,
            max_patches=128, projector_hidden_dim=128, predictor_layers=1,
            predictor_heads=4, max_context=4, num_actions=10, dropout=0.0,
        )
    if scale == "medium":
        return ModelConfig(
            observation_mode="pixels", latent_dim=256, hidden_dim=256,
            vit_dim=128, vit_layers=4, vit_heads=8, pixel_patch_size=16,
            max_patches=128, projector_hidden_dim=512, predictor_layers=4,
            predictor_heads=8, max_context=4, num_actions=10, dropout=0.0,
        )
    raise ValueError(f"unknown pixel model scale: {scale}")


def _render_frame(pixels, label: str) -> Image.Image:
    image = Image.fromarray(pixels).resize((576, 576), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (576, 624), "#111722")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
    draw.text((12, 590), label, fill="#e7ebf2", font=font)
    return canvas


def render_media(data_path: Path, output: Path, episode_key: str = "15") -> dict:
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    with h5py.File(data_path, "r") as handle:
        group = handle[episode_key]
        pixels = group["pixels"][:]
        stages = group["stages"][:]
        key_position = group.attrs["key_position"].tolist()
        goal_position = group.attrs["goal_position"].tolist()
    indices = {
        "start": 0,
        "key": int(next((i for i, stage in enumerate(stages) if stage >= 1), 0)),
        "door": int(next((i for i, stage in enumerate(stages) if stage >= 2), len(stages) - 1)),
        "goal": len(stages) - 1,
    }
    for name, index in indices.items():
        stage = STAGE_NAMES.get(int(stages[index]), name)
        _render_frame(pixels[index], f"frame {index} · {stage}").save(assets / f"{name}.png")
    video = assets / "trajectory.mp4"
    with tempfile.TemporaryDirectory() as temporary:
        frame_directory = Path(temporary)
        for index, (frame, stage_id) in enumerate(zip(pixels, stages)):
            _render_frame(frame, f"frame {index} · {STAGE_NAMES[int(stage_id)]}").save(
                frame_directory / f"frame-{index:04d}.png"
            )
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-framerate", "3",
                "-i", str(frame_directory / "frame-%04d.png"), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "18", str(video),
            ],
            check=True,
        )
    return {
        "episode": int(episode_key),
        "frames": len(pixels),
        "key_position": key_position,
        "goal_position": goal_position,
        "video": "assets/trajectory.mp4",
        **{name: f"assets/{name}.png" for name in indices},
    }


def write_report(payload: dict, output: Path) -> Path:
    timeline = payload["timeline"]
    initial, final = timeline[0]["diagnostics"], timeline[-1]["diagnostics"]
    normalized_improvement = initial["normalized_prediction_loss"] / max(
        final["normalized_prediction_loss"], 1e-12
    )
    shuffled_penalty = final["shuffled_goal_loss"] / max(final["prediction_loss"], 1e-12)
    zero_penalty = final["zero_goal_loss"] / max(final["prediction_loss"], 1e-12)
    copy_ratio = final["prediction_vs_copy"]
    rank = final["latent_effective_rank"]
    healthy = rank > 4 and final["latent_feature_std_mean"] > 0.1
    goal_used = final["correct_vs_shuffled"] < 0.9 and final["correct_vs_zero"] < 0.9
    verdict = "Promising representation signal" if healthy and goal_used else "Diagnostic warning remains"
    verdict_class = "good" if healthy and goal_used else "warn"
    loss_chart = _svg_chart(
        [
            ("held-out prediction", [(x["step"], x["diagnostics"]["prediction_loss"]) for x in timeline], "#79a8ff"),
            ("copy current", [(x["step"], x["diagnostics"]["copy_current_loss"]) for x in timeline], "#f0b35a"),
            ("shuffled goal", [(x["step"], x["diagnostics"]["shuffled_goal_loss"]) for x in timeline], "#e9788f"),
        ],
        "Held-out losses",
    )
    geometry_chart = _svg_chart(
        [
            ("effective rank", [(x["step"], x["diagnostics"]["latent_effective_rank"]) for x in timeline], "#d28cff"),
            ("feature std", [(x["step"], x["diagnostics"]["latent_feature_std_mean"]) for x in timeline], "#70d6a5"),
        ],
        "Latent geometry",
    )
    rows = "".join(
        f"<tr><td>{entry['step']}</td><td>{entry['diagnostics']['prediction_loss']:.4f}</td>"
        f"<td>{entry['diagnostics']['copy_current_loss']:.4f}</td>"
        f"<td>{entry['diagnostics']['shuffled_goal_loss']:.4f}</td>"
        f"<td>{entry['diagnostics']['zero_goal_loss']:.4f}</td>"
        f"<td>{entry['diagnostics']['latent_feature_std_mean']:.3f}</td>"
        f"<td>{entry['diagnostics']['latent_effective_rank']:.2f}</td></tr>"
        for entry in timeline
    )
    stage_rows = "".join(
        f"<tr><td>{html.escape(STAGE_NAMES.get(int(stage), stage))}</td><td>{loss:.4f}</td></tr>"
        for stage, loss in final.get("stage_prediction_loss", {}).items()
    )
    media = payload["media"]
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MiniHack KeyRoom · goal-conditioned report</title>
<style>
:root{{--bg:#080b11;--panel:#111722;--text:#e7ebf2;--muted:#98a3b5;--green:#70d6a5;--amber:#f0b35a;--blue:#79a8ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 Inter,system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:52px 24px 80px}}
h1{{font-size:44px;line-height:1.08;margin:8px 0}}h2{{margin-top:42px}}.eyebrow{{color:var(--green);letter-spacing:.14em;text-transform:uppercase;font-size:12px}}.muted{{color:var(--muted)}}
.panel,.card{{background:var(--panel);border:1px solid #202938;border-radius:14px;padding:18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px}}.metric{{font-size:27px;font-weight:700}}.label{{color:var(--muted);font-size:12px}}.good{{color:var(--green)}}.warn{{color:var(--amber)}}
.media{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}img,video,svg{{width:100%;border-radius:12px;background:#0b0e14}}video{{margin-top:14px;max-height:680px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #273040;text-align:right}}td:first-child,th:first-child{{text-align:left}}code{{color:#b9cbff}}a{{color:var(--blue)}}
@media(max-width:760px){{.media{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:32px}}}}
</style></head><body><main><div class="eyebrow">MiniHack · tiled causal environment</div><h1>Goal-conditioned KeyRoom prediction</h1>
<p class="muted">{payload['config']['steps']} steps · {payload['config']['training_windows']:,} sampled windows · 16 key/goal layouts · held-out episode split</p>
<div class="panel"><strong class="{verdict_class}">{verdict}</strong><br>The task requires reaching the key, unlocking the central door, and navigating to the goal staircase.</div>
<h2>Reading the result</h2><div class="panel"><ul>
<li>Scale-normalized held-out prediction error fell <strong>{normalized_improvement:.1f}×</strong>, from {initial['normalized_prediction_loss']:.2f} to {final['normalized_prediction_loss']:.3f}. Raw MSE is not comparable over time because SIGReg expands latent variance.</li>
<li>Shuffled goals cause <strong>{shuffled_penalty:.2f}×</strong> error and zero goals cause <strong>{zero_penalty:.2f}×</strong> error relative to the correct goal.</li>
<li>The predictor is <strong>{abs(1-copy_ratio)*100:.1f}% {'worse' if copy_ratio > 1 else 'better'}</strong> than copying the current latent.</li>
<li>Mean feature std is {final['latent_feature_std_mean']:.3f}; effective rank is <strong>{rank:.2f}/64</strong>.</li></ul></div>
<h2>Final diagnostics</h2><div class="grid">
<div class="card"><div class="metric">{final['prediction_loss']:.4f}</div><div class="label">held-out prediction MSE</div></div>
<div class="card"><div class="metric">{copy_ratio:.2f}×</div><div class="label">prediction / copy error</div></div>
<div class="card"><div class="metric">{final['correct_vs_shuffled']:.3f}×</div><div class="label">correct / shuffled goal</div></div>
<div class="card"><div class="metric">{final['correct_vs_zero']:.3f}×</div><div class="label">correct / zero goal</div></div>
<div class="card"><div class="metric">{final['latent_feature_std_mean']:.3f}</div><div class="label">mean latent feature std</div></div>
<div class="card"><div class="metric">{rank:.2f}</div><div class="label">effective rank / 64</div></div></div>
<h2>What the model sees</h2><p class="muted">Episode {media['episode']}: key at {media['key_position']}, goal at {media['goal_position']}.</p>
<div class="media"><img src="{media['start']}" alt="start"><img src="{media['key']}" alt="key acquired"><img src="{media['door']}" alt="door unlocked"><img src="{media['goal']}" alt="goal reached"></div><video controls loop muted playsinline src="{media['video']}"></video>
<h2>Learning curves</h2><div class="panel">{loss_chart}</div><div class="panel" style="margin-top:12px">{geometry_chart}</div>
<h2>Final error by causal stage</h2><div class="panel"><table><thead><tr><th>Stage</th><th>Prediction MSE</th></tr></thead><tbody>{stage_rows}</tbody></table></div>
<h2>Diagnostic timeline</h2><div class="panel" style="overflow:auto"><table><thead><tr><th>Step</th><th>Prediction</th><th>Copy</th><th>Shuffled</th><th>Zero</th><th>Std</th><th>Rank</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Method</h2><div class="panel"><p>A 144×144 tiled crop is split into 16×16 patches and passed through a tiny ViT. Its CLS token goes through the required BatchNorm MLP before SIGReg. The predictor receives <code>z_t</code> and the successful episode's final <code>goal_z</code>, then predicts <code>z_(t+1)</code>.</p><p><a href="metrics.json">Raw metrics</a> · <a href="checkpoint.pt">Checkpoint</a></p></div>
</main></body></html>'''
    path = output / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def run_report(
    data_path: Path,
    output: Path,
    *,
    steps: int,
    batch_size: int,
    sigreg_slices: int,
    eval_every: int,
    validation_samples: int,
    device: str,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    model_scale: str = "small",
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    with h5py.File(data_path, "r") as handle:
        keys = sorted(handle.keys(), key=int)
    validation_keys = keys[::5]
    validation_set = set(validation_keys)
    training_keys = [key for key in keys if key not in validation_set]
    train_data = MiniHackPixelGoalDataset(
        data_path, episode_keys=training_keys, context_length=1,
        samples_per_epoch=steps * batch_size,
    )
    validation_data = MiniHackPixelGoalDataset(
        data_path, episode_keys=validation_keys, context_length=1,
        samples_per_epoch=validation_samples, seed=20_000,
    )
    use_cuda = device.startswith("cuda")
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        loader_options.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
        )
    validation_batches = [
        move_batch(batch, device, non_blocking=use_cuda)
        for batch in DataLoader(validation_data, batch_size=batch_size, **loader_options)
    ]
    model_config = pixel_model_config(model_scale)
    model = GoalConditionedLeWorldModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    timeline = [{"step": 0, "diagnostics": diagnose(model, validation_batches)}]
    started = time.perf_counter()
    loader = DataLoader(train_data, batch_size=batch_size, **loader_options)
    for step, batch in enumerate(loader, start=1):
        metrics = pretrain_step(
            model,
            move_batch(batch, device, non_blocking=use_cuda),
            optimizer,
            sigreg_slices=sigreg_slices,
        )
        if step % eval_every == 0 or step == steps:
            entry = {
                "step": step,
                "train_prediction_loss": metrics["prediction_loss"],
                "train_sigreg_loss": metrics["sigreg_loss"],
                "train_total_loss": metrics["loss"],
                "diagnostics": diagnose(model, validation_batches),
            }
            timeline.append(entry)
            diagnostic = entry["diagnostics"]
            print(
                f"step {step:4d}/{steps} | train {metrics['prediction_loss']:.4f} | "
                f"held-out {diagnostic['prediction_loss']:.4f} | "
                f"goal/shuffle {diagnostic['correct_vs_shuffled']:.3f} | "
                f"rank {diagnostic['latent_effective_rank']:.2f}",
                flush=True,
            )
            torch.save(
                {"model": model.state_dict(), "config": asdict(model_config), "step": step},
                output / "checkpoint.pt",
            )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    torch.save(
        {"model": model.state_dict(), "config": asdict(model_config), "step": steps},
        output / "checkpoint.pt",
    )
    media = render_media(data_path, output)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "config": {
            "steps": steps,
            "model_scale": model_scale,
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "batch_size": batch_size,
            "training_windows": steps * batch_size,
            "sigreg_slices": sigreg_slices,
            "eval_every": eval_every,
            "validation_samples": validation_samples,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "device": device,
            "training_episodes": len(training_keys),
            "validation_episodes": len(validation_keys),
            "elapsed_seconds": elapsed,
            "samples_per_second": steps * batch_size / max(elapsed, 1e-12),
        },
        "timeline": timeline,
        "media": media,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return write_report(payload, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and report the tiled MiniHack KeyRoom experiment")
    parser.add_argument("--data", default="data/minihack/keyroom-s5-16v-v2.hdf5")
    parser.add_argument("--output", default="reports/minihack-keyroom")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sigreg-slices", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--model-scale", choices=("small", "medium"), default="small")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    path = run_report(
        Path(args.data), Path(args.output), steps=args.steps, batch_size=args.batch_size,
        sigreg_slices=args.sigreg_slices, eval_every=args.eval_every,
        validation_samples=args.validation_samples, device=args.device,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
        model_scale=args.model_scale,
    )
    print(path)


if __name__ == "__main__":
    main()
