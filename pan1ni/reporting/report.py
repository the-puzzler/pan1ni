from __future__ import annotations

import argparse
import html
import json
import math
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pan1ni.models.config import ModelConfig
from pan1ni.models.model import GoalConditionedLeWorldModel
from pan1ni.data.nld import NLDHDF5GoalDataset, nld_episode_keys
from pan1ni.training.primitives import move_batch, pretrain_step

NLE_COLORS = (
    "#111318", "#b94a48", "#5cab63", "#b58b45", "#6577c8", "#a960b8", "#55a7ac", "#c8ccd4",
    "#626773", "#ff6b68", "#7fe089", "#f5d76e", "#8298ff", "#df83ef", "#76e5eb", "#ffffff",
)


def report_model_config() -> ModelConfig:
    return ModelConfig(
        latent_dim=64,
        cell_dim=32,
        message_dim=16,
        hidden_dim=64,
        vit_dim=32,
        vit_layers=1,
        vit_heads=4,
        terminal_patch_size=2,
        projector_hidden_dim=128,
        predictor_layers=1,
        predictor_heads=4,
        max_context=8,
        num_actions=8,
        dropout=0.0,
    )


@torch.no_grad()
def diagnose(
    model: GoalConditionedLeWorldModel,
    batches: Iterable[Mapping],
    *,
    objective: str = "mse",
) -> dict:
    model.eval()
    predictions: list[Tensor] = []
    shuffled_predictions: list[Tensor] = []
    zero_goal_predictions: list[Tensor] = []
    targets: list[Tensor] = []
    currents: list[Tensor] = []
    embeddings: list[Tensor] = []
    stages: list[Tensor] = []
    for batch_index, batch in enumerate(batches):
        grouped = model.encode_group(batch["history"], batch["goal"], batch["target"])
        history_z, goal_z, target_z = grouped[:, :-2], grouped[:, -2], grouped[:, -1]
        if objective == "flow":
            generator = torch.Generator(device=target_z.device).manual_seed(40_000 + batch_index)
            source_noise = torch.randn(
                history_z.shape,
                generator=generator,
                device=history_z.device,
                dtype=history_z.dtype,
            )
            prediction, _ = model.one_step_flow(history_z, goal_z, source_noise)
            shuffled, _ = model.one_step_flow(history_z, goal_z.roll(1, 0), source_noise)
            zero_goal, _ = model.one_step_flow(history_z, torch.zeros_like(goal_z), source_noise)
            predictions.append(prediction[:, -1].cpu())
            shuffled_predictions.append(shuffled[:, -1].cpu())
            zero_goal_predictions.append(zero_goal[:, -1].cpu())
        else:
            predictions.append(model.predict_latents(history_z, goal_z).next_latent.cpu())
            shuffled_predictions.append(model.predict_latents(history_z, goal_z.roll(1, 0)).next_latent.cpu())
            zero_goal_predictions.append(model.predict_latents(history_z, torch.zeros_like(goal_z)).next_latent.cpu())
        targets.append(target_z.cpu())
        currents.append(history_z[:, -1].cpu())
        embeddings.append(grouped.flatten(0, 1).cpu())
        if "stage" in batch:
            stages.append(batch["stage"].cpu())
    prediction = torch.cat(predictions)
    shuffled = torch.cat(shuffled_predictions)
    zero_goal = torch.cat(zero_goal_predictions)
    target = torch.cat(targets)
    current = torch.cat(currents)
    flat_z = torch.cat(embeddings)
    centered = flat_z - flat_z.mean(0)
    covariance = centered.T @ centered / max(flat_z.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    feature_std = flat_z.std(0)
    prediction_loss = F.mse_loss(prediction, target)
    shuffled_loss = F.mse_loss(shuffled, target)
    zero_goal_loss = F.mse_loss(zero_goal, target)
    copy_loss = F.mse_loss(current, target)
    target_variance = target.var(0).mean().clamp_min(1e-12)
    result = {
        "prediction_loss": prediction_loss.item(),
        "copy_current_loss": copy_loss.item(),
        "shuffled_goal_loss": shuffled_loss.item(),
        "zero_goal_loss": zero_goal_loss.item(),
        "prediction_vs_copy": (prediction_loss / copy_loss.clamp_min(1e-12)).item(),
        "correct_vs_shuffled": (prediction_loss / shuffled_loss.clamp_min(1e-12)).item(),
        "correct_vs_zero": (prediction_loss / zero_goal_loss.clamp_min(1e-12)).item(),
        "normalized_prediction_loss": (prediction_loss / target_variance).item(),
        "cosine_similarity": F.cosine_similarity(prediction, target).mean().item(),
        "latent_feature_std_mean": feature_std.mean().item(),
        "latent_feature_std_min": feature_std.min().item(),
        "latent_feature_std_median": feature_std.median().item(),
        "latent_feature_std_max": feature_std.max().item(),
        "latent_effective_rank": effective_rank.item(),
        "covariance_eigenvalues": eigenvalues.flip(0).tolist(),
    }
    if stages:
        stage = torch.cat(stages)
        per_sample = (prediction - target).square().mean(-1)
        result["stage_prediction_loss"] = {
            str(int(value)): per_sample[stage == value].mean().item()
            for value in stage.unique(sorted=True)
        }
    return result


def render_terminal(chars: Tensor, colors: Tensor, label: str, scale: int = 1) -> Image.Image:
    chars = chars.cpu()
    colors = colors.cpu()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    font = ImageFont.truetype(font_path, 14 * scale)
    cell_width, cell_height = 9 * scale, 17 * scale
    rows, columns = chars.shape
    footer = 32 * scale
    image = Image.new("RGB", (columns * cell_width, rows * cell_height + footer), "#090b10")
    draw = ImageDraw.Draw(image)
    for row in range(rows):
        for column in range(columns):
            code = int(chars[row, column])
            glyph = bytes((code,)).decode("cp437", errors="replace") if code else " "
            color = NLE_COLORS[int(colors[row, column]) & 15]
            draw.text((column * cell_width, row * cell_height), glyph, font=font, fill=color)
    draw.rectangle((0, rows * cell_height, image.width, image.height), fill="#151923")
    draw.text((12 * scale, rows * cell_height + 7 * scale), label, font=font, fill="#d8dee9")
    return image


def render_media(data_path: Path, dataset: NLDHDF5GoalDataset, output: Path) -> dict[str, str | int]:
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    sample = dataset[7]
    episode_key = str(int(sample["trajectory_id"]))
    timestep = int(sample["timestep"])
    horizon = int(sample["goal_offset"])
    frames = (
        (sample["history"]["chars"][-1], sample["history"]["colors"][-1], "Current frame"),
        (sample["target"]["chars"], sample["target"]["colors"], "Immediate next frame"),
        (sample["goal"]["chars"], sample["goal"]["colors"], f"Goal frame (+{horizon})"),
    )
    names = ("current.png", "target.png", "goal.png")
    for (chars, colors, label), name in zip(frames, names):
        render_terminal(chars, colors, f"Episode {episode_key} · t={timestep} · {label}").save(assets / name)

    video_path = assets / "trajectory.mp4"
    with h5py.File(data_path, "r") as handle, tempfile.TemporaryDirectory() as temporary:
        group = handle[episode_key]
        frame_directory = Path(temporary)
        for frame_number, frame_t in enumerate(range(timestep, timestep + horizon + 1)):
            chars = torch.from_numpy(group["tty_chars"][frame_t])
            colors = torch.from_numpy(group["tty_colors"][frame_t])
            label = f"Episode {episode_key} · frame {frame_t} · goal in {timestep + horizon - frame_t}"
            render_terminal(chars, colors, label).save(frame_directory / f"frame-{frame_number:04d}.png")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-framerate", "8",
                "-i", str(frame_directory / "frame-%04d.png"), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "20", str(video_path),
            ],
            check=True,
        )
    return {
        "episode": int(episode_key),
        "timestep": timestep,
        "horizon": horizon,
        "current_image": "assets/current.png",
        "target_image": "assets/target.png",
        "goal_image": "assets/goal.png",
        "video": "assets/trajectory.mp4",
    }


def _svg_chart(series: list[tuple[str, list[tuple[float, float]], str]], title: str) -> str:
    width, height, margin = 760, 300, 48
    all_points = [point for _, points, _ in series for point in points]
    if not all_points:
        return ""
    xs, ys = [p[0] for p in all_points], [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if math.isclose(y_min, y_max):
        y_max = y_min + 1
    def sx(value: float) -> float:
        return margin + (value - x_min) / max(x_max - x_min, 1e-12) * (width - 2 * margin)
    def sy(value: float) -> float:
        return height - margin - (value - y_min) / (y_max - y_min) * (height - 2 * margin)
    lines = []
    legends = []
    for index, (name, points, color) in enumerate(series):
        path = " ".join(("M" if i == 0 else "L") + f" {sx(x):.1f} {sy(y):.1f}" for i, (x, y) in enumerate(points))
        lines.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legends.append(f'<text x="{margin + index * 190}" y="22" fill="{color}" font-size="13">{html.escape(name)}</text>')
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
      <rect width="100%" height="100%" rx="12" fill="#111722"/>
      <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#394252"/>
      <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#394252"/>
      {''.join(legends)}{''.join(lines)}
      <text x="{margin}" y="{height-12}" fill="#8d98aa" font-size="11">step {x_min:g}</text>
      <text x="{width-margin-70}" y="{height-12}" fill="#8d98aa" font-size="11">step {x_max:g}</text>
      <text x="6" y="{margin}" fill="#8d98aa" font-size="11">{y_max:.3g}</text>
      <text x="6" y="{height-margin}" fill="#8d98aa" font-size="11">{y_min:.3g}</text>
    </svg>'''


def write_report(payload: dict, output: Path) -> Path:
    timeline = payload["timeline"]
    initial = timeline[0]["diagnostics"]
    final = timeline[-1]
    diagnostic = final["diagnostics"]
    rank = diagnostic["latent_effective_rank"]
    std = diagnostic["latent_feature_std_mean"]
    no_collapse = std > 0.1 and rank > 2.0
    goal_used = diagnostic["correct_vs_shuffled"] < 0.98
    prediction_improvement = initial["prediction_loss"] / max(diagnostic["prediction_loss"], 1e-12)
    shuffled_penalty = diagnostic["shuffled_goal_loss"] / max(diagnostic["prediction_loss"], 1e-12)
    copy_comparison = "worse" if diagnostic["prediction_vs_copy"] > 1 else "better"
    loss_chart = _svg_chart(
        [
            ("training prediction", [(x["step"], x["train_prediction_loss"]) for x in timeline if "train_prediction_loss" in x], "#70d6a5"),
            ("held-out prediction", [(x["step"], x["diagnostics"]["prediction_loss"]) for x in timeline], "#79a8ff"),
            ("copy current", [(x["step"], x["diagnostics"]["copy_current_loss"]) for x in timeline], "#f0b35a"),
        ],
        "Losses over training",
    )
    geometry_chart = _svg_chart(
        [
            ("effective rank", [(x["step"], x["diagnostics"]["latent_effective_rank"]) for x in timeline], "#d28cff"),
            ("mean feature std", [(x["step"], x["diagnostics"]["latent_feature_std_mean"]) for x in timeline], "#70d6a5"),
        ],
        "Latent geometry",
    )
    rows = "".join(
        f"<tr><td>{entry['step']}</td><td>{entry['diagnostics']['prediction_loss']:.4f}</td>"
        f"<td>{entry['diagnostics']['copy_current_loss']:.4f}</td>"
        f"<td>{entry['diagnostics']['shuffled_goal_loss']:.4f}</td>"
        f"<td>{entry['diagnostics']['latent_feature_std_mean']:.3f}</td>"
        f"<td>{entry['diagnostics']['latent_effective_rank']:.2f}</td></tr>"
        for entry in timeline
    )
    media = payload["media"]
    verdict_class = "good" if no_collapse else "warn"
    verdict = "No trivial collapse detected" if no_collapse else "Collapse risk remains"
    goal_text = "Correct goals outperform shuffled goals" if goal_used else "No clear goal-use signal yet"
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Goal-conditioned LeWorldModel · NLD report</title>
<style>
:root{{--bg:#080b11;--panel:#111722;--text:#e7ebf2;--muted:#98a3b5;--green:#70d6a5;--amber:#f0b35a;--blue:#79a8ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 Inter,system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:52px 24px 80px}} h1{{font-size:42px;line-height:1.1;margin:8px 0}} h2{{margin-top:42px}}
.eyebrow{{color:var(--green);text-transform:uppercase;letter-spacing:.14em;font-size:12px}} .muted{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}} .card,.panel{{background:var(--panel);border:1px solid #202938;border-radius:14px;padding:18px}}
.metric{{font-size:28px;font-weight:700}} .label{{color:var(--muted);font-size:12px}} .good{{color:var(--green)}} .warn{{color:var(--amber)}}
.media{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}} img,video,svg{{width:100%;border-radius:12px;background:#0b0e14}} video{{margin-top:14px}}
table{{width:100%;border-collapse:collapse}} td,th{{padding:9px;border-bottom:1px solid #273040;text-align:right}} td:first-child,th:first-child{{text-align:left}}
code{{color:#b9cbff}} @media(max-width:760px){{.media{{grid-template-columns:1fr}} h1{{font-size:32px}}}}
</style></head><body><main>
<div class="eyebrow">NLD-AA · real NetHack trajectories</div><h1>Goal-conditioned latent prediction</h1>
<p class="muted">Generated {html.escape(payload['generated_at'])} · {payload['config']['steps']} steps · horizon {payload['config']['goal_horizon']} · episode-level held-out split</p>
<div class="panel"><strong class="{verdict_class}">{verdict}</strong><br>{html.escape(goal_text)}. This is a diagnostic run, not a downstream-control result.</div>
<h2>Reading the result</h2><div class="panel"><ul>
<li>Held-out prediction error improved <strong>{prediction_improvement:.2f}×</strong>, from {initial['prediction_loss']:.4f} to {diagnostic['prediction_loss']:.4f}.</li>
<li>Shuffling goal embeddings makes error <strong>{shuffled_penalty:.2f}×</strong> larger, evidence that the predictor is using the goal input.</li>
<li>The learned predictor is <strong>{abs(1 - diagnostic['prediction_vs_copy']) * 100:.1f}% {copy_comparison}</strong> than copying the current latent ({diagnostic['prediction_loss']:.4f} vs {diagnostic['copy_current_loss']:.4f} MSE).</li>
<li>Effective rank moved from {initial['latent_effective_rank']:.2f} to <strong>{rank:.2f} / 64</strong>. Nonzero feature variance alone therefore does not rule out a low-dimensional collapse.</li>
</ul></div>
<h2>Final diagnostics</h2><div class="grid">
<div class="card"><div class="metric">{diagnostic['prediction_loss']:.4f}</div><div class="label">held-out prediction MSE</div></div>
<div class="card"><div class="metric">{diagnostic['prediction_vs_copy']:.2f}×</div><div class="label">prediction / copy error</div></div>
<div class="card"><div class="metric">{diagnostic['correct_vs_shuffled']:.3f}×</div><div class="label">correct / shuffled goal</div></div>
<div class="card"><div class="metric">{std:.3f}</div><div class="label">mean latent feature std</div></div>
<div class="card"><div class="metric">{rank:.2f}</div><div class="label">effective latent rank / 64</div></div>
<div class="card"><div class="metric">{diagnostic['cosine_similarity']:.3f}</div><div class="label">prediction cosine similarity</div></div></div>
<h2>Learning curves</h2><div class="panel">{loss_chart}</div><div class="panel" style="margin-top:12px">{geometry_chart}</div>
<h2>What the model sees</h2><p class="muted">Real episode {media['episode']}, current frame {media['timestep']}; goal is {media['horizon']} transitions later.</p>
<div class="media"><img src="{media['current_image']}" alt="Current NetHack terminal"><img src="{media['target_image']}" alt="Next NetHack terminal"><img src="{media['goal_image']}" alt="Goal NetHack terminal"></div>
<video controls loop muted playsinline src="{media['video']}"></video>
<h2>Diagnostic timeline</h2><div class="panel" style="overflow:auto"><table><thead><tr><th>Step</th><th>Prediction</th><th>Copy</th><th>Shuffled goal</th><th>Feature std</th><th>Eff. rank</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Method</h2><div class="panel"><p>The model receives <code>z_t</code> and <code>goal_z = E(o_{{t+{payload['config']['goal_horizon']}}})</code>, and predicts <code>z_{{t+1}}</code>. SIGReg uses {payload['config']['sigreg_slices']} random projections. Diagnostics aggregate {payload['config']['validation_samples']} held-out windows from episodes never used for training.</p><p><a href="metrics.json" style="color:var(--blue)">Raw metrics JSON</a> · <a href="checkpoint.pt" style="color:var(--blue)">Checkpoint</a></p></div>
</main></body></html>'''
    report_path = output / "index.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def run_report(
    data_path: Path,
    output: Path,
    *,
    steps: int,
    batch_size: int,
    goal_horizon: int,
    sigreg_slices: int,
    eval_every: int,
    validation_samples: int,
    device: str,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    keys = nld_episode_keys(data_path)
    validation_count = max(1, len(keys) // 5)
    train_keys, validation_keys = keys[:-validation_count], keys[-validation_count:]
    train_data = NLDHDF5GoalDataset(
        data_path, episode_keys=train_keys, context_length=1, goal_horizon=goal_horizon,
        samples_per_epoch=steps * batch_size,
    )
    validation_data = NLDHDF5GoalDataset(
        data_path, episode_keys=validation_keys, context_length=1, goal_horizon=goal_horizon,
        samples_per_epoch=validation_samples, seed=20_000,
    )
    validation_batches = [move_batch(batch, device) for batch in DataLoader(validation_data, batch_size=batch_size)]
    model = GoalConditionedLeWorldModel(report_model_config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    timeline: list[dict] = [{"step": 0, "diagnostics": diagnose(model, validation_batches)}]
    started = time.perf_counter()
    for step, batch in enumerate(DataLoader(train_data, batch_size=batch_size), start=1):
        train_metrics = pretrain_step(model, move_batch(batch, device), optimizer, sigreg_slices=sigreg_slices)
        if step % eval_every == 0 or step == steps:
            entry = {
                "step": step,
                "train_prediction_loss": train_metrics["prediction_loss"],
                "train_sigreg_loss": train_metrics["sigreg_loss"],
                "train_total_loss": train_metrics["loss"],
                "diagnostics": diagnose(model, validation_batches),
            }
            timeline.append(entry)
            diagnostics = entry["diagnostics"]
            print(
                f"step {step:4d}/{steps} | train {train_metrics['prediction_loss']:.4f} | "
                f"held-out {diagnostics['prediction_loss']:.4f} | "
                f"std {diagnostics['latent_feature_std_mean']:.3f} | "
                f"rank {diagnostics['latent_effective_rank']:.2f}",
                flush=True,
            )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    torch.save(
        {"model": model.state_dict(), "config": asdict(report_model_config())},
        output / "checkpoint.pt",
    )
    media = render_media(data_path, validation_data, output)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "config": {
            "steps": steps, "batch_size": batch_size, "goal_horizon": goal_horizon,
            "sigreg_slices": sigreg_slices, "eval_every": eval_every,
            "validation_samples": validation_samples, "device": device,
            "train_episodes": len(train_keys), "validation_episodes": len(validation_keys),
            "elapsed_seconds": elapsed, "samples_per_second": steps * batch_size / max(elapsed, 1e-12),
        },
        "timeline": timeline,
        "media": media,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return write_report(payload, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on NLD and generate an HTML diagnostic report")
    parser.add_argument("--data", default="data/downloads/nld-aa-taster.hdf5")
    parser.add_argument("--output", default="reports/nld-long-run")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--goal-horizon", type=int, default=64)
    parser.add_argument("--sigreg-slices", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    path = run_report(
        Path(args.data), Path(args.output), steps=args.steps, batch_size=args.batch_size,
        goal_horizon=args.goal_horizon, sigreg_slices=args.sigreg_slices,
        eval_every=args.eval_every, validation_samples=args.validation_samples, device=args.device,
    )
    print(path)


if __name__ == "__main__":
    main()
