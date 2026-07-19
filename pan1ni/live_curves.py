from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt


def render(metrics_path: Path, output_path: Path) -> int:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    timeline = payload["timeline"]
    trained = [entry for entry in timeline if "train_sigreg_loss" in entry]
    diagnostic_key = (
        "simulator_diagnostics"
        if "simulator_diagnostics" in timeline[0]
        else "diagnostics"
    )
    steps = [entry["step"] for entry in timeline]
    train_steps = [entry["step"] for entry in trained]

    plt.style.use("dark_background")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    latest_step = timeline[-1]["step"]
    total_steps = payload["config"]["steps"]
    is_flow = payload["config"].get("objective") == "flow"
    figure.suptitle(f"Training progress — step {latest_step:,}/{total_steps:,}", fontsize=14)

    axes[0].plot(
        steps,
        [entry[diagnostic_key]["latent_effective_rank"] for entry in timeline],
        marker="o",
        markersize=3,
        label="simulator" if "player_diagnostics" in timeline[0] else "held-out",
        color="#c084fc",
    )
    if "player_diagnostics" in timeline[0]:
        axes[0].plot(
            steps,
            [entry["player_diagnostics"]["latent_effective_rank"] for entry in timeline],
            marker="o",
            markersize=3,
            label="player",
            color="#22d3ee",
        )
    axes[0].set(title="Effective rank by held-out source", xlabel="Step", ylabel="Rank")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        train_steps,
        [entry["train_sigreg_loss"] for entry in trained],
        marker="o",
        markersize=3,
        color="#34d399",
    )
    axes[1].set(title="SIGReg", xlabel="Step", ylabel="Loss")

    diagnostics = [entry[diagnostic_key] for entry in timeline]
    axes[2].plot(
        steps,
        [item["prediction_loss"] for item in diagnostics],
        marker="o",
        markersize=3,
        label="held-out predictor",
        color="#fb7185",
    )
    if not is_flow:
        axes[2].plot(
            train_steps,
            [entry["train_prediction_loss"] for entry in trained],
            label="train predictor",
            color="#fbbf24",
        )
    if "player_diagnostics" in timeline[0]:
        axes[2].plot(
            steps,
            [entry["player_diagnostics"]["prediction_loss"] for entry in timeline],
            marker="o",
            markersize=3,
            label="held-out player",
            color="#22d3ee",
        )
    axes[2].plot(
        steps,
        [item["copy_current_loss"] for item in diagnostics],
        label="copy current",
        color="#60a5fa",
    )
    axes[2].plot(
        steps,
        [item["zero_goal_loss"] for item in diagnostics],
        label="zero goal",
        color="#a3e635",
    )
    axes[2].plot(
        steps,
        [item["shuffled_goal_loss"] for item in diagnostics],
        label="shuffled goal",
        color="#f472b6",
        alpha=0.7,
    )
    axes[2].set(
        title="One-step next-state MSE" if is_flow else "Next-state MSE",
        xlabel="Step",
        ylabel="MSE",
    )
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    figure.savefig(temporary, format="png", dpi=180)
    plt.close(figure)
    os.replace(temporary, output_path)
    return latest_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously refresh training curves")
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    output = args.output or args.metrics.with_name("learning-curves.png")
    last_mtime = -1
    while True:
        try:
            mtime = args.metrics.stat().st_mtime_ns
            if mtime != last_mtime:
                step = render(args.metrics, output)
                last_mtime = mtime
                print(f"updated {output} at step {step}", flush=True)
                payload = json.loads(args.metrics.read_text(encoding="utf-8"))
                if step >= payload["config"]["steps"]:
                    break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
