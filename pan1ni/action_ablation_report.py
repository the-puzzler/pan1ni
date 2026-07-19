from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def run(root: Path, output: Path) -> tuple[Path, Path]:
    rows = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        action_metrics = directory / "metrics.json"
        closed_loop = directory / "closed-loop-eval.json"
        if not action_metrics.exists() or not closed_loop.exists():
            continue
        action = json.loads(action_metrics.read_text(encoding="utf-8"))
        evaluation = json.loads(closed_loop.read_text(encoding="utf-8"))
        latest = action["timeline"][-1]
        rows.append(
            {
                "feature": action["config"]["feature"],
                "player_move_accuracy": latest["player_validation"]["accuracy"],
                "player_move_balanced_accuracy": latest["player_validation"]["balanced_accuracy"],
                "special_accuracy": latest["special_validation"]["accuracy"],
                "special_balanced_accuracy": latest["special_validation"]["balanced_accuracy"],
                "closed_loop_episodes": evaluation["episodes"],
                "closed_loop_successes": evaluation["successes"],
                "closed_loop_success_rate": evaluation["success_rate"],
                "median_success_steps": evaluation["median_success_steps"],
            }
        )
    if not rows:
        raise ValueError(f"no completed ablations under {root}")
    rows.sort(key=lambda row: row["closed_loop_success_rate"], reverse=True)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps({"ablations": rows}, indent=2), encoding="utf-8")

    plt.style.use("dark_background")
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    labels = [row["feature"] for row in rows]
    x = range(len(rows))
    axes[0].bar(x, [row["closed_loop_success_rate"] for row in rows], color="#70d6a5")
    axes[0].set(title="Closed-loop KeyRoom success", ylabel="Success rate", ylim=(0, 1))
    axes[1].bar(
        [index - 0.2 for index in x],
        [row["player_move_balanced_accuracy"] for row in rows],
        width=0.4,
        label="human movement",
        color="#22d3ee",
    )
    axes[1].bar(
        [index + 0.2 for index in x],
        [row["special_balanced_accuracy"] for row in rows],
        width=0.4,
        label="pickup/apply",
        color="#c084fc",
    )
    axes[1].set(title="Held-out action balanced accuracy", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(list(x), labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize action feature ablations")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(run(args.root, args.output))


if __name__ == "__main__":
    main()
