from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def run(root: Path, output: Path, evaluation_name: str) -> tuple[Path, Path]:
    rows = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        evaluation_path = directory / evaluation_name
        if directory.name == "uniform_random" and not evaluation_path.exists():
            evaluation_path = directory / "nle-closed-loop-eval.json"
        metrics_path = directory / "metrics.json"
        if not evaluation_path.exists():
            continue
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        records = evaluation["records"]
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            best = max(
                metrics["timeline"][1:],
                key=lambda record: record.get("selection_balanced_accuracy", -1),
            )
            heldout_accuracy = best["player_validation"]["accuracy"]
            heldout_balanced_accuracy = best["player_validation"]["balanced_accuracy"]
        else:
            heldout_accuracy = None
            heldout_balanced_accuracy = None
        # Softer, navigation-fair success: reaching the goal cell OR its immediate
        # neighbourhood counts, even if a later step oversteps it. The strict
        # success_rate requires landing on the exact cell (and stops on contact).
        near_goal = sum(record["best_distance"] <= 1 for record in records) / len(records)
        touched_goal = sum(record["best_distance"] == 0 for record in records) / len(records)
        rows.append(
            {
                "feature": evaluation["feature"],
                "episodes": evaluation["episodes"],
                "successes": evaluation["successes"],
                "success_rate": evaluation["success_rate"],
                "touched_goal_rate": touched_goal,
                "near_goal_rate": near_goal,
                "mean_best_distance": sum(record["best_distance"] for record in records) / len(records),
                "mean_best_progress": evaluation["mean_best_progress"],
                "mean_final_progress": sum(
                    max(-1.0, min(1.0, record["final_progress"])) for record in records
                ) / len(records),
                "collision_rate": evaluation["collision_rate"],
                "mean_revisit_rate": evaluation["mean_revisit_rate"],
                "termination_rate": sum(record["terminated"] for record in records) / len(records),
                "heldout_human_accuracy": heldout_accuracy,
                "heldout_human_balanced_accuracy": heldout_balanced_accuracy,
            }
        )
    if not rows:
        raise ValueError(f"no completed full-NLE ablations under {root}")
    rows.sort(key=lambda row: (row["near_goal_rate"], row["success_rate"]), reverse=True)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps({"ablations": rows}, indent=2), encoding="utf-8")

    plt.style.use("dark_background")
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    labels = [row["feature"] for row in rows]
    x = list(range(len(rows)))
    axes[0].bar(
        [index - 0.2 for index in x],
        [row["near_goal_rate"] for row in rows],
        width=0.4,
        label="reached goal (<=1 cell)",
        color="#70d6a5",
    )
    axes[0].bar(
        [index + 0.2 for index in x],
        [row["success_rate"] for row in rows],
        width=0.4,
        label="exact cell",
        color="#2f855a",
    )
    axes[0].set(title="Full-NLE goal success", ylabel="Rate", ylim=(0, 1))
    axes[0].legend(frameon=False)
    axes[1].bar(
        [index - 0.2 for index in x],
        [row["mean_best_progress"] for row in rows],
        width=0.4,
        label="best",
        color="#22d3ee",
    )
    axes[1].bar(
        [index + 0.2 for index in x],
        [row["mean_final_progress"] for row in rows],
        width=0.4,
        label="final",
        color="#f0b35a",
    )
    axes[1].set(title="Distance progress toward goal", ylabel="Normalized progress", ylim=(-1, 1))
    axes[1].legend(frameon=False)
    axes[2].bar(
        [index - 0.2 for index in x],
        [row["collision_rate"] for row in rows],
        width=0.4,
        label="collision",
        color="#fb7185",
    )
    axes[2].bar(
        [index + 0.2 for index in x],
        [row["mean_revisit_rate"] for row in rows],
        width=0.4,
        label="revisit",
        color="#c084fc",
    )
    axes[2].set(title="Failure behavior", ylabel="Rate", ylim=(0, 1))
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize full-NLE action feature ablations")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluation-name", default="nle-closed-loop-eval.json")
    args = parser.parse_args()
    print(run(args.root, args.output, args.evaluation_name))


if __name__ == "__main__":
    main()
