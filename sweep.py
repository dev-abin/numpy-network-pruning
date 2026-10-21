"""Compare two pruning criteria at five levels across three seeds."""

import copy
import csv
import json
from pathlib import Path

from main import SmallNetwork, make_data, pruning_schedule, np, plt


def write_csv(path, rows):
    """Save experiment rows with one header containing their field names."""
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Compare both pruning methods across budgets/seeds; save results and a plot."""
    x_train, y_train, x_test, y_test = make_data(seed=42)
    levels = [0, 0.25, 0.5, 0.75, 0.9]
    seeds = [7, 8, 9]
    criteria = ["magnitude", "taylor"]
    rows = []
    for seed in seeds:
        baseline = SmallNetwork(seed)
        baseline.train(x_train, y_train)
        _, before_accuracy = baseline.evaluate(x_test, y_test)
        for criterion in criteria:
            for target in levels:
                # Same warmup weights, momentum and shuffle state for both methods.
                model = copy.deepcopy(baseline)
                model.train(x_train, y_train, schedule=pruning_schedule(target),
                            criterion=criterion)
                # Evaluate after equal training budgets, including the 0% control.
                loss, accuracy = model.evaluate(x_test, y_test)
                active = int(model.mask1.sum() + model.mask2.sum())
                rows.append({
                    "criterion": criterion, "seed": seed, "target_sparsity": target,
                    "extra_full_training_gradient_passes":
                        3 if criterion == "taylor" and target > 0 else 0,
                    "achieved_sparsity": 1 - active / 32, "active_weights": active,
                    "accuracy_before": before_accuracy,
                    "accuracy_after_gradual_pruning": accuracy,
                    "loss_after_gradual_pruning": loss,
                })

    summary = []
    for criterion in criteria:
        for target in levels:
            group = [r for r in rows if r["criterion"] == criterion
                     and r["target_sparsity"] == target]
            accuracies = [r["accuracy_after_gradual_pruning"] for r in group]
            summary.append({
                "criterion": criterion, "target_sparsity": target,
                "achieved_sparsity": group[0]["achieved_sparsity"],
                "active_weights": group[0]["active_weights"],
                "mean_accuracy": float(np.mean(accuracies)),
                "sample_std_accuracy": float(np.std(accuracies, ddof=1)),
                "seeds": seeds,
            })

    differences = []
    for target in levels:
        paired_deltas = []
        for seed in seeds:
            pair = {r["criterion"]: r["accuracy_after_gradual_pruning"] for r in rows
                    if r["seed"] == seed and r["target_sparsity"] == target}
            paired_deltas.append(100 * (pair["taylor"] - pair["magnitude"]))
        differences.append({
            "target_sparsity": target, "seeds": seeds,
            "taylor_minus_magnitude_percentage_points": paired_deltas,
            "mean_difference_percentage_points": float(np.mean(paired_deltas)),
            "sample_std_difference_percentage_points": float(np.std(paired_deltas, ddof=1)),
        })

    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    write_csv(output / "comparison.csv", rows)
    report = {
        "settings": {
            "data_seed": 42, "seeds": seeds, "batch_size": 64, "momentum": 0.9,
            "warmup_epochs": 150, "pruning_epochs": 150, "learning_rate": 0.01,
            "score_data": "All 900 training samples at each pruning checkpoint; no test data.",
            "budget_note": "Equal optimizer updates and batch orders. Taylor uses three extra full-training gradient evaluations per nonzero target, without updates; training compute/time is not equal.",
        },
        "method_results": summary,
        "paired_differences": differences,
    }
    (output / "comparison_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for criterion in criteria:
        group = [r for r in summary if r["criterion"] == criterion]
        ax.errorbar(
            [100 * r["achieved_sparsity"] for r in group],
            [100 * r["mean_accuracy"] for r in group],
            yerr=[100 * r["sample_std_accuracy"] for r in group],
            marker="o", capsize=4, label=criterion, alpha=0.85,
        )
    ax.set(xlabel="Achieved weight sparsity (%)", ylabel="Final test accuracy (%)",
           title="Gradual pruning: mean ± sample SD, 3 seeds", ylim=(0, 100))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "sparsity_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"Saved {len(rows)} runs, comparison_summary.json and sparsity_accuracy.png to {output}")


if __name__ == "__main__":
    main()
