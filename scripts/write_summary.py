"""Regenerate human-readable measured tables from the saved experiment only."""

import json
from pathlib import Path


def main():
    result = json.loads(Path("reports/real/results.json").read_text())
    lines = [
        "# Measured real-data results",
        "",
        "Source: `reports/real/results.json`.",
        "",
        "| Model | Validation cost | Test AP | Precision | Recall | Alerts | P@100 | R@100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in result["test"].items():
        cost = result["policies"][name]["validation_cost"]
        lines.append(
            f"| {name} | {cost:.0f} | {metrics['average_precision']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['alerts']} | {metrics['precision_at_k']:.4f} | "
            f"{metrics['recall_at_k']:.4f} |"
        )
    lines += [
        "",
        f"Selected on validation only: **{result['selected_model']}**.",
        "",
        "Top-100 is a fixed budget for the entire test partition, not a daily budget.",
        "Costs are hypothetical FP=1, FN=100. All outputs are uncalibrated scores.",
        "",
    ]
    Path("reports/real/SUMMARY.md").write_text("\n".join(lines))
    print("Wrote reports/real/SUMMARY.md from measured results")


if __name__ == "__main__":
    main()
