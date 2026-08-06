from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt


def generate_plot(output_dir: Path | None = None) -> None:
    output_dir = output_dir or Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "evaluation_result.json"
    if not data_path.exists():
        raise FileNotFoundError("No evaluation_result.json found. Run the evaluation script first.")

    with data_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    metrics = [
        ("Detection", result["detection_accuracy"]),
        ("Recognition", result["recognition_accuracy"]),
        ("Memory", result["memory_consistency"]),
        ("Narration", result["narration_quality"]),
    ]
    labels = [name for name, _ in metrics]
    values = [value for _, value in metrics]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#EECA3B"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("VisionMate Evaluation Metrics")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_plot.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    generate_plot()
