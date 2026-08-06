from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import VisionMateEvaluator


def main() -> None:
    evaluator = VisionMateEvaluator(output_dir=Path(__file__).resolve().parent / "results")
    result = evaluator.run_experiment()

    print("Evaluation complete")
    print(evaluator.to_markdown_table(result))
    print("Artifacts written to", evaluator.output_dir)


if __name__ == "__main__":
    main()
