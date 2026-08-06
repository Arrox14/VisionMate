from __future__ import annotations

from pathlib import Path

from evaluation.metrics import VisionMateEvaluator


def test_evaluation_pipeline_generates_real_artifacts(tmp_path: Path) -> None:
    evaluator = VisionMateEvaluator(output_dir=tmp_path)
    result = evaluator.run_experiment()

    assert result.detection_accuracy >= 0.0
    assert result.recognition_accuracy >= 0.0
    assert result.memory_consistency >= 0.0
    assert result.latency_ms >= 0.0
    assert result.response_time_ms >= 0.0
    assert result.narration_quality >= 0.0

    assert (tmp_path / "evaluation_result.json").exists()
    assert (tmp_path / "evaluation_results.csv").exists()
    assert (tmp_path / "evaluation_metrics.tex").exists()
    assert (tmp_path / "evaluation_plot.png").exists()
