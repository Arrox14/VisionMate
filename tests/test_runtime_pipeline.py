from __future__ import annotations

from agent.runtime_pipeline import VisionMateRuntimePipeline


def test_runtime_pipeline_generates_narration_and_context() -> None:
    pipeline = VisionMateRuntimePipeline()
    decision = pipeline.process_detections(
        ["chair", "door", "teacher"],
        person_info={"name": "Alice", "confidence": 0.93},
        scene_context="classroom",
    )

    assert decision.narration
    assert decision.context_summary
    assert decision.scene_context == "classroom"
    assert decision.priority_objects
