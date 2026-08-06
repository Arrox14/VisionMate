from __future__ import annotations

from research.mdl import MultimodalDecisionLayer


def test_mdl_fuses_outputs_into_unified_context() -> None:
    engine = MultimodalDecisionLayer()
    fused = engine.fuse(
        objects=["person", "chair", "car"],
        people=["Alice"],
        priorities=[{"object_name": "person", "priority_score": 0.96, "level": "Critical"}],
        filtered_context=[{"name": "person", "reason": "socially important in classroom"}],
        hazards=["vehicle nearby"],
        interactions=["social seating interaction"],
        scene="classroom",
    )

    assert fused.scene == "classroom"
    assert fused.detected_objects == ["person", "chair", "car"]
    assert fused.recognized_people == ["Alice"]
    assert fused.priorities[0]["object_name"] == "person"
    assert "vehicle nearby" in fused.hazards
    assert "TopPriority=person" in fused.decision_summary
