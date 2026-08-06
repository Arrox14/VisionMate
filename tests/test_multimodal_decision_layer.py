from __future__ import annotations

from decision.multimodal_decision_layer import MultimodalDecisionLayer


def test_multimodal_decision_layer_fuses_inputs_into_unified_context() -> None:
    engine = MultimodalDecisionLayer()
    context = engine.fuse(
        person={"name": "Alice", "confidence": 0.93},
        scene={"context": "classroom", "summary": "teacher standing near whiteboard"},
        objects=[
            {"name": "chair", "priority_score": 0.91, "object_class": "furniture"},
            {"name": "door", "priority_score": 0.80, "object_class": "utility"},
        ],
        hazards=[{"description": "chair blocks doorway", "severity": "high"}],
        memory={"encounter_count": 2, "relationship": "friend"},
        priority={"ranked_objects": [{"object_name": "chair", "priority_score": 0.91, "level": "Critical"}]},
        recommendations=["guide user to the exit"],
    )

    assert context.person.name == "Alice"
    assert context.scene.context == "classroom"
    assert len(context.objects) == 2
    assert context.hazards[0].description == "chair blocks doorway"
    assert context.memory["relationship"] == "friend"
    assert context.priority.rankings[0].object_name == "chair"
    assert context.recommendations[0] == "guide user to the exit"
