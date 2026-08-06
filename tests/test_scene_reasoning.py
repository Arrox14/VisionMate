from __future__ import annotations

from reasoning.scene_reasoning import HierarchicalSceneReasoningModule


def test_scene_reasoning_infers_relationships_and_hazards() -> None:
    engine = HierarchicalSceneReasoningModule()
    result = engine.reason(
        [
            {"name": "chair", "priority_score": 0.95, "object_class": "furniture", "position": (0.2, 0.1)},
            {"name": "door", "priority_score": 0.88, "object_class": "utility", "position": (0.8, 0.1)},
            {"name": "teacher", "priority_score": 0.90, "object_class": "person", "position": (0.4, 0.5)},
            {"name": "whiteboard", "priority_score": 0.82, "object_class": "object", "position": (0.4, 0.1)},
            {"name": "bottle", "priority_score": 0.40, "object_class": "object", "position": (0.3, 0.3)},
        ]
    )

    assert any("blocks" in relation.description.lower() for relation in result.relationships)
    assert any("near" in relation.description.lower() for relation in result.relationships)
    assert any(hazard.description.lower().startswith("hazard") for hazard in result.hazards)
    assert any(interaction.description.lower().startswith("interaction") for interaction in result.interactions)
    assert result.summary
