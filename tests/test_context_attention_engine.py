from __future__ import annotations

from reasoning.context_attention_engine import ContextAwareAttentionEngine


def test_classroom_context_focuses_on_teaching_and_exit_objects() -> None:
    engine = ContextAwareAttentionEngine()
    result = engine.filter_objects(
        [
            {"name": "teacher", "priority_score": 0.95, "object_class": "person"},
            {"name": "whiteboard", "priority_score": 0.80, "object_class": "object"},
            {"name": "exit", "priority_score": 0.75, "object_class": "utility"},
            {"name": "empty seat", "priority_score": 0.55, "object_class": "furniture"},
            {"name": "bottle", "priority_score": 0.40, "object_class": "object"},
            {"name": "laptop", "priority_score": 0.50, "object_class": "electronic"},
        ],
        context="classroom",
    )

    relevant_names = {item.object_name for item in result.relevant_objects}
    assert relevant_names >= {"teacher", "whiteboard", "exit", "empty seat"}
    assert result.context == "classroom"
    assert any(item.object_name == "teacher" and item.relevance_score >= 0.8 for item in result.relevant_objects)
    assert any(item.object_name == "bottle" for item in result.ignored_objects)


def test_road_context_focuses_on_vehicles_and_crossings() -> None:
    engine = ContextAwareAttentionEngine()
    result = engine.filter_objects(
        [
            {"name": "car", "priority_score": 0.92, "object_class": "vehicle"},
            {"name": "traffic light", "priority_score": 0.84, "object_class": "utility"},
            {"name": "crossing", "priority_score": 0.76, "object_class": "utility"},
            {"name": "tree", "priority_score": 0.30, "object_class": "object"},
            {"name": "bottle", "priority_score": 0.25, "object_class": "object"},
        ],
        context="road",
    )

    relevant_names = {item.object_name for item in result.relevant_objects}
    assert relevant_names >= {"car", "traffic light", "crossing"}
    assert {item.object_name for item in result.ignored_objects} >= {"tree", "bottle"}
