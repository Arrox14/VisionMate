from __future__ import annotations

from research.ape import AssistivePriorityEngine


def test_ape_ranks_objects_by_assistive_priority() -> None:
    engine = AssistivePriorityEngine()
    result = engine.rank_objects(
        [
            {"object_name": "person", "distance": 0.8, "movement": 0.9, "hazard": True, "object_class": "person"},
            {"object_name": "chair", "distance": 2.0, "movement": 0.1, "hazard": False, "object_class": "furniture"},
            {"object_name": "bottle", "distance": 1.5, "movement": 0.0, "hazard": False, "object_class": "object"},
        ]
    )

    assert result.ranked_objects[0].object_name == "person"
    assert result.ranked_objects[0].level == "Critical"
    assert result.critical_objects == ["person"]
    assert set(result.medium_priority_objects + result.low_priority_objects) == {"chair", "bottle"}
