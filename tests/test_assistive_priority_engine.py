from __future__ import annotations

from reasoning.assistive_priority_engine import AssistivePriorityEngine


def test_ape_ranks_common_indoor_objects_into_expected_priority_levels() -> None:
    engine = AssistivePriorityEngine()
    result = engine.rank_detections(
        [
            {
                "name": "chair",
                "distance": 0.9,
                "movement": 0.0,
                "hazard": False,
                "object_class": "furniture",
                "context": "indoor",
                "blocking_path": True,
            },
            {
                "name": "door",
                "distance": 2.0,
                "movement": 0.1,
                "hazard": False,
                "object_class": "utility",
                "context": "indoor",
            },
            {
                "name": "laptop",
                "distance": 1.8,
                "movement": 0.05,
                "hazard": False,
                "object_class": "object",
                "context": "indoor",
            },
            {
                "name": "bottle",
                "distance": 3.0,
                "movement": 0.0,
                "hazard": False,
                "object_class": "object",
                "context": "indoor",
            },
        ]
    )

    assert result.ranked_objects[0].object_name == "chair"
    assert result.ranked_objects[0].level == "Critical"
    assert result.critical_objects == ["chair"]
    assert result.high_priority_objects == ["door"]
    assert result.medium_priority_objects == ["laptop"]
    assert result.low_priority_objects == ["bottle"]
