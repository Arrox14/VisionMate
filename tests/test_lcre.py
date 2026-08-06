from __future__ import annotations

import re

from intelligence.lcre import LLMReasoningEngine


def test_lcre_generates_guidance_from_unified_context() -> None:
    engine = LLMReasoningEngine()
    result = engine.reason(
        {
            "person": {"name": "Alice", "confidence": 0.93},
            "scene": {"context": "road", "summary": "pedestrian crossing"},
            "objects": [{"name": "person", "priority_score": 0.96, "object_class": "person"}],
            "hazards": [{"description": "vehicle nearby", "severity": "high"}],
            "memory": {"relationship": "friend"},
            "priority": {"rankings": [{"object_name": "person", "priority_score": 0.96, "level": "Critical"}]},
            "recommendations": ["guide user to the sidewalk"],
        }
    )

    assert "vehicle nearby" in result.hazard_assessment
    assert "road" in result.context_reasoning.lower()
    assert "Alice" in result.answer or "Alice" in result.narration
    assert result.recommendations
    assert result.narration


def test_lcre_narration_is_personalized_and_non_repetitive() -> None:
    engine = LLMReasoningEngine()
    context = {
        "person": {"name": "Alice", "confidence": 0.93},
        "scene": {"context": "road", "summary": "pedestrian crossing"},
        "objects": [{"name": "car", "priority_score": 0.97, "object_class": "vehicle"}],
        "hazards": [{"description": "vehicle nearby", "severity": "high"}],
        "memory": {"relationship": "friend"},
        "priority": {"rankings": [{"object_name": "car", "priority_score": 0.97, "level": "Critical"}]},
        "recommendations": ["guide user to the sidewalk"],
    }

    first = engine.reason(context)
    second = engine.reason(context)

    assert "Alice" in first.narration
    assert "vehicle nearby" in first.narration.lower()
    assert 2 <= len(re.findall(r"[.!?]+", first.narration)) <= 4
    assert first.narration != second.narration
