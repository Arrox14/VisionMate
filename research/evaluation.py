from __future__ import annotations

from typing import Any, Dict

from research.mpam import PerceptionSnapshot


class EvaluationTracker:
    """Evaluation utilities for the research pipeline."""

    def __init__(self) -> None:
        self._events: list[Dict[str, Any]] = []

    def record(
        self,
        snapshot: PerceptionSnapshot,
        context: Dict[str, Any],
        narrative: str,
        suggestion: str,
    ) -> None:
        self._events.append(
            {
                "objects_detected": len(snapshot.objects),
                "people_detected": len(snapshot.recognized_people),
                "scene_type": context.get("scene_type", "unknown"),
                "narrative": narrative,
                "suggestion": suggestion,
            }
        )

    def snapshot(self) -> Dict[str, Any]:
        if not self._events:
            return {"scene_count": 0, "objects_detected": 0, "people_detected": 0}

        total_objects = sum(event["objects_detected"] for event in self._events)
        total_people = sum(event["people_detected"] for event in self._events)
        return {
            "scene_count": len(self._events),
            "objects_detected": total_objects,
            "people_detected": total_people,
        }
