from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class UnifiedContext:
    """Unified decision object produced by MDL fusion."""

    scene: str
    detected_objects: List[str] = field(default_factory=list)
    recognized_people: List[str] = field(default_factory=list)
    priorities: List[Dict[str, Any]] = field(default_factory=list)
    filtered_context: List[Dict[str, Any]] = field(default_factory=list)
    hazards: List[str] = field(default_factory=list)
    interactions: List[str] = field(default_factory=list)
    decision_summary: str = ""


class MultimodalDecisionLayer:
    """MDL: fuses perception, memory, priority, context, and scene reasoning into one structured representation."""

    def fuse(
        self,
        objects: Optional[List[str]] = None,
        people: Optional[List[str]] = None,
        priorities: Optional[List[Dict[str, Any]]] = None,
        filtered_context: Optional[List[Dict[str, Any]]] = None,
        hazards: Optional[List[str]] = None,
        interactions: Optional[List[str]] = None,
        scene: str = "home",
    ) -> UnifiedContext:
        detected_objects = [obj for obj in (objects or []) if obj]
        recognized_people = [person for person in (people or []) if person]
        priority_items = list(priorities or [])
        context_items = list(filtered_context or [])
        hazard_items = list(hazards or [])
        interaction_items = list(interactions or [])

        decision_summary = self._build_summary(
            detected_objects=detected_objects,
            recognized_people=recognized_people,
            priority_items=priority_items,
            hazard_items=hazard_items,
            interaction_items=interaction_items,
        )

        return UnifiedContext(
            scene=scene,
            detected_objects=detected_objects,
            recognized_people=recognized_people,
            priorities=priority_items,
            filtered_context=context_items,
            hazards=hazard_items,
            interactions=interaction_items,
            decision_summary=decision_summary,
        )

    def _build_summary(
        self,
        detected_objects: List[str],
        recognized_people: List[str],
        priority_items: List[Dict[str, Any]],
        hazard_items: List[str],
        interaction_items: List[str],
    ) -> str:
        object_summary = ", ".join(detected_objects) if detected_objects else "none"
        people_summary = ", ".join(recognized_people) if recognized_people else "none"
        top_priority = priority_items[0].get("object_name", "none") if priority_items else "none"
        hazard_summary = ", ".join(hazard_items) if hazard_items else "none"
        interaction_summary = ", ".join(interaction_items) if interaction_items else "none"
        return (
            f"Objects={object_summary}; People={people_summary}; "
            f"TopPriority={top_priority}; Hazards={hazard_summary}; Interactions={interaction_summary}"
        )
