from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class AttentionItem:
    """A filtered object or face with its relevance score for the current scene."""

    item_type: str
    name: str
    relevance_score: float
    reason: str


@dataclass(slots=True)
class AttentionResult:
    """Structured output from the CAE module."""

    scene: str
    filtered_items: List[AttentionItem]
    summary: str
    ignored_items: List[str]


class ContextAwareAttentionEngine:
    """CAE: filters assistive information based on the current environment."""

    def __init__(self) -> None:
        self._environment_rules = {
            "classroom": {
                "relevant": {"person", "book", "chair", "laptop", "backpack", "bottle", "cup"},
                "priority": {"person": 1.0, "book": 0.9, "chair": 0.7, "laptop": 0.85, "backpack": 0.75, "bottle": 0.5, "cup": 0.5},
            },
            "office": {
                "relevant": {"person", "laptop", "chair", "book", "cup", "bottle", "cell phone"},
                "priority": {"person": 0.95, "laptop": 0.9, "chair": 0.7, "book": 0.75, "cup": 0.55, "bottle": 0.5, "cell phone": 0.6},
            },
            "road": {
                "relevant": {"person", "car", "truck", "motorcycle", "bicycle", "dog", "traffic light", "crosswalk"},
                "priority": {"person": 1.0, "car": 0.98, "truck": 0.99, "motorcycle": 0.95, "bicycle": 0.8, "dog": 0.75, "traffic light": 0.7, "crosswalk": 0.65},
            },
            "home": {
                "relevant": {"person", "chair", "cup", "bottle", "book", "laptop", "cell phone", "backpack", "handbag"},
                "priority": {"person": 0.95, "chair": 0.7, "cup": 0.6, "bottle": 0.6, "book": 0.7, "laptop": 0.8, "cell phone": 0.65, "backpack": 0.6, "handbag": 0.6},
            },
        }

    def filter_scene(
        self,
        objects: Optional[List[str]] = None,
        faces: Optional[List[str]] = None,
        scene: Optional[str] = None,
    ) -> AttentionResult:
        scene_name = (scene or "home").lower().strip()
        if scene_name not in self._environment_rules:
            scene_name = "home"

        known_objects = [obj.lower().strip() for obj in (objects or []) if obj]
        known_faces = [face.lower().strip() for face in (faces or []) if face]

        rules = self._environment_rules[scene_name]
        relevant = rules["relevant"]
        priority_map = rules["priority"]

        filtered_items: List[AttentionItem] = []
        ignored_items: List[str] = []

        for obj in known_objects:
            if obj in relevant:
                score = priority_map.get(obj, 0.5)
                reason = self._reason_for_object(scene_name, obj)
                filtered_items.append(
                    AttentionItem(item_type="object", name=obj, relevance_score=round(score, 3), reason=reason)
                )
            else:
                ignored_items.append(obj)

        for face in known_faces:
            if face in relevant or scene_name in {"classroom", "office", "home"}:
                score = priority_map.get("person", 0.9)
                filtered_items.append(
                    AttentionItem(item_type="face", name=face, relevance_score=round(score, 3), reason="recognized person in scene")
                )
            else:
                ignored_items.append(face)

        filtered_items.sort(key=lambda item: item.relevance_score, reverse=True)

        summary = self._build_summary(scene_name, filtered_items, ignored_items)
        return AttentionResult(
            scene=scene_name,
            filtered_items=filtered_items,
            summary=summary,
            ignored_items=ignored_items,
        )

    def _reason_for_object(self, scene: str, obj: str) -> str:
        if obj == "person":
            return f"socially important in {scene}"
        if obj in {"laptop", "book", "chair"}:
            return f"common functional object in {scene}"
        if obj in {"car", "truck", "motorcycle"}:
            return f"movement hazard in {scene}"
        return f"contextually relevant in {scene}"

    def _build_summary(self, scene: str, filtered_items: List[AttentionItem], ignored_items: List[str]) -> str:
        if not filtered_items:
            return f"No assistive information for {scene}."
        names = ", ".join(item.name for item in filtered_items)
        ignored = ", ".join(ignored_items) if ignored_items else "none"
        return f"In {scene}, the system focuses on {names}; ignored {ignored}."
