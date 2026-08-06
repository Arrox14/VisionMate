from __future__ import annotations

"""Hierarchical Scene Reasoning Module (HSRM).

This module transforms a filtered set of objects into semantically meaningful
scene understanding using geometry and heuristics only. It does not rely on an
LLM or learned model.

The pipeline is intentionally simple and reusable:
1. Estimate relative positions from object coordinates.
2. Infer spatial relationships such as near, on, blocking, and aligned with.
3. Detect hazards such as obstruction or unsafe placement.
4. Extract human-object interactions such as standing near or using an object.
5. Produce a structured summary that can be consumed by later modules.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(slots=True)
class SceneRelation:
    """A relation inferred between two scene objects."""

    subject: str
    object: str
    description: str


@dataclass(slots=True)
class SceneHazard:
    """A hazard detected from the scene geometry."""

    description: str
    severity: str


@dataclass(slots=True)
class SceneInteraction:
    """A human-object interaction inferred from the scene."""

    description: str
    confidence: float


@dataclass(slots=True)
class SceneSummary:
    """Structured output from the scene reasoning module."""

    relationships: List[SceneRelation] = field(default_factory=list)
    hazards: List[SceneHazard] = field(default_factory=list)
    interactions: List[SceneInteraction] = field(default_factory=list)
    summary: str = ""


class HierarchicalSceneReasoningModule:
    """Heuristic scene parser for assistive understanding."""

    def __init__(self) -> None:
        self._hazard_keywords = {"chair", "door", "bottle", "knife", "fire", "cable"}

    def reason(self, objects: List[Dict[str, Any]]) -> SceneSummary:
        """Reason over a list of objects and generate scene structures."""
        normalized = [self._normalize_object(item) for item in objects if item.get("name")]
        relationships: List[SceneRelation] = []
        hazards: List[SceneHazard] = []
        interactions: List[SceneInteraction] = []

        for item in normalized:
            for other in normalized:
                if item["name"] == other["name"]:
                    continue
                relation = self._infer_relation(item, other)
                if relation is not None:
                    relationships.append(relation)

        for item in normalized:
            hazard = self._infer_hazard(item, normalized)
            if hazard is not None:
                hazards.append(hazard)

        for item in normalized:
            interaction = self._infer_interaction(item, normalized)
            if interaction is not None:
                interactions.append(interaction)

        summary = self._build_summary(normalized, relationships, hazards, interactions)
        return SceneSummary(
            relationships=relationships,
            hazards=hazards,
            interactions=interactions,
            summary=summary,
        )

    def _normalize_object(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize keys to support both APE and CAE-style input."""
        return {
            "name": str(item.get("name") or item.get("object_name") or "").strip().lower(),
            "object_class": str(item.get("object_class") or "object").strip().lower(),
            "priority_score": float(item.get("priority_score", 0.0) or 0.0),
            "position": tuple(item.get("position") or (0.0, 0.0)),
        }

    def _infer_relation(self, left: Dict[str, Any], right: Dict[str, Any]) -> Optional[SceneRelation]:
        """Infer a simple relation using spatial distance and object semantics."""
        distance = self._distance(left["position"], right["position"])
        if left["name"] == "chair" and right["name"] == "door":
            if distance < 0.45 or (left["position"][0] < right["position"][0] and left["position"][0] > 0.1):
                return SceneRelation(left["name"], right["name"], "chair blocks doorway")
        if left["name"] == "teacher" and right["name"] == "whiteboard" and distance < 0.35:
            return SceneRelation(left["name"], right["name"], "teacher standing near whiteboard")
        if left["name"] == "bottle" and right["name"] == "desk" and distance < 0.25:
            return SceneRelation(left["name"], right["name"], "bottle on desk")
        if distance < 0.25:
            return SceneRelation(left["name"], right["name"], f"{left['name']} near {right['name']}")
        return None

    def _infer_hazard(self, item: Dict[str, Any], objects: List[Dict[str, Any]]) -> Optional[SceneHazard]:
        """Detect obvious hazards from object semantics and position."""
        if item["name"] in self._hazard_keywords:
            severity = "high" if item["name"] in {"fire", "knife"} else "medium"
            return SceneHazard(f"hazard detected: {item['name']}", severity)
        if item["name"] == "chair":
            for other in objects:
                if other["name"] == "door" and self._distance(item["position"], other["position"]) < 0.35:
                    return SceneHazard("hazard detected: chair blocks doorway", "high")
        return None

    def _infer_interaction(self, item: Dict[str, Any], objects: List[Dict[str, Any]]) -> Optional[SceneInteraction]:
        """Infer a human-object interaction using object type and proximity."""
        if item["name"] == "teacher" and any(other["name"] == "whiteboard" for other in objects):
            return SceneInteraction("interaction: teacher is presenting at whiteboard", 0.9)
        if item["name"] == "teacher" and any(other["name"] == "door" for other in objects):
            return SceneInteraction("interaction: teacher near doorway", 0.7)
        if item["name"] == "bottle" and any(other["name"] == "desk" for other in objects):
            return SceneInteraction("interaction: bottle placed on desk", 0.75)
        return None

    def _build_summary(
        self,
        objects: List[Dict[str, Any]],
        relationships: List[SceneRelation],
        hazards: List[SceneHazard],
        interactions: List[SceneInteraction],
    ) -> str:
        """Produce a concise natural-language scene summary."""
        parts: List[str] = []
        if relationships:
            parts.append("Scene contains " + ", ".join(rel.description for rel in relationships[:3]))
        if hazards:
            parts.append("Hazards include " + ", ".join(hazard.description for hazard in hazards))
        if interactions:
            parts.append("Interactions include " + ", ".join(interaction.description for interaction in interactions))
        if not parts:
            return "Scene contains no strong semantic relationships."
        return " ".join(parts)

    def _distance(self, left: Tuple[float, float], right: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two positions."""
        dx = left[0] - right[0]
        dy = left[1] - right[1]
        return (dx * dx + dy * dy) ** 0.5
