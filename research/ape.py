from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class PriorityItem:
    """Structured priority result for one detected object."""

    object_name: str
    priority_score: float
    level: str
    reasons: List[str] = field(default_factory=list)


@dataclass(slots=True)
class PriorityResult:
    """Structured ranking output from the APE module."""

    ranked_objects: List[PriorityItem]
    critical_objects: List[str]
    high_priority_objects: List[str]
    medium_priority_objects: List[str]
    low_priority_objects: List[str]


class AssistivePriorityEngine:
    """APE: ranks perceived objects by assistive importance and hazard."""

    def __init__(self) -> None:
        self._hazard_weights = {
            "person": 0.95,
            "car": 1.0,
            "truck": 1.0,
            "motorcycle": 0.98,
            "bicycle": 0.85,
            "dog": 0.88,
            "knife": 1.0,
            "fire": 1.0,
            "cell phone": 0.55,
            "bottle": 0.45,
            "cup": 0.35,
            "chair": 0.40,
            "book": 0.30,
            "laptop": 0.50,
            "backpack": 0.45,
            "handbag": 0.42,
        }
        self._movement_weights = {
            "person": 0.20,
            "car": 0.25,
            "truck": 0.25,
            "motorcycle": 0.22,
            "bicycle": 0.18,
            "dog": 0.18,
            "knife": 0.05,
            "fire": 0.10,
        }

    def score_object(
        self,
        object_name: str,
        distance: Optional[float] = None,
        movement: Optional[float] = None,
        hazard: Optional[bool] = None,
        object_class: Optional[str] = None,
    ) -> PriorityItem:
        normalized_name = object_name.lower().strip()
        distance_score = self._distance_score(distance)
        movement_score = self._movement_score(normalized_name, movement)
        hazard_score = self._hazard_score(normalized_name, hazard)
        class_score = self._class_score(normalized_name, object_class)

        priority_score = round(
            min(1.0, 0.35 * distance_score + 0.25 * movement_score + 0.25 * hazard_score + 0.15 * class_score),
            3,
        )
        level = self._level_for_score(priority_score)

        reasons: List[str] = []
        if distance is not None:
            reasons.append(f"distance={distance:.2f}")
        if movement is not None:
            reasons.append(f"movement={movement:.2f}")
        if hazard:
            reasons.append("hazard=true")
        if object_class:
            reasons.append(f"class={object_class}")

        return PriorityItem(
            object_name=object_name,
            priority_score=priority_score,
            level=level,
            reasons=reasons,
        )

    def rank_objects(self, objects: List[Dict[str, Any]]) -> PriorityResult:
        ranked = [
            self.score_object(
                object_name=item.get("object_name", ""),
                distance=item.get("distance"),
                movement=item.get("movement"),
                hazard=item.get("hazard"),
                object_class=item.get("object_class"),
            )
            for item in objects
            if item.get("object_name")
        ]
        ranked.sort(key=lambda item: item.priority_score, reverse=True)

        critical = [item.object_name for item in ranked if item.level == "Critical"]
        high = [item.object_name for item in ranked if item.level == "High"]
        medium = [item.object_name for item in ranked if item.level == "Medium"]
        low = [item.object_name for item in ranked if item.level == "Low"]

        return PriorityResult(
            ranked_objects=ranked,
            critical_objects=critical,
            high_priority_objects=high,
            medium_priority_objects=medium,
            low_priority_objects=low,
        )

    def _distance_score(self, distance: Optional[float]) -> float:
        if distance is None:
            return 0.5
        if distance <= 1.0:
            return 1.0
        if distance <= 3.0:
            return 0.8
        if distance <= 6.0:
            return 0.6
        return 0.3

    def _movement_score(self, object_name: str, movement: Optional[float]) -> float:
        if movement is None:
            return self._movement_weights.get(object_name, 0.1)
        return min(1.0, 0.5 + movement * 0.5)

    def _hazard_score(self, object_name: str, hazard: Optional[bool]) -> float:
        if hazard is True:
            return 1.0
        return self._hazard_weights.get(object_name, 0.2)

    def _class_score(self, object_name: str, object_class: Optional[str]) -> float:
        if object_class is None:
            return 0.5
        class_weight = {
            "person": 0.95,
            "vehicle": 0.9,
            "animal": 0.8,
            "hazard": 1.0,
            "utility": 0.6,
            "furniture": 0.4,
            "object": 0.35,
        }.get(object_class.lower(), 0.5)
        return class_weight

    def _level_for_score(self, score: float) -> str:
        if score >= 0.85:
            return "Critical"
        if score >= 0.65:
            return "High"
        if score >= 0.4:
            return "Medium"
        return "Low"
