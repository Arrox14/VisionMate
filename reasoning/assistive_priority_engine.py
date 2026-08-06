from __future__ import annotations

"""Assistive Priority Engine (APE).

This module converts raw YOLO-like detections into assistive-aware priorities.
It is intentionally independent from YOLO so the detector can remain unchanged
while the reasoning layer produces structured output for downstream modules.

Scoring formula
---------------
The engine computes a score in the range [0.0, 1.0] using a weighted sum:

priority_score =
    0.35 * distance_score
  + 0.25 * movement_score
  + 0.20 * hazard_score
  + 0.10 * context_score
  + 0.10 * category_score
  + 0.15 * blocking_path_bonus

where each component is normalized to [0.0, 1.0].

Interpretation
-------------
- Critical: score >= 0.85
- High: score >= 0.65
- Medium: score >= 0.40
- Low: otherwise

The score is designed to prioritize objects that are close, moving, hazardous,
or likely to interfere with safe navigation or task assistance.
"""

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
    """APE: ranks detections by assistive importance.

    The engine accepts YOLO-style detection dictionaries and returns structured
    priorities. It prepares the data for later CAE integration by exposing the
    context of each detection and by keeping the scoring logic explicit and
    documented.
    """

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
            "bottle": 0.10,
            "cup": 0.35,
            "chair": 0.40,
            "book": 0.30,
            "laptop": 0.50,
            "backpack": 0.45,
            "handbag": 0.42,
            "door": 0.60,
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
            "chair": 0.05,
            "door": 0.10,
            "laptop": 0.02,
            "bottle": 0.01,
        }
        self._category_weights = {
            "person": 0.95,
            "vehicle": 0.90,
            "animal": 0.80,
            "hazard": 1.00,
            "utility": 0.70,
            "furniture": 0.55,
            "object": 0.30,
            "electronic": 0.45,
        }
        self._context_weights = {
            "indoor": 0.70,
            "outdoor": 0.55,
            "unknown": 0.50,
        }

    def score_detection(
        self,
        object_name: str,
        distance: Optional[float] = None,
        movement: Optional[float] = None,
        hazard: Optional[bool] = None,
        object_class: Optional[str] = None,
        context: Optional[str] = None,
        blocking_path: bool = False,
    ) -> PriorityItem:
        """Score a single detection based on assistive relevance."""
        normalized_name = object_name.lower().strip()
        distance_score = self._distance_score(distance)
        movement_score = self._movement_score(normalized_name, movement)
        hazard_score = self._hazard_score(normalized_name, hazard)
        context_score = self._context_score(context)
        category_score = self._category_score(normalized_name, object_class)

        blocking_path_bonus = 1.0 if blocking_path else 0.0
        if blocking_path:
            distance_score = max(distance_score, 0.95)
            movement_score = max(movement_score, 0.65)
            hazard_score = max(hazard_score, 0.75)

        priority_score = round(
            min(
                1.0,
                0.35 * distance_score
                + 0.25 * movement_score
                + 0.20 * hazard_score
                + 0.10 * context_score
                + 0.10 * category_score
                + 0.15 * blocking_path_bonus,
            ),
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
        if context:
            reasons.append(f"context={context}")
        if blocking_path:
            reasons.append("blocking_path=true")

        return PriorityItem(
            object_name=object_name,
            priority_score=priority_score,
            level=level,
            reasons=reasons,
        )

    def rank_detections(self, detections: List[Dict[str, Any]]) -> PriorityResult:
        """Rank a list of detections and group them by priority level."""
        ranked = [
            self.score_detection(
                object_name=item.get("name") or item.get("object_name") or "",
                distance=item.get("distance"),
                movement=item.get("movement"),
                hazard=item.get("hazard"),
                object_class=item.get("object_class"),
                context=item.get("context"),
                blocking_path=bool(item.get("blocking_path", False)),
            )
            for item in detections
            if item.get("name") or item.get("object_name")
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

    def rank_objects(self, objects: List[Dict[str, Any]]) -> PriorityResult:
        """Backward-compatible wrapper for the older APE interface."""
        return self.rank_detections(
            [
                {
                    "name": item.get("object_name", ""),
                    "distance": item.get("distance"),
                    "movement": item.get("movement"),
                    "hazard": item.get("hazard"),
                    "object_class": item.get("object_class"),
                    "context": item.get("context"),
                    "blocking_path": bool(item.get("blocking_path", False)),
                }
                for item in objects
            ]
        )

    def _distance_score(self, distance: Optional[float]) -> float:
        """Convert distance into a normalized urgency signal."""
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
        """Weight dynamic motion so moving objects become more urgent."""
        if object_name == "bottle":
            return 0.0
        if movement is None:
            return self._movement_weights.get(object_name, 0.1)
        return min(1.0, 0.5 + movement * 0.5)

    def _hazard_score(self, object_name: str, hazard: Optional[bool]) -> float:
        """Map hazard labels and object semantics to a hazard risk factor."""
        if hazard is True:
            return 1.0
        return self._hazard_weights.get(object_name, 0.2)

    def _context_score(self, context: Optional[str]) -> float:
        """Adjust importance based on indoor/outdoor context."""
        if context is None:
            return self._context_weights["unknown"]
        return self._context_weights.get(context.lower(), self._context_weights["unknown"])

    def _category_score(self, object_name: str, object_class: Optional[str]) -> float:
        """Score relevance based on object category semantics."""
        if object_class is None:
            return 0.5
        category_weight = self._category_weights.get(object_class.lower(), 0.5)
        if object_name in {"chair", "door"} and object_class.lower() in {"furniture", "utility"}:
            category_weight = max(category_weight, 0.6)
        if object_name == "bottle":
            category_weight = 0.1
        return category_weight

    def _level_for_score(self, score: float) -> str:
        """Map a numeric score to the assistive priority level."""
        if score >= 0.85:
            return "Critical"
        if score >= 0.65:
            return "High"
        if score >= 0.4:
            return "Medium"
        return "Low"
