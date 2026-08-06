from __future__ import annotations

"""Context-Aware Attention Engine (CAE).

This module filters a list of prioritized detections into the subset that is
relevant for assistive narration in a given environment. The implementation is
rule-based first so it remains interpretable and easy to extend before more
complex learning-based reasoning is introduced.

The engine supports a fixed set of contexts:
- classroom
- office
- road
- home
- hospital
- laboratory

Each context uses a small set of hand-authored rules that boost or suppress
objects based on semantic category, name, and assistive relevance.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class AttentionItem:
    """A single object retained or discarded by the attention engine."""

    object_name: str
    relevance_score: float
    reason: str


@dataclass(slots=True)
class AttentionResult:
    """Structured output from the CAE module."""

    context: str
    relevant_objects: List[AttentionItem]
    ignored_objects: List[AttentionItem]


class ContextAwareAttentionEngine:
    """Rule-based filtering engine for assistive context selection."""

    def __init__(self) -> None:
        self._context_rules = {
            "classroom": {
                "keep": {
                    "teacher",
                    "whiteboard",
                    "exit",
                    "empty seat",
                    "student",
                    "desk",
                    "chair",
                },
                "boost": {"teacher", "whiteboard", "exit", "empty seat"},
                "suppress": {"bottle", "cup", "book", "laptop"},
            },
            "office": {
                "keep": {"computer", "desk", "door", "person", "exit", "meeting table"},
                "boost": {"computer", "door", "exit"},
                "suppress": {"bottle", "cup", "chair"},
            },
            "road": {
                "keep": {"car", "truck", "bike", "bicycle", "traffic light", "crossing", "pedestrian", "vehicle"},
                "boost": {"car", "traffic light", "crossing", "pedestrian"},
                "suppress": {"bottle", "tree", "bag", "laptop"},
            },
            "home": {
                "keep": {"door", "stove", "couch", "bed", "person", "exit", "kitchen"},
                "boost": {"door", "stove", "bed", "exit"},
                "suppress": {"bottle", "cup", "remote"},
            },
            "hospital": {
                "keep": {"patient", "bed", "doctor", "exit", "wheelchair", "nurse"},
                "boost": {"patient", "doctor", "exit", "wheelchair"},
                "suppress": {"bottle", "cup", "bag"},
            },
            "laboratory": {
                "keep": {"experiment", "equipment", "chemical", "door", "exit", "person"},
                "boost": {"experiment", "chemical", "door", "exit"},
                "suppress": {"bottle", "cup", "book"},
            },
        }

    def filter_objects(
        self,
        objects: List[Dict[str, Any]],
        context: str,
    ) -> AttentionResult:
        """Filter an object list into context-relevant and ignored items."""
        normalized_context = self._normalize_context(context)
        rules = self._context_rules.get(normalized_context, self._context_rules["home"])

        relevant: List[AttentionItem] = []
        ignored: List[AttentionItem] = []

        for item in objects:
            name = str(item.get("name") or item.get("object_name") or "").strip().lower()
            priority_score = float(item.get("priority_score", 0.0) or 0.0)
            object_class = str(item.get("object_class") or "").strip().lower()

            relevance_score = self._compute_relevance_score(name, object_class, priority_score, rules)
            if relevance_score >= 0.6:
                relevant.append(
                    AttentionItem(
                        object_name=str(item.get("name") or item.get("object_name") or ""),
                        relevance_score=relevance_score,
                        reason=self._reason_for_score(name, object_class, rules),
                    )
                )
            else:
                ignored.append(
                    AttentionItem(
                        object_name=str(item.get("name") or item.get("object_name") or ""),
                        relevance_score=relevance_score,
                        reason=self._reason_for_score(name, object_class, rules),
                    )
                )

        relevant.sort(key=lambda item: item.relevance_score, reverse=True)
        ignored.sort(key=lambda item: item.relevance_score)

        return AttentionResult(
            context=normalized_context,
            relevant_objects=relevant,
            ignored_objects=ignored,
        )

    def _compute_relevance_score(
        self,
        name: str,
        object_class: str,
        priority_score: float,
        rules: Dict[str, set[str]],
    ) -> float:
        """Assign a relevance score based on context rules and priority."""
        score = 0.0
        if name in rules["keep"]:
            score += 0.45
        if name in rules["boost"]:
            score += 0.25
        if object_class in {"person", "vehicle", "utility", "furniture"}:
            score += 0.15
        if name in rules["suppress"]:
            score -= 0.35
        if priority_score >= 0.75:
            score += 0.15
        if priority_score >= 0.9:
            score += 0.10

        return min(1.0, max(0.0, score))

    def _reason_for_score(self, name: str, object_class: str, rules: Dict[str, set[str]]) -> str:
        """Explain why an object was kept or suppressed."""
        if name in rules["boost"]:
            return "context-critical-object"
        if name in rules["suppress"]:
            return "context-irrelevant-object"
        if object_class in {"person", "vehicle", "utility"}:
            return "context-semantic-category"
        return "default"

    def _normalize_context(self, context: str) -> str:
        """Normalize a requested context to the supported vocabulary."""
        normalized = (context or "home").strip().lower()
        if normalized in self._context_rules:
            return normalized
        return "home"
