from __future__ import annotations

"""Multimodal Decision Layer (MDL).

This module performs information fusion only. It does not generate narration.
It aggregates outputs from perception and reasoning modules into a single,
structured context object that can be consumed by later narration, guidance,
or planning layers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class PersonContext:
    """Person-level context extracted from face recognition and memory."""

    name: str = "unknown"
    confidence: float = 0.0
    identity: Optional[str] = None


@dataclass(slots=True)
class SceneContext:
    """Scene-level context extracted from the environment."""

    context: str = "unknown"
    summary: str = ""


@dataclass(slots=True)
class ObjectContext:
    """A single object entry in the unified context."""

    name: str
    priority_score: float = 0.0
    object_class: str = "object"


@dataclass(slots=True)
class HazardContext:
    """Hazard information fused from HSRM and other modules."""

    description: str
    severity: str = "medium"


@dataclass(slots=True)
class PriorityRanking:
    """A single ranked object entry for the decision layer."""

    object_name: str
    priority_score: float = 0.0
    level: str = "Low"


@dataclass(slots=True)
class PriorityContext:
    """Priority information from APE."""

    rankings: List[PriorityRanking] = field(default_factory=list)


@dataclass(slots=True)
class UnifiedContext:
    """The final fused decision object for downstream use."""

    person: PersonContext
    scene: SceneContext
    objects: List[ObjectContext]
    hazards: List[HazardContext]
    memory: Dict[str, Any]
    priority: PriorityContext
    recommendations: List[str] = field(default_factory=list)


class MultimodalDecisionLayer:
    """Fuse outputs from YOLO, InsightFace, MPAM, APE, CAE, and HSRM."""

    def fuse(
        self,
        person: Optional[Dict[str, Any]] = None,
        scene: Optional[Dict[str, Any]] = None,
        objects: Optional[List[Dict[str, Any]]] = None,
        hazards: Optional[List[Dict[str, Any]]] = None,
        memory: Optional[Dict[str, Any]] = None,
        priority: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[str]] = None,
    ) -> UnifiedContext:
        """Merge heterogeneous inputs into one structured context object."""
        person_payload = person or {}
        scene_payload = scene or {}
        objects_payload = objects or []
        hazards_payload = hazards or []
        memory_payload = memory or {}
        priority_payload = priority or {}
        recommendations_payload = recommendations or []

        person_context = PersonContext(
            name=str(person_payload.get("name") or "unknown"),
            confidence=float(person_payload.get("confidence", 0.0) or 0.0),
            identity=str(person_payload.get("identity") or person_payload.get("name") or "unknown"),
        )

        scene_context = SceneContext(
            context=str(scene_payload.get("context") or "unknown"),
            summary=str(scene_payload.get("summary") or ""),
        )

        object_contexts = [
            ObjectContext(
                name=str(item.get("name") or item.get("object_name") or "object"),
                priority_score=float(item.get("priority_score", 0.0) or 0.0),
                object_class=str(item.get("object_class") or "object"),
            )
            for item in objects_payload
        ]

        hazard_contexts = [
            HazardContext(
                description=str(item.get("description") or "hazard detected"),
                severity=str(item.get("severity") or "medium"),
            )
            for item in hazards_payload
        ]

        priority_rankings: List[PriorityRanking] = []
        if isinstance(priority_payload.get("ranked_objects"), list):
            priority_rankings = [
                PriorityRanking(
                    object_name=str(item.get("object_name") or item.get("name") or "object"),
                    priority_score=float(item.get("priority_score", 0.0) or 0.0),
                    level=str(item.get("level") or "Low"),
                )
                for item in priority_payload["ranked_objects"]
            ]
        elif isinstance(priority_payload.get("rankings"), list):
            priority_rankings = [
                PriorityRanking(
                    object_name=str(item.get("object_name") or item.get("name") or "object"),
                    priority_score=float(item.get("priority_score", 0.0) or 0.0),
                    level=str(item.get("level") or "Low"),
                )
                for item in priority_payload["rankings"]
            ]

        priority_context = PriorityContext(rankings=priority_rankings)

        return UnifiedContext(
            person=person_context,
            scene=scene_context,
            objects=object_contexts,
            hazards=hazard_contexts,
            memory=memory_payload,
            priority=priority_context,
            recommendations=recommendations_payload,
        )
