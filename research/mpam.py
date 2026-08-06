from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class PerceptionSnapshot:
    """Normalized snapshot of perceived objects and context."""

    objects: List[str] = field(default_factory=list)
    recognized_people: List[str] = field(default_factory=list)
    memory_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryAugmentedPerceptionModule:
    """MPAM: organizes raw detections into a structured perception snapshot."""

    def __init__(self) -> None:
        self._history: List[PerceptionSnapshot] = []

    def update(
        self,
        objects: Optional[List[str]] = None,
        recognized_people: Optional[List[str]] = None,
        memory_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerceptionSnapshot:
        normalized_objects = sorted({obj for obj in (objects or []) if obj})
        normalized_people = [person for person in (recognized_people or []) if person]
        snapshot = PerceptionSnapshot(
            objects=normalized_objects,
            recognized_people=normalized_people,
            memory_context=memory_context or {},
            metadata=metadata or {},
        )
        self._history.append(snapshot)
        return snapshot

    def history(self) -> List[PerceptionSnapshot]:
        return list(self._history)
