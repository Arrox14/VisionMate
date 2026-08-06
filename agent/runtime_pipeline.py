from __future__ import annotations

"""Runtime orchestration for the VisionMate reasoning stack.

This module connects the perception and reasoning components into a single
assistive pipeline without replacing the existing YOLO and TTS functionality.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from database.person_memory import get_all_people, update_person_seen
from decision.multimodal_decision_layer import MultimodalDecisionLayer
from intelligence.lcre import LLMReasoningEngine
from reasoning.assistive_priority_engine import AssistivePriorityEngine
from reasoning.context_attention_engine import ContextAwareAttentionEngine
from reasoning.scene_reasoning import HierarchicalSceneReasoningModule


@dataclass(slots=True)
class RuntimeDecision:
    """Structured output produced by the runtime reasoning pipeline."""

    narration: str
    context_summary: str
    hazards: List[str] = field(default_factory=list)
    priority_objects: List[str] = field(default_factory=list)
    scene_context: str = "home"
    memory_summary: str = ""


class VisionMateRuntimePipeline:
    """Coordinate APE, CAE, HSRM, MDL, and LCRE at runtime."""

    def __init__(self) -> None:
        self.ape = AssistivePriorityEngine()
        self.cae = ContextAwareAttentionEngine()
        self.hsrm = HierarchicalSceneReasoningModule()
        self.mdl = MultimodalDecisionLayer()
        self.lcre = LLMReasoningEngine()

    def process_detections(
        self,
        detections: List[str],
        person_info: Optional[Dict[str, Any]] = None,
        scene_context: Optional[str] = None,
    ) -> RuntimeDecision:
        """Run the full reasoning chain for a set of YOLO-style detections."""
        if not detections:
            return RuntimeDecision(narration="Nothing significant detected.", context_summary="No objects visible.")

        objects = [
            {
                "name": label,
                "priority_score": 0.0,
                "object_class": self._infer_object_class(label),
            }
            for label in detections
        ]

        priority_result = self.ape.rank_detections(objects)
        context_name = self._infer_context(objects, scene_context)
        filtered_objects = self.cae.filter_objects(
            [
                {
                    "name": item.object_name,
                    "priority_score": item.priority_score,
                    "object_class": self._infer_object_class(item.object_name),
                }
                for item in priority_result.ranked_objects
            ],
            context=context_name,
        )

        scene_payload = [
            {
                "name": item.object_name,
                "priority_score": 0.0,
                "object_class": self._infer_object_class(item.object_name),
                "position": self._position_for_object(item.object_name, index),
            }
            for index, item in enumerate(filtered_objects.relevant_objects)
        ]
        scene_result = self.hsrm.reason(scene_payload)

        if "person" in {item.lower() for item in detections}:
            update_person_seen(
                person_name=str(person_info.get("name") or "detected_person"),
                confidence=float(person_info.get("confidence", 0.6) or 0.6),
                location=context_name,
                nearby_objects=detections,
                relationship="unknown",
                note="Runtime scene update",
            )

        people = get_all_people()
        memory_payload = {
            "encounter_count": len(people),
            "relationship": "memory_active",
            "people": [person.name for person in people],
        }

        context = self.mdl.fuse(
            person=person_info or {"name": "unknown", "confidence": 0.0},
            scene={"context": context_name, "summary": scene_result.summary},
            objects=[
                {"name": item["name"], "priority_score": 0.0, "object_class": item["object_class"]}
                for item in objects
            ],
            hazards=[{"description": hazard.description, "severity": hazard.severity} for hazard in scene_result.hazards],
            memory=memory_payload,
            priority={
                "ranked_objects": [
                    {
                        "object_name": item.object_name,
                        "priority_score": item.priority_score,
                        "level": item.level,
                    }
                    for item in priority_result.ranked_objects
                ]
            },
            recommendations=[
                guidance for guidance in [
                    self._recommendation_for_context(context_name),
                    self._recommendation_for_hazards(scene_result.hazards),
                ]
                if guidance
            ],
        )

        guidance = self.lcre.reason(context=self._serialize_context(context), mode="guidance")
        narration = guidance.personalized_guidance or guidance.answer or guidance.context_reasoning

        return RuntimeDecision(
            narration=narration,
            context_summary=scene_result.summary,
            hazards=[hazard.description for hazard in scene_result.hazards],
            priority_objects=[item.object_name for item in priority_result.ranked_objects[:3]],
            scene_context=context_name,
            memory_summary=memory_payload.get("people", ["none"])[0] if memory_payload.get("people") else "none",
        )

    def _infer_object_class(self, label: str) -> str:
        label_lower = label.lower()
        if label_lower in {"person", "teacher", "student"}:
            return "person"
        if label_lower in {"car", "truck", "motorcycle", "bike", "bicycle"}:
            return "vehicle"
        if label_lower in {"chair", "desk", "table", "sofa", "bed"}:
            return "furniture"
        if label_lower in {"door", "exit", "window"}:
            return "utility"
        if label_lower in {"laptop", "phone", "computer"}:
            return "electronic"
        return "object"

    def _infer_context(self, objects: List[Dict[str, Any]], scene_context: Optional[str]) -> str:
        if scene_context:
            return scene_context.lower()
        labels = {item.get("name", "").lower() for item in objects}
        if labels & {"teacher", "whiteboard", "student", "desk", "chair"}:
            return "classroom"
        if labels & {"car", "truck", "traffic light", "crossing", "pedestrian", "bike", "bicycle"}:
            return "road"
        if labels & {"patient", "doctor", "bed", "wheelchair", "nurse"}:
            return "hospital"
        if labels & {"experiment", "chemical", "equipment", "lab"}:
            return "laboratory"
        if labels & {"computer", "meeting", "desk", "door"}:
            return "office"
        return "home"

    def _position_for_object(self, name: str, index: int) -> tuple[float, float]:
        presets = {
            "chair": (0.2, 0.1),
            "door": (0.8, 0.1),
            "teacher": (0.4, 0.5),
            "whiteboard": (0.4, 0.1),
            "bottle": (0.3, 0.3),
        }
        return presets.get(name, (0.1 + 0.1 * index, 0.1 + 0.1 * index))

    def _recommendation_for_context(self, context_name: str) -> Optional[str]:
        if context_name == "road":
            return "Stay alert at crossings and near vehicles."
        if context_name == "classroom":
            return "Keep track of the teacher, the exit, and the main route."
        if context_name == "hospital":
            return "Be mindful of patient movement and accessible pathways."
        if context_name == "laboratory":
            return "Avoid chemical or equipment hazards and follow safe routes."
        return "Proceed carefully and monitor the surroundings."

    def _recommendation_for_hazards(self, hazards: List[Any]) -> Optional[str]:
        if not hazards:
            return None
        return "Avoid the identified hazards and keep a safe distance."

    def _serialize_context(self, context: Any) -> Dict[str, Any]:
        return {
            "person": {"name": context.person.name, "confidence": context.person.confidence},
            "scene": {"context": context.scene.context, "summary": context.scene.summary},
            "objects": [
                {"name": item.name, "priority_score": item.priority_score, "object_class": item.object_class}
                for item in context.objects
            ],
            "hazards": [{"description": item.description, "severity": item.severity} for item in context.hazards],
            "memory": context.memory,
            "priority": {"rankings": [{"object_name": item.object_name, "priority_score": item.priority_score, "level": item.level} for item in context.priority.rankings]},
            "recommendations": context.recommendations,
        }
