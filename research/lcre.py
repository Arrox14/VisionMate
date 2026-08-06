from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class GuidanceResult:
    """Structured natural-language guidance output from LCRE."""

    hazard_assessment: str
    context_reasoning: str
    answer: str
    personalized_guidance: str
    recommendations: List[str] = field(default_factory=list)
    narration: str = ""


class LLMReasoningEngine:
    """LCRE: generates offline-first guidance from a unified context object."""

    def __init__(self, local_model: Optional[Any] = None) -> None:
        self.local_model = local_model
        self._prompts = {
            "hazard": "Assess hazards in the current scene and mention the highest-risk items.",
            "context": "Explain the situation in short, clear assistive language.",
            "answer": "Answer the user's likely question using the observed scene context.",
            "guidance": "Provide personalized and safe guidance for the user.",
        }

    def reason(self, unified_context: Dict[str, Any]) -> GuidanceResult:
        scene = unified_context.get("scene", "home")
        objects = unified_context.get("detected_objects", [])
        people = unified_context.get("recognized_people", [])
        hazards = unified_context.get("hazards", [])
        interactions = unified_context.get("interactions", [])
        priorities = unified_context.get("priorities", [])

        hazard_assessment = self._compose_hazard_assessment(hazards, priorities)
        context_reasoning = self._compose_context_reasoning(scene, objects, people, interactions)
        answer = self._compose_answer(scene, objects, people)
        personalized_guidance = self._compose_personalized_guidance(scene, objects, people, hazards)
        recommendations = self._compose_recommendations(hazards, interactions, priorities)
        narration = self._compose_narration(personalized_guidance, recommendations)

        return GuidanceResult(
            hazard_assessment=hazard_assessment,
            context_reasoning=context_reasoning,
            answer=answer,
            personalized_guidance=personalized_guidance,
            recommendations=recommendations,
            narration=narration,
        )

    def use_local_model(self, model: Any) -> None:
        """Attach an optional local model for future offline inference."""
        self.local_model = model

    def _compose_hazard_assessment(self, hazards: List[str], priorities: List[Dict[str, Any]]) -> str:
        if hazards:
            return f"Hazard assessment: {', '.join(hazards)}."
        if priorities:
            top = priorities[0]
            return f"No immediate hazard detected, but {top.get('object_name', 'the scene')} is the highest-priority object."
        return "No immediate hazard detected."

    def _compose_context_reasoning(self, scene: str, objects: List[str], people: List[str], interactions: List[str]) -> str:
        object_summary = ", ".join(objects) if objects else "nothing significant"
        people_summary = ", ".join(people) if people else "no recognized people"
        interaction_summary = ", ".join(interactions) if interactions else "no notable interactions"
        return (
            f"In {scene}, the system observes {object_summary} and recognizes {people_summary}. "
            f"The current interaction pattern is {interaction_summary}."
        )

    def _compose_answer(self, scene: str, objects: List[str], people: List[str]) -> str:
        if people:
            return f"The user is with {', '.join(people)} in {scene}."
        if objects:
            return f"The current scene in {scene} contains {', '.join(objects)}."
        return f"The scene in {scene} is currently sparse."

    def _compose_personalized_guidance(self, scene: str, objects: List[str], people: List[str], hazards: List[str]) -> str:
        if hazards:
            return f"Please stay alert and avoid the hazardous elements visible in {scene}."
        if people:
            return f"Continue observing the environment while staying aware of {', '.join(people)} in {scene}."
        if objects:
            return f"The main items to notice are {', '.join(objects)}."
        return "Continue monitoring the environment."

    def _compose_recommendations(self, hazards: List[str], interactions: List[str], priorities: List[Dict[str, Any]]) -> List[str]:
        recommendations: List[str] = []
        if hazards:
            recommendations.append("Avoid hazardous objects and keep a safe distance.")
        if interactions:
            recommendations.append("Continue monitoring social or functional interactions.")
        if priorities:
            top = priorities[0]
            recommendations.append(f"Prioritize {top.get('object_name', 'the scene')} because it has the highest urgency.")
        if not recommendations:
            recommendations.append("Continue monitoring the scene for changes.")
        return recommendations

    def _compose_narration(self, guidance: str, recommendations: List[str]) -> str:
        return f"{guidance} {' '.join(recommendations)}"
