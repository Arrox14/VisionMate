from __future__ import annotations

"""LLM-Based Contextual Reasoning Engine (LCRE).

This module is designed as an offline-first reasoning interface for a unified
context object. It does not hardcode task-specific prompts in the reasoning
logic; instead, it uses a prompt builder and a response parser so it can later
support local models, Ollama, or future multimodal LLM backends.
"""

from dataclasses import dataclass, field
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class GuidanceResult:
    """Structured natural-language guidance produced by the engine."""

    hazard_assessment: str
    context_reasoning: str
    answer: str
    personalized_guidance: str
    recommendations: List[str] = field(default_factory=list)
    narration: str = ""


class PromptBuilder:
    """Build structured prompts without embedding task text directly in the engine."""

    def build(self, context: Dict[str, Any], question: Optional[str] = None, mode: str = "guidance") -> str:
        """Create a prompt from a unified context object and an optional question."""
        person = context.get("person", {})
        scene = context.get("scene", {})
        objects = context.get("objects", [])
        hazards = context.get("hazards", [])
        memory = context.get("memory", {})
        priority = context.get("priority", {})
        recommendations = context.get("recommendations", [])
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        prompt_parts = [
            f"Task: {mode}",
            "You are an assistive reasoning engine for an autonomous vision agent.",
            f"Person: {person.get('name', 'unknown')} ({person.get('confidence', 0.0):.2f})",
            f"Current Scene: {scene.get('context', 'unknown')} - {scene.get('summary', '')}",
            f"Object List: {', '.join(self._stringify_objects(objects)) if objects else 'none'}",
            f"Priority Ranking: {self._stringify_priority(priority)}",
            f"Hazards: {', '.join(self._stringify_hazards(hazards)) if hazards else 'none'}",
            f"Memory Summary: {self._stringify_memory(memory)}",
            f"Recommendations: {', '.join(recommendations) if recommendations else 'none'}",
            f"Current Time: {timestamp}",
            "Conversation History: future support",
            "Return a concise result with sections Hazard, Context, Answer, Guidance, Recommendation, and Narration.",
            "The Narration should be 2-3 sentences, conversational, and sound like a professional accessibility assistant.",
            "Mention the person's name when known, mention hazards first, mention only the most important object, and avoid repetitive wording.",
        ]
        if question:
            prompt_parts.append(f"Question: {question}")
        return "\n".join(prompt_parts)

    def _stringify_objects(self, objects: List[Dict[str, Any]]) -> List[str]:
        return [f"{item.get('name', 'object')}({item.get('priority_score', 0.0):.2f})" for item in objects]

    def _stringify_hazards(self, hazards: List[Dict[str, Any]]) -> List[str]:
        return [f"{item.get('description', 'hazard')}[{item.get('severity', 'medium')}]" for item in hazards]

    def _stringify_memory(self, memory: Dict[str, Any]) -> str:
        if not memory:
            return "none"
        return ", ".join(f"{key}={value}" for key, value in memory.items())

    def _stringify_priority(self, priority: Dict[str, Any]) -> str:
        rankings = priority.get("rankings", [])
        if not rankings:
            return "none"
        return ", ".join(f"{item.get('object_name', 'object')}:{item.get('level', 'Low')}" for item in rankings)


class ResponseParser:
    """Parse a raw model response into structured guidance fields."""

    def parse(self, raw_response: str) -> Dict[str, Any]:
        """Convert the raw text into structured fields for downstream use."""
        lines = [line.strip() for line in raw_response.splitlines() if line.strip()]
        if not lines:
            return {
                "hazard_assessment": "No immediate hazard detected.",
                "context_reasoning": "The scene appears manageable.",
                "answer": "Continue monitoring the environment.",
                "personalized_guidance": "Stay alert and proceed safely.",
                "recommendations": ["Continue monitoring the scene."],
            }

        return {
            "hazard_assessment": self._extract_section(lines, "Hazard") or lines[0],
            "context_reasoning": self._extract_section(lines, "Context") or (lines[1] if len(lines) > 1 else "The scene appears manageable."),
            "answer": self._extract_section(lines, "Answer") or (lines[2] if len(lines) > 2 else "Continue monitoring the environment."),
            "personalized_guidance": self._extract_section(lines, "Guidance") or (lines[3] if len(lines) > 3 else "Stay alert and proceed safely."),
            "recommendations": self._extract_recommendations(lines),
            "narration": self._extract_section(lines, "Narration") or self._extract_section(lines, "Narrative"),
        }

    def _extract_section(self, lines: List[str], prefix: str) -> Optional[str]:
        for line in lines:
            if line.lower().startswith(prefix.lower()):
                return line.split(":", 1)[1].strip() if ":" in line else line
        return None

    def _extract_recommendations(self, lines: List[str]) -> List[str]:
        recommendations = []
        for line in lines:
            if line.lower().startswith(("recommendation", "recommendations")):
                value = line.split(":", 1)[1].strip() if ":" in line else line
                recommendations.append(value)
        return recommendations or ["Continue monitoring the scene."]


class LLMReasoningEngine:
    """Offline-first reasoning engine with prompt construction and response parsing."""

    def __init__(self, backend: Optional[Any] = None) -> None:
        self.backend = backend
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()
        self._last_signature: Optional[str] = None
        self._last_narration: Optional[str] = None

    def reason(self, context: Dict[str, Any], question: Optional[str] = None, mode: str = "guidance") -> GuidanceResult:
        """Create a prompt, invoke the configured backend if available, and parse the output."""
        prompt = self.prompt_builder.build(context, question=question, mode=mode)
        raw_response = self._invoke_backend(prompt)
        parsed = self.response_parser.parse(raw_response)
        scene_signature = self._scene_signature(context)

        narration = self._select_narration(context, parsed, scene_signature)
        self._last_signature = scene_signature
        self._last_narration = narration

        return GuidanceResult(
            hazard_assessment=parsed.get("hazard_assessment", "No immediate hazard detected."),
            context_reasoning=parsed.get("context_reasoning", "The scene appears manageable."),
            answer=parsed.get("answer", "Continue monitoring the environment."),
            personalized_guidance=parsed.get("personalized_guidance", "Stay alert and proceed safely."),
            recommendations=parsed.get("recommendations", ["Continue monitoring the scene."]),
            narration=narration,
        )

    def use_backend(self, backend: Any) -> None:
        """Attach a future local or remote model backend."""
        self.backend = backend

    def _invoke_backend(self, prompt: str) -> str:
        """Invoke a local backend if present; otherwise generate fallback guidance from the prompt."""
        if self.backend is not None:
            return self.backend(prompt)

        return self._fallback_reason_from_prompt(prompt)

    def _select_narration(self, context: Dict[str, Any], parsed: Dict[str, Any], scene_signature: Optional[str]) -> str:
        if parsed.get("narration"):
            narration = parsed.get("narration")
        else:
            narration = self._compose_refined_narration(context, parsed)

        if self._last_signature == scene_signature and self._last_narration:
            return self._follow_up_narration(context, narration)

        return narration

    def _compose_refined_narration(self, context: Dict[str, Any], parsed: Dict[str, Any]) -> str:
        person_name = self._extract_person_name(context)
        scene_summary = self._extract_scene_summary(context)
        hazard_text = self._normalize_phrase(parsed.get("hazard_assessment", "No immediate hazard detected."))
        guidance = self._normalize_phrase(parsed.get("personalized_guidance", "Stay alert and proceed safely."))
        recommendation = parsed.get("recommendations", ["Continue monitoring the scene."])
        recommendation_text = self._normalize_phrase(recommendation[0] if recommendation else "Continue monitoring the scene.")
        top_object = self._normalize_phrase(self._extract_top_object(context))
        memory_hint = self._extract_memory_hint(context)

        sentences: List[str] = []
        if person_name and hazard_text:
            sentences.append(f"{person_name}, {hazard_text}.")
        elif hazard_text:
            sentences.append(f"{hazard_text}.")

        scene_parts: List[str] = []
        if scene_summary:
            scene_parts.append(f"We’re in the {scene_summary}.")
        if top_object:
            scene_parts.append(f"The main thing to notice is {top_object}.")
        if scene_parts:
            sentences.append(" ".join(scene_parts))

        detail_parts: List[str] = []
        if memory_hint:
            detail_parts.append(memory_hint)
        if guidance:
            detail_parts.append(guidance)
        if recommendation_text:
            detail_parts.append(self._format_recommendation(recommendation_text))
        if detail_parts:
            sentences.append(" ".join(detail_parts))

        if len(sentences) < 2:
            sentences.append("Please continue carefully.")
        if len(sentences) > 3:
            sentences = sentences[:3]
        return " ".join(sentence.strip().rstrip(".") + "." for sentence in sentences)

    def _split_sentences(self, narration: str) -> List[str]:
        cleaned = narration.replace("\n", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        parts = []
        for chunk in re.split(r"(?<=[.!?])\s+", cleaned):
            sentence = chunk.strip()
            if sentence:
                parts.append(sentence)
        return parts

    def _follow_up_narration(self, context: Dict[str, Any], previous_narration: str) -> str:
        person_name = self._extract_person_name(context)
        if person_name:
            return f"I’m still monitoring the scene for you, {person_name}. Please continue carefully."
        return "I’m still monitoring the scene for you. Please continue carefully."

    def _fallback_reason_from_prompt(self, prompt: str) -> str:
        """Create a concise fallback response with a polished narration section."""
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        context_details = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                context_details[key.strip().lower()] = value.strip()

        person_name = self._normalize_phrase(context_details.get("person", "unknown"))
        scene = self._normalize_phrase(context_details.get("current scene", "unknown"))
        object_list = self._normalize_phrase(context_details.get("object list", "none"))
        priority = self._normalize_phrase(context_details.get("priority ranking", "none"))
        hazards = self._normalize_phrase(context_details.get("hazards", "none"))
        memory = self._normalize_phrase(context_details.get("memory summary", "none"))
        recommendations = self._normalize_phrase(context_details.get("recommendations", "none"))

        person_name = person_name.split("(", 1)[0].strip() if person_name else ""
        scene_summary = scene.split("-", 1)[0].strip() if scene and scene != "unknown" else ""
        object_items = [item.strip() for item in object_list.split(",") if item.strip() and item.strip() != "none"]
        priority_items = [item.strip() for item in priority.split(",") if item.strip() and item.strip() != "none"]
        hazard_items = [item.strip() for item in hazards.split(",") if item.strip() and item.strip() != "none"]
        recommendation_items = [item.strip() for item in recommendations.split(",") if item.strip() and item.strip() != "none"]

        hazard_text = self._normalize_phrase(hazard_items[0]) if hazard_items else "No immediate hazard detected"
        top_object = self._normalize_phrase(object_items[0]) if object_items else self._normalize_phrase(priority_items[0]) if priority_items else ""
        top_priority = self._normalize_phrase(priority_items[0]) if priority_items else ""
        guidance_text = f"Please {recommendation_items[0].lower()}" if recommendation_items else "Please proceed carefully."
        memory_text = "I remember this is a familiar setting." if memory and memory != "none" else ""

        if person_name:
            lead = f"{person_name},"
        else:
            lead = ""

        narration = []
        if hazard_text:
            narration.append(f"{lead} {hazard_text}.".strip())
        if scene_summary:
            narration.append(f"We’re in the {scene_summary}.")
        if top_object or top_priority:
            narration.append(f"The main thing to notice is {top_object or top_priority}.")
        if memory_text:
            narration.append(memory_text)
        if guidance_text:
            narration.append(guidance_text + ".")

        if len(narration) > 4:
            narration = narration[:4]

        return "\n".join(
            [
                f"Hazard: {hazard_text}",
                f"Context: {' '.join(narration[:2]) if len(narration) > 1 else 'The scene is being monitored.'}",
                f"Answer: {' '.join(narration[:2]) if len(narration) > 1 else 'The scene is being monitored.'}",
                f"Guidance: {' '.join(narration[2:4]) if len(narration) > 2 else 'Please proceed carefully.'}",
                f"Recommendation: {'; '.join(recommendation_items) if recommendation_items else 'Continue monitoring the scene.'}",
                f"Narration: {' '.join(narration)}",
            ]
        )

    def _scene_signature(self, context: Dict[str, Any]) -> Optional[str]:
        person = context.get("person", {})
        scene = context.get("scene", {})
        objects = context.get("objects", [])
        hazards = context.get("hazards", [])
        priority = context.get("priority", {})
        return "|".join(
            [
                str(person.get("name", "")),
                str(scene.get("context", "")),
                str(scene.get("summary", "")),
                str(hazards[0].get("description", "") if hazards else ""),
                str(objects[0].get("name", "") if objects else ""),
                str(priority.get("rankings", [{}])[0].get("object_name", "") if priority.get("rankings") else ""),
            ]
        )

    def _extract_person_name(self, context: Dict[str, Any]) -> str:
        person = context.get("person", {})
        name = person.get("name") if isinstance(person, dict) else None
        return self._clean_value(name)

    def _extract_scene_summary(self, context: Dict[str, Any]) -> str:
        scene = context.get("scene", {})
        if isinstance(scene, dict):
            summary = scene.get("summary") or scene.get("context")
            return self._clean_value(summary)
        return self._clean_value(scene)

    def _extract_top_object(self, context: Dict[str, Any]) -> str:
        objects = context.get("objects", [])
        if objects and isinstance(objects[0], dict):
            return self._clean_value(objects[0].get("name"))
        return ""

    def _extract_memory_hint(self, context: Dict[str, Any]) -> str:
        memory = context.get("memory", {})
        if not memory:
            return ""
        if isinstance(memory, dict) and any(str(value).lower() in {"friend", "family", "known", "familiar"} for value in memory.values()):
            return "I remember this is a familiar setting."
        return ""

    def _clean_sentence(self, value: str) -> str:
        cleaned = self._normalize_phrase(value)
        if not cleaned:
            return ""
        if cleaned.endswith("."):
            return cleaned
        return f"{cleaned}."

    def _format_recommendation(self, value: str) -> str:
        cleaned = self._normalize_phrase(value)
        if not cleaned:
            return ""
        if cleaned.lower().startswith(("please", "i recommend", "you should")):
            return cleaned
        return f"Please {cleaned.lower()}."

    def _normalize_phrase(self, value: Any) -> str:
        text = self._clean_value(value)
        if not text:
            return ""
        text = re.sub(r"\[[^\]]*\]", "", text)
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text.startswith("Hazard:"):
            text = text[len("Hazard:"):].strip()
        if text.lower().startswith("context:"):
            text = text[len("Context:"):].strip()
        if text.lower().startswith("answer:"):
            text = text[len("Answer:"):].strip()
        if text.lower().startswith("guidance:"):
            text = text[len("Guidance:"):].strip()
        if text.lower().startswith("recommendation:"):
            text = text[len("Recommendation:"):].strip()
        if text.lower().startswith("narration:"):
            text = text[len("Narration:"):].strip()
        return text.rstrip(".")

    def _clean_value(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        return text.replace("\n", " ").replace("\t", " ")
