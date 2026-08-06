from __future__ import annotations

from typing import Any, Dict, List, Optional

from research.ape import AdaptivePerceptionEngine
from research.cae import ContextAggregationEngine
from research.evaluation import EvaluationTracker
from research.hsrm import HierarchicalSceneRepresentationModule
from research.lcre import LanguageConditionedReasoningEngine
from research.mdl import MemoryDrivenLanguageModule
from research.mpam import MemoryAugmentedPerceptionModule
from voice.text_to_speech import speak


class VisionMateResearchOrchestrator:
    """Coordinates the full research architecture for perception and reasoning."""

    def __init__(self) -> None:
        self.mpam = MemoryAugmentedPerceptionModule()
        self.ape = AdaptivePerceptionEngine()
        self.cae = ContextAggregationEngine()
        self.hsrm = HierarchicalSceneRepresentationModule()
        self.mdl = MemoryDrivenLanguageModule()
        self.lcre = LanguageConditionedReasoningEngine()
        self.evaluation = EvaluationTracker()

    def handle_detection(
        self,
        objects: Optional[List[str]] = None,
        recognized_people: Optional[List[str]] = None,
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        snapshot = self.mpam.update(
            objects=list(objects or []),
            recognized_people=list(recognized_people or []),
            memory_context=memory_context or {},
        )
        salience = self.ape.filter(snapshot)
        context = self.cae.build_context(snapshot, salience)
        self.hsrm.observe(snapshot)
        narrative = self.mdl.render(snapshot, context)
        suggestion = self.lcre.reason(snapshot, context)
        self.evaluation.record(snapshot, context, narrative, suggestion)

        print(f"[Research] {narrative}")
        print(f"[Research] {suggestion}")

        try:
            speak(narrative)
        except Exception as exc:  # pragma: no cover - defensive runtime logging
            print(f"Research speech warning: {exc}")

        return {
            "snapshot": snapshot,
            "salience": salience,
            "context": context,
            "narrative": narrative,
            "suggestion": suggestion,
        }
