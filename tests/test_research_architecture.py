from __future__ import annotations

from research.ape import AdaptivePerceptionEngine
from research.cae import ContextAggregationEngine
from research.evaluation import EvaluationTracker
from research.hsrm import HierarchicalSceneRepresentationModule
from research.lcre import LanguageConditionedReasoningEngine
from research.mdl import MemoryDrivenLanguageModule
from research.mpam import MemoryAugmentedPerceptionModule


def test_research_modules_produce_context_and_reasoning() -> None:
    mpam = MemoryAugmentedPerceptionModule()
    snapshot = mpam.update(
        objects=["person", "chair", "bottle"],
        recognized_people=["Alice"],
        memory_context={"Alice": {"seen_count": 2}},
    )

    assert snapshot.objects == ["bottle", "chair", "person"]
    assert snapshot.recognized_people == ["Alice"]

    ape = AdaptivePerceptionEngine()
    filtered = ape.filter(snapshot)
    assert filtered["person"] == 1.0
    assert filtered["chair"] >= 0.6

    cae = ContextAggregationEngine()
    context = cae.build_context(snapshot, filtered)
    assert "Alice" in context["summary"]

    hsrm = HierarchicalSceneRepresentationModule()
    hsrm.observe(snapshot)
    representation = hsrm.get_current_representation()
    assert representation["scene_type"] == "social"

    mdl = MemoryDrivenLanguageModule()
    narrative = mdl.render(snapshot, context)
    assert "Alice" in narrative

    lcre = LanguageConditionedReasoningEngine()
    suggestion = lcre.reason(snapshot, context)
    assert "assist" in suggestion.lower() or "observe" in suggestion.lower()

    evaluation = EvaluationTracker()
    evaluation.record(snapshot, context, narrative, suggestion)
    metrics = evaluation.snapshot()
    assert metrics["scene_count"] == 1
    assert metrics["objects_detected"] >= 3
