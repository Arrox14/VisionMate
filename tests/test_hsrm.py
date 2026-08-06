from __future__ import annotations

from research.hsrm import HierarchicalSceneRepresentationModule


def test_hsrm_builds_scene_graph() -> None:
    engine = HierarchicalSceneRepresentationModule()
    graph = engine.observe(objects=["person", "chair", "car"], people=["Alice"], scene_type="social")

    assert graph.scene_type == "social"
    assert any(node.name == "person" for node in graph.nodes)
    assert any(edge.relation == "near" for edge in graph.edges)
    assert "vehicle nearby" in graph.hazards
    assert any("social" in interaction for interaction in graph.interactions)
