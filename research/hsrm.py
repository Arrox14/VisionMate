from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class SceneNode:
    """Represents a detected object or person in the scene graph."""

    name: str
    node_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SceneEdge:
    """Represents a relationship between two scene nodes."""

    source: str
    target: str
    relation: str
    description: str


@dataclass(slots=True)
class SceneGraph:
    """Structured semantic scene understanding produced by HSRM."""

    scene_type: str
    nodes: List[SceneNode]
    edges: List[SceneEdge]
    hazards: List[str]
    interactions: List[str]
    summary: str


class HierarchicalSceneRepresentationModule:
    """HSRM: builds a semantic scene graph from prioritized objects."""

    def __init__(self) -> None:
        self._scene_graph: Optional[SceneGraph] = None

    def observe(self, objects: Optional[List[str]] = None, people: Optional[List[str]] = None, scene_type: Optional[str] = None) -> SceneGraph:
        object_names = [obj for obj in (objects or []) if obj]
        person_names = [person for person in (people or []) if person]
        inferred_scene = scene_type or self._infer_scene_type(object_names, person_names)

        nodes = [SceneNode(name=obj, node_type="object") for obj in object_names]
        nodes.extend(SceneNode(name=person, node_type="person") for person in person_names)

        edges = self._build_edges(object_names, person_names)
        hazards = self._detect_hazards(object_names)
        interactions = self._detect_interactions(object_names, person_names)
        summary = self._summarize(object_names, person_names, hazards, interactions)

        self._scene_graph = SceneGraph(
            scene_type=inferred_scene,
            nodes=nodes,
            edges=edges,
            hazards=hazards,
            interactions=interactions,
            summary=summary,
        )
        return self._scene_graph

    def get_current_representation(self) -> Optional[SceneGraph]:
        return self._scene_graph

    def _infer_scene_type(self, objects: List[str], people: List[str]) -> str:
        if people:
            return "social"
        if any(obj in {"chair", "book", "laptop"} for obj in objects):
            return "indoor"
        if any(obj in {"car", "truck", "motorcycle", "bicycle"} for obj in objects):
            return "mobile"
        return "general"

    def _build_edges(self, objects: List[str], people: List[str]) -> List[SceneEdge]:
        edges: List[SceneEdge] = []
        if people and objects:
            for person in people:
                for obj in objects:
                    relation = "near"
                    description = f"{person} is near {obj}"
                    edges.append(SceneEdge(source=person, target=obj, relation=relation, description=description))

        if "person" in objects and "chair" in objects:
            edges.append(SceneEdge(source="person", target="chair", relation="sits_on", description="person may be sitting on chair"))
        if "person" in objects and "book" in objects:
            edges.append(SceneEdge(source="person", target="book", relation="holds", description="person may be holding a book"))
        if "person" in objects and "car" in objects:
            edges.append(SceneEdge(source="person", target="car", relation="near", description="person is near a vehicle"))
        return edges

    def _detect_hazards(self, objects: List[str]) -> List[str]:
        hazards: List[str] = []
        if "car" in objects or "truck" in objects or "motorcycle" in objects:
            hazards.append("vehicle nearby")
        if "knife" in objects or "fire" in objects:
            hazards.append("sharp or hazardous object")
        if "dog" in objects:
            hazards.append("animal nearby")
        return hazards

    def _detect_interactions(self, objects: List[str], people: List[str]) -> List[str]:
        interactions: List[str] = []
        if people and "chair" in objects:
            interactions.append("social seating interaction")
        if people and "book" in objects:
            interactions.append("reading interaction")
        if people and "laptop" in objects:
            interactions.append("working interaction")
        return interactions

    def _summarize(self, objects: List[str], people: List[str], hazards: List[str], interactions: List[str]) -> str:
        object_summary = ", ".join(objects) if objects else "no objects"
        person_summary = ", ".join(people) if people else "no recognized people"
        hazard_summary = ", ".join(hazards) if hazards else "no notable hazards"
        interaction_summary = ", ".join(interactions) if interactions else "no notable interactions"
        return f"Scene includes {object_summary}; persons: {person_summary}; hazards: {hazard_summary}; interactions: {interaction_summary}."
