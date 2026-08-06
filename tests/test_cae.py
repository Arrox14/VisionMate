from __future__ import annotations

from research.cae import ContextAwareAttentionEngine


def test_cae_filters_scene_information_by_environment() -> None:
    engine = ContextAwareAttentionEngine()

    result = engine.filter_scene(
        objects=["person", "book", "car", "cup", "dog"],
        faces=["Alice"],
        scene="classroom",
    )

    assert result.scene == "classroom"
    assert any(item.name == "person" for item in result.filtered_items)
    assert any(item.name == "book" for item in result.filtered_items)
    assert "car" in result.ignored_items
    assert "dog" in result.ignored_items
