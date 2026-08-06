from __future__ import annotations

from pathlib import Path

from database.person_memory import (
    PersonMemoryStore,
    create_person,
    get_all_people,
    get_average_confidence,
    get_person,
    get_person_seen_count,
    load_memory,
    save_memory,
    update_location,
    update_nearby_objects,
    update_person,
    update_person_seen,
)


def test_mpam_supports_rich_person_memory_and_backward_compatibility(tmp_path: Path) -> None:
    store = PersonMemoryStore(db_path=tmp_path / "person_memory.json")
    profile = store.create_person(name="Alice", person_id="p001", relationship="friend")
    assert profile.person_id == "p001"

    store.update_person_seen("Alice", confidence=0.93, location="kitchen", nearby_objects=["cup"], note="Met in kitchen")
    store.update_location("Alice", "office")
    store.update_nearby_objects("Alice", ["book", "cup"])

    saved = store.get_person("Alice")
    assert saved is not None
    assert saved.encounter_count == 1
    assert saved.average_confidence == 0.93
    assert "kitchen" in saved.locations_history
    assert "office" in saved.locations_history
    assert "cup" in saved.nearby_objects_history
    assert saved.relationship == "friend"

    assert get_person_seen_count("Alice") == 1
    assert get_average_confidence("Alice") == 0.93

    module_profile = update_person("Alice", relationship="colleague")
    assert module_profile.relationship == "colleague"

    all_people = get_all_people()
    assert len(all_people) >= 1

    save_memory(load_memory())
