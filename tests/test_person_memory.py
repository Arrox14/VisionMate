from __future__ import annotations

import json
from pathlib import Path

from database.person_memory import (
    PersonMemoryStore,
    get_person_seen_count,
    get_person_profile,
    update_person_seen,
)


def test_person_memory_records_rich_profile_and_preserves_helpers(tmp_path: Path) -> None:
    store = PersonMemoryStore(db_path=tmp_path / "person_memory.json")

    profile = store.record_encounter(
        name="Alice",
        confidence=0.93,
        location="kitchen",
        nearby_objects=["cup", "bottle"],
        relationship="friend",
        note="Met in the kitchen",
    )

    assert profile.person_id == "Alice"
    assert profile.name == "Alice"
    assert profile.encounter_count == 1
    assert profile.locations == ["kitchen"]
    assert profile.nearby_objects == ["cup", "bottle"]
    assert profile.relationship == "friend"
    assert profile.notes == ["Met in the kitchen"]
    assert profile.confidence_history[0]["confidence"] == 0.93

    reloaded = PersonMemoryStore(db_path=tmp_path / "person_memory.json")
    persisted = reloaded.get_person_profile("Alice")
    assert persisted is not None
    assert persisted.encounter_count == 1
    assert get_person_seen_count("Alice") == 1


def test_person_memory_migrates_legacy_seen_count_format(tmp_path: Path) -> None:
    db_path = tmp_path / "person_memory.json"
    db_path.write_text(json.dumps({"Alice": {"seen_count": 2}}), encoding="utf-8")

    store = PersonMemoryStore(db_path=db_path)
    profile = store.get_person_profile("Alice")

    assert profile is not None
    assert profile.encounter_count == 2
    assert profile.name == "Alice"

    updated_count = update_person_seen("Alice", confidence=0.81, location="office")
    assert updated_count == 3
    assert get_person_profile("Alice") is not None
