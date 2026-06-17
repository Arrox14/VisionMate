from __future__ import annotations

from pathlib import Path
import json
from typing import Dict

DB_PATH = Path(__file__).with_name("person_memory.json")


def _load_memory() -> Dict[str, Dict[str, object]]:
    if not DB_PATH.exists():
        return {}
    try:
        with DB_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_memory(data: Dict[str, Dict[str, object]]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DB_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def get_person_seen_count(person_name: str) -> int:
    data = _load_memory()
    person = data.get(person_name)
    if isinstance(person, dict):
        count = person.get("seen_count")
        if isinstance(count, int):
            return count
    return 0


def update_person_seen(person_name: str) -> int:
    data = _load_memory()
    person = data.get(person_name)
    if not isinstance(person, dict):
        person = {}

    count = person.get("seen_count")
    if not isinstance(count, int):
        count = 0

    count += 1
    person["seen_count"] = count
    data[person_name] = person
    _save_memory(data)
    return count
