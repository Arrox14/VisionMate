from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).with_name("person_memory.json")
_ACTIVE_DB_PATH = DB_PATH


@dataclass(slots=True)
class PersonProfile:
    """Persistent profile for a recognized person with structured memory fields."""

    person_id: str
    name: str
    first_seen: str
    last_seen: str
    encounter_count: int = 0
    confidence_history: List[Dict[str, Any]] = field(default_factory=list)
    average_confidence: float = 0.0
    locations_history: List[str] = field(default_factory=list)
    nearby_objects_history: List[str] = field(default_factory=list)
    relationship: str = "unknown"
    notes: List[str] = field(default_factory=list)

    @property
    def locations(self) -> List[str]:
        """Backward-compatible alias for location history."""
        return list(self.locations_history)

    @property
    def nearby_objects(self) -> List[str]:
        """Backward-compatible alias for nearby-object history."""
        return list(self.nearby_objects_history)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the profile into a JSON-safe dictionary."""
        return {
            "person_id": self.person_id,
            "name": self.name,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "encounter_count": self.encounter_count,
            "confidence_history": self.confidence_history,
            "average_confidence": self.average_confidence,
            "locations_history": self.locations_history,
            "nearby_objects_history": self.nearby_objects_history,
            "relationship": self.relationship,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], fallback_name: str) -> "PersonProfile":
        """Create a profile from legacy or future JSON payloads."""
        now = datetime.now(timezone.utc).isoformat()
        confidence_history = list(data.get("confidence_history") or [])
        if not confidence_history and isinstance(data.get("seen_count"), int):
            confidence_history = []

        encounter_count = int(data.get("encounter_count") or data.get("seen_count") or 0)
        if encounter_count < len(confidence_history):
            encounter_count = len(confidence_history)

        average_confidence = data.get("average_confidence")
        if average_confidence is None:
            average_confidence = cls._compute_average_confidence(confidence_history)

        locations_history = list(data.get("locations_history") or data.get("locations") or [])
        nearby_objects_history = list(data.get("nearby_objects_history") or data.get("nearby_objects") or [])

        return cls(
            person_id=str(data.get("person_id") or fallback_name),
            name=str(data.get("name") or fallback_name),
            first_seen=str(data.get("first_seen") or now),
            last_seen=str(data.get("last_seen") or now),
            encounter_count=encounter_count,
            confidence_history=confidence_history,
            average_confidence=float(average_confidence),
            locations_history=locations_history,
            nearby_objects_history=nearby_objects_history,
            relationship=str(data.get("relationship") or "unknown"),
            notes=list(data.get("notes") or []),
        )

    @staticmethod
    def _compute_average_confidence(confidence_history: List[Dict[str, Any]]) -> float:
        if not confidence_history:
            return 0.0
        values = [float(item.get("confidence", 0.0)) for item in confidence_history if isinstance(item, dict)]
        if not values:
            return 0.0
        return round(sum(values) / len(values), 3)


class PersonMemoryStore:
    """Persistent JSON-backed memory store with backward-compatible helpers."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        global _ACTIVE_DB_PATH
        self.db_path = db_path or _ACTIVE_DB_PATH
        if db_path is not None:
            _ACTIVE_DB_PATH = db_path

    def load_memory(self) -> Dict[str, Dict[str, Any]]:
        """Load memory from disk, validating and normalizing the JSON payload."""
        if not self.db_path.exists():
            return {}

        try:
            with self.db_path.open("r", encoding="utf-8") as handle:
                raw_data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            backup_path = self.db_path.with_suffix(".corrupt.json")
            if self.db_path.exists():
                self.db_path.replace(backup_path)
            return {}

        if not isinstance(raw_data, dict):
            return {}

        normalized: Dict[str, Dict[str, Any]] = {}
        for key, value in raw_data.items():
            if isinstance(value, dict):
                profile = PersonProfile.from_dict(value, fallback_name=str(key))
                normalized[profile.person_id] = profile.to_dict()
        return normalized

    def save_memory(self, data: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Persist memory to disk as indented JSON."""
        payload = data if data is not None else self.load_memory()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.db_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def create_person(
        self,
        name: str,
        person_id: Optional[str] = None,
        relationship: str = "unknown",
        notes: Optional[List[str]] = None,
    ) -> PersonProfile:
        """Create a new person profile if one does not already exist."""
        existing = self.get_person(name)
        if existing is not None:
            return existing

        now = datetime.now(timezone.utc).isoformat()
        profile = PersonProfile(
            person_id=person_id or name,
            name=name,
            first_seen=now,
            last_seen=now,
            relationship=relationship,
            notes=list(notes or []),
        )
        self._persist_profile(profile)
        return profile

    def update_person(
        self,
        person_name_or_id: str,
        name: Optional[str] = None,
        relationship: Optional[str] = None,
        notes: Optional[List[str]] = None,
    ) -> PersonProfile:
        """Update a profile's mutable metadata."""
        profile = self.get_person(person_name_or_id)
        if profile is None:
            profile = self.create_person(name=person_name_or_id, person_id=person_name_or_id)

        if name is not None:
            profile.name = name
        if relationship is not None:
            profile.relationship = relationship
        if notes is not None:
            profile.notes = list(notes)

        self._persist_profile(profile)
        return profile

    def get_person(self, person_name_or_id: str) -> Optional[PersonProfile]:
        """Retrieve a person by name or person ID."""
        data = self.load_memory()
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            profile = PersonProfile.from_dict(value, fallback_name=str(key))
            if profile.person_id == person_name_or_id or profile.name == person_name_or_id:
                return profile
        return None

    def get_all_people(self) -> List[PersonProfile]:
        """Return all known person profiles."""
        data = self.load_memory()
        profiles = []
        for key, value in data.items():
            if isinstance(value, dict):
                profiles.append(PersonProfile.from_dict(value, fallback_name=str(key)))
        return profiles

    def update_person_seen(
        self,
        person_name: str,
        confidence: Optional[float] = None,
        location: Optional[str] = None,
        nearby_objects: Optional[List[str]] = None,
        relationship: Optional[str] = None,
        note: Optional[str] = None,
    ) -> int:
        """Record a new encounter while preserving the previous helper behavior."""
        profile = self.get_person(person_name)
        now = datetime.now(timezone.utc).isoformat()

        if profile is None:
            profile = PersonProfile(person_id=person_name, name=person_name, first_seen=now, last_seen=now)
        else:
            profile.last_seen = now

        if not profile.first_seen:
            profile.first_seen = now

        profile.encounter_count += 1
        confidence_value = float(confidence if confidence is not None else 0.0)
        profile.confidence_history.append({"timestamp": now, "confidence": confidence_value})
        profile.average_confidence = PersonProfile._compute_average_confidence(profile.confidence_history)

        if location:
            profile.locations_history = list(dict.fromkeys(profile.locations_history + [location]))
        if nearby_objects:
            profile.nearby_objects_history = list(dict.fromkeys(profile.nearby_objects_history + [obj for obj in nearby_objects if obj]))
        if relationship:
            profile.relationship = relationship
        if note:
            profile.notes = profile.notes + [note]

        self._persist_profile(profile)
        return profile.encounter_count

    def update_location(self, person_name: str, location: str) -> PersonProfile:
        """Add a location to a person's history."""
        profile = self.get_person(person_name)
        if profile is None:
            profile = self.create_person(name=person_name, person_id=person_name)
        profile.locations_history = list(dict.fromkeys(profile.locations_history + [location]))
        self._persist_profile(profile)
        return profile

    def update_nearby_objects(self, person_name: str, nearby_objects: List[str]) -> PersonProfile:
        """Add nearby objects to a person's history."""
        profile = self.get_person(person_name)
        if profile is None:
            profile = self.create_person(name=person_name, person_id=person_name)
        filtered = [obj for obj in nearby_objects if obj]
        profile.nearby_objects_history = list(dict.fromkeys(profile.nearby_objects_history + filtered))
        self._persist_profile(profile)
        return profile

    def get_person_seen_count(self, person_name: str) -> int:
        """Backward-compatible helper for the old seen_count-based API."""
        profile = self.get_person(person_name)
        return profile.encounter_count if profile is not None else 0

    def get_average_confidence(self, person_name: str) -> float:
        """Return the average confidence for a person."""
        profile = self.get_person(person_name)
        return profile.average_confidence if profile is not None else 0.0

    def get_person_profile(self, person_name: str) -> Optional[PersonProfile]:
        """Backward-compatible alias for a person lookup."""
        return self.get_person(person_name)

    def record_encounter(
        self,
        name: str,
        confidence: float,
        location: Optional[str] = None,
        nearby_objects: Optional[List[str]] = None,
        relationship: Optional[str] = None,
        note: Optional[str] = None,
    ) -> PersonProfile:
        """Backward-compatible helper for recording a new encounter."""
        return self.update_person_seen(
            person_name=name,
            confidence=confidence,
            location=location,
            nearby_objects=nearby_objects,
            relationship=relationship,
            note=note,
        )

    def _persist_profile(self, profile: PersonProfile) -> None:
        data = self.load_memory()
        data[profile.person_id] = profile.to_dict()
        self.save_memory(data)


def create_person(
    name: str,
    person_id: Optional[str] = None,
    relationship: str = "unknown",
    notes: Optional[List[str]] = None,
) -> PersonProfile:
    """Create or retrieve a person profile through the module-level API."""
    return PersonMemoryStore(db_path=_ACTIVE_DB_PATH).create_person(name=name, person_id=person_id, relationship=relationship, notes=notes)


def update_person(
    person_name_or_id: str,
    name: Optional[str] = None,
    relationship: Optional[str] = None,
    notes: Optional[List[str]] = None,
) -> PersonProfile:
    """Update a stored person's metadata through the module-level API."""
    return PersonMemoryStore(db_path=_ACTIVE_DB_PATH).update_person(person_name_or_id=person_name_or_id, name=name, relationship=relationship, notes=notes)


def get_person(person_name_or_id: str) -> Optional[PersonProfile]:
    """Retrieve a person profile by name or person ID."""
    return PersonMemoryStore(db_path=_ACTIVE_DB_PATH).get_person(person_name_or_id)


def get_all_people() -> List[PersonProfile]:
    """Return every known person profile."""
    return PersonMemoryStore(db_path=_ACTIVE_DB_PATH).get_all_people()


def update_person_seen(
    person_name: str,
    confidence: Optional[float] = None,
    location: Optional[str] = None,
    nearby_objects: Optional[List[str]] = None,
    relationship: Optional[str] = None,
    note: Optional[str] = None,
) -> int:
    """Backward-compatible module-level wrapper for recording a new encounter."""
    return PersonMemoryStore(db_path=_ACTIVE_DB_PATH).update_person_seen(
        person_name=person_name,
        confidence=confidence,
        location=location,
        nearby_objects=nearby_objects,
        relationship=relationship,
        note=note,
    )


def get_person_seen_count(person_name: str) -> int:
    """Backward-compatible module-level wrapper for the old count API."""
    return PersonMemoryStore(db_path=_ACTIVE_DB_PATH).get_person_seen_count(person_name)


def get_average_confidence(person_name: Optional[str] = None) -> float:
    """Return the average confidence for a person or all people when no name is given."""
    if person_name is None:
        profiles = get_all_people()
        if not profiles:
            return 0.0
        return round(sum(profile.average_confidence for profile in profiles) / len(profiles), 3)
    return PersonMemoryStore(db_path=_ACTIVE_DB_PATH).get_average_confidence(person_name)


def update_location(person_name: str, location: str) -> PersonProfile:
    """Add a location to a person's history through the module-level API."""
    return PersonMemoryStore(db_path=_ACTIVE_DB_PATH).update_location(person_name, location)


def update_nearby_objects(person_name: str, nearby_objects: List[str]) -> PersonProfile:
    """Add nearby objects to a person's history through the module-level API."""
    return PersonMemoryStore(db_path=_ACTIVE_DB_PATH).update_nearby_objects(person_name, nearby_objects)


def save_memory(data: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """Persist the active memory store to disk."""
    PersonMemoryStore(db_path=_ACTIVE_DB_PATH).save_memory(data)


def load_memory() -> Dict[str, Dict[str, Any]]:
    """Load and normalize the active memory store from disk."""
    return PersonMemoryStore(db_path=_ACTIVE_DB_PATH).load_memory()


def get_person_profile(person_name: str) -> Optional[PersonProfile]:
    """Backward-compatible alias for a person lookup."""
    return get_person(person_name)
