"""Text-to-speech utilities for VisionMate.

This module provides a simple wrapper around the ``pyttsx3`` engine so other
parts of the application can speak notifications, descriptions, and detected
person names without needing to manage engine setup directly.
"""

from __future__ import annotations

from typing import Optional

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


_ENGINE: Optional[object] = None


def _get_engine() -> Optional[object]:
    """Create and cache a speech engine if the dependency is available."""
    global _ENGINE

    if _ENGINE is not None:
        return _ENGINE

    if pyttsx3 is None:
        print("Warning: pyttsx3 is not installed. Speech output is unavailable.")
        return None

    try:
        _ENGINE = pyttsx3.init()
        return _ENGINE
    except Exception as exc:
        print(f"Warning: Unable to initialize text-to-speech engine: {exc}")
        return None


def speak(text: str) -> None:
    """Speak the provided text using the configured speech engine.

    Args:
        text: The text to be spoken aloud.

    Notes:
        - The function is safe to call even if the engine is unavailable.
        - It does not raise exceptions for initialization/runtime issues;
          instead, it prints a warning and returns.
    """
    if not text or not text.strip():
        return

    engine = _get_engine()
    if engine is None:
        return

    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as exc:
        print(f"Warning: Unable to speak text: {exc}")
