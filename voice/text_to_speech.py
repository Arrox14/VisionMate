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

import pyttsx3
import time

def speak(text: str) -> None:
    if not text:
        return

    try:
        engine = pyttsx3.init()

        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)

        engine.say(text)
        engine.runAndWait()

        time.sleep(0.5)

        engine.stop()

        del engine

    except Exception as exc:
        print(f"TTS Error: {exc}")