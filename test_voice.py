"""Simple test script for the VisionMate voice system."""

from voice.text_to_speech import speak


if __name__ == "__main__":
    try:
        print("Testing voice...")
        speak("VisionMate voice system is working.")
        print("Done.")
    except ImportError as exc:
        print(f"Error: pyttsx3 is not installed or could not be imported. {exc}")
    except Exception as exc:
        print(f"Error: Unable to run voice test: {exc}")
