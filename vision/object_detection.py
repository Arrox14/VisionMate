import time
import traceback

import cv2
from ultralytics import YOLO

from agent.scene_narrator import generate_scene_description
from voice.text_to_speech import speak

SPEECH_COOLDOWN = 5
STABLE_SCENE_SECONDS = 2
WARMUP_ATTEMPTS = 10
WARMUP_DELAY_SECONDS = 0.1


def _initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """Initialize and validate the webcam for Windows using DirectShow."""
    print("Initializing webcam...")
    camera = cv2.VideoCapture(
        camera_index,
        cv2.CAP_DSHOW
    )
    print("Camera backend:", camera.getBackendName())
    print("Camera opened:", camera.isOpened())

    if not camera.isOpened():
        backend_name = "unknown"
        try:
            backend_name = camera.getBackendName()
        except Exception:
            pass
        raise RuntimeError(
            f"Camera open failed for index {camera_index} using CAP_DSHOW. "
            f"Backend: {backend_name}. Please ensure the webcam is connected "
            "and not being used by another application."
        )

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(WARMUP_ATTEMPTS):
        success, _ = camera.read()
        if success:
            return camera
        time.sleep(WARMUP_DELAY_SECONDS)

    camera.release()
    raise RuntimeError(
        "Camera opened but failed to deliver frames during warmup. "
        "Please ensure webcam drivers and permissions are correct."
    )


def run_object_detection(camera_index: int = 0) -> None:
    model = None
    camera = None

    try:
        print("Loading YOLO model...")
        model = YOLO("yolov8n.pt")
        print("YOLO model loaded successfully")

        camera = _initialize_camera(camera_index)

        print("\nStarting object detection...")
        print("Press 'Q' to quit\n")

        last_detected_objects: tuple[str, ...] | None = None
        last_spoken_scene: tuple[str, ...] | None = None
        last_speech_time = 0
        pending_scene: tuple[str, ...] | None = None
        pending_scene_since = 0.0

        while True:
            success, frame = camera.read()

            if not success:
                print("Error: Failed to read frame from webcam.")
                break

            results = model(frame, verbose=False)

            detected_objects = [
                model.names[int(class_id)]
                for class_id in results[0].boxes.cls.tolist()
            ]
            current_detected_objects = tuple(
                sorted({obj for obj in detected_objects if obj})
            )

            if current_detected_objects != last_detected_objects:
                print("Scene changed")
                if current_detected_objects:
                    if current_detected_objects != pending_scene:
                        pending_scene = current_detected_objects
                        pending_scene_since = time.time()
                    elif time.time() - pending_scene_since >= STABLE_SCENE_SECONDS:
                        print("Stable scene reached")
                        description = generate_scene_description(list(current_detected_objects))
                        print(description)

                        if current_detected_objects != last_spoken_scene:
                            print("Calling speak()")
                            print("Speaking narration...")
                            speak(description)
                            last_spoken_scene = current_detected_objects
                            last_speech_time = time.time()
                        elif time.time() - last_speech_time >= SPEECH_COOLDOWN:
                            print("Calling speak()")
                            print("Speaking narration...")
                            speak(description)
                            last_speech_time = time.time()

                        last_detected_objects = current_detected_objects
                        pending_scene = None
                else:
                    last_detected_objects = current_detected_objects
                    pending_scene = None
            else:
                pending_scene = None

            annotated_frame = results[0].plot()

            cv2.imshow("VisionMate - Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nExiting object detection...")
                break

    except RuntimeError as e:
        print(f"\n{e}")

    except KeyboardInterrupt:
        print("\n\nObject detection interrupted by user.")

    except Exception:
        print("Error: An unexpected error occurred. Details:")
        traceback.print_exc()

    finally:
        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()
        print("\n✓ Object detection module closed")


if __name__ == "__main__":
    run_object_detection()
