from ultralytics import YOLO
import cv2


def run_object_detection(camera_index: int = 0) -> None:
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(camera_index)

    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("VisionMate - Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
