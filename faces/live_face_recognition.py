"""
Live Face Recognition Module for VisionMate

This module performs real-time face recognition using InsightFace embeddings.
It loads registered face images, generates embeddings, and compares detected faces
against the registered faces in real-time using cosine similarity.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import traceback
import warnings
import time

from database.person_memory import get_person_seen_count, update_person_seen
from voice.text_to_speech import speak

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError(
        "InsightFace is not installed. Please install it using:\n"
        "pip install insightface"
    )


# Configuration
CONFIDENCE_THRESHOLD = 0.6  # Minimum cosine similarity to recognize a face
KNOWN_FACES_DIR = Path("known_faces")
MODEL_NAME = "buffalo_l"  # InsightFace model
COOLDOWN_SECONDS = 30
VOICE_COOLDOWN = 10


def load_known_faces() -> Tuple[Dict[str, np.ndarray], FaceAnalysis]:
    """
    Load all known faces from the known_faces directory and generate embeddings.
    
    Supports multiple images per person. Extracts person name from filename
    (e.g., "Arro_1.jpg" -> "Arro") and averages embeddings for the same person.
    
    Returns:
        Tuple[Dict[str, np.ndarray], FaceAnalysis]: 
            - Dictionary mapping person names to averaged embeddings
            - FaceAnalysis model instance
            
    Raises:
        RuntimeError: If FaceAnalysis cannot be initialized or no faces found.
    """
    print("\n" + "=" * 60)
    print("Loading Face Recognition Model...")
    print("=" * 60)
    
    try:
        # Initialize FaceAnalysis model
        face_analyzer = FaceAnalysis(
            name=MODEL_NAME,
            root=".",
            providers=["CPUExecutionProvider"]
        )
        face_analyzer.prepare(ctx_id=-1, det_size=(640, 480))
        print("✓ FaceAnalysis model loaded successfully")
        
    except Exception as e:
        raise RuntimeError(
            f"Error: Failed to initialize FaceAnalysis model. Details: {e}\n"
            "Please ensure InsightFace is properly installed."
        )
    
    # Check if known_faces directory exists
    if not KNOWN_FACES_DIR.exists():
        raise RuntimeError(
            f"Error: '{KNOWN_FACES_DIR}' directory not found.\n"
            "Please run face registration first to create known faces."
        )
    
    # Get all JPG files from known_faces directory
    face_images = list(KNOWN_FACES_DIR.glob("*.jpg"))
    
    if not face_images:
        raise RuntimeError(
            f"Error: No face images found in '{KNOWN_FACES_DIR}' directory.\n"
            "Please run face registration first."
        )
    
    print(f"\n✓ Found {len(face_images)} face image(s)")
    
    # Dictionary to store embeddings grouped by person name
    person_embeddings = defaultdict(list)
    
    print("\nProcessing face images...")
    processed_count = 0
    failed_count = 0
    
    # Process each face image
    for face_image_path in face_images:
        try:
            # Extract person name from filename
            # "Arro_1.jpg" -> "Arro"
            # "Arro.jpg" -> "Arro"
            stem = face_image_path.stem
            person_name = stem.rsplit('_', 1)[0]  # Split on last underscore
            
            # Read image
            image = cv2.imread(str(face_image_path))
            
            if image is None:
                print(f"  ⚠ Warning: Could not read image {face_image_path.name}")
                failed_count += 1
                continue
            
            # Convert BGR to RGB for InsightFace
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces and generate embeddings
            faces = face_analyzer.get(image_rgb)
            
            if not faces:
                print(f"  ⚠ Warning: No face detected in {face_image_path.name}")
                failed_count += 1
                continue
            
            # Use the first (usually largest) face detected
            face = faces[0]
            embedding = face.embedding
            
            # Store embedding for this person
            person_embeddings[person_name].append(embedding)
            processed_count += 1
            print(f"  ✓ {face_image_path.name} -> {person_name}")
            
        except Exception as e:
            print(f"  ✗ Error processing {face_image_path.name}: {e}")
            failed_count += 1
            continue
    
    # Average embeddings for each person
    print("\nAveraging embeddings per person...")
    known_faces_db = {}
    
    for person_name, embeddings_list in person_embeddings.items():
        if embeddings_list:
            # Calculate mean embedding for better recognition
            averaged_embedding = np.mean(embeddings_list, axis=0)
            # Normalize the embedding
            averaged_embedding = averaged_embedding / np.linalg.norm(averaged_embedding)
            known_faces_db[person_name] = averaged_embedding
            print(f"  ✓ {person_name}: {len(embeddings_list)} image(s)")
    
    print("\n" + "=" * 60)
    print(f"Successfully loaded {processed_count} face(s) for {len(known_faces_db)} person(s)")
    if failed_count > 0:
        print(f"Failed to process {failed_count} image(s)")
    print("=" * 60 + "\n")
    
    return known_faces_db, face_analyzer


def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First face embedding.
        embedding2: Second face embedding.
        
    Returns:
        float: Cosine similarity score (0-1).
    """
    # Normalize embeddings
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2)
    
    return float(similarity)


def find_best_match(
    embedding: np.ndarray,
    known_faces_db: Dict[str, np.ndarray],
    threshold: float = CONFIDENCE_THRESHOLD
) -> Tuple[str, float]:
    """
    Find the best matching person for a given embedding.
    
    Args:
        embedding: Embedding of detected face.
        known_faces_db: Dictionary of known person embeddings.
        threshold: Minimum similarity threshold to recognize.
        
    Returns:
        Tuple[str, float]: (person_name, similarity_score)
    """
    best_match = "Unknown"
    best_similarity = 0.0
    
    for person_name, known_embedding in known_faces_db.items():
        similarity = calculate_cosine_similarity(embedding, known_embedding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = person_name
    
    # Return "Unknown" if similarity below threshold
    if best_similarity < threshold:
        best_match = "Unknown"
    
    return best_match, best_similarity


def recognize_faces(
    known_faces_db: Dict[str, np.ndarray],
    face_analyzer: FaceAnalysis
) -> None:
    """
    Main function for real-time face recognition.
    
    Captures video from webcam, detects faces, and recognizes them
    against the known faces database.
    
    Args:
        known_faces_db: Dictionary of known person embeddings.
        face_analyzer: FaceAnalysis model instance.
    """
    camera = None
    
    try:
        # Initialize webcam
        print("Initializing webcam...")
        camera = cv2.VideoCapture(0)
        
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        if not camera.isOpened():
            raise RuntimeError(
                "Error: Could not access webcam. Please ensure:\n"
                "  - Webcam is connected\n"
                "  - No other application is using the webcam"
            )
        
        print("✓ Webcam initialized successfully")
        print("\nStarting live face recognition...")
        print("Press 'Q' to quit\n")
        
        frame_count = 0
        last_memory_update = {}
        last_voice_update = {}
        print("Entering recognition loop")
        
        while True:
            success, frame = camera.read()
            
            if not success:
                print("Error: Failed to read frame from webcam.")
                break
            
            print("Frame captured")
            frame_count += 1
            
            # Mirror frame for natural user experience
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Convert BGR to RGB for InsightFace
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # Detect faces in the frame
                faces = face_analyzer.get(frame_rgb)
                print(f"Detected {len(faces)} faces")
                
                # Process each detected face
                for face in faces:
                    # Get face bounding box
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # Get face embedding
                    embedding = face.embedding
                    
                    # Find best match
                    person_name, similarity = find_best_match(embedding, known_faces_db)
                    print(f"Match: {person_name} Similarity: {similarity:.2f}")
                    
                    # Color code: green for recognized, red for unknown
                    if person_name != "Unknown":
                        color = (0, 255, 0)  # Green
                        label = f"{person_name}: {similarity:.2f}"

                        now = time.time()
                        last_seen = last_memory_update.get(person_name, 0)
                        if now - last_seen >= COOLDOWN_SECONDS:
                            update_person_seen(
                                person_name,
                                confidence=float(similarity),
                                location="live_scene",
                                nearby_objects=[obj for obj in []],
                                relationship="unknown",
                                note=f"Recognized in live camera feed with similarity {similarity:.2f}",
                            )
                            last_memory_update[person_name] = now

                        should_speak = (
                            person_name not in last_voice_update or
                            now - last_voice_update[person_name] >= VOICE_COOLDOWN
                        )

                        if should_speak:
                            print(f"Speaking: {person_name}")
                            speak(f"{person_name} is in front of you.")
                            last_voice_update[person_name] = now

                        seen_count = get_person_seen_count(person_name)
                        memory_label = f"Seen: {seen_count}"
                    else:
                        color = (0, 0, 255)  # Red
                        label = f"Unknown: {similarity:.2f}"
                        memory_label = None
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    label_size, _ = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        2
                    )
                    cv2.rectangle(
                        display_frame,
                        (x1, y1 - 25),
                        (x1 + label_size[0], y1),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        display_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    if memory_label is not None:
                        memory_size, _ = cv2.getTextSize(
                            memory_label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            1
                        )
                        memory_y = y1 - 45
                        cv2.rectangle(
                            display_frame,
                            (x1, memory_y - 12),
                            (x1 + memory_size[0], memory_y + 6),
                            (0, 0, 0),
                            -1
                        )
                        cv2.putText(
                            display_frame,
                            memory_label,
                            (x1, memory_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )
                
                # Display info on frame
                info_text = f"Faces detected: {len(faces)} | Frame: {frame_count}"
                cv2.putText(
                    display_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Display threshold info
                threshold_text = f"Threshold: {CONFIDENCE_THRESHOLD:.2f}"
                cv2.putText(
                    display_frame,
                    threshold_text,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 0),
                    1
                )
                
            except Exception:
                print("Error during face detection:")
                traceback.print_exc()
            
            # Display frame
            cv2.imshow("VisionMate - Live Face Recognition", display_frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Q pressed")
                print("\nExiting face recognition...")
                break
    
    except RuntimeError as e:
        print(f"\n{e}")
    
    except KeyboardInterrupt:
        print("\n\nFace recognition interrupted by user.")
    
    except Exception:
        print("Error: An unexpected error occurred. Details:")
        traceback.print_exc()
    
    finally:
        # Clean up resources
        if camera is not None:
            camera.release()
        
        cv2.destroyAllWindows()
        print("\n✓ Face recognition module closed")


def main() -> None:
    """
    Entry point for the live face recognition module.
    """
    print("\n" + "=" * 60)
    print("VisionMate - Live Face Recognition")
    print("=" * 60)
    
    try:
        # Load known faces
        known_faces_db, face_analyzer = load_known_faces()
        
        # Start real-time recognition
        recognize_faces(known_faces_db, face_analyzer)
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("  1. InsightFace is installed: pip install insightface")
        print("  2. Face images exist in 'known_faces/' directory")
        print("  3. Webcam is connected and accessible")


if __name__ == "__main__":
    main()
