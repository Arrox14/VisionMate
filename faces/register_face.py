"""
Face Registration Module for VisionMate

This module handles the registration of new faces by capturing images
from the webcam and saving them to the known_faces directory.
"""

import cv2
import os
from pathlib import Path


def get_person_name() -> str:
    """
    Get the person's name from user input with validation.
    
    Returns:
        str: The validated person's name.
        
    Raises:
        ValueError: If the name is empty or invalid.
    """
    while True:
        name = input("Enter name: ").strip()
        
        if not name:
            print("Error: Name cannot be empty. Please try again.")
            continue
            
        if len(name) < 2:
            print("Error: Name must be at least 2 characters long. Please try again.")
            continue
            
        if not name.replace(" ", "").replace("-", "").isalnum():
            print("Error: Name can only contain alphanumeric characters, spaces, and hyphens.")
            continue
            
        return name


def initialize_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    """
    Initialize the webcam with error handling.
    
    Args:
        camera_index: The index of the camera device (default: 0).
        
    Returns:
        cv2.VideoCapture: The initialized camera object.
        
    Raises:
        RuntimeError: If the webcam cannot be accessed.
    """
    camera = cv2.VideoCapture(camera_index)
    
    # Allow time for camera to initialize
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not camera.isOpened():
        raise RuntimeError(
            "Error: Could not access webcam. Please ensure:\n"
            "  - Webcam is connected\n"
            "  - No other application is using the webcam\n"
            "  - Camera permissions are granted"
        )
    
    return camera


def create_known_faces_directory() -> Path:
    """
    Create the known_faces directory if it doesn't exist.
    
    Returns:
        Path: The path to the known_faces directory.
    """
    known_faces_dir = Path("known_faces")
    known_faces_dir.mkdir(exist_ok=True)
    return known_faces_dir


def register_face(name: str) -> None:
    """
    Main function to register a face. Captures from webcam and saves the image.
    
    Args:
        name: The name of the person to register.
    """
    camera = None
    frame_count = 0
    
    try:
        # Initialize webcam
        print("\nInitializing webcam...")
        camera = initialize_webcam()
        
        # Set camera resolution for better quality
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✓ Webcam initialized successfully")
        print("\nInstructions:")
        print("  - Press 'S' to save face")
        print("  - Press 'Q' to quit")
        print("\nStarting live feed...\n")        
        while True:
            success, frame = camera.read()
            
            if not success:
                print("Error: Failed to read frame from webcam.")
                break
            
            # Mirror the frame horizontally for better user experience
            frame = cv2.flip(frame, 1)
            
            # Create a copy for display with text overlay
            display_frame = frame.copy()
            
            # Add instructions text on the frame
            cv2.putText(
                display_frame,
                "Press 'S' to save | 'Q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                display_frame,
                f"Name: {name}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                display_frame,
                f"Frames captured: {frame_count}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display the frame
            cv2.imshow("Face Registration - VisionMate", display_frame)
            
            # Wait for key press (1ms timeout)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord('S'):
                # Save face
                try:
                    known_faces_dir = create_known_faces_directory()
                    
                    # Create file path
                    face_path = known_faces_dir / f"{name}_{frame_count}.jpg"
                    
                    # Save the original frame (not the display frame)
                    cv2.imwrite(str(face_path), frame)
                    
                    frame_count += 1
                    print(f"✓ Face registered successfully: {face_path}")                    
                except IOError as e:
                    print(f"Error: Failed to save face image. Details: {e}")
                except Exception as e:
                    print(f"Error: Unexpected error while saving. Details: {e}")
            
            elif key == ord('q') or key == ord('Q'):
                # Quit
                print("\nExiting face registration...")
                break
    
    except RuntimeError as e:
        print(f"\n{e}")
    
    except KeyboardInterrupt:
        print("\n\nFace registration cancelled by user.")
    
    except Exception as e:
        print(f"Error: An unexpected error occurred. Details: {e}")
    
    finally:
        # Clean up resources
        if camera is not None:
            camera.release()
        
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            print(f"\n✓ Successfully registered {frame_count} face image(s) for: {name}")
        else:
            print("\nNo faces were registered.")


def main() -> None:
    """
    Entry point for the face registration module.
    """
    print("=" * 50)
    print("VisionMate - Face Registration Module")
    print("=" * 50)
    
    try:
        # Get person's name
        name = get_person_name()
        
        # Register face
        register_face(name)
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
