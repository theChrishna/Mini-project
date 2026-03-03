import cv2
import mediapipe as mp
import time

# Use the Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand connections for drawing (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index finger
    (9, 10), (10, 11), (11, 12), # Middle finger
    (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17) # Palm
]

# This callback receives the results from the async process
def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # We will process results in the main thread instead of in a callback for simplicity
    pass

def main():
    # Initialize the Hand Landmarker
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    detector = HandLandmarker.create_from_options(options)

    # Open webcam
    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press 'q' to quit.")

    # Variables for calculating FPS
    frame_timestamp_ms = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_timestamp_ms += int(1000 / cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

        # Flip the image horizontally for a selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Process the image and detect hands
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        # Draw the hand landmarks on the image.
        if detection_result and detection_result.hand_landmarks:
            height, width, _ = image.shape
            for hand_landmarks in detection_result.hand_landmarks:
                # Convert normalized coordinates to pixel coordinates
                pixel_landmarks = []
                for landmark in hand_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    pixel_landmarks.append((x, y))
                    # Draw points (White)
                    cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
                
                # Draw connections (Black)
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks):
                        cv2.line(image, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (0, 0, 0), 2)

        # Show the image
        cv2.imshow('Hand Tracking', image)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Clean up
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
