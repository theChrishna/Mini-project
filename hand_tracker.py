import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config
import os


class HandTracker:
    """
    Wraps MediaPipe GestureRecognizer which provides hand landmarks
    AND gesture classification in a single model pass — more efficient
    than running HandLandmarker + a separate gesture model.
    """

    # Map MediaPipe category names → human-readable word signs
    GESTURE_MAP = {
        "Thumb_Up":    "YES",
        "Thumb_Down":  "NO",
        "Open_Palm":   "STOP",
        "Pointing_Up": "ATTENTION",
        "Victory":     "PEACE",
        "ILoveYou":    "I LOVE YOU",
        "Closed_Fist": "HOLD ON",
    }

    def __init__(self):
        model_path = os.path.join("models", "gesture_recognizer.task")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.GestureRecognizer.create_from_options(options)

    def process_frame(self, frame):
        """Run detection on a BGR frame. Returns a GestureRecognizerResult."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.recognize(mp_image)

    def get_gesture_info(self, results):
        """
        Returns (word_sign, confidence) for the primary hand's top gesture,
        or (None, 0.0) if no confident gesture is detected.
        Filters out the 'None' category (no gesture) and low-confidence hits.
        """
        if results.gestures:
            top = results.gestures[0][0]
            if top.category_name != "None" and top.score >= 0.60:
                word = self.GESTURE_MAP.get(top.category_name)
                return word, top.score
        return None, 0.0

    def draw_landmarks(self, frame, results):
        if not results.hand_landmarks:
            return frame

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),           # index
            (0, 9), (9, 10), (10, 11), (11, 12),      # middle
            (0, 13), (13, 14), (14, 15), (15, 16),    # ring
            (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
            (5, 9), (9, 13), (13, 17),                 # palm cross
        ]

        h, w = frame.shape[:2]
        for hand_landmarks in results.hand_landmarks:
            for lm in hand_landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (255, 255, 255), -1)
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (100, 100, 100), -1)
            for start, end in connections:
                p1 = hand_landmarks[start]
                p2 = hand_landmarks[end]
                cv2.line(frame,
                         (int(p1.x * w), int(p1.y * h)),
                         (int(p2.x * w), int(p2.y * h)),
                         (200, 200, 200), 2)
        return frame

    def get_landmarks(self, results):
        """Returns list-of-lists: [[( x, y, z ), ...], ...] one per hand."""
        if not results.hand_landmarks:
            return []
        return [
            [(lm.x, lm.y, lm.z) for lm in hand]
            for hand in results.hand_landmarks
        ]
