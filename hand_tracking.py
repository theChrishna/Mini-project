import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(
        self,
        mode=mp.tasks.vision.RunningMode.VIDEO,
        maxHands=2,
        detectionCon=0.5,
        trackCon=0.5,
        model_path="hand_landmarker.task",
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=self.mode,
            num_hands=self.maxHands,
            min_hand_detection_confidence=self.detectionCon,
            min_hand_presence_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )

        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(
            self.options
        )

        self.results = None

        # Keep your connections (unchanged)
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]

    def findHands(self, image, frame_timestamp_ms=0, draw=True):
        if image is None:
            return image

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detection (same logic, cleaner)
            if self.mode == mp.tasks.vision.RunningMode.VIDEO:
                if frame_timestamp_ms == 0:
                    frame_timestamp_ms = int(time.time() * 1000)
                self.results = self.detector.detect_for_video(
                    mp_image, frame_timestamp_ms
                )
            else:
                self.results = self.detector.detect(mp_image)

            # Drawing (unchanged behavior)
            if self.results and self.results.hand_landmarks and draw:
                h, w, _ = image.shape

                for hand_landmarks in self.results.hand_landmarks:
                    pixel_landmarks = []

                    for lm in hand_landmarks:
                        x, y = int(lm.x * w), int(lm.y * h)
                        pixel_landmarks.append((x, y))
                        cv2.circle(image, (x, y), 5, (255, 255, 255), -1)

                    for start, end in self.HAND_CONNECTIONS:
                        if start < len(pixel_landmarks) and end < len(pixel_landmarks):
                            cv2.line(
                                image,
                                pixel_landmarks[start],
                                pixel_landmarks[end],
                                (0, 0, 0),
                                2,
                            )

            return image

        except Exception as e:
            print(f"Error: {e}")
            return image

    def findPosition(self, image, handNo=0, draw=True):
        lmList = []

        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                h, w, _ = image.shape
                myHand = self.results.hand_landmarks[handNo]

                for idx, lm in enumerate(myHand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([idx, cx, cy])

                    if draw:
                        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList

    def close(self):
        """Safe cleanup (optional, won't break anything)"""
        if self.detector:
            self.detector.close()