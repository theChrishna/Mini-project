import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=mp.tasks.vision.RunningMode.VIDEO, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        
        self.options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=self.mode,
            num_hands=self.maxHands,
            min_hand_detection_confidence=self.detectionCon,
            min_hand_presence_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        
        self.detector = self.HandLandmarker.create_from_options(self.options)
        
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        self.results = None

    def findHands(self, image, frame_timestamp_ms=0, draw=True):
        if image is None:
            return image
            
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            if self.mode == mp.tasks.vision.RunningMode.VIDEO:
                self.results = self.detector.detect_for_video(mp_image, frame_timestamp_ms)
            elif self.mode == mp.tasks.vision.RunningMode.IMAGE:
                self.results = self.detector.detect(mp_image)

            if self.results and self.results.hand_landmarks:
                if draw:
                    height, width, _ = image.shape
                    for hand_landmarks in self.results.hand_landmarks:
                        pixel_landmarks = []
                        for landmark in hand_landmarks:
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)
                            pixel_landmarks.append((x, y))
                            cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
                        
                        for connection in self.HAND_CONNECTIONS:
                            start_idx, end_idx = connection
                            if start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks):
                                cv2.line(image, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (0, 0, 0), 2)
                                
            return image
        except Exception as e:
            print(f"Error processing frame: {e}")
            return image

    def findPosition(self, image, handNo=0, draw=True):
        lmList = []
        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                myHand = self.results.hand_landmarks[handNo]
                height, width, _ = image.shape
                for id, landmark in enumerate(myHand):
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        return
        
    print("Starting webcam... Press 'q' to quit.")
    
    detector = HandDetector()
    frame_timestamp_ms = 0
    pTime = 0

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            fps_prop = cap.get(cv2.CAP_PROP_FPS)
            frame_timestamp_ms += int(1000 / fps_prop) if fps_prop > 0 else 30

            image = cv2.flip(image, 1)
            image = detector.findHands(image, frame_timestamp_ms)
            
            lmList = detector.findPosition(image, draw=False)
            if len(lmList) > 4:
                print(f"Detected hands on screen. Thumb tip: {lmList[4]}")

            cTime = time.time()
            fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
            pTime = cTime

            cv2.putText(image, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow('Hand Tracking', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
