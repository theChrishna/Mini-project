import requests
import os

models = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
}

os.makedirs("models", exist_ok=True)

for name, url in models.items():
    path = os.path.join("models", name)
    if not os.path.exists(path):
        print(f"Downloading {name}...")
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)
        print("Done.")
    else:
        print(f"{name} already exists.")
