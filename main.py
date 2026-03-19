import cv2
import pickle
import numpy as np
import time
from hand_tracking import HandDetector


with open('model.p', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

print("Starting Camera...")
print("Translator is LIVE. Press 'q' to exit.")

frame_timestamp_ms = 0
pTime = 0

while True:
    success, img = cap.read()
    if not success: break
    
    frame_timestamp_ms = time.time_ns() // 1_000_000
    
    img = cv2.flip(img, 1)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    
    img = detector.findHands(img, frame_timestamp_ms, draw=True)
    lmList = detector.findPosition(img, draw=False)
    
    if lmList:
        h, w, _ = img.shape
     
        features = np.array([lm[1]/w for lm in lmList] + [lm[2]/h for lm in lmList]).reshape(1, -1)
        
        prediction = model.predict(features)
        letter = str(prediction[0])
        
        
        cv2.rectangle(img, (20, 20), (150, 120), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, letter, (45, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Real-Time Sign Translator", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()