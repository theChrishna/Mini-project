import cv2
import pickle
import numpy as np
from hand_tracking import HandDetector

# 1. Load the trained "Brain"
with open('model.p', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

print("Translator is LIVE. Press 'q' to exit.")

frame_timestamp_ms = 0

while True:
    success, img = cap.read()
    if not success: break
    
    fps_prop = cap.get(cv2.CAP_PROP_FPS)
    frame_timestamp_ms += int(1000 / fps_prop) if fps_prop > 0 else 30
    
    img = cv2.flip(img, 1) # Mirror the image for natural movement
    
    # Extract landmarks (ensure timestamp is handled as per your module)
    img = detector.findHands(img, frame_timestamp_ms, draw=True)
    lmList = detector.findPosition(img, draw=False)
    
    if lmList:
        h, w, _ = img.shape
        # Normalize coordinates exactly as you did in data collection
        x_coords = [lm[1]/w for lm in lmList]
        y_coords = [lm[2]/h for lm in lmList]
        
        # Prepare data for the model
        features = np.array(x_coords + y_coords).reshape(1, -1)
        
        # 2. Predict the letter
        prediction = model.predict(features)
        letter = str(prediction[0])
        
        # 3. Visual Feedback
        # Draw a box and the predicted letter
        cv2.rectangle(img, (20, 20), (150, 120), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, letter, (45, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

    cv2.imshow("Real-Time Sign Translator", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()