import cv2
import pickle
import numpy as np
import time
from flask import Flask, render_template, Response
from hand_tracking import HandDetector

app = Flask(__name__)  # Initialize the Flask application

# 1. Load the trained model
with open('model.p', 'rb') as f:
    model = pickle.load(f)

detector = HandDetector(maxHands=1)

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    frame_timestamp_ms = 0
    
    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Error: Failed to read frame from camera.")
                break
                
            frame_timestamp_ms = time.time_ns() // 1_000_000
            
            # Mirror the image for natural movement
            img = cv2.flip(img, 1)
            
            # Process hands
            img = detector.findHands(img, frame_timestamp_ms, draw=True)
            lmList = detector.findPosition(img, draw=False)
            
            if lmList:
                h, w, _ = img.shape
                # Normalize and extract coordinates directly
                features = np.array([lm[1]/w for lm in lmList] + [lm[2]/h for lm in lmList]).reshape(1, -1)
                
                # Predict the gesture
                prediction = model.predict(features)
                letter = str(prediction[0])
                
                # Draw the UI on the frame
                cv2.rectangle(img, (20, 20), (150, 120), (255, 99, 71), cv2.FILLED)  # Tomato color
                cv2.putText(img, letter, (45, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
                
            # Format the image into JPEG for web streaming
            ret, buffer = cv2.imencode('.jpg', img)
            if not ret:
                continue
                
            frame = buffer.tobytes()
            
            # Yield the multipart byte stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        # Ensure the camera is released even if an error occurs
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(debug=True, port=5000)
