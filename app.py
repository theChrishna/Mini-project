import cv2
import pickle
import numpy as np
import time
from flask import Flask, render_template, Response, request
from hand_tracking import HandDetector

app = Flask(__name__)  # Initialize the Flask application
current_prediction = ""

# 1. Load the trained model
with open("model.p", "rb") as f:
    model = pickle.load(f)

detector = HandDetector(maxHands=1)


def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    frame_timestamp_ms = 0
    prev_time = 0

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
                features = np.array(
                    [lm[1] / w for lm in lmList] + [lm[2] / h for lm in lmList]
                ).reshape(1, -1)

                # Predict the gesture
                prediction = model.predict(features)
                letter = str(prediction[0])

                global current_prediction
                current_prediction = letter

                # Draw the UI on the frame
                cv2.rectangle(
                    img, (20, 20), (150, 120), (255, 99, 71), cv2.FILLED
                )  # Tomato color
                cv2.putText(
                    img,
                    letter,
                    (45, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (255, 255, 255),
                    5,
                )

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            
            # Put FPS on screen
            cv2.putText(img, f"FPS: {int(fps)} 👽", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Add live time telecast
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cv2.putText(
                img,
                current_time,
                (10, img.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Format the image into JPEG for web streaming
            ret, buffer = cv2.imencode(".jpg", img)
            if not ret:
                continue

            frame = buffer.tobytes()

            # Yield the multipart byte stream
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    finally:
        # Ensure the camera is released even if an error occurs
        cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route('/report', methods=['POST'])
def report_false_detection():
    data = request.json or {}
    message = data.get('message', 'No details provided by user.')
    
    # Notify the owner via terminal log
    print("\n" + "="*60)
    print("🚨 ALERT: False detection reported by user at front-end! 🚨")
    print(f"📝 User Feedback: {message}")
    print("="*60 + "\n")
    return {"status": "success", "message": "Report sent"}

@app.route('/prediction')
def get_prediction():
    return {"letter": current_prediction}

if __name__ == "__main__":
    print("Starting Flask server on port 5000...")
    app.run(debug=True, port=5000)
    # End of application
