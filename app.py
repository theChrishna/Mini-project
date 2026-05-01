from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera

app = Flask(__name__)

# Global camera object to maintain state across requests
video_camera = None

def get_camera():
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera()
    return video_camera

@app.route('/')
def index():
    """Render the main UI page."""
    return render_template('index.html')

import time

def gen(camera):
    """Video streaming generator function yielding multipart JPEG frames."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(get_camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/toggle', methods=['POST'])
def toggle_feature():
    feature = request.json.get('feature')
    if feature:
        cam = get_camera()
        states = cam.toggle_feature(feature)
        return jsonify({"status": "success", "state": states[feature]})
    return jsonify({"status": "error", "message": "Invalid feature"}), 400

@app.route('/api/record', methods=['POST'])
def record_custom_sign():
    word = request.json.get('word')
    if word:
        get_camera().start_custom_recording(word)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "No word provided"}), 400

@app.route('/api/clear', methods=['POST'])
def clear_buffer():
    get_camera().clear_buffer()
    return jsonify({"status": "success"})

@app.route('/api/speak', methods=['POST'])
def speak_manual():
    get_camera().speak_manual()
    return jsonify({"status": "success"})

if __name__ == '__main__':
    # Threaded ensures multiple requests can be handled (e.g. streaming + API calls)
    app.run(host='0.0.0.0', port=5000, threaded=True)
