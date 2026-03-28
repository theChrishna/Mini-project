# Sign Language / Gesture Translator

A real-time hand gesture recognition and sign language translation system. This project uses your webcam to track hand movements using **MediaPipe**, collects custom dataset features, and classifies the gestures in real time using a **Random Forest Classifier** built with scikit-learn.

## Project Structure

- **`hand_tracking.py`**: A helper module that wraps around the MediaPipe Tasks API (`hand_landmarker.task`) for accurate hand tracking. It extracts 21 hand landmarks and visualizes them on the frame. *Recently refactored for improved error handling and clean, comment-free architecture.*
- **`data_collection.py`**: A script used to build your custom gesture dataset. It captures webcam frames, normalizes the 21 `(x, y)` hand landmarks, and saves them to a CSV file (`hand_data.csv`).
- **`check_model.py` / `train_model.py`**: Reads your collected data, splits it into training and testing sets (using stratification), and trains a Random Forest Classifier. The trained model is saved as a Python pickle file (`model.p`).
- **`main.py`**: The real-time translation application. It loads your trained model, processes live webcam video to extract normalized hand landmarks, and predicts the specific gesture/sign, rendering the result directly on the screen along with a live FPS counter.

## Prerequisites

Ensure you have Python 3.8+ installed on your system. You can install all required dependencies using `pip`:

```bash
pip install opencv-python mediapipe pandas scikit-learn numpy
```

> **Note:** The MediaPipe Hand Landmarker relies on the `hand_landmarker.task` model file, which must be present in your project directory.

## How to use

### 1. Collect Data for Custom Gestures
Open `data_collection.py` and modify the `label = "..."` variable near the top (e.g., set it to "A", "B", "Hello").
Run the script:
```bash
python data_collection.py
```
- Show the corresponding gesture to the camera.
- Press **`s`** to save a frame to your dataset (`hand_data.csv`). Save multiple frames at various angles and distances for better accuracy!
- Press **`q`** to quit when you are done. *Repeat this step for every new gesture you want to add to your dataset.*

### 2. Train the Machine Learning Model
Once you have collected sufficient data for all your target gestures, train the Random Forest Classifier by running:
```bash
python train_model.py
```
This script will output the accuracy of your model on the test data split and save the compiled parameters into `model.p`.

### 3. Run the Real-Time Web Translator
Start the live prediction Flask pipeline:
```bash
python app.py
```
After the server boots, visit `http://localhost:5000` in your web browser. The webcam will open inside a beautiful UI and the program will begin recognizing and displaying your gestures in real-time.

> **Latest Updates:** 
> - **FPS Optimization:** The `app.py` script runs at a smoother 640x480 resolution for drastically improved track-speed.
> - **Real-time FPS Telecast:** The video feed securely prints the rendering frames-per-second and live clock to directly benchmark system speed.
> - **Catchy UI Redesign:** Fully revamped the web UI using glassmorphism, dynamic gradients, and an animated pre-detection loading screen.
> - **Light/Dark Toggle:** Added an interactive slider toggle allowing users to instantly swap between sleek dark mode and bright light mode themes.
> - **Interactive Experiences:** Introduced a sliding AI chatbot interface and a localized Community Discussion chat section for maximum user interactions.
> - **Advanced Feedback Reporting:** Implemented an interactive prompt that transmits highly detailed review payloads about false detections directly to the backend maintainer.

## How It Works

1. **Hand Tracking:** MediaPipe identifies 21 spatial points representing the joints and fingertips of a hand.
2. **Normalization:** The `(x, y)` pixel coordinates are normalized against the width and height of the image frame so that gestures are recognized accurately regardless of how close or far your hand is from the camera.
3. **Classification:** A pre-trained Random Forest model takes the flattened array of those formatted 42 coordinate features and maps it to the learned gesture label.